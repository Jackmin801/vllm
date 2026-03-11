# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the fused MoE + LoRA Triton kernel.

Tests both the standalone kernel (invoke_fused_moe_lora_kernel) and the
TritonExpertsWithLoRA integration against a PyTorch reference.
"""

import pytest
import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    invoke_fused_moe_lora_kernel,
)
from vllm.lora.ops.triton_ops.moe_lora_align import (
    moe_lora_align_block_size_fused,
)
from vllm.triton_utils import tl


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


def reference_moe_with_lora(
    hidden_states: torch.Tensor,  # (num_tokens, K)
    w: torch.Tensor,  # (E, N_total, K)
    lora_a: torch.Tensor,  # (NUM_SLICES, max_loras, E, rank, K)
    lora_b: torch.Tensor,  # (NUM_SLICES, max_loras, E, N_per_slice, rank)
    topk_weights: torch.Tensor | None,  # (num_tokens, top_k)
    topk_ids: torch.Tensor,  # (num_tokens, top_k)
    lora_ids: torch.Tensor,  # (num_tokens,)
    mul_routed_weight: bool,
    num_slices: int,
) -> torch.Tensor:
    """PyTorch reference for fused MoE + LoRA (float32 for precision)."""
    num_tokens, top_k = topk_ids.shape
    N_total = w.shape[1]
    N = N_total // num_slices

    x = hidden_states.float()
    W = w.float()
    A = lora_a.float()
    B = lora_b.float()
    if topk_weights is not None:
        tw = topk_weights.float()

    output = torch.zeros(
        (num_tokens * top_k, N_total),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    for t in range(num_tokens):
        for k in range(top_k):
            e = topk_ids[t, k].item()
            lo = lora_ids[t].item()
            flat_idx = t * top_k + k

            # Base GEMM: W[e] @ x[t]
            result = W[e] @ x[t]

            # LoRA delta per slice
            if lo >= 0:
                for s in range(num_slices):
                    # A[s, lo, e] is (rank, K), B[s, lo, e] is (N, rank)
                    intermediate = A[s, lo, e] @ x[t]  # (rank,)
                    delta = B[s, lo, e] @ intermediate  # (N,)
                    result[s * N : (s + 1) * N] += delta

            if mul_routed_weight and topk_weights is not None:
                result = result * tw[t, k]

            output[flat_idx] = result

    return output


@pytest.mark.parametrize("num_tokens", [16, 64, 512])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("top_k", [2, 6])
@pytest.mark.parametrize("max_loras", [1, 4])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_lora_kernel_correctness(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    max_loras: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
):
    if top_k > num_experts:
        pytest.skip("top_k > num_experts")

    torch.manual_seed(42)
    device = "cuda"
    K = 256
    N_per_slice = 128
    N_total = N_per_slice * num_slices

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N_total, K, device=device, dtype=dtype) * 0.01
    lora_a = (
        torch.randn(
            num_slices, max_loras, num_experts, max_lora_rank, K,
            device=device, dtype=dtype,
        )
        * 0.01
    )
    lora_b = (
        torch.randn(
            num_slices, max_loras, num_experts, N_per_slice, max_lora_rank,
            device=device, dtype=dtype,
        )
        * 0.01
    )

    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device=device)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )

    # LoRA assignment: each token gets a random adapter
    lora_ids_per_token = torch.randint(
        0, max_loras, (num_tokens,), device=device, dtype=torch.int64
    )

    # Token alignment
    block_size_m = 32
    sorted_token_ids, expert_ids, lora_ids_per_blk, num_post = (
        moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, block_size_m, num_experts, max_loras
        )
    )

    num_valid_tokens = num_tokens * top_k

    # Output buffer
    output = torch.zeros(
        (num_valid_tokens, N_total), device=device, dtype=dtype
    )

    if dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    else:
        compute_type = tl.float16

    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    invoke_fused_moe_lora_kernel(
        hidden_states,
        w,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        lora_ids_per_blk,
        num_post,
        lora_a,
        lora_b,
        True,  # mul_routed_weight
        top_k,
        num_slices,
        config,
        compute_type=compute_type,
    )

    # Reference
    ref_output = reference_moe_with_lora(
        hidden_states,
        w,
        lora_a,
        lora_b,
        topk_weights,
        topk_ids,
        lora_ids_per_token,
        mul_routed_weight=True,
        num_slices=num_slices,
    )

    # Compare
    fused_f32 = output.float()
    atol = 1e-2
    rtol = 5e-2
    assert torch.allclose(fused_f32, ref_output, atol=atol, rtol=rtol), (
        f"Max diff: {(fused_f32 - ref_output).abs().max().item():.6f}, "
        f"Mean diff: {(fused_f32 - ref_output).abs().mean().item():.6f}"
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_kernel_mixed_lora_no_lora(num_tokens: int, dtype: torch.dtype):
    """Test with a mix of LoRA and no-LoRA tokens."""
    torch.manual_seed(42)
    device = "cuda"
    num_experts = 8
    top_k = 2
    K = 256
    N = 128
    max_loras = 4
    max_lora_rank = 16
    num_slices = 1

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = (
        torch.randn(
            num_slices, max_loras, num_experts, max_lora_rank, K,
            device=device, dtype=dtype,
        )
        * 0.01
    )
    lora_b = (
        torch.randn(
            num_slices, max_loras, num_experts, N, max_lora_rank,
            device=device, dtype=dtype,
        )
        * 0.01
    )

    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device=device)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )

    # Mix: half tokens have LoRA, half don't
    lora_ids_per_token = torch.full(
        (num_tokens,), -1, device=device, dtype=torch.int64
    )
    lora_ids_per_token[: num_tokens // 2] = torch.randint(
        0, max_loras, (num_tokens // 2,), device=device, dtype=torch.int64
    )

    block_size_m = 32
    sorted_token_ids, expert_ids, lora_ids_per_blk, num_post = (
        moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, block_size_m, num_experts, max_loras
        )
    )

    num_valid_tokens = num_tokens * top_k
    output = torch.zeros(
        (num_valid_tokens, N), device=device, dtype=dtype
    )

    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    invoke_fused_moe_lora_kernel(
        hidden_states,
        w,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        lora_ids_per_blk,
        num_post,
        lora_a,
        lora_b,
        True,
        top_k,
        num_slices,
        config,
        compute_type=tl.bfloat16,
    )

    ref_output = reference_moe_with_lora(
        hidden_states,
        w,
        lora_a,
        lora_b,
        topk_weights,
        topk_ids,
        lora_ids_per_token,
        mul_routed_weight=True,
        num_slices=num_slices,
    )

    fused_f32 = output.float()
    assert torch.allclose(fused_f32, ref_output, atol=1e-2, rtol=5e-2), (
        f"Max diff: {(fused_f32 - ref_output).abs().max().item():.6f}"
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
def test_fused_kernel_no_lora_at_all(num_tokens: int):
    """Test with HAS_LORA=False (all lora_ids == -1)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    num_experts = 8
    top_k = 2
    K = 256
    N = 128
    max_loras = 4
    max_lora_rank = 16

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01

    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device=device)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )

    # All tokens: no LoRA
    lora_ids_per_token = torch.full(
        (num_tokens,), -1, device=device, dtype=torch.int64
    )

    block_size_m = 32
    sorted_token_ids, expert_ids, lora_ids_per_blk, num_post = (
        moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, block_size_m, num_experts, max_loras
        )
    )

    num_valid_tokens = num_tokens * top_k
    output = torch.zeros(
        (num_valid_tokens, N), device=device, dtype=dtype
    )

    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    invoke_fused_moe_lora_kernel(
        hidden_states,
        w,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        lora_ids_per_blk,
        num_post,
        None,  # no lora_a
        None,  # no lora_b
        True,
        top_k,
        1,  # num_slices
        config,
        compute_type=tl.bfloat16,
    )

    # Reference: base MoE only (no LoRA)
    dummy_a = torch.zeros(
        1, max_loras, num_experts, max_lora_rank, K, device=device, dtype=dtype
    )
    dummy_b = torch.zeros(
        1, max_loras, num_experts, N, max_lora_rank, device=device, dtype=dtype
    )
    ref_output = reference_moe_with_lora(
        hidden_states,
        w,
        dummy_a,
        dummy_b,
        topk_weights,
        topk_ids,
        lora_ids_per_token,
        mul_routed_weight=True,
        num_slices=1,
    )

    fused_f32 = output.float()
    assert torch.allclose(fused_f32, ref_output, atol=1e-2, rtol=5e-2), (
        f"Max diff: {(fused_f32 - ref_output).abs().max().item():.6f}"
    )


def test_moe_lora_align_block_size_fused():
    """Test the alignment function produces valid blocks."""
    device = "cuda"
    num_tokens = 32
    top_k = 2
    num_experts = 4
    max_loras = 2
    block_size = 16

    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device=device)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int64)
    lora_ids = torch.randint(
        -1, max_loras, (num_tokens,), device=device, dtype=torch.int64
    )

    sorted_token_ids, expert_ids, lora_ids_blk, num_post = (
        moe_lora_align_block_size_fused(
            topk_ids, lora_ids, block_size, num_experts, max_loras
        )
    )

    total_padded = num_post.item()
    assert sorted_token_ids.shape[0] == total_padded
    assert total_padded % block_size == 0

    num_blocks = total_padded // block_size
    assert expert_ids.shape[0] == num_blocks
    assert lora_ids_blk.shape[0] == num_blocks

    num_valid = num_tokens * top_k
    # Check each block has consistent (expert, lora)
    for b in range(num_blocks):
        block_tokens = sorted_token_ids[
            b * block_size : (b + 1) * block_size
        ]
        valid_in_block = block_tokens[block_tokens < num_valid]
        if len(valid_in_block) == 0:
            continue
        # All valid tokens in this block should map to the same expert
        flat_experts = topk_ids.flatten()
        block_expert_id = expert_ids[b].item()
        for tid in valid_in_block:
            assert flat_experts[tid.item()].item() == block_expert_id
