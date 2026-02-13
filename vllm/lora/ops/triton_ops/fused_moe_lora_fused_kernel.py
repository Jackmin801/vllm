# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused MoE + LoRA Triton Kernel.

Fuses the LoRA delta computation (x @ A^T @ B^T) into the base expert GEMM
(x @ W^T) during the K-reduction loop, avoiding separate LoRA kernel launches
and redundant activation memory traffic.

The fused computation per (token, expert, lora):
    output = x @ W^T + x @ A^T @ B^T

Key insight: the input tile 'a' loaded for the base GEMM is reused for the
LoRA-A shrink at zero extra memory cost. After the K-loop, the tiny LoRA-B
expand ([BLOCK_M, r] @ [r, BLOCK_N]) is applied before the store.

Supports:
- Multi-slice (NUM_SLICES constexpr) for w1 gated MoE (gate+up)
- Per-block no-LoRA check (lora_id == -1 → base GEMM only)
- Expert parallelism (expert_id == -1 → write zeros)
- HAS_LORA constexpr to compile out LoRA paths when no LoRA in batch
"""

import functools
import json
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


def get_fused_lora_default_config(
    M: int, E: int, max_loras: int, top_k: int
) -> dict[str, int]:
    """Heuristic tile config for the fused MoE+LoRA kernel.

    Chooses BLOCK_SIZE_M based on expected tokens per (expert, lora) group
    to keep tile utilization >= 50% and reduce padding inflation at small
    batch sizes.
    """
    tokens_per_group = max(1, (M * top_k) // (E * max_loras))
    if tokens_per_group <= 16:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    elif tokens_per_group <= 64:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }


@functools.lru_cache
def _get_fused_lora_configs(
    E: int, N: int, dtype: str | None
) -> dict[int, Any] | None:
    """Load tuned fused-LoRA configs from JSON, mirroring get_moe_configs().

    Looks in ``vllm/lora/ops/triton_ops/configs/`` for files named like
    ``E=8,N=4096,device_name=NVIDIA_H100,...json``.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_config_file_name,
    )
    from vllm.model_executor.layers.batch_invariant import (
        vllm_is_batch_invariant,
    )

    if vllm_is_batch_invariant():
        return None

    json_file_name = get_config_file_name(E, N, dtype)

    config_file_paths = []

    user_folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
    if user_folder is not None:
        config_file_paths.append(os.path.join(user_folder, json_file_name))

    default_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs"
    )
    config_file_paths.append(os.path.join(default_dir, json_file_name))

    for path in config_file_paths:
        if os.path.exists(path):
            with open(path) as f:
                logger.info_once(
                    "Using configuration from %s for fused MoE+LoRA layer.",
                    path,
                    scope="global",
                )
                tuned = json.load(f)
                tuned.pop("triton_version", None)
                return {int(k): v for k, v in tuned.items()}

    return None


def try_get_optimal_fused_lora_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    max_loras: int,
) -> dict[str, int]:
    """Get the best available config for the fused MoE+LoRA kernel.

    Tries GPU-specific JSON configs first, falls back to the heuristic.
    """
    E, _, N = w2_shape
    configs = _get_fused_lora_configs(E, N, dtype)

    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = get_fused_lora_default_config(M, E, max_loras, top_k)

    return config


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_with_lora_kernel(
    # Base MoE pointers
    a_ptr,  # input activations: (num_tokens, K)
    w_ptr,  # expert weights:    (E, N_total, K)  N_total = NUM_SLICES * N
    c_ptr,  # output:            written at sorted token positions, width N_total
    topk_weights_ptr,  # router weights: (num_tokens * top_k,)  flat
    sorted_token_ids_ptr,  # (num_tokens_post_padded,)
    expert_ids_ptr,  # (num_m_blocks,)
    num_tokens_post_padded_ptr,  # scalar
    # LoRA pointers
    lora_a_ptr,  # (NUM_SLICES, max_loras, E, LORA_RANK, K)
    lora_b_ptr,  # (NUM_SLICES, max_loras, E, N, LORA_RANK)
    lora_ids_ptr,  # (num_m_blocks,) — per-block lora adapter id, -1 = none
    # Dimensions
    N: tl.int64,  # per-slice output dim (N_total = NUM_SLICES * N)
    K: tl.int64,
    EM: tl.int64,
    num_valid_tokens: tl.int64,
    # A strides
    stride_am: tl.int64,
    stride_ak: tl.int64,
    # W strides  (W is (E, N_total, K))
    stride_we: tl.int64,
    stride_wn: tl.int64,
    stride_wk: tl.int64,
    # C strides
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    # LoRA-A strides (NUM_SLICES, max_loras, E, LORA_RANK, K)
    stride_la_s: tl.int64,
    stride_la_l: tl.int64,
    stride_la_e: tl.int64,
    stride_la_r: tl.int64,
    stride_la_k: tl.int64,
    # LoRA-B strides (NUM_SLICES, max_loras, E, N, LORA_RANK)
    stride_lb_s: tl.int64,
    stride_lb_l: tl.int64,
    stride_lb_e: tl.int64,
    stride_lb_n: tl.int64,
    stride_lb_r: tl.int64,
    # Constexprs
    NUM_SLICES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LORA_RANK: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HAS_LORA: tl.constexpr,
):
    # --- Map 1-D pid to (pid_m, pid_n) with grouped ordering for L2 reuse ---
    pid = tl.program_id(axis=0)
    N_total = N * NUM_SLICES
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_total, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Early exit for padding blocks ---
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # --- Load token indices and expert id for this M-block ---
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # --- Expert parallel: expert_id == -1 means not in this rank ---
    if off_expert == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N_total,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # --- Determine which slice this N-block belongs to (for multi-slice) ---
    global_n_offset = pid_n * BLOCK_SIZE_N
    if NUM_SLICES > 1:
        slice_idx = global_n_offset // N
        local_n_start = global_n_offset - slice_idx * N
    else:
        slice_idx = 0
        local_n_start = global_n_offset

    # --- Base GEMM pointer setup ---
    offs_bn = (
        global_n_offset + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    ) % N_total
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A tile pointers: [BLOCK_M, BLOCK_K]
    # offs_token // top_k recovers the original token index
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am
        + offs_k[None, :] * stride_ak
    )

    # W tile pointers: [BLOCK_K, BLOCK_N]
    w_ptrs = (
        w_ptr
        + off_expert * stride_we
        + offs_k[:, None] * stride_wk
        + offs_bn[None, :] * stride_wn
    )

    # --- LoRA pointer setup ---
    if HAS_LORA:
        off_lora = tl.load(lora_ids_ptr + pid_m).to(tl.int64)
        has_lora_for_block = off_lora >= 0

        offs_r = tl.arange(0, LORA_RANK)

        # Local N offsets within the slice for LoRA-B
        local_offs_n = (
            local_n_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        )

        # LoRA-A base for this (slice, lora, expert) pair
        lora_a_base = (
            lora_a_ptr
            + slice_idx * stride_la_s
            + off_lora * stride_la_l
            + off_expert * stride_la_e
        )
        # Tile pointers: [BLOCK_K, LORA_RANK]
        lora_a_ptrs = (
            lora_a_base
            + offs_k[:, None] * stride_la_k
            + offs_r[None, :] * stride_la_r
        )

        # Shrink accumulator: [BLOCK_M, LORA_RANK]
        lora_acc = tl.zeros((BLOCK_SIZE_M, LORA_RANK), dtype=tl.float32)

    # --- Main K-reduction loop ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining

        # Load input tile (shared between base GEMM and LoRA shrink)
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]

        # Load weight tile
        w = tl.load(
            w_ptrs, mask=k_mask[:, None], other=0.0
        )  # [BLOCK_K, BLOCK_N]

        # Base GEMM accumulation
        accumulator += tl.dot(a, w)

        # LoRA-A shrink: reuse 'a' tile already in registers
        if HAS_LORA:
            if has_lora_for_block:
                la = tl.load(
                    lora_a_ptrs, mask=k_mask[:, None], other=0.0
                )  # [BLOCK_K, LORA_RANK]
                lora_acc += tl.dot(a, la)  # [BLOCK_M, LORA_RANK]
                lora_a_ptrs += BLOCK_SIZE_K * stride_la_k

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # --- LoRA-B expand: accumulator += lora_acc @ B^T ---
    if HAS_LORA:
        if has_lora_for_block:
            # Load full LoRA-B tile: [LORA_RANK, BLOCK_N]
            lora_b_base = (
                lora_b_ptr
                + slice_idx * stride_lb_s
                + off_lora * stride_lb_l
                + off_expert * stride_lb_e
            )
            lora_b_tile = tl.load(
                lora_b_base
                + offs_r[:, None] * stride_lb_r
                + local_offs_n[None, :] * stride_lb_n,
                mask=local_offs_n[None, :] < N,
                other=0.0,
            )  # [LORA_RANK, BLOCK_N]

            # Expand: [BLOCK_M, LORA_RANK] @ [LORA_RANK, BLOCK_N]
            accumulator += tl.dot(lora_acc.to(compute_type), lora_b_tile)

    # --- Router weight multiplication ---
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # --- Store output ---
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_total)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_lora_kernel(
    A: torch.Tensor,  # (num_tokens, K)
    B: torch.Tensor,  # (E, N_total, K) — base expert weights
    C: torch.Tensor,  # output buffer indexed by sorted token positions
    topk_weights: torch.Tensor | None,  # (num_tokens * top_k,) flat
    sorted_token_ids: torch.Tensor,  # (num_tokens_post_padded,)
    expert_ids: torch.Tensor,  # (num_m_blocks,)
    lora_ids: torch.Tensor,  # (num_m_blocks,) — per-block lora id, -1=none
    num_tokens_post_padded: torch.Tensor,  # (1,) scalar
    lora_a_stacked: torch.Tensor | None,  # (NUM_SLICES, max_loras, E, rank, K)
    lora_b_stacked: torch.Tensor | None,  # (NUM_SLICES, max_loras, E, N, rank)
    mul_routed_weight: bool,
    top_k: int,
    num_slices: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """
    Launch the fused MoE + LoRA Triton kernel.

    Mirrors invoke_fused_moe_triton_kernel but with added LoRA arguments.
    """
    assert topk_weights is not None or not mul_routed_weight

    num_tokens = A.size(0)
    K = A.size(1)
    N_total = B.size(1)
    N = N_total // num_slices
    num_valid_tokens = num_tokens * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(
            sorted_token_ids.size(0),
            A.size(0) * top_k * config["BLOCK_SIZE_M"],
        )

    HAS_LORA = lora_a_stacked is not None and bool(
        (lora_ids >= 0).any().item()
    )

    if HAS_LORA:
        LORA_RANK = max(lora_a_stacked.shape[-2], 16)
    else:
        LORA_RANK = 16  # dummy, won't be used

    # Create dummy tensors for the kernel when no LoRA
    if lora_a_stacked is None:
        lora_a_stacked = A  # dummy, won't be accessed
        lora_b_stacked = A  # dummy, won't be accessed

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N_total, META["BLOCK_SIZE_N"]),
    )

    config = config.copy()
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K", 32)
    # Remove SPLIT_K which the base MoE config may contain but our kernel
    # doesn't use. Preserve num_warps/num_stages so tuned values reach Triton.
    config.pop("SPLIT_K", None)

    fused_moe_with_lora_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        lora_a_stacked,
        lora_b_stacked,
        lora_ids,
        # Dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # A strides
        A.stride(0),
        A.stride(1),
        # W strides (E, N_total, K)
        B.stride(0),
        B.stride(1),
        B.stride(2),
        # C strides — C may be 3D (num_tokens, top_k, N) or 2D (M*top_k, N)
        C.stride(-2),
        C.stride(-1),
        # LoRA-A strides (NUM_SLICES, max_loras, E, LORA_RANK, K)
        lora_a_stacked.stride(0) if lora_a_stacked.dim() == 5 else 0,
        lora_a_stacked.stride(1) if lora_a_stacked.dim() == 5 else 0,
        lora_a_stacked.stride(2) if lora_a_stacked.dim() == 5 else 0,
        lora_a_stacked.stride(3) if lora_a_stacked.dim() == 5 else 0,
        lora_a_stacked.stride(4) if lora_a_stacked.dim() == 5 else 0,
        # LoRA-B strides (NUM_SLICES, max_loras, E, N, LORA_RANK)
        lora_b_stacked.stride(0) if lora_b_stacked.dim() == 5 else 0,
        lora_b_stacked.stride(1) if lora_b_stacked.dim() == 5 else 0,
        lora_b_stacked.stride(2) if lora_b_stacked.dim() == 5 else 0,
        lora_b_stacked.stride(3) if lora_b_stacked.dim() == 5 else 0,
        lora_b_stacked.stride(4) if lora_b_stacked.dim() == 5 else 0,
        # Constexprs
        NUM_SLICES=num_slices,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        LORA_RANK=LORA_RANK,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        HAS_LORA=HAS_LORA,
        **config,
    )
