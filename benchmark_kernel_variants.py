"""
Kernel variants to reduce register pressure in fused MoE+LoRA.

Variant 1 (baseline): Current fused kernel - lora_acc lives across entire K-loop
Variant 2 (two-phase): Separate K-loops for base GEMM and LoRA shrink -
    accumulators never coexist. Reloads 'a' tiles from L2 cache.
Variant 3 (bf16 acc): Use bf16 for lora_acc instead of fp32 - halves register usage
Variant 4 (scratchpad): Spill lora_acc to a small global buffer (hits L1) before
    the base GEMM dot, reload after - frees registers during tl.dot(a, w)
"""

import json
import sys
import time

import torch

from vllm.triton_utils import tl, triton

# ============================================================================
# Variant 2: Two-phase K-loop
# ============================================================================
@triton.jit
def _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, compute_type: tl.constexpr):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def fused_moe_lora_twophase(
    a_ptr, w_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, lora_ids_ptr, num_tokens_post_padded_ptr,
    lora_a_ptr, lora_b_ptr,
    N: tl.int64, K: tl.int64, EM: tl.int64, num_valid_tokens: tl.int64,
    stride_am: tl.int64, stride_ak: tl.int64,
    stride_we: tl.int64, stride_wn: tl.int64, stride_wk: tl.int64,
    stride_cm: tl.int64, stride_cn: tl.int64,
    stride_la_s: tl.int64, stride_la_l: tl.int64, stride_la_e: tl.int64,
    stride_la_r: tl.int64, stride_la_k: tl.int64,
    stride_lb_s: tl.int64, stride_lb_l: tl.int64, stride_lb_e: tl.int64,
    stride_lb_n: tl.int64, stride_lb_r: tl.int64,
    NUM_SLICES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, LORA_RANK: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
    HAS_LORA: tl.constexpr,
):
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

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_expert == -1:
        _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N_total, offs_token, token_mask,
                     BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return

    global_n_offset = pid_n * BLOCK_SIZE_N
    if NUM_SLICES > 1:
        slice_idx = global_n_offset // N
        local_n_start = global_n_offset - slice_idx * N
    else:
        slice_idx = 0
        local_n_start = global_n_offset

    offs_bn = (global_n_offset + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N_total
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + off_expert * stride_we + offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn

    # === PHASE 1: Base GEMM only — no lora_acc in registers ===
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator += tl.dot(a, w)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # === PHASE 2: LoRA shrink — reload 'a' from L2 cache, no base accumulator pressure ===
    if HAS_LORA:
        off_lora = tl.load(lora_ids_ptr + pid_m).to(tl.int64)
        if off_lora >= 0:
            offs_r = tl.arange(0, LORA_RANK)
            local_offs_n = local_n_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

            lora_a_base = (lora_a_ptr + slice_idx * stride_la_s
                          + off_lora * stride_la_l + off_expert * stride_la_e)
            lora_a_ptrs = lora_a_base + offs_k[:, None] * stride_la_k + offs_r[None, :] * stride_la_r

            # Reset a_ptrs for second pass
            a_ptrs2 = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

            lora_acc = tl.zeros((BLOCK_SIZE_M, LORA_RANK), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_remaining = K - k * BLOCK_SIZE_K
                k_mask = offs_k < k_remaining
                a = tl.load(a_ptrs2, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
                la = tl.load(lora_a_ptrs, mask=k_mask[:, None], other=0.0)
                lora_acc += tl.dot(a, la)
                a_ptrs2 += BLOCK_SIZE_K * stride_ak
                lora_a_ptrs += BLOCK_SIZE_K * stride_la_k

            # LoRA-B expand
            lora_b_base = (lora_b_ptr + slice_idx * stride_lb_s
                          + off_lora * stride_lb_l + off_expert * stride_lb_e)
            lora_b_tile = tl.load(
                lora_b_base + offs_r[:, None] * stride_lb_r + local_offs_n[None, :] * stride_lb_n,
                mask=local_offs_n[None, :] < N, other=0.0)
            accumulator += tl.dot(lora_acc.to(compute_type), lora_b_tile)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_total)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ============================================================================
# Variant 3: bf16 lora_acc
# ============================================================================
@triton.jit
def fused_moe_lora_bf16acc(
    a_ptr, w_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, lora_ids_ptr, num_tokens_post_padded_ptr,
    lora_a_ptr, lora_b_ptr,
    N: tl.int64, K: tl.int64, EM: tl.int64, num_valid_tokens: tl.int64,
    stride_am: tl.int64, stride_ak: tl.int64,
    stride_we: tl.int64, stride_wn: tl.int64, stride_wk: tl.int64,
    stride_cm: tl.int64, stride_cn: tl.int64,
    stride_la_s: tl.int64, stride_la_l: tl.int64, stride_la_e: tl.int64,
    stride_la_r: tl.int64, stride_la_k: tl.int64,
    stride_lb_s: tl.int64, stride_lb_l: tl.int64, stride_lb_e: tl.int64,
    stride_lb_n: tl.int64, stride_lb_r: tl.int64,
    NUM_SLICES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, LORA_RANK: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
    HAS_LORA: tl.constexpr,
):
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

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_expert == -1:
        _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N_total, offs_token, token_mask,
                     BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return

    global_n_offset = pid_n * BLOCK_SIZE_N
    if NUM_SLICES > 1:
        slice_idx = global_n_offset // N
        local_n_start = global_n_offset - slice_idx * N
    else:
        slice_idx = 0
        local_n_start = global_n_offset

    offs_bn = (global_n_offset + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N_total
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + off_expert * stride_we + offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn

    if HAS_LORA:
        off_lora = tl.load(lora_ids_ptr + pid_m).to(tl.int64)
        has_lora_for_block = off_lora >= 0
        offs_r = tl.arange(0, LORA_RANK)
        local_offs_n = local_n_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        lora_a_base = (lora_a_ptr + slice_idx * stride_la_s
                      + off_lora * stride_la_l + off_expert * stride_la_e)
        lora_a_ptrs = lora_a_base + offs_k[:, None] * stride_la_k + offs_r[None, :] * stride_la_r
        # KEY CHANGE: bf16 accumulator instead of fp32 — halves register usage
        lora_acc = tl.zeros((BLOCK_SIZE_M, LORA_RANK), dtype=compute_type)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator += tl.dot(a, w)
        if HAS_LORA:
            if has_lora_for_block:
                la = tl.load(lora_a_ptrs, mask=k_mask[:, None], other=0.0)
                lora_acc = (lora_acc + tl.dot(a, la)).to(compute_type)
                lora_a_ptrs += BLOCK_SIZE_K * stride_la_k
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    if HAS_LORA:
        if has_lora_for_block:
            lora_b_base = (lora_b_ptr + slice_idx * stride_lb_s
                          + off_lora * stride_lb_l + off_expert * stride_lb_e)
            lora_b_tile = tl.load(
                lora_b_base + offs_r[:, None] * stride_lb_r + local_offs_n[None, :] * stride_lb_n,
                mask=local_offs_n[None, :] < N, other=0.0)
            accumulator += tl.dot(lora_acc, lora_b_tile)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_total)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ============================================================================
# Variant 4: Explicit scratchpad spill
# ============================================================================
@triton.jit
def fused_moe_lora_scratchpad(
    a_ptr, w_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, lora_ids_ptr, num_tokens_post_padded_ptr,
    lora_a_ptr, lora_b_ptr,
    scratch_ptr,  # (max_programs, BLOCK_SIZE_M, LORA_RANK) scratchpad
    N: tl.int64, K: tl.int64, EM: tl.int64, num_valid_tokens: tl.int64,
    stride_am: tl.int64, stride_ak: tl.int64,
    stride_we: tl.int64, stride_wn: tl.int64, stride_wk: tl.int64,
    stride_cm: tl.int64, stride_cn: tl.int64,
    stride_la_s: tl.int64, stride_la_l: tl.int64, stride_la_e: tl.int64,
    stride_la_r: tl.int64, stride_la_k: tl.int64,
    stride_lb_s: tl.int64, stride_lb_l: tl.int64, stride_lb_e: tl.int64,
    stride_lb_n: tl.int64, stride_lb_r: tl.int64,
    stride_sc_p: tl.int64, stride_sc_m: tl.int64, stride_sc_r: tl.int64,
    NUM_SLICES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, LORA_RANK: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
    HAS_LORA: tl.constexpr,
):
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

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_expert == -1:
        _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N_total, offs_token, token_mask,
                     BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return

    global_n_offset = pid_n * BLOCK_SIZE_N
    if NUM_SLICES > 1:
        slice_idx = global_n_offset // N
        local_n_start = global_n_offset - slice_idx * N
    else:
        slice_idx = 0
        local_n_start = global_n_offset

    offs_bn = (global_n_offset + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N_total
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + off_expert * stride_we + offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn

    if HAS_LORA:
        off_lora = tl.load(lora_ids_ptr + pid_m).to(tl.int64)
        has_lora_for_block = off_lora >= 0
        offs_r = tl.arange(0, LORA_RANK)
        local_offs_n = local_n_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        lora_a_base = (lora_a_ptr + slice_idx * stride_la_s
                      + off_lora * stride_la_l + off_expert * stride_la_e)
        lora_a_ptrs = lora_a_base + offs_k[:, None] * stride_la_k + offs_r[None, :] * stride_la_r

        # Scratchpad pointers for this program
        sc_base = scratch_ptr + pid * stride_sc_p
        sc_ptrs = sc_base + offs_m[:, None] * stride_sc_m + offs_r[None, :] * stride_sc_r

        # Initialize scratchpad to zero
        tl.store(sc_ptrs, tl.zeros((BLOCK_SIZE_M, LORA_RANK), dtype=tl.float32))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator += tl.dot(a, w)

        if HAS_LORA:
            if has_lora_for_block:
                # Load lora_acc from scratchpad (L1 cache)
                lora_acc = tl.load(sc_ptrs)
                la = tl.load(lora_a_ptrs, mask=k_mask[:, None], other=0.0)
                lora_acc += tl.dot(a, la)
                # Spill back to scratchpad
                tl.store(sc_ptrs, lora_acc)
                lora_a_ptrs += BLOCK_SIZE_K * stride_la_k

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    if HAS_LORA:
        if has_lora_for_block:
            lora_acc = tl.load(sc_ptrs)
            lora_b_base = (lora_b_ptr + slice_idx * stride_lb_s
                          + off_lora * stride_lb_l + off_expert * stride_lb_e)
            lora_b_tile = tl.load(
                lora_b_base + offs_r[:, None] * stride_lb_r + local_offs_n[None, :] * stride_lb_n,
                mask=local_offs_n[None, :] < N, other=0.0)
            accumulator += tl.dot(lora_acc.to(compute_type), lora_b_tile)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_total)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ============================================================================
# Launcher helpers
# ============================================================================
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


def launch_variant(kernel_fn, A, B, C, topk_weights, sorted_token_ids, expert_ids,
                   lora_ids, num_tokens_post_padded, lora_a, lora_b, top_k, num_slices,
                   config, compute_type, scratch=None):
    num_tokens = A.size(0)
    K = A.size(1)
    N_total = B.size(1)
    N = N_total // num_slices
    num_valid_tokens = num_tokens * top_k
    EM = sorted_token_ids.size(0)

    HAS_LORA = lora_a is not None and bool((lora_ids >= 0).any().item())
    LORA_RANK = max(lora_a.shape[-2], 16) if HAS_LORA else 16

    cfg = config.copy()
    BLOCK_SIZE_K = cfg.pop("BLOCK_SIZE_K", 32)
    cfg.pop("SPLIT_K", None)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N_total, META["BLOCK_SIZE_N"]),)

    # Match the kernel signature order: ..., num_tokens_post_padded, lora_a, lora_b, lora_ids, ...
    args = [A, B, C, topk_weights, sorted_token_ids, expert_ids,
            lora_ids, num_tokens_post_padded, lora_a, lora_b]

    if scratch is not None:
        args.append(scratch)

    args.extend([N, K, EM, num_valid_tokens,
                 A.stride(0), A.stride(1),
                 B.stride(0), B.stride(1), B.stride(2),
                 C.stride(-2), C.stride(-1),
                 lora_a.stride(0), lora_a.stride(1), lora_a.stride(2), lora_a.stride(3), lora_a.stride(4),
                 lora_b.stride(0), lora_b.stride(1), lora_b.stride(2), lora_b.stride(3), lora_b.stride(4)])

    if scratch is not None:
        args.extend([scratch.stride(0), scratch.stride(1), scratch.stride(2)])

    kwargs = dict(NUM_SLICES=num_slices, BLOCK_SIZE_K=BLOCK_SIZE_K, LORA_RANK=LORA_RANK,
                  MUL_ROUTED_WEIGHT=True, top_k=top_k, compute_type=compute_type, HAS_LORA=HAS_LORA)
    kwargs.update(cfg)

    kernel_fn[grid](*args, **kwargs)


def bench(num_tokens, num_experts=8, top_k=2, K=4096, N=4096, max_loras=1,
          rank=16, dtype=torch.bfloat16, n_warmup=8, n_iter=40):
    device = "cuda"
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1).reshape(-1)
    lora_ids_per_token = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    compute_type = tl.bfloat16

    # Use base-optimal config for base, and test each variant with several configs
    base_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}

    # Try both NW=4 and NW=8 for fused variants
    fused_configs = [
        ("NW4", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                 "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
        ("NW8", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                 "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}),
    ]

    bm = base_config["BLOCK_SIZE_M"]
    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, bm, num_experts)
    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, bm, num_experts, max_loras)

    out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)
    out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

    # Max grid size for scratchpad
    max_programs = (sorted_f.size(0) // bm) * (N // 128 + 1)
    scratch = torch.zeros(max_programs, bm, rank, device=device, dtype=torch.float32)

    results = {}

    # Base
    for _ in range(n_warmup):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights.view(num_tokens, top_k),
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights.view(num_tokens, top_k),
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    results["base"] = (time.perf_counter() - t0) / n_iter * 1000

    # Test each variant with each config
    from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel as invoke_orig

    # v1_original: use the proper launcher
    for cname, cfg in fused_configs:
        key = f"v1_original_{cname}"
        full_cfg = {**cfg, "SPLIT_K": 1}
        try:
            for _ in range(n_warmup):
                out_fused.zero_()
                invoke_orig(hidden, w, out_fused, topk_weights.view(num_tokens, top_k),
                           sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                           True, top_k, 1, full_cfg, compute_type=compute_type)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                invoke_orig(hidden, w, out_fused, topk_weights.view(num_tokens, top_k),
                           sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                           True, top_k, 1, full_cfg, compute_type=compute_type)
            torch.cuda.synchronize()
            results[key] = (time.perf_counter() - t0) / n_iter * 1000
        except Exception as e:
            results[key] = None
            print(f"  {key}: ERROR - {e}", file=sys.stderr)

    # v2, v3, v4: use launch_variant
    variants = [
        ("v2_twophase", fused_moe_lora_twophase, False),
        ("v3_bf16acc", fused_moe_lora_bf16acc, False),
        ("v4_scratch", fused_moe_lora_scratchpad, True),
    ]

    for vname, kernel, needs_scratch in variants:
        for cname, cfg in fused_configs:
            key = f"{vname}_{cname}"
            try:
                for _ in range(n_warmup):
                    out_fused.zero_()
                    launch_variant(kernel, hidden, w, out_fused, topk_weights,
                                   sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                                   top_k, 1, cfg, compute_type,
                                   scratch=scratch if needs_scratch else None)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_iter):
                    launch_variant(kernel, hidden, w, out_fused, topk_weights,
                                   sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                                   top_k, 1, cfg, compute_type,
                                   scratch=scratch if needs_scratch else None)
                torch.cuda.synchronize()
                results[key] = (time.perf_counter() - t0) / n_iter * 1000
            except Exception as e:
                results[key] = None
                print(f"  {key}: ERROR - {e}", file=sys.stderr)

    return results


def main():
    batch_sizes = [16, 64, 256, 1024, 4096, 65536]
    all_keys = None

    print("Kernel variant comparison (max_loras=1, same tile sizes)")
    print("All fused variants use BM=16 BN=128 BK=32 GM=16")
    print()

    all_results = {}
    for M in batch_sizes:
        nw, ni = (3, 15) if M >= 16384 else (5, 25) if M >= 4096 else (8, 40)
        r = bench(M, n_warmup=nw, n_iter=ni)
        all_results[M] = r
        if all_keys is None:
            all_keys = list(r.keys())
        torch.cuda.empty_cache()

    # Print table
    header = f"{'Tokens':>8s}"
    for k in all_keys:
        header += f" | {k:>16s}"
    print(header)
    print("-" * len(header))

    for M in batch_sizes:
        r = all_results[M]
        base = r["base"]
        row = f"{M:>8d}"
        for k in all_keys:
            v = r[k]
            if v is None:
                row += f" | {'ERR':>16s}"
            elif k == "base":
                row += f" | {v:>10.3f}      "
            else:
                oh = (v - base) / base * 100
                row += f" | {v:>7.3f} {oh:>+5.0f}%"
        print(row)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
