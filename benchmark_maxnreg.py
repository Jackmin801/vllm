"""Test if maxnreg can cap register count and improve occupancy."""
import sys
import time

import torch

from vllm.triton_utils import tl, triton

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    _write_zeros_to_output,
    fused_moe_with_lora_kernel,
    invoke_fused_moe_lora_kernel,
)
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


# Create a copy of the fused kernel with maxnreg constraint
@triton.jit
def fused_moe_lora_maxnreg(
    a_ptr, w_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    lora_a_ptr, lora_b_ptr, lora_ids_ptr,
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
    """Same as fused_moe_with_lora_kernel but will be launched with maxnreg."""
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
        _write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N_total,
                               offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
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
        lora_acc = tl.zeros((BLOCK_SIZE_M, LORA_RANK), dtype=tl.float32)

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
                lora_acc += tl.dot(a, la)
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
            accumulator += tl.dot(lora_acc.to(compute_type), lora_b_tile)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_total)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def launch_maxnreg(A, B, C, topk_weights, sorted_token_ids, expert_ids,
                   lora_ids, num_tokens_post_padded, lora_a, lora_b,
                   top_k, num_slices, config, compute_type, maxnreg=None):
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

    extra = {}
    if maxnreg is not None:
        extra["maxnreg"] = maxnreg

    fused_moe_lora_maxnreg[grid](
        A, B, C, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        lora_a, lora_b, lora_ids,
        N, K, EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(-2), C.stride(-1),
        lora_a.stride(0), lora_a.stride(1), lora_a.stride(2), lora_a.stride(3), lora_a.stride(4),
        lora_b.stride(0), lora_b.stride(1), lora_b.stride(2), lora_b.stride(3), lora_b.stride(4),
        NUM_SLICES=num_slices, BLOCK_SIZE_K=BLOCK_SIZE_K, LORA_RANK=LORA_RANK,
        MUL_ROUTED_WEIGHT=True, top_k=top_k, compute_type=compute_type, HAS_LORA=HAS_LORA,
        **cfg, **extra,
    )


def bench(num_tokens, num_experts=8, top_k=2, K=4096, N=4096, max_loras=1,
          rank=16, dtype=torch.bfloat16, n_warmup=8, n_iter=40):
    device = "cuda"
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
    lora_ids_per_token = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    compute_type = tl.bfloat16

    base_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}
    fused_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}
    bm = 16

    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, bm, num_experts)
    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, bm, num_experts, max_loras)
    out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)
    out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

    results = {}

    # Base
    for _ in range(n_warmup):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    results["base_NW4"] = (time.perf_counter() - t0) / n_iter * 1000

    # Fused without maxnreg (current)
    for _ in range(n_warmup):
        out_fused.zero_()
        launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                       sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                       top_k, 1, fused_config, compute_type, maxnreg=None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                       sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                       top_k, 1, fused_config, compute_type, maxnreg=None)
    torch.cuda.synchronize()
    results["fused_noreg"] = (time.perf_counter() - t0) / n_iter * 1000

    # Test different maxnreg values
    for maxnreg in [64, 72, 80, 85, 96, 104, 112]:
        key = f"fused_maxnreg{maxnreg}"
        try:
            for _ in range(n_warmup):
                out_fused.zero_()
                launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                               sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                               top_k, 1, fused_config, compute_type, maxnreg=maxnreg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                               sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                               top_k, 1, fused_config, compute_type, maxnreg=maxnreg)
            torch.cuda.synchronize()
            results[key] = (time.perf_counter() - t0) / n_iter * 1000
        except Exception as e:
            results[key] = None
            print(f"  {key}: ERROR - {e}", file=sys.stderr)

    return results


# Also get register counts for each maxnreg variant
def get_reg_counts():
    """Compile all variants and report register counts."""
    num_tokens = 64
    num_experts = 8
    top_k = 2
    K = 4096
    N = 4096
    rank = 16
    max_loras = 1
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
    lora_ids_per_token = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    compute_type = tl.bfloat16

    fused_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}
    bm = 16

    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, bm, num_experts, max_loras)
    out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

    # Compile without maxnreg
    launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                   sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                   top_k, 1, fused_config, compute_type, maxnreg=None)

    # Compile with different maxnreg
    for maxnreg in [64, 72, 80, 85, 96, 104, 112]:
        try:
            launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                           sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                           top_k, 1, fused_config, compute_type, maxnreg=maxnreg)
        except Exception as e:
            print(f"  maxnreg={maxnreg}: compile error - {e}", file=sys.stderr)
    torch.cuda.synchronize()

    # Report
    print("\nRegister counts for fused_moe_lora_maxnreg variants:")
    caches = fused_moe_lora_maxnreg.device_caches
    for dk, dv in caches.items():
        compiled_cache = dv[0]
        for key, compiled in compiled_cache.items():
            nw = compiled.packed_metadata[0]
            shared = compiled.packed_metadata[2]
            blks = compiled.n_max_threads // (nw * 32)
            warps = blks * nw
            print(f"  NW={nw} regs={compiled.n_regs:>3} spills={compiled.n_spills:>2} "
                  f"max_threads={compiled.n_max_threads:>4} blocks/SM={blks} warps/SM={warps} "
                  f"shared={shared}B")


print("Getting register counts...")
get_reg_counts()

print("\nRunning benchmarks...")
batch_sizes = [256, 1024, 4096, 65536]

all_results = {}
all_keys = None
for M in batch_sizes:
    nw, ni = (3, 15) if M >= 16384 else (5, 25) if M >= 4096 else (8, 40)
    print(f"  M={M}...", end=" ", flush=True)
    all_results[M] = bench(M, n_warmup=nw, n_iter=ni)
    if all_keys is None:
        all_keys = list(all_results[M].keys())
    print("done")
    torch.cuda.empty_cache()

# Print table
print()
print(f"{'Config':<25s}", end="")
for M in batch_sizes:
    print(f" | {M:>8d}", end="")
print()
print("-" * (25 + 11 * len(batch_sizes)))

for k in all_keys:
    print(f"{k:<25s}", end="")
    for M in batch_sizes:
        v = all_results[M].get(k)
        if v is None:
            print(f" | {'ERR':>8s}", end="")
        elif k == "base_NW4":
            print(f" | {v:>7.3f}ms", end="")
        else:
            base = all_results[M]["base_NW4"]
            oh = (v - base) / base * 100
            print(f" |  {oh:>+5.1f}%  ", end="")
    print()
