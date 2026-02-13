"""
Config sweep for the BASE MoE Triton kernel (no LoRA).

Mirrors the fused sweep structure so we get a tuned config for B200
to enable a fair apples-to-apples comparison.
"""

import json
import os
import sys
import time
from itertools import product

import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def bench_base_config(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    K: int,
    N: int,
    dtype: torch.dtype,
    config: dict,
    n_warmup: int = 5,
    n_iter: int = 30,
) -> float:
    device = "cuda"
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k]
         for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )

    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_size_m, num_experts
    )
    num_valid_tokens = num_tokens * top_k
    output = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)

    for _ in range(n_warmup):
        output.zero_()
        invoke_fused_moe_triton_kernel(
            hidden_states, w, output,
            None, None, topk_weights,
            sorted_token_ids, expert_ids, num_post,
            True, top_k, config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        output.zero_()
        invoke_fused_moe_triton_kernel(
            hidden_states, w, output,
            None, None, topk_weights,
            sorted_token_ids, expert_ids, num_post,
            True, top_k, config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


def get_iters(num_tokens: int):
    if num_tokens >= 16384:
        return 3, 15
    elif num_tokens >= 4096:
        return 5, 20
    elif num_tokens >= 256:
        return 5, 30
    else:
        return 8, 40


def main():
    num_experts = 8
    top_k = 2
    K = 4096
    N = 4096
    dtype = torch.bfloat16

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536]

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: E={num_experts}, top_k={top_k}, K={K}, N={N}")
    print()

    best_configs: dict[int, dict] = {}

    # Phase 1: BLOCK_SIZE_M
    print("=" * 80)
    print("PHASE 1: BLOCK_SIZE_M sweep")
    print("=" * 80)
    bm_values = [16, 32, 64, 128]

    header = f"{'Tokens':>8s}"
    for bm in bm_values:
        header += f" | BM={bm:>3d} (ms)"
    header += " | Best"
    print(header)
    print("-" * len(header))

    for M in batch_sizes:
        n_warmup, n_iter = get_iters(M)
        row = f"{M:>8d}"
        times = {}
        for bm in bm_values:
            config = {
                "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3, "SPLIT_K": 1,
            }
            try:
                t = bench_base_config(M, num_experts, top_k, K, N, dtype, config, n_warmup, n_iter)
                times[bm] = t
                row += f" | {t:>10.3f}"
            except Exception:
                row += f" | {'ERR':>10s}"
        best_bm = min(times, key=times.get) if times else 64
        row += f" | BM={best_bm}"
        print(row)
        sys.stdout.flush()
        best_configs[M] = {
            "BLOCK_SIZE_M": best_bm, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3, "SPLIT_K": 1,
        }
        torch.cuda.empty_cache()
    print()

    # Phase 2: BK x BN
    print("=" * 80)
    print("PHASE 2: BLOCK_SIZE_K x BLOCK_SIZE_N sweep")
    print("=" * 80)
    bk_values = [32, 64, 128]
    bn_values = [32, 64, 128]
    combos = list(product(bk_values, bn_values))

    header = f"{'Tokens':>8s} | {'BM':>3s}"
    for bk, bn in combos:
        header += f" | K{bk}N{bn}"
    header += " | Best"
    print(header)
    print("-" * len(header))

    for M in batch_sizes:
        n_warmup, n_iter = get_iters(M)
        best_bm = best_configs[M]["BLOCK_SIZE_M"]
        row = f"{M:>8d} | {best_bm:>3d}"
        times = {}
        for bk, bn in combos:
            config = {
                "BLOCK_SIZE_M": best_bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": best_configs[M]["GROUP_SIZE_M"],
                "num_warps": 4, "num_stages": 3, "SPLIT_K": 1,
            }
            try:
                t = bench_base_config(M, num_experts, top_k, K, N, dtype, config, n_warmup, n_iter)
                times[(bk, bn)] = t
                row += f" | {t:>6.3f}"
            except Exception:
                row += f" | {'ERR':>6s}"
        if times:
            best_bk, best_bn = min(times, key=times.get)
            row += f" | K{best_bk}N{best_bn}"
            best_configs[M]["BLOCK_SIZE_K"] = best_bk
            best_configs[M]["BLOCK_SIZE_N"] = best_bn
        print(row)
        sys.stdout.flush()
        torch.cuda.empty_cache()
    print()

    # Phase 3: GM x NW x NS
    print("=" * 80)
    print("PHASE 3: GROUP_SIZE_M x num_warps x num_stages sweep")
    print("=" * 80)
    gm_values = [1, 4, 8, 16]
    nw_values = [4, 8]
    ns_values = [2, 3, 4, 5]

    header = f"{'Tokens':>8s} | {'BM':>3s} | {'BK':>3s} | {'BN':>3s} | Best GM/NW/NS | Time (ms)"
    print(header)
    print("-" * len(header))

    for M in batch_sizes:
        n_warmup, n_iter = get_iters(M)
        bc = best_configs[M]
        row = f"{M:>8d} | {bc['BLOCK_SIZE_M']:>3d} | {bc['BLOCK_SIZE_K']:>3d} | {bc['BLOCK_SIZE_N']:>3d}"
        times = {}
        for gm, nw, ns in product(gm_values, nw_values, ns_values):
            config = {
                "BLOCK_SIZE_M": bc["BLOCK_SIZE_M"],
                "BLOCK_SIZE_N": bc["BLOCK_SIZE_N"],
                "BLOCK_SIZE_K": bc["BLOCK_SIZE_K"],
                "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns, "SPLIT_K": 1,
            }
            try:
                t = bench_base_config(M, num_experts, top_k, K, N, dtype, config, n_warmup, n_iter)
                times[(gm, nw, ns)] = t
            except Exception:
                pass
        if times:
            best_gm, best_nw, best_ns = min(times, key=times.get)
            best_t = times[(best_gm, best_nw, best_ns)]
            row += f" | GM={best_gm:>2d} NW={best_nw} NS={best_ns} | {best_t:.3f}"
            best_configs[M]["GROUP_SIZE_M"] = best_gm
            best_configs[M]["num_warps"] = best_nw
            best_configs[M]["num_stages"] = best_ns
        print(row)
        sys.stdout.flush()
        torch.cuda.empty_cache()
    print()

    # Summary
    print("=" * 80)
    print("FINAL BEST CONFIGS")
    print("=" * 80)
    header = f"{'Tokens':>8s} | {'BM':>4s} | {'BN':>4s} | {'BK':>4s} | {'GM':>3s} | {'NW':>3s} | {'NS':>3s}"
    print(header)
    print("-" * len(header))
    for M in batch_sizes:
        c = best_configs[M]
        print(f"{M:>8d} | {c['BLOCK_SIZE_M']:>4d} | {c['BLOCK_SIZE_N']:>4d} | "
              f"{c['BLOCK_SIZE_K']:>4d} | {c['GROUP_SIZE_M']:>3d} | "
              f"{c['num_warps']:>3d} | {c['num_stages']:>3d}")

    # Save to vLLM config dir
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    E = num_experts
    output = {str(M): best_configs[M] for M in batch_sizes}
    configs_dir = "vllm/model_executor/layers/fused_moe/configs"
    filename = f"E={E},N={N},device_name={device_name},dtype=bfloat16.json"
    path = os.path.join(configs_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
