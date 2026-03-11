"""
Config sweep benchmark for the fused MoE+LoRA Triton kernel.

Three-phase sweep to find optimal tile configs per batch size:
  Phase 1: BLOCK_SIZE_M sweep (tests padding hypothesis)
  Phase 2: BLOCK_SIZE_K × BLOCK_SIZE_N sweep (using best BM per batch)
  Phase 3: num_warps × num_stages × GROUP_SIZE_M sweep (using best BM/BK/BN)

Saves best-per-batch-size configs to JSON compatible with vLLM config loader.
"""

import json
import sys
import time
from itertools import product

import torch
import triton.language as tl

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    invoke_fused_moe_lora_kernel,
)
from vllm.lora.ops.triton_ops.moe_lora_align import (
    moe_lora_align_block_size_fused,
)


def bench_config(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    K: int,
    N: int,
    max_loras: int,
    rank: int,
    num_slices: int,
    dtype: torch.dtype,
    config: dict,
    n_warmup: int = 5,
    n_iter: int = 30,
) -> float:
    """Run fused kernel with a specific config and return time in ms."""
    device = "cuda"
    torch.manual_seed(0)

    N_total = N * num_slices
    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N_total, K, device=device, dtype=dtype) * 0.01

    lora_a = (
        torch.randn(num_slices, max_loras, num_experts, rank, K,
                     device=device, dtype=dtype) * 0.01
    )
    lora_b = (
        torch.randn(num_slices, max_loras, num_experts, N, rank,
                     device=device, dtype=dtype) * 0.01
    )

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k]
         for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )
    lora_ids_per_token = torch.randint(
        0, max_loras, (num_tokens,), device=device, dtype=torch.int64
    )

    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, expert_ids, lora_ids, num_post = (
        moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, block_size_m, num_experts, max_loras
        )
    )
    num_valid_tokens = num_tokens * top_k
    output = torch.zeros(num_valid_tokens, N_total, device=device, dtype=dtype)

    # Warmup
    for _ in range(n_warmup):
        output.zero_()
        invoke_fused_moe_lora_kernel(
            hidden_states, w, output, topk_weights,
            sorted_token_ids, expert_ids, lora_ids, num_post,
            lora_a, lora_b,
            True, top_k, num_slices, config,
            compute_type=compute_type,
        )
    torch.cuda.synchronize()

    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        output.zero_()
        invoke_fused_moe_lora_kernel(
            hidden_states, w, output, topk_weights,
            sorted_token_ids, expert_ids, lora_ids, num_post,
            lora_a, lora_b,
            True, top_k, num_slices, config,
            compute_type=compute_type,
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
    # Model config: Mixtral-like
    num_experts = 8
    top_k = 2
    K = 4096
    N = 4096
    max_loras = 4
    rank = 16
    num_slices = 1
    dtype = torch.bfloat16

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536]

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: E={num_experts}, top_k={top_k}, K={K}, N={N}, "
          f"max_loras={max_loras}, rank={rank}, slices={num_slices}")
    print()

    # Track best configs per batch size
    best_configs: dict[int, dict] = {}

    # =================================================================
    # Phase 1: BLOCK_SIZE_M sweep
    # =================================================================
    print("=" * 80)
    print("PHASE 1: BLOCK_SIZE_M sweep")
    print("Fixed: BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, "
          "num_warps=4, num_stages=3")
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
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
            }
            try:
                t = bench_config(
                    M, num_experts, top_k, K, N, max_loras, rank,
                    num_slices, dtype, config, n_warmup, n_iter,
                )
                times[bm] = t
                row += f" | {t:>10.3f}"
            except Exception as e:
                row += f" | {'ERR':>10s}"
        best_bm = min(times, key=times.get) if times else 64
        row += f" | BM={best_bm}"
        print(row)
        sys.stdout.flush()

        best_configs[M] = {
            "BLOCK_SIZE_M": best_bm,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
        torch.cuda.empty_cache()

    print()

    # =================================================================
    # Phase 2: BLOCK_SIZE_K × BLOCK_SIZE_N sweep
    # =================================================================
    print("=" * 80)
    print("PHASE 2: BLOCK_SIZE_K × BLOCK_SIZE_N sweep (using best BM per batch)")
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
                "BLOCK_SIZE_M": best_bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": best_configs[M]["GROUP_SIZE_M"],
                "num_warps": 4,
                "num_stages": 3,
            }
            try:
                t = bench_config(
                    M, num_experts, top_k, K, N, max_loras, rank,
                    num_slices, dtype, config, n_warmup, n_iter,
                )
                times[(bk, bn)] = t
                row += f" | {t:>6.3f}"
            except Exception as e:
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

    # =================================================================
    # Phase 3: num_warps × num_stages × GROUP_SIZE_M sweep
    # =================================================================
    print("=" * 80)
    print("PHASE 3: num_warps × num_stages × GROUP_SIZE_M sweep")
    print("=" * 80)

    gm_values = [1, 4, 8, 16]
    nw_values = [4, 8]
    ns_values = [2, 3, 4, 5]
    phase3_combos = list(product(gm_values, nw_values, ns_values))

    header = f"{'Tokens':>8s} | {'BM':>3s} | {'BK':>3s} | {'BN':>3s}"
    header += " | Best GM/NW/NS | Time (ms)"
    print(header)
    print("-" * len(header))

    for M in batch_sizes:
        n_warmup, n_iter = get_iters(M)
        bc = best_configs[M]
        row = f"{M:>8d} | {bc['BLOCK_SIZE_M']:>3d} | {bc['BLOCK_SIZE_K']:>3d} | {bc['BLOCK_SIZE_N']:>3d}"
        times = {}
        for gm, nw, ns in phase3_combos:
            config = {
                "BLOCK_SIZE_M": bc["BLOCK_SIZE_M"],
                "BLOCK_SIZE_N": bc["BLOCK_SIZE_N"],
                "BLOCK_SIZE_K": bc["BLOCK_SIZE_K"],
                "GROUP_SIZE_M": gm,
                "num_warps": nw,
                "num_stages": ns,
            }
            try:
                t = bench_config(
                    M, num_experts, top_k, K, N, max_loras, rank,
                    num_slices, dtype, config, n_warmup, n_iter,
                )
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

    # =================================================================
    # Summary
    # =================================================================
    print("=" * 80)
    print("FINAL BEST CONFIGS PER BATCH SIZE")
    print("=" * 80)
    header = (f"{'Tokens':>8s} | {'BM':>4s} | {'BN':>4s} | {'BK':>4s} | "
              f"{'GM':>3s} | {'NW':>3s} | {'NS':>3s}")
    print(header)
    print("-" * len(header))
    for M in batch_sizes:
        c = best_configs[M]
        print(f"{M:>8d} | {c['BLOCK_SIZE_M']:>4d} | {c['BLOCK_SIZE_N']:>4d} | "
              f"{c['BLOCK_SIZE_K']:>4d} | {c['GROUP_SIZE_M']:>3d} | "
              f"{c['num_warps']:>3d} | {c['num_stages']:>3d}")

    # Save to JSON
    output_json = {str(M): best_configs[M] for M in batch_sizes}
    output_path = "fused_lora_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Also save in vLLM config format
    E = num_experts
    N_cfg = K  # w2 shape for config naming
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    vllm_config = {str(M): best_configs[M] for M in batch_sizes}
    vllm_path = f"E={E},N={N_cfg},device_name={device_name},dtype=bfloat16.json"
    configs_dir = "vllm/lora/ops/triton_ops/configs"
    import os
    os.makedirs(configs_dir, exist_ok=True)
    vllm_config_path = os.path.join(configs_dir, vllm_path)
    with open(vllm_config_path, "w") as f:
        json.dump(vllm_config, f, indent=2)
    print(f"Saved vLLM config to {vllm_config_path}")


if __name__ == "__main__":
    main()
