"""
Focused tuning of the bf16acc variant (best performer from kernel_variants).

Tests different BLOCK_SIZE_K and BLOCK_SIZE_N to find the sweet spot:
- Larger BK = fewer K-iterations = fewer LoRA-A loads
- Smaller BN = smaller main accumulator = more register headroom
"""
import json
import sys
import time

import torch

from vllm.triton_utils import tl, triton

from benchmark_kernel_variants import fused_moe_lora_bf16acc, launch_variant
from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


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

    # Base kernel: use its best tuned config
    base_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}
    bm = base_config["BLOCK_SIZE_M"]

    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, bm, num_experts)

    out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)

    # Bench base
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
    base_ms = (time.perf_counter() - t0) / n_iter * 1000

    results = {"base": base_ms}

    # Test configs for bf16acc variant
    configs = [
        # (label, config_dict)
        # Baseline: same as base kernel config but NW=8
        ("BN128_BK32_NW4_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                                  "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
        ("BN128_BK32_NW8_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                                  "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}),
        # Larger BK: fewer iterations, fewer LoRA-A loads
        ("BN128_BK64_NW4_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                                  "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
        ("BN128_BK64_NW8_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                                  "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}),
        ("BN128_BK128_NW4_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                                   "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
        ("BN128_BK128_NW8_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                                   "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}),
        # Smaller BN: smaller main accumulator
        ("BN64_BK32_NW4_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                                 "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
        ("BN64_BK64_NW4_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                                 "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
        ("BN64_BK64_NW4_NS3", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                                 "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3}),
        # More pipeline stages to hide LoRA-A latency
        ("BN128_BK32_NW8_NS3", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                                  "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3}),
        ("BN128_BK32_NW8_NS4", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                                  "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4}),
        ("BN128_BK64_NW8_NS3", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                                  "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3}),
        # Larger BM to amortize LoRA overhead across more tokens
        ("BM32_BN128_BK32_NW8_NS2", {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                                      "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}),
        ("BM32_BN64_BK64_NW4_NS2", {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                                      "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
    ]

    # Also test the original (fp32 acc) kernel with same configs for comparison
    orig_configs = [
        ("ORIG_BN128_BK32_NW8_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                                       "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2, "SPLIT_K": 1}),
        ("ORIG_BN128_BK64_NW8_NS2", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                                       "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2, "SPLIT_K": 1}),
    ]

    for label, cfg in configs:
        bm_cfg = cfg.get("BLOCK_SIZE_M", 16)
        sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, bm_cfg, num_experts, max_loras)
        out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)
        try:
            for _ in range(n_warmup):
                out_fused.zero_()
                launch_variant(fused_moe_lora_bf16acc, hidden, w, out_fused, topk_weights.reshape(-1),
                               sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                               top_k, 1, cfg, compute_type)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                launch_variant(fused_moe_lora_bf16acc, hidden, w, out_fused, topk_weights.reshape(-1),
                               sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                               top_k, 1, cfg, compute_type)
            torch.cuda.synchronize()
            results[label] = (time.perf_counter() - t0) / n_iter * 1000
        except Exception as e:
            results[label] = None
            print(f"  {label}: ERROR - {e}", file=sys.stderr)

    # Original kernel with different BK
    for label, cfg in orig_configs:
        sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, 16, num_experts, max_loras)
        out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)
        try:
            for _ in range(n_warmup):
                out_fused.zero_()
                invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
                    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                    True, top_k, 1, cfg, compute_type=compute_type)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
                    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                    True, top_k, 1, cfg, compute_type=compute_type)
            torch.cuda.synchronize()
            results[label] = (time.perf_counter() - t0) / n_iter * 1000
        except Exception as e:
            results[label] = None
            print(f"  {label}: ERROR - {e}", file=sys.stderr)

    return results


def main():
    batch_sizes = [64, 256, 1024, 4096, 65536]

    print("bf16acc variant tuning: testing BK, BN, NW, NS combinations")
    print("Base kernel: BM=16 BN=128 BK=32 GM=16 NW=4 NS=2 (tuned for B200)")
    print()

    all_results = {}
    all_keys = None
    for M in batch_sizes:
        nw, ni = (3, 15) if M >= 16384 else (5, 25) if M >= 4096 else (8, 40)
        print(f"  Running M={M}...", end=" ", flush=True)
        r = bench(M, n_warmup=nw, n_iter=ni)
        all_results[M] = r
        if all_keys is None:
            all_keys = list(r.keys())
        print("done")
        torch.cuda.empty_cache()

    # Print compact table
    print()
    print(f"{'Config':<30s}", end="")
    for M in batch_sizes:
        print(f" | {M:>8d}", end="")
    print()
    print("-" * (30 + 11 * len(batch_sizes)))

    for k in all_keys:
        print(f"{k:<30s}", end="")
        for M in batch_sizes:
            v = all_results[M][k]
            if v is None:
                print(f" | {'ERR':>8s}", end="")
            elif k == "base":
                print(f" | {v:>7.3f}ms", end="")
            else:
                base = all_results[M]["base"]
                oh = (v - base) / base * 100
                print(f" |  {oh:>+5.1f}%  ", end="")
        print()

    # Also print absolute times for best configs
    print()
    print("Absolute times (ms) for top configs:")
    print(f"{'Config':<30s}", end="")
    for M in batch_sizes:
        print(f" | {M:>8d}", end="")
    print()
    print("-" * (30 + 11 * len(batch_sizes)))
    for k in all_keys:
        print(f"{k:<30s}", end="")
        for M in batch_sizes:
            v = all_results[M][k]
            if v is None:
                print(f" | {'ERR':>8s}", end="")
            else:
                print(f" | {v:>7.3f}ms", end="")
        print()


if __name__ == "__main__":
    main()
