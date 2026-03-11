"""Isolate the NW=4 vs NW=8 penalty for the base kernel alone.
Then compare: does fused@NW8 beat base@NW8 + lora_overhead?"""
import json
import sys
import time

import torch

import triton.language as tl
from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from benchmark_kernel_variants import fused_moe_lora_bf16acc, launch_variant

def run(num_tokens, num_experts=8, top_k=2, K=4096, N=4096, max_loras=1,
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

    results = {}

    # Test base kernel at NW=4 and NW=8 (same tile sizes otherwise)
    for nw in [4, 8]:
        for ns in [2, 3, 4]:
            cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 16, "num_warps": nw, "num_stages": ns, "SPLIT_K": 1}
            bm = cfg["BLOCK_SIZE_M"]
            sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, bm, num_experts)
            out = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)

            for _ in range(n_warmup):
                invoke_fused_moe_triton_kernel(hidden, w, out, None, None, topk_weights,
                    sorted_b, expert_b, npost_b, True, top_k, cfg, compute_type=compute_type,
                    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
                    use_int4_w4a16=False, per_channel_quant=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                invoke_fused_moe_triton_kernel(hidden, w, out, None, None, topk_weights,
                    sorted_b, expert_b, npost_b, True, top_k, cfg, compute_type=compute_type,
                    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
                    use_int4_w4a16=False, per_channel_quant=False)
            torch.cuda.synchronize()
            results[f"base_NW{nw}_NS{ns}"] = (time.perf_counter() - t0) / n_iter * 1000

    # Test fused kernel (fp32 acc and bf16 acc) at NW=4 and NW=8
    for nw in [4, 8]:
        for ns in [2, 3, 4]:
            cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 16, "num_warps": nw, "num_stages": ns}
            bm = cfg["BLOCK_SIZE_M"]
            sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
                topk_ids, lora_ids_per_token, bm, num_experts, max_loras)
            out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

            # fp32 acc (original)
            key = f"fused_fp32_NW{nw}_NS{ns}"
            try:
                full_cfg = {**cfg, "SPLIT_K": 1}
                for _ in range(n_warmup):
                    out_fused.zero_()
                    invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
                        sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                        True, top_k, 1, full_cfg, compute_type=compute_type)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_iter):
                    invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
                        sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                        True, top_k, 1, full_cfg, compute_type=compute_type)
                torch.cuda.synchronize()
                results[key] = (time.perf_counter() - t0) / n_iter * 1000
            except Exception as e:
                results[key] = None
                print(f"  {key}: ERROR - {e}", file=sys.stderr)

            # bf16 acc
            key = f"fused_bf16_NW{nw}_NS{ns}"
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
                results[key] = (time.perf_counter() - t0) / n_iter * 1000
            except Exception as e:
                results[key] = None
                print(f"  {key}: ERROR - {e}", file=sys.stderr)

    # Also test fused with HAS_LORA=False (no lora computation) at NW=8
    # This tells us if the NW=8 penalty is from registers or just from warp count
    for nw in [4, 8]:
        cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
               "GROUP_SIZE_M": 16, "num_warps": nw, "num_stages": 2}
        bm = cfg["BLOCK_SIZE_M"]
        sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, bm, num_experts, max_loras)
        out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)
        # Set all lora_ids to -1 to disable LoRA
        lora_f_nolora = torch.full_like(lora_f, -1)

        key = f"fused_NOLORA_NW{nw}"
        try:
            full_cfg = {**cfg, "SPLIT_K": 1}
            for _ in range(n_warmup):
                out_fused.zero_()
                invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
                    sorted_f, expert_f, lora_f_nolora, npost_f, lora_a, lora_b,
                    True, top_k, 1, full_cfg, compute_type=compute_type)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
                    sorted_f, expert_f, lora_f_nolora, npost_f, lora_a, lora_b,
                    True, top_k, 1, full_cfg, compute_type=compute_type)
            torch.cuda.synchronize()
            results[key] = (time.perf_counter() - t0) / n_iter * 1000
        except Exception as e:
            results[key] = None
            print(f"  {key}: ERROR - {e}", file=sys.stderr)

    return results


batch_sizes = [64, 1024, 4096, 65536]
print("NW=4 vs NW=8 penalty analysis (max_loras=1)")
print("All configs: BM=16 BN=128 BK=32 GM=16")
print()

all_results = {}
for M in batch_sizes:
    nw, ni = (3, 15) if M >= 16384 else (5, 25) if M >= 4096 else (8, 40)
    print(f"  M={M}...", end=" ", flush=True)
    all_results[M] = run(M, n_warmup=nw, n_iter=ni)
    print("done")
    torch.cuda.empty_cache()

# Print organized table
print()
for M in batch_sizes:
    r = all_results[M]
    base4 = r["base_NW4_NS2"]
    print(f"=== M={M} (base@NW4={base4:.3f}ms) ===")

    # Group: base at different NW/NS
    print(f"  {'Kernel':<30s} {'Time(ms)':>10s} {'vs base@NW4':>12s}")
    print(f"  {'-'*55}")
    for key in sorted(r.keys()):
        v = r[key]
        if v is None:
            print(f"  {key:<30s} {'ERROR':>10s}")
        else:
            oh = (v - base4) / base4 * 100
            print(f"  {key:<30s} {v:>9.3f}ms  {oh:>+6.1f}%")
    print()
