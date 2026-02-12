"""
Benchmark: Fused MoE+LoRA kernel vs Separate (base GEMM + standalone LoRA).

Sweeps num_tokens in powers of 2 from 1 to 65536.
Prints results to stdout and writes a markdown table to benchmark_results.md.
"""

import sys
import time

import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    invoke_fused_moe_lora_kernel,
)
from vllm.lora.ops.triton_ops.moe_lora_align import (
    moe_lora_align_block_size_fused,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

# Import the separate LoRA op
from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _fused_moe_lora_shrink,
    _fused_moe_lora_expand,
)


def benchmark_one(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    K: int,
    N: int,
    max_loras: int,
    rank: int,
    num_slices: int,
    dtype: torch.dtype,
    n_warmup: int = 10,
    n_iter: int = 50,
):
    device = "cuda"
    torch.manual_seed(0)

    N_total = N * num_slices

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N_total, K, device=device, dtype=dtype) * 0.01

    # LoRA weights for fused path: (NUM_SLICES, max_loras, E, rank, K)
    lora_a_fused = (
        torch.randn(num_slices, max_loras, num_experts, rank, K,
                     device=device, dtype=dtype) * 0.01
    )
    lora_b_fused = (
        torch.randn(num_slices, max_loras, num_experts, N, rank,
                     device=device, dtype=dtype) * 0.01
    )

    # LoRA weights for separate path: list of (max_loras, E, rank, K)
    lora_a_separate = [
        lora_a_fused[s] for s in range(num_slices)
    ]
    lora_b_separate = [
        lora_b_fused[s] for s in range(num_slices)
    ]

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k]
         for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )

    # Token-level LoRA assignment
    lora_ids_per_token = torch.randint(
        0, max_loras, (num_tokens,), device=device, dtype=torch.int64
    )
    # adapter_enabled (all adapters active)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int, device=device)

    num_valid_tokens = num_tokens * top_k
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    # Get config
    w2_shape = (num_experts, K, N_total)  # dummy for config
    config = try_get_optimal_moe_config(
        w.size(), w2_shape, top_k, "bfloat16", num_tokens
    )
    block_size_m = config["BLOCK_SIZE_M"]

    # =====================================================================
    # Setup for FUSED path
    # =====================================================================
    sorted_token_ids_f, expert_ids_f, lora_ids_f, num_post_f = (
        moe_lora_align_block_size_fused(
            topk_ids, lora_ids_per_token, block_size_m, num_experts, max_loras
        )
    )
    output_fused = torch.zeros(
        num_valid_tokens, N_total, device=device, dtype=dtype
    )

    # =====================================================================
    # Setup for SEPARATE path (base GEMM + separate LoRA)
    # =====================================================================
    sorted_token_ids_s, expert_ids_s, num_post_s = moe_align_block_size(
        topk_ids, block_size_m, num_experts
    )
    output_base = torch.zeros(
        num_tokens, top_k, N_total, device=device, dtype=dtype
    )

    token_lora_mapping = lora_ids_per_token.to(torch.int32)
    lora_ids_int = lora_ids_per_token.to(torch.int64)

    # Shrink output buffer: (num_slices, num_tokens, top_k, rank)
    a_intermediate = torch.zeros(
        num_slices, num_tokens, top_k, rank, device=device, dtype=dtype
    )

    # =====================================================================
    # Warmup FUSED
    # =====================================================================
    for _ in range(n_warmup):
        output_fused.zero_()
        invoke_fused_moe_lora_kernel(
            hidden_states, w, output_fused, topk_weights,
            sorted_token_ids_f, expert_ids_f, lora_ids_f, num_post_f,
            lora_a_fused, lora_b_fused,
            True, top_k, num_slices, config,
            compute_type=compute_type,
        )
    torch.cuda.synchronize()

    # =====================================================================
    # Warmup SEPARATE (base GEMM only, then standalone LoRA)
    # =====================================================================
    for _ in range(n_warmup):
        output_base.zero_()
        invoke_fused_moe_triton_kernel(
            hidden_states, w, output_base,
            None, None,
            topk_weights,
            sorted_token_ids_s, expert_ids_s, num_post_s,
            True, top_k, config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )
        a_intermediate.zero_()
        _fused_moe_lora_shrink(
            a_intermediate,
            hidden_states,
            lora_a_separate,
            topk_weights,
            None, topk_ids.reshape(-1), None,
            token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled,
            device=torch.device(device),
            N=rank, M=num_tokens,
            EM=num_tokens * top_k * block_size_m,
            K=K, num_tokens=num_tokens * top_k,
            num_experts=num_experts, num_slices=num_slices,
            block_size_m=block_size_m,
            block_size_n=min(64, rank),
            block_size_k=32,
            group_size_m=8,
            num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras,
            mul_routed_weight=False,
        )
        _fused_moe_lora_expand(
            output_base,
            a_intermediate,
            lora_b_separate,
            topk_weights,
            None, topk_ids.reshape(-1), None,
            token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled,
            device=torch.device(device),
            N=rank, M=num_tokens,
            EM=num_tokens * top_k * block_size_m,
            K=K, num_tokens=num_tokens * top_k,
            num_experts=num_experts, num_slices=num_slices,
            max_lora_rank=rank, w1_output_dim_size=N,
            block_size_m=block_size_m,
            block_size_n=min(64, N),
            block_size_k=min(32, rank),
            group_size_m=8,
            num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras,
            mul_routed_weight=True,
        )
    torch.cuda.synchronize()

    # =====================================================================
    # Benchmark FUSED
    # =====================================================================
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        output_fused.zero_()
        invoke_fused_moe_lora_kernel(
            hidden_states, w, output_fused, topk_weights,
            sorted_token_ids_f, expert_ids_f, lora_ids_f, num_post_f,
            lora_a_fused, lora_b_fused,
            True, top_k, num_slices, config,
            compute_type=compute_type,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    fused_ms = (t1 - t0) / n_iter * 1000

    # =====================================================================
    # Benchmark SEPARATE (base GEMM + separate LoRA shrink + expand)
    # =====================================================================
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        output_base.zero_()
        invoke_fused_moe_triton_kernel(
            hidden_states, w, output_base,
            None, None,
            topk_weights,
            sorted_token_ids_s, expert_ids_s, num_post_s,
            True, top_k, config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )
        a_intermediate.zero_()
        _fused_moe_lora_shrink(
            a_intermediate,
            hidden_states,
            lora_a_separate,
            topk_weights,
            None, topk_ids.reshape(-1), None,
            token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled,
            device=torch.device(device),
            N=rank, M=num_tokens,
            EM=num_tokens * top_k * block_size_m,
            K=K, num_tokens=num_tokens * top_k,
            num_experts=num_experts, num_slices=num_slices,
            block_size_m=block_size_m,
            block_size_n=min(64, rank),
            block_size_k=32,
            group_size_m=8,
            num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras,
            mul_routed_weight=False,
        )
        _fused_moe_lora_expand(
            output_base,
            a_intermediate,
            lora_b_separate,
            topk_weights,
            None, topk_ids.reshape(-1), None,
            token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled,
            device=torch.device(device),
            N=rank, M=num_tokens,
            EM=num_tokens * top_k * block_size_m,
            K=K, num_tokens=num_tokens * top_k,
            num_experts=num_experts, num_slices=num_slices,
            max_lora_rank=rank, w1_output_dim_size=N,
            block_size_m=block_size_m,
            block_size_n=min(64, N),
            block_size_k=min(32, rank),
            group_size_m=8,
            num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras,
            mul_routed_weight=True,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    separate_ms = (t1 - t0) / n_iter * 1000

    # =====================================================================
    # Benchmark BASE GEMM ONLY (no LoRA at all, for reference)
    # =====================================================================
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        output_base.zero_()
        invoke_fused_moe_triton_kernel(
            hidden_states, w, output_base,
            None, None,
            topk_weights,
            sorted_token_ids_s, expert_ids_s, num_post_s,
            True, top_k, config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    base_only_ms = (t1 - t0) / n_iter * 1000

    return base_only_ms, separate_ms, fused_ms


def main():
    # Model config: Mixtral-like (8 experts, top_k=2, K=4096, N=4096)
    num_experts = 8
    top_k = 2
    K = 4096
    N = 4096
    max_loras = 4
    rank = 16
    num_slices = 1
    dtype = torch.bfloat16

    # Powers of 2 from 1 to 65536
    token_counts = [2**i for i in range(17)]  # 1, 2, 4, ..., 65536

    print("=" * 95)
    print("Benchmark: Fused MoE+LoRA vs Separate (Base GEMM + Standalone LoRA)")
    print(f"Config: E={num_experts}, top_k={top_k}, K={K}, N={N}, "
          f"max_loras={max_loras}, rank={rank}, slices={num_slices}, "
          f"dtype=bf16")
    print("=" * 95)
    print()

    header = (
        f"{'Tokens':>8s} | {'Base (ms)':>10s} | {'Separate (ms)':>14s} | "
        f"{'Fused (ms)':>11s} | {'Sep OH (ms)':>12s} | {'Fused OH (ms)':>14s} | "
        f"{'Speedup':>7s}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for num_tokens in token_counts:
        # Adjust iterations for very large token counts to keep runtime sane
        if num_tokens >= 16384:
            n_warmup, n_iter = 5, 20
        elif num_tokens >= 4096:
            n_warmup, n_iter = 5, 30
        else:
            n_warmup, n_iter = 10, 50

        try:
            base_ms, sep_ms, fused_ms = benchmark_one(
                num_tokens, num_experts, top_k, K, N, max_loras, rank,
                num_slices, dtype, n_warmup=n_warmup, n_iter=n_iter,
            )
        except Exception as e:
            print(f"{num_tokens:>8d} | {'ERROR':>10s} | {str(e)[:50]}")
            results.append((num_tokens, None, None, None))
            continue

        sep_overhead = sep_ms - base_ms
        fused_overhead = fused_ms - base_ms
        speedup = sep_ms / fused_ms if fused_ms > 0 else float("inf")

        print(
            f"{num_tokens:>8d} | {base_ms:>9.3f}  | {sep_ms:>13.3f}  | "
            f"{fused_ms:>10.3f}  | {sep_overhead:>+11.3f}  | {fused_overhead:>+13.3f}  | "
            f"{speedup:>6.2f}x"
        )
        sys.stdout.flush()

        results.append((num_tokens, base_ms, sep_ms, fused_ms))

        # Free GPU memory between runs
        torch.cuda.empty_cache()

    print()
    print("Legend:")
    print("  Base       = base expert GEMM only (no LoRA)")
    print("  Separate   = base GEMM + separate LoRA shrink + expand kernels")
    print("  Fused      = single fused GEMM+LoRA kernel")
    print("  Sep OH     = overhead of separate LoRA over base (Separate - Base)")
    print("  Fused OH   = overhead of fused LoRA over base (Fused - Base)")
    print("  Speedup    = Separate / Fused (overall, higher is better)")
    print()

    # =====================================================================
    # Write markdown table
    # =====================================================================
    gpu_name = torch.cuda.get_device_name(0)
    md_lines = [
        "# Fused MoE+LoRA Kernel Benchmark Results",
        "",
        f"**GPU**: {gpu_name}  ",
        f"**Config**: E={num_experts}, top_k={top_k}, K={K}, N={N}, "
        f"max_loras={max_loras}, rank={rank}, slices={num_slices}, dtype=bf16  ",
        "",
        "| Tokens | Base (ms) | Separate (ms) | Fused (ms) | Sep OH (ms) | Fused OH (ms) | Speedup |",
        "|-------:|----------:|--------------:|-----------:|------------:|--------------:|--------:|",
    ]

    for num_tokens, base_ms, sep_ms, fused_ms in results:
        if base_ms is None:
            md_lines.append(
                f"| {num_tokens:,} | ERROR | — | — | — | — | — |"
            )
        else:
            sep_oh = sep_ms - base_ms
            fused_oh = fused_ms - base_ms
            speedup = sep_ms / fused_ms if fused_ms > 0 else float("inf")
            md_lines.append(
                f"| {num_tokens:,} | {base_ms:.3f} | {sep_ms:.3f} | "
                f"{fused_ms:.3f} | {sep_oh:+.3f} | {fused_oh:+.3f} | "
                f"{speedup:.2f}x |"
            )

    md_lines.extend([
        "",
        "### Legend",
        "",
        "- **Base** — base expert GEMM only (no LoRA)",
        "- **Separate** — base GEMM + separate LoRA shrink + expand kernels",
        "- **Fused** — single fused GEMM+LoRA kernel",
        "- **Sep OH** — overhead of separate LoRA over base (Separate − Base)",
        "- **Fused OH** — overhead of fused LoRA over base (Fused − Base)",
        "- **Speedup** — Separate / Fused (overall, >1 means fused wins)",
        "",
    ])

    md_path = "benchmark_results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown table written to {md_path}")


if __name__ == "__main__":
    main()
