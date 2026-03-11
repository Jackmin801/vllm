"""
Full config tuning sweep v2: base MoE, fused MoE+LoRA, and separate LoRA paths.
Now includes BM=128 and high maxnreg values for fused kernel (Blackwell TMEM).

Phase 1: Sweep base MoE kernel configs -> best per batch size
Phase 2: Sweep fused MoE+LoRA kernel configs -> best per batch size
Phase 3: Compare tuned base vs tuned separate vs tuned fused
"""
import itertools
import json
import sys
import time
import traceback
import importlib

import torch
import triton.language as tl

import vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel as fmod
from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _fused_moe_lora_shrink,
    _fused_moe_lora_expand,
)
from vllm.lora.ops.triton_ops.moe_lora_align import (
    moe_lora_align_block_size_fused,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

# ----- Problem dimensions (Llama-style MoE) -----
E = 8
N = 4096
K = 4096
RANK = 16
TOP_K = 2
DTYPE = torch.bfloat16
COMPUTE_TYPE = tl.bfloat16

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536]

# ----- Config search spaces -----
BASE_SEARCH = list(itertools.product(
    [16, 64, 128],       # BLOCK_SIZE_M
    [32, 64, 128],       # BLOCK_SIZE_N
    [32, 64],            # BLOCK_SIZE_K
    [1, 4, 8, 16],      # GROUP_SIZE_M
    [4, 8],              # num_warps
    [2, 3, 4, 5],       # num_stages
))

# Fused: now includes BM=128 and high maxnreg for Blackwell TMEM
FUSED_SEARCH = list(itertools.product(
    [16, 64, 128],       # BLOCK_SIZE_M
    [64, 128],           # BLOCK_SIZE_N
    [32, 64],            # BLOCK_SIZE_K
    [1, 4, 8, 16],      # GROUP_SIZE_M
    [4, 8],              # num_warps
    [2, 3, 4, 5],       # num_stages
    [96, 128, 192, 255], # maxnreg (high values to reduce spills on Blackwell)
))


def make_base_cfg(bm, bn, bk, gm, nw, ns):
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns, "SPLIT_K": 1}


def make_fused_cfg(bm, bn, bk, gm, nw, ns, maxnreg):
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns,
            "maxnreg": maxnreg}


def bench_base_kernel(hidden, w, out, topk_weights,
                      sorted_ids, expert_ids, npost, config, nw, ni):
    try:
        for _ in range(nw):
            invoke_fused_moe_triton_kernel(
                hidden, w, out, None, None, topk_weights,
                sorted_ids, expert_ids, npost, True, TOP_K, config,
                compute_type=COMPUTE_TYPE,
                use_fp8_w8a8=False, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            invoke_fused_moe_triton_kernel(
                hidden, w, out, None, None, topk_weights,
                sorted_ids, expert_ids, npost, True, TOP_K, config,
                compute_type=COMPUTE_TYPE,
                use_fp8_w8a8=False, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ni * 1000
    except Exception:
        return float('inf')


def bench_fused_kernel(hidden, w, out, topk_weights,
                       sorted_f, expert_f, lora_f, npost_f,
                       lora_a, lora_b, config, nw, ni):
    try:
        importlib.reload(fmod)
        for _ in range(nw):
            fmod.invoke_fused_moe_lora_kernel(
                hidden, w, out, topk_weights,
                sorted_f, expert_f, lora_f, npost_f,
                lora_a, lora_b,
                True, TOP_K, 1, config.copy(), compute_type=COMPUTE_TYPE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            fmod.invoke_fused_moe_lora_kernel(
                hidden, w, out, topk_weights,
                sorted_f, expert_f, lora_f, npost_f,
                lora_a, lora_b,
                True, TOP_K, 1, config.copy(), compute_type=COMPUTE_TYPE)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ni * 1000
    except Exception:
        return float('inf')


def bench_separate(hidden, w, out_base, topk_weights, topk_ids,
                   sorted_b, expert_b, npost_b,
                   lora_a_sep, lora_b_sep, lora_ids_per_token,
                   base_config, max_loras, num_tokens, nw, ni):
    base_bm = base_config["BLOCK_SIZE_M"]
    token_lora_mapping = lora_ids_per_token.to(torch.int32)
    lora_ids_int = lora_ids_per_token.to(torch.int64)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int, device="cuda")
    a_intermediate = torch.zeros(1, num_tokens, TOP_K, RANK,
                                 device="cuda", dtype=DTYPE)

    def run_once():
        invoke_fused_moe_triton_kernel(
            hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, TOP_K, base_config,
            compute_type=COMPUTE_TYPE,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False)
        a_intermediate.zero_()
        _fused_moe_lora_shrink(
            a_intermediate, hidden, lora_a_sep, topk_weights,
            None, topk_ids.reshape(-1), None, token_lora_mapping,
            TOP_K, lora_ids_int, adapter_enabled,
            device=torch.device("cuda"),
            N=RANK, M=num_tokens, EM=num_tokens * TOP_K * base_bm, K=K,
            num_tokens=num_tokens * TOP_K, num_experts=E, num_slices=1,
            block_size_m=base_bm, block_size_n=min(64, RANK),
            block_size_k=32, group_size_m=8, num_warps=4, num_stages=3,
            split_k=1, num_active_loras=max_loras, mul_routed_weight=False)
        _fused_moe_lora_expand(
            out_base, a_intermediate, lora_b_sep, topk_weights,
            None, topk_ids.reshape(-1), None, token_lora_mapping,
            TOP_K, lora_ids_int, adapter_enabled,
            device=torch.device("cuda"),
            N=RANK, M=num_tokens, EM=num_tokens * TOP_K * base_bm, K=K,
            num_tokens=num_tokens * TOP_K, num_experts=E, num_slices=1,
            max_lora_rank=RANK, w1_output_dim_size=N,
            block_size_m=base_bm, block_size_n=min(64, N),
            block_size_k=min(32, RANK), group_size_m=8,
            num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras, mul_routed_weight=True)

    try:
        for _ in range(nw):
            run_once()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            run_once()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ni * 1000
    except Exception as e:
        traceback.print_exc()
        return float('inf')


def iter_params(M):
    if M >= 16384:
        return 3, 10
    elif M >= 1024:
        return 5, 20
    else:
        return 5, 30


def main():
    device = "cuda"
    torch.manual_seed(42)
    max_loras = 1

    best_base = {}
    best_fused = {}

    # ========== PHASE 1: Sweep base MoE kernel ==========
    print("=" * 100)
    print("PHASE 1: Sweeping base MoE kernel configs")
    print("=" * 100)

    for M in BATCH_SIZES:
        nw, ni = iter_params(M)
        hidden = torch.randn(M, K, device=device, dtype=DTYPE)
        w = torch.randn(E, N, K, device=device, dtype=DTYPE) * 0.01
        topk_ids = torch.stack([
            torch.randperm(E, device=device)[:TOP_K] for _ in range(M)
        ]).to(torch.int64)
        topk_weights = torch.softmax(
            torch.randn(M, TOP_K, device=device, dtype=DTYPE), dim=-1)

        best_ms = float('inf')
        best_cfg = None
        tested = 0
        errors = 0

        from collections import defaultdict
        by_bm = defaultdict(list)
        for bm, bn, bk, gm, nw_c, ns in BASE_SEARCH:
            if bm > max(M * TOP_K, 16) and bm > 16:
                continue
            by_bm[bm].append((bm, bn, bk, gm, nw_c, ns))

        for bm, cfgs in by_bm.items():
            sorted_b, expert_b, npost_b = moe_align_block_size(
                topk_ids, bm, E)
            out = torch.zeros(M, TOP_K, N, device=device, dtype=DTYPE)

            for _, bn, bk, gm, nw_c, ns in cfgs:
                config = make_base_cfg(bm, bn, bk, gm, nw_c, ns)
                ms = bench_base_kernel(hidden, w, out, topk_weights,
                                       sorted_b, expert_b, npost_b,
                                       config, nw, ni)
                tested += 1
                if ms == float('inf'):
                    errors += 1
                elif ms < best_ms:
                    best_ms = ms
                    best_cfg = config.copy()

        best_base[M] = (best_ms, best_cfg)
        cfg_str = (f"BM={best_cfg['BLOCK_SIZE_M']} BN={best_cfg['BLOCK_SIZE_N']} "
                   f"BK={best_cfg['BLOCK_SIZE_K']} GM={best_cfg['GROUP_SIZE_M']} "
                   f"NW={best_cfg['num_warps']} NS={best_cfg['num_stages']}")
        print(f"  M={M:>6d}: {best_ms:>8.3f} ms  ({tested} tested, {errors} errors)  "
              f"best: {cfg_str}")
        sys.stdout.flush()
        torch.cuda.empty_cache()

    # ========== PHASE 2: Sweep fused MoE+LoRA kernel ==========
    print()
    print("=" * 100)
    print("PHASE 2: Sweeping fused MoE+LoRA kernel configs (Blackwell TMEM-aware)")
    print("=" * 100)

    for M in BATCH_SIZES:
        nw, ni = iter_params(M)
        hidden = torch.randn(M, K, device=device, dtype=DTYPE)
        w = torch.randn(E, N, K, device=device, dtype=DTYPE) * 0.01
        lora_a = torch.randn(1, max_loras, E, RANK, K,
                             device=device, dtype=DTYPE) * 0.01
        lora_b = torch.randn(1, max_loras, E, N, RANK,
                             device=device, dtype=DTYPE) * 0.01
        topk_ids = torch.stack([
            torch.randperm(E, device=device)[:TOP_K] for _ in range(M)
        ]).to(torch.int64)
        topk_weights = torch.softmax(
            torch.randn(M, TOP_K, device=device, dtype=DTYPE), dim=-1)
        lora_ids_per_token = torch.zeros(M, device=device, dtype=torch.int64)

        best_ms = float('inf')
        best_cfg = None
        tested = 0
        errors = 0

        from collections import defaultdict
        by_bm = defaultdict(list)
        for bm, bn, bk, gm, nw_c, ns, maxnreg in FUSED_SEARCH:
            if bm > max(M * TOP_K, 16) and bm > 16:
                continue
            by_bm[bm].append((bm, bn, bk, gm, nw_c, ns, maxnreg))

        for bm, cfgs in by_bm.items():
            sorted_f, expert_f, lora_f, npost_f = \
                moe_lora_align_block_size_fused(
                    topk_ids, lora_ids_per_token, bm, E, max_loras)
            out_f = torch.zeros(M * TOP_K, N, device=device, dtype=DTYPE)

            for _, bn, bk, gm, nw_c, ns, maxnreg in cfgs:
                config = make_fused_cfg(bm, bn, bk, gm, nw_c, ns, maxnreg)
                ms = bench_fused_kernel(
                    hidden, w, out_f, topk_weights,
                    sorted_f, expert_f, lora_f, npost_f,
                    lora_a, lora_b, config, nw, ni)
                tested += 1
                if ms == float('inf'):
                    errors += 1
                elif ms < best_ms:
                    best_ms = ms
                    best_cfg = config.copy()

        best_fused[M] = (best_ms, best_cfg)
        cfg_str = (f"BM={best_cfg['BLOCK_SIZE_M']} BN={best_cfg['BLOCK_SIZE_N']} "
                   f"BK={best_cfg['BLOCK_SIZE_K']} GM={best_cfg['GROUP_SIZE_M']} "
                   f"NW={best_cfg['num_warps']} NS={best_cfg['num_stages']} "
                   f"maxnreg={best_cfg.get('maxnreg', 'none')}")
        print(f"  M={M:>6d}: {best_ms:>8.3f} ms  ({tested} tested, {errors} errors)  "
              f"best: {cfg_str}")
        sys.stdout.flush()
        torch.cuda.empty_cache()

    # ========== PHASE 3: Final comparison ==========
    print()
    print("=" * 100)
    print("PHASE 3: Final comparison with tuned configs (max_loras=1)")
    print("=" * 100)
    print(f"\n{'Tokens':>8s} | {'Base':>8s} | {'Sep':>8s} {'SepOH%':>8s} | "
          f"{'Fused':>8s} {'FusOH%':>8s} | {'Winner':>10s} {'vsSep%':>8s}")
    print("-" * 100)

    for M in BATCH_SIZES:
        if M >= 16384:
            nw_f, ni_f = 5, 20
        elif M >= 1024:
            nw_f, ni_f = 8, 40
        else:
            nw_f, ni_f = 10, 60

        hidden = torch.randn(M, K, device=device, dtype=DTYPE)
        w = torch.randn(E, N, K, device=device, dtype=DTYPE) * 0.01
        lora_a = torch.randn(1, max_loras, E, RANK, K,
                             device=device, dtype=DTYPE) * 0.01
        lora_b = torch.randn(1, max_loras, E, N, RANK,
                             device=device, dtype=DTYPE) * 0.01
        lora_a_sep = [lora_a[0]]
        lora_b_sep = [lora_b[0]]
        topk_ids = torch.stack([
            torch.randperm(E, device=device)[:TOP_K] for _ in range(M)
        ]).to(torch.int64)
        topk_weights = torch.softmax(
            torch.randn(M, TOP_K, device=device, dtype=DTYPE), dim=-1)
        lora_ids_per_token = torch.zeros(M, device=device, dtype=torch.int64)

        _, base_cfg = best_base[M]
        _, fused_cfg = best_fused[M]

        base_bm = base_cfg["BLOCK_SIZE_M"]
        fused_bm = fused_cfg["BLOCK_SIZE_M"]

        sorted_b, expert_b, npost_b = moe_align_block_size(
            topk_ids, base_bm, E)
        out_base = torch.zeros(M, TOP_K, N, device=device, dtype=DTYPE)

        sorted_f, expert_f, lora_f, npost_f = \
            moe_lora_align_block_size_fused(
                topk_ids, lora_ids_per_token, fused_bm, E, max_loras)
        out_fused = torch.zeros(M * TOP_K, N, device=device, dtype=DTYPE)

        base_ms = bench_base_kernel(
            hidden, w, out_base, topk_weights,
            sorted_b, expert_b, npost_b, base_cfg, nw_f, ni_f)

        sep_ms = bench_separate(
            hidden, w, out_base, topk_weights, topk_ids,
            sorted_b, expert_b, npost_b,
            lora_a_sep, lora_b_sep, lora_ids_per_token,
            base_cfg, max_loras, M, nw_f, ni_f)

        importlib.reload(fmod)
        fused_ms = bench_fused_kernel(
            hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f,
            lora_a, lora_b, fused_cfg, nw_f, ni_f)

        sep_oh = (sep_ms - base_ms) / base_ms * 100
        fused_oh = (fused_ms - base_ms) / base_ms * 100
        winner = "FUSED" if fused_ms < sep_ms else "SEPARATE"
        vs_sep = (fused_ms - sep_ms) / sep_ms * 100

        print(f"{M:>8d} | {base_ms:>7.3f}  | {sep_ms:>7.3f}  {sep_oh:>+7.1f}% | "
              f"{fused_ms:>7.3f}  {fused_oh:>+7.1f}% | {winner:>10s} {vs_sep:>+7.1f}%")
        sys.stdout.flush()
        torch.cuda.empty_cache()

    # ========== Print tuned configs ==========
    print()
    print("=" * 100)
    print("Tuned base configs:")
    base_json = {}
    for M in BATCH_SIZES:
        ms, cfg = best_base[M]
        base_json[str(M)] = cfg
        print(f"  {M:>6d}: {cfg}")
    print()
    print("Tuned fused configs:")
    fused_json = {}
    for M in BATCH_SIZES:
        ms, cfg = best_fused[M]
        fused_json[str(M)] = cfg
        print(f"  {M:>6d}: {cfg}")

    with open("tuned_base_configs.json", "w") as f:
        json.dump(base_json, f, indent=2)
    with open("tuned_fused_configs.json", "w") as f:
        json.dump(fused_json, f, indent=2)
    print("\nSaved tuned configs to tuned_base_configs.json and tuned_fused_configs.json")


if __name__ == "__main__":
    main()
