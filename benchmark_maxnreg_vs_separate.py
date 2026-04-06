"""Head-to-head: fused with maxnreg=80 vs separate path."""
import json
import sys
import time

import torch
import triton.language as tl

from benchmark_maxnreg import fused_moe_lora_maxnreg, launch_maxnreg
from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.fused_moe_lora_op import _fused_moe_lora_shrink, _fused_moe_lora_expand
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

with open("vllm/model_executor/layers/fused_moe/configs/E=8,N=4096,device_name=NVIDIA_B200,dtype=bfloat16.json") as f:
    base_configs = {int(k): v for k, v in json.load(f).items()}

def nearest(configs, M):
    return configs[min(configs.keys(), key=lambda x: abs(x - M))]


def run(num_tokens, max_loras, num_experts=8, top_k=2, K=4096, N=4096, rank=16,
        dtype=torch.bfloat16, n_warmup=10, n_iter=50):
    device = "cuda"
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
    lora_a_sep = [lora_a[0]]
    lora_b_sep = [lora_b[0]]
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
    lora_ids_per_token = torch.randint(0, max_loras, (num_tokens,), device=device, dtype=torch.int64)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int, device=device)
    compute_type = tl.bfloat16

    bc = nearest(base_configs, num_tokens)
    base_bm = bc["BLOCK_SIZE_M"]

    # Fused kernel config: NW=8 with maxnreg=80
    fused_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}
    fused_bm = fused_config["BLOCK_SIZE_M"]

    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, base_bm, num_experts)
    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, fused_bm, num_experts, max_loras)

    out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)
    out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)
    token_lora_mapping = lora_ids_per_token.to(torch.int32)
    lora_ids_int = lora_ids_per_token.to(torch.int64)
    a_intermediate = torch.zeros(1, num_tokens, top_k, rank, device=device, dtype=dtype)

    # Warmup all paths
    for _ in range(n_warmup):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, bc, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
        # Fused maxnreg=80
        out_fused.zero_()
        launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                       sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                       top_k, 1, fused_config, compute_type, maxnreg=80)
        # Separate
        out_base.zero_()
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, bc, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
        a_intermediate.zero_()
        _fused_moe_lora_shrink(a_intermediate, hidden, lora_a_sep, topk_weights,
            None, topk_ids.reshape(-1), None, token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled, device=torch.device(device),
            N=rank, M=num_tokens, EM=num_tokens*top_k*base_bm, K=K,
            num_tokens=num_tokens*top_k, num_experts=num_experts, num_slices=1,
            block_size_m=base_bm, block_size_n=min(64,rank), block_size_k=32,
            group_size_m=8, num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras, mul_routed_weight=False)
        _fused_moe_lora_expand(out_base, a_intermediate, lora_b_sep, topk_weights,
            None, topk_ids.reshape(-1), None, token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled, device=torch.device(device),
            N=rank, M=num_tokens, EM=num_tokens*top_k*base_bm, K=K,
            num_tokens=num_tokens*top_k, num_experts=num_experts, num_slices=1,
            max_lora_rank=rank, w1_output_dim_size=N,
            block_size_m=base_bm, block_size_n=min(64,N), block_size_k=min(32,rank),
            group_size_m=8, num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras, mul_routed_weight=True)
    torch.cuda.synchronize()

    # Bench base
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iter):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, bc, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    base_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Bench separate
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iter):
        out_base.zero_()
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, bc, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
        a_intermediate.zero_()
        _fused_moe_lora_shrink(a_intermediate, hidden, lora_a_sep, topk_weights,
            None, topk_ids.reshape(-1), None, token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled, device=torch.device(device),
            N=rank, M=num_tokens, EM=num_tokens*top_k*base_bm, K=K,
            num_tokens=num_tokens*top_k, num_experts=num_experts, num_slices=1,
            block_size_m=base_bm, block_size_n=min(64,rank), block_size_k=32,
            group_size_m=8, num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras, mul_routed_weight=False)
        _fused_moe_lora_expand(out_base, a_intermediate, lora_b_sep, topk_weights,
            None, topk_ids.reshape(-1), None, token_lora_mapping,
            top_k, lora_ids_int, adapter_enabled, device=torch.device(device),
            N=rank, M=num_tokens, EM=num_tokens*top_k*base_bm, K=K,
            num_tokens=num_tokens*top_k, num_experts=num_experts, num_slices=1,
            max_lora_rank=rank, w1_output_dim_size=N,
            block_size_m=base_bm, block_size_n=min(64,N), block_size_k=min(32,rank),
            group_size_m=8, num_warps=4, num_stages=3, split_k=1,
            num_active_loras=max_loras, mul_routed_weight=True)
    torch.cuda.synchronize()
    sep_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Bench fused maxnreg=80
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iter):
        out_fused.zero_()
        launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                       sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                       top_k, 1, fused_config, compute_type, maxnreg=80)
    torch.cuda.synchronize()
    fused80_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Bench fused no maxnreg (current production)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iter):
        out_fused.zero_()
        launch_maxnreg(hidden, w, out_fused, topk_weights.reshape(-1),
                       sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                       top_k, 1, fused_config, compute_type, maxnreg=None)
    torch.cuda.synchronize()
    fused_orig_ms = (time.perf_counter() - t0) / n_iter * 1000

    return base_ms, sep_ms, fused_orig_ms, fused80_ms


batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536]

for ml in [1, 4]:
    print(f"\n{'='*115}")
    print(f"max_loras={ml}")
    print(f"{'='*115}")
    print(f"{'Tokens':>8s} | {'Base':>7s} | {'Sep':>7s} {'SepOH%':>7s} | {'Fused':>8s} {'FusOH%':>7s} | {'Fus+MNR80':>9s} {'MNR80%':>7s} | {'Best':>12s}  {'vs Sep':>8s}")
    print("-" * 112)
    for M in batch_sizes:
        nw, ni = (5, 15) if M >= 16384 else (5, 25) if M >= 4096 else (8, 40)
        b, s, fo, f80 = run(M, ml, n_warmup=nw, n_iter=ni)
        soh = (s - b) / b * 100
        fooh = (fo - b) / b * 100
        f80oh = (f80 - b) / b * 100

        best_fused = min(fo, f80)
        best_name = "FUSED" if fo <= f80 else "FUSED+MNR80"
        winner = "SEPARATE" if s < best_fused else best_name
        vs_sep = (best_fused - s) / s * 100

        print(f"{M:>8d} | {b:>6.3f}  | {s:>6.3f}  {soh:>+6.1f}% | {fo:>7.3f}  {fooh:>+6.1f}% | {f80:>8.3f}  {f80oh:>+6.1f}% | {winner:>12s}  {vs_sep:>+6.1f}%")
        sys.stdout.flush()
        torch.cuda.empty_cache()
