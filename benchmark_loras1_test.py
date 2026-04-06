"""Quick A/B: max_loras=1 vs max_loras=4 to isolate grouping cost."""
import sys, time, torch
import triton.language as tl
from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

def run(num_tokens, max_loras, num_experts=8, top_k=2, K=4096, N=4096, rank=16,
        dtype=torch.bfloat16, n_warmup=10, n_iter=50):
    device = "cuda"
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
    lora_ids_per_token = torch.randint(0, max_loras, (num_tokens,), device=device, dtype=torch.int64)
    compute_type = tl.bfloat16

    # Base config (tuned)
    w2_shape = (num_experts, K, N)
    base_config = try_get_optimal_moe_config(w.size(), w2_shape, top_k, "bfloat16", num_tokens)
    base_bm = base_config["BLOCK_SIZE_M"]

    # Fused: use same BM=16 config as sweep found
    fused_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3}
    if num_tokens <= 256:
        fused_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}

    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, fused_config["BLOCK_SIZE_M"], num_experts, max_loras)
    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, base_bm, num_experts)

    out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)
    out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(n_warmup):
        out_base.zero_()
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
        out_fused.zero_()
        invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
            True, top_k, 1, fused_config, compute_type=compute_type)
    torch.cuda.synchronize()

    # Bench base
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_base.zero_()
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    base_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Bench fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_fused.zero_()
        invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
            True, top_k, 1, fused_config, compute_type=compute_type)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / n_iter * 1000

    n_groups = num_experts * max_loras
    tpg = max(1, (num_tokens * top_k) // n_groups)
    return base_ms, fused_ms, n_groups, tpg

batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536]
print(f"{'':>8s} | {'--- max_loras=1 ---':^34s} | {'--- max_loras=4 ---':^34s}")
print(f"{'Tokens':>8s} | {'Groups':>6s} {'TPG':>4s} {'Base':>7s} {'Fused':>7s} {'OH%':>7s} | {'Groups':>6s} {'TPG':>4s} {'Base':>7s} {'Fused':>7s} {'OH%':>7s}")
print("-" * 95)

for M in batch_sizes:
    nw, ni = (5, 20) if M >= 16384 else (5, 30) if M >= 4096 else (10, 50)
    b1, f1, g1, t1 = run(M, 1, n_warmup=nw, n_iter=ni)
    b4, f4, g4, t4 = run(M, 4, n_warmup=nw, n_iter=ni)
    oh1 = (f1 - b1) / b1 * 100
    oh4 = (f4 - b4) / b4 * 100
    print(f"{M:>8d} | {g1:>6d} {t1:>4d} {b1:>7.3f} {f1:>7.3f} {oh1:>+6.1f}% | {g4:>6d} {t4:>4d} {b4:>7.3f} {f4:>7.3f} {oh4:>+6.1f}%")
    sys.stdout.flush()
    torch.cuda.empty_cache()
