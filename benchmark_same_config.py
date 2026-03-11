"""Run fused and base with IDENTICAL configs to measure true fusion overhead."""
import sys, time, torch
import triton.language as tl
from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

def run(num_tokens, config, max_loras=1, num_experts=8, top_k=2, K=4096, N=4096, rank=16,
        dtype=torch.bfloat16, n_warmup=10, n_iter=50):
    device = "cuda"
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
    lora_ids_per_token = torch.zeros(num_tokens, device=device, dtype=torch.int64)  # all lora 0
    compute_type = tl.bfloat16
    bm = config["BLOCK_SIZE_M"]

    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, bm, num_experts, max_loras)
    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, bm, num_experts)
    out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)
    out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)

    # Add SPLIT_K for base kernel
    base_config = {**config, "SPLIT_K": 1}

    # Warmup both
    for _ in range(n_warmup):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
        invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
            True, top_k, 1, config, compute_type=compute_type)
    torch.cuda.synchronize()

    # Bench base
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iter):
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, base_config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    torch.cuda.synchronize()
    base_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Bench fused
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iter):
        invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
            True, top_k, 1, config, compute_type=compute_type)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / n_iter * 1000

    return base_ms, fused_ms

# Test a few configs at 65536 tokens
configs_to_test = [
    ("Base-tuned", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
    ("Fused-tuned", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4}),
    ("NW4-NS2-GM16", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2}),
    ("NW4-NS3-GM8", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}),
    ("NW8-NS2-GM16", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2}),
]

batch_sizes = [64, 256, 1024, 4096, 65536]

print("Same-config comparison (max_loras=1): base vs fused kernel with IDENTICAL tile config")
print()

for name, cfg in configs_to_test:
    print(f"Config: {name} — BM={cfg['BLOCK_SIZE_M']} BN={cfg['BLOCK_SIZE_N']} BK={cfg['BLOCK_SIZE_K']} GM={cfg['GROUP_SIZE_M']} NW={cfg['num_warps']} NS={cfg['num_stages']}")
    print(f"{'Tokens':>8s} | {'Base (ms)':>9s} | {'Fused (ms)':>10s} | {'OH%':>7s}")
    print("-" * 45)
    for M in batch_sizes:
        nw, ni = (5, 20) if M >= 16384 else (5, 30) if M >= 4096 else (10, 50)
        b, f = run(M, cfg, max_loras=1, n_warmup=nw, n_iter=ni)
        oh = (f - b) / b * 100
        print(f"{M:>8d} | {b:>9.3f} | {f:>10.3f} | {oh:>+6.1f}%")
        sys.stdout.flush()
        torch.cuda.empty_cache()
    print()
