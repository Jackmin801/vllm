"""Get register counts for both base and fused kernels."""
import torch
import triton.language as tl

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    fused_moe_with_lora_kernel, invoke_fused_moe_lora_kernel
)
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel, invoke_fused_moe_triton_kernel
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

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

sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, 16, num_experts)
sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
    topk_ids, lora_ids_per_token, 16, num_experts, max_loras)
out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)
out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

# Also compile HAS_LORA=False variant
lora_f_nolora = torch.full_like(lora_f, -1)

# Compile all variants
for nw in [4, 8]:
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
           "GROUP_SIZE_M": 16, "num_warps": nw, "num_stages": 2, "SPLIT_K": 1}
    invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
        sorted_b, expert_b, npost_b, True, top_k, cfg, compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        use_int4_w4a16=False, per_channel_quant=False)
    # HAS_LORA=True
    invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
        sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
        True, top_k, 1, cfg, compute_type=tl.bfloat16)
    # HAS_LORA=False
    invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
        sorted_f, expert_f, lora_f_nolora, npost_f, lora_a, lora_b,
        True, top_k, 1, cfg, compute_type=tl.bfloat16)
torch.cuda.synchronize()

# Print register info
def print_cache_info(name, kernel):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    caches = kernel.device_caches
    for dk, dv in caches.items():
        compiled_cache = dv[0]  # first element is the compiled kernel dict
        for key, compiled in compiled_cache.items():
            # Extract num_warps from packed_metadata
            nw = compiled.packed_metadata[0] if hasattr(compiled, 'packed_metadata') else '?'
            shared = compiled.packed_metadata[2] if hasattr(compiled, 'packed_metadata') else '?'

            # Try to determine HAS_LORA from the key
            key_str = str(key)
            has_lora = "?"

            print(f"  NW={nw:>2}  n_regs={compiled.n_regs:>3}  n_spills={compiled.n_spills:>2}  "
                  f"n_max_threads={compiled.n_max_threads:>4}  shared={shared:>5}B  "
                  f"blocks/SM={compiled.n_max_threads // (nw*32) if isinstance(nw, int) else '?'}")

print_cache_info("BASE KERNEL (fused_moe_kernel)", fused_moe_kernel)
print_cache_info("FUSED KERNEL (fused_moe_with_lora_kernel)", fused_moe_with_lora_kernel)

# Summary comparison
print(f"\n{'='*60}")
print("SUMMARY: Occupancy comparison (B200: 65536 regs/SM, 2048 threads/SM)")
print(f"{'='*60}")

base_caches = list(fused_moe_kernel.device_caches.values())[0][0]
fused_caches = list(fused_moe_with_lora_kernel.device_caches.values())[0][0]

print(f"\n{'Kernel':<35s} {'NW':>3} {'Regs':>5} {'Spills':>6} {'MaxThr':>7} {'Blks/SM':>8} {'Warps/SM':>9}")
print("-" * 75)

for key, c in base_caches.items():
    nw = c.packed_metadata[0]
    blks = c.n_max_threads // (nw * 32)
    warps = blks * nw
    print(f"{'base':<35s} {nw:>3} {c.n_regs:>5} {c.n_spills:>6} {c.n_max_threads:>7} {blks:>8} {warps:>9}")

for key, c in fused_caches.items():
    nw = c.packed_metadata[0]
    blks = c.n_max_threads // (nw * 32)
    warps = blks * nw
    # Try to identify if HAS_LORA is true or false
    key_str = str(key)
    label = "fused"
    if "False" in key_str[-100:]:
        label = "fused (NOLORA)"
    elif "True" in key_str[-100:]:
        label = "fused (HAS_LORA)"
    print(f"{label:<35s} {nw:>3} {c.n_regs:>5} {c.n_spills:>6} {c.n_max_threads:>7} {blks:>8} {warps:>9}")
