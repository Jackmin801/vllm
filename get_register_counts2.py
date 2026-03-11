"""Get register counts via Triton's device_caches."""
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

# Compile all variants
for nw in [4, 8]:
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
           "GROUP_SIZE_M": 16, "num_warps": nw, "num_stages": 2, "SPLIT_K": 1}
    invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
        sorted_b, expert_b, npost_b, True, top_k, cfg, compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        use_int4_w4a16=False, per_channel_quant=False)
    invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
        sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
        True, top_k, 1, cfg, compute_type=tl.bfloat16)
torch.cuda.synchronize()

# Inspect device_caches
print("=== Base kernel (fused_moe_kernel) device_caches ===")
for device_key, cache in fused_moe_kernel.device_caches.items():
    print(f"Device: {device_key}")
    for key, compiled in cache.items():
        # Inspect compiled kernel object
        print(f"  Key: {key[:80]}...")
        for attr in dir(compiled):
            if 'reg' in attr.lower() or 'spill' in attr.lower() or 'shared' in attr.lower() or 'occupancy' in attr.lower():
                print(f"    {attr}: {getattr(compiled, attr, 'N/A')}")
        # Check metadata
        if hasattr(compiled, 'metadata'):
            meta = compiled.metadata
            print(f"    metadata type: {type(meta)}")
            if isinstance(meta, dict):
                for mk, mv in meta.items():
                    if 'reg' in str(mk).lower() or 'spill' in str(mk).lower() or 'shared' in str(mk).lower():
                        print(f"    metadata[{mk}]: {mv}")
            else:
                for attr in dir(meta):
                    if not attr.startswith('_'):
                        val = getattr(meta, attr, None)
                        if val is not None and not callable(val):
                            print(f"    metadata.{attr}: {val}")

print("\n=== Fused kernel (fused_moe_with_lora_kernel) device_caches ===")
for device_key, cache in fused_moe_with_lora_kernel.device_caches.items():
    print(f"Device: {device_key}")
    for key, compiled in cache.items():
        print(f"  Key: {key[:80]}...")
        for attr in dir(compiled):
            if 'reg' in attr.lower() or 'spill' in attr.lower() or 'shared' in attr.lower() or 'occupancy' in attr.lower():
                print(f"    {attr}: {getattr(compiled, attr, 'N/A')}")
        if hasattr(compiled, 'metadata'):
            meta = compiled.metadata
            print(f"    metadata type: {type(meta)}")
            if isinstance(meta, dict):
                for mk, mv in meta.items():
                    if 'reg' in str(mk).lower() or 'spill' in str(mk).lower() or 'shared' in str(mk).lower():
                        print(f"    metadata[{mk}]: {mv}")
            else:
                for attr in dir(meta):
                    if not attr.startswith('_'):
                        val = getattr(meta, attr, None)
                        if val is not None and not callable(val):
                            print(f"    metadata.{attr}: {val}")
