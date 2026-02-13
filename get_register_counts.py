"""Get register counts from Triton compiled kernels."""
import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import fused_moe_with_lora_kernel
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel

# Compile the base kernel with NW=4 and NW=8
# We need to call it to trigger compilation, then inspect

# Helper to get register count from a compiled kernel
def get_kernel_info(compiled_kernel):
    """Extract register count and other info from compiled Triton kernel."""
    if hasattr(compiled_kernel, 'n_regs'):
        return compiled_kernel.n_regs
    if hasattr(compiled_kernel, 'metadata'):
        meta = compiled_kernel.metadata
        if hasattr(meta, 'num_regs'):
            return meta.num_regs
    return None

# Alternative: inspect from the cache
import os
import glob
import json

triton_cache = os.path.expanduser("~/.triton/cache")
print(f"Triton cache at: {triton_cache}")

# Trigger compilation by running the kernels
num_tokens = 64
num_experts = 8
top_k = 2
K = 4096
N = 4096
rank = 16
max_loras = 1
dtype = torch.bfloat16
device = "cuda"

torch.manual_seed(0)
hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
lora_a = torch.randn(1, max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
lora_b = torch.randn(1, max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01
topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]).to(torch.int64)
topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
lora_ids_per_token = torch.zeros(num_tokens, device=device, dtype=torch.int64)

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, 16, num_experts)
sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
    topk_ids, lora_ids_per_token, 16, num_experts, max_loras)
out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)
out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

# Run each variant to trigger compilation
configs_to_test = [
    ("base_NW4", "base", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                           "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}),
    ("base_NW8", "base", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                           "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2, "SPLIT_K": 1}),
    ("fused_NW4", "fused", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                              "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}),
    ("fused_NW8", "fused", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                              "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2, "SPLIT_K": 1}),
]

for name, ktype, cfg in configs_to_test:
    if ktype == "base":
        invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
            sorted_b, expert_b, npost_b, True, top_k, cfg, compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False)
    else:
        invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
            True, top_k, 1, cfg, compute_type=tl.bfloat16)
torch.cuda.synchronize()

# Now try to inspect Triton kernel metadata
# Method 1: Check the kernel's compiled attributes
print("\n=== Method 1: Kernel object attributes ===")
for attr in ['n_regs', 'n_spills', 'shared', 'num_regs', 'num_warps']:
    val_base = getattr(fused_moe_kernel, attr, 'N/A')
    val_fused = getattr(fused_moe_with_lora_kernel, attr, 'N/A')
    print(f"  {attr}: base={val_base}, fused={val_fused}")

# Method 2: Check cache entries
print("\n=== Method 2: Inspecting Triton cache ===")
cache_dirs = glob.glob(os.path.join(triton_cache, "**"), recursive=False)
print(f"  Found {len(cache_dirs)} cache dirs")

# Method 3: Use triton compilation directly
print("\n=== Method 3: Direct compilation info ===")
# Triton stores compilation metadata that we can access
try:
    # Look for the compiled kernel in the JIT cache
    base_kernel = fused_moe_kernel
    fused_kernel = fused_moe_with_lora_kernel

    # Check all attributes
    print(f"\nBase kernel attributes: {[a for a in dir(base_kernel) if not a.startswith('__')]}")
    print(f"\nFused kernel attributes: {[a for a in dir(fused_kernel) if not a.startswith('__')]}")

    # Try to access the cache
    if hasattr(base_kernel, 'cache'):
        print(f"\nBase cache keys: {list(base_kernel.cache.keys())[:5]}")
        for key, val in list(base_kernel.cache.items())[:2]:
            print(f"  Key: {key}")
            if hasattr(val, 'metadata'):
                print(f"  Metadata: {val.metadata}")
            if hasattr(val, 'n_regs'):
                print(f"  n_regs: {val.n_regs}")
            for a in dir(val):
                if 'reg' in a.lower() or 'spill' in a.lower() or 'shared' in a.lower():
                    print(f"  {a}: {getattr(val, a)}")

    if hasattr(fused_kernel, 'cache'):
        print(f"\nFused cache keys (first 5): {list(fused_kernel.cache.keys())[:5]}")
        for key, val in list(fused_kernel.cache.items())[:5]:
            print(f"  Key: {key}")
            if hasattr(val, 'metadata'):
                print(f"  Metadata: {val.metadata}")
            for a in dir(val):
                if 'reg' in a.lower() or 'spill' in a.lower() or 'shared' in a.lower():
                    print(f"  {a}: {getattr(val, a)}")
except Exception as e:
    print(f"Error: {e}")

# Method 4: Use CUDA to get function attributes
print("\n=== Method 4: CUDA function attributes ===")
try:
    import triton.runtime as tr
    if hasattr(base_kernel, 'cache'):
        for key, compiled in list(base_kernel.cache.items())[:2]:
            if hasattr(compiled, 'function'):
                fn = compiled.function
                print(f"Base [{key}]: type={type(fn)}")
                for a in dir(fn):
                    if not a.startswith('_'):
                        print(f"  {a}: {getattr(fn, a, 'N/A')}")
except Exception as e:
    print(f"Error: {e}")
