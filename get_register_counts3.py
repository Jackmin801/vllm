"""Get register counts via Triton compiled kernel inspection."""
import torch
import triton
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

# Inspect device_caches structure
print("=== Base kernel device_caches structure ===")
caches = fused_moe_kernel.device_caches
print(f"Type: {type(caches)}")
for dk, dv in caches.items():
    print(f"  Device {dk}: type={type(dv)}")
    if isinstance(dv, tuple):
        for i, item in enumerate(dv):
            print(f"    [{i}]: type={type(item)}")
            if isinstance(item, dict):
                for k, v in list(item.items())[:3]:
                    print(f"      Key: {k[:60]}...")
                    print(f"      Val type: {type(v)}")
                    for a in dir(v):
                        if not a.startswith('_'):
                            val = getattr(v, a, None)
                            if val is not None and not callable(val):
                                print(f"        {a}: {val}" if len(str(val)) < 200 else f"        {a}: <long>")
    elif isinstance(dv, dict):
        for k, v in list(dv.items())[:3]:
            print(f"    Key: {k[:60]}...")
            print(f"    Val type: {type(v)}")
