"""Get register counts for base vs fused kernels via ncu."""
import torch
import triton.language as tl
from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import invoke_fused_moe_lora_kernel
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

num_tokens = 1024
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
compute_type = tl.bfloat16

cfg_nw8 = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
           "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2, "SPLIT_K": 1}
cfg_nw4 = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
           "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}

sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, 16, num_experts)
sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
    topk_ids, lora_ids_per_token, 16, num_experts, max_loras)
out_base = torch.zeros(num_tokens, top_k, N, device=device, dtype=dtype)
out_fused = torch.zeros(num_tokens * top_k, N, device=device, dtype=dtype)

# Warmup to compile
invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
    sorted_b, expert_b, npost_b, True, top_k, cfg_nw4, compute_type=compute_type,
    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
    sorted_b, expert_b, npost_b, True, top_k, cfg_nw8, compute_type=compute_type,
    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
    True, top_k, 1, cfg_nw8, compute_type=compute_type)
invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
    True, top_k, 1, cfg_nw4, compute_type=compute_type)

# All lora -1 to get HAS_LORA=False variant compiled
lora_f_nolora = torch.full_like(lora_f, -1)
invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f_nolora, npost_f, lora_a, lora_b,
    True, top_k, 1, cfg_nw8, compute_type=compute_type)

torch.cuda.synchronize()

# Now run for profiling
print("PHASE: base_NW4")
invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
    sorted_b, expert_b, npost_b, True, top_k, cfg_nw4, compute_type=compute_type,
    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
torch.cuda.synchronize()

print("PHASE: base_NW8")
invoke_fused_moe_triton_kernel(hidden, w, out_base, None, None, topk_weights,
    sorted_b, expert_b, npost_b, True, top_k, cfg_nw8, compute_type=compute_type,
    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
torch.cuda.synchronize()

print("PHASE: fused_HAS_LORA_NW8")
invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
    True, top_k, 1, cfg_nw8, compute_type=compute_type)
torch.cuda.synchronize()

print("PHASE: fused_NOLORA_NW8")
invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f_nolora, npost_f, lora_a, lora_b,
    True, top_k, 1, cfg_nw8, compute_type=compute_type)
torch.cuda.synchronize()

print("PHASE: fused_HAS_LORA_NW4")
invoke_fused_moe_lora_kernel(hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
    True, top_k, 1, cfg_nw4, compute_type=compute_type)
torch.cuda.synchronize()

print("Done")
