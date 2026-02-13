"""Check if fused alignment produces more padding than base alignment."""
import torch
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

for num_tokens in [64, 256, 1024, 4096, 65536]:
    num_experts = 8
    top_k = 2
    max_loras = 1
    bm = 16
    device = "cuda"

    torch.manual_seed(0)
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k]
                           for _ in range(num_tokens)]).to(torch.int64)
    lora_ids = torch.zeros(num_tokens, device=device, dtype=torch.int64)

    sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, bm, num_experts)
    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids, bm, num_experts, max_loras)

    nb = npost_b.item()
    nf = npost_f.item()
    valid = num_tokens * top_k

    base_mblocks = nb // bm
    fused_mblocks = nf // bm
    base_nblocks = 4096 // 128  # BN=128
    fused_nblocks = 4096 // 128

    base_grid = base_mblocks * base_nblocks
    fused_grid = fused_mblocks * fused_nblocks

    print(f"M={num_tokens:>6d}: valid={valid:>7d}  "
          f"base: post_pad={nb:>7d} mblocks={base_mblocks:>5d} grid={base_grid:>6d}  |  "
          f"fused: post_pad={nf:>7d} mblocks={fused_mblocks:>5d} grid={fused_grid:>6d}  |  "
          f"fused/base={nf/nb:.3f}x")
