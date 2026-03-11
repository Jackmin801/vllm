"""Check fused kernel with and without maxnreg - compare spills and instructions."""
import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    fused_moe_with_lora_kernel,
    invoke_fused_moe_lora_kernel,
)
from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused

E, N, K, RANK, TOP_K = 8, 4096, 4096, 16, 2
M = 64
DTYPE = torch.bfloat16
device = "cuda"

torch.manual_seed(42)
hidden = torch.randn(M, K, device=device, dtype=DTYPE)
w = torch.randn(E, N, K, device=device, dtype=DTYPE) * 0.01
lora_a = torch.randn(1, 1, E, RANK, K, device=device, dtype=DTYPE) * 0.01
lora_b = torch.randn(1, 1, E, N, RANK, device=device, dtype=DTYPE) * 0.01
topk_ids = torch.stack([torch.randperm(E, device=device)[:TOP_K] for _ in range(M)]).to(torch.int64)
topk_weights = torch.softmax(torch.randn(M, TOP_K, device=device, dtype=DTYPE), dim=-1)
lora_ids_per_token = torch.zeros(M, device=device, dtype=torch.int64)

configs_to_test = [
    ("maxnreg=96 (current best)", {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
     "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3, "maxnreg": 96}),
    ("maxnreg=80", {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
     "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3, "maxnreg": 80}),
    ("NO maxnreg", {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
     "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3}),
    ("NO maxnreg BM=128", {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
     "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3}),
    ("NO maxnreg BM=128 NW=4", {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
     "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2}),
]

# Need to clear the cache between runs since constexprs change
import importlib
import vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel as fmod

for label, cfg in configs_to_test:
    # Re-import to get fresh kernel object
    importlib.reload(fmod)
    kernel_fn = fmod.fused_moe_with_lora_kernel

    bm = cfg["BLOCK_SIZE_M"]
    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, bm, E, 1)
    out_fused = torch.zeros(M * TOP_K, N, device=device, dtype=DTYPE)

    try:
        fmod.invoke_fused_moe_lora_kernel(
            hidden, w, out_fused, topk_weights,
            sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
            True, TOP_K, 1, cfg.copy(), compute_type=tl.bfloat16)
        torch.cuda.synchronize()

        # Analyze
        found = False
        for device_key, cache in kernel_fn.device_caches.items():
            compiled_dict = cache[0]
            for k, compiled in compiled_dict.items():
                ptx = compiled.asm.get('ptx', '') if hasattr(compiled, 'asm') else ''
                tcgen05 = ptx.lower().count('tcgen05.mma')
                wgmma = ptx.lower().count('wgmma')
                mma_legacy = ptx.lower().count('mma.sync')
                tmem_alloc = ptx.lower().count('tcgen05.alloc')

                print(f"{label:35s}: regs={compiled.n_regs:3d}  spills={compiled.n_spills:3d}  "
                      f"tcgen05.mma={tcgen05}  wgmma={wgmma}  mma.sync={mma_legacy}  "
                      f"tmem_alloc={tmem_alloc}")
                found = True
                break
            if found:
                break
        if not found:
            print(f"{label:35s}: NO COMPILED KERNEL FOUND")
    except Exception as e:
        print(f"{label:35s}: ERROR: {e}")
