"""Check what instructions the ACTUAL fused kernel generates on B200.
Compare base kernel vs fused kernel PTX to see if both use tcgen05 (Blackwell MMA+TMEM)."""
import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel import (
    fused_moe_with_lora_kernel,
    invoke_fused_moe_lora_kernel,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel,
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
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

# Run base kernel
base_cfg = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}
sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, 128, E)
out_base = torch.zeros(M, TOP_K, N, device=device, dtype=DTYPE)
invoke_fused_moe_triton_kernel(
    hidden, w, out_base, None, None, topk_weights,
    sorted_b, expert_b, npost_b, True, TOP_K, base_cfg,
    compute_type=tl.bfloat16,
    use_fp8_w8a8=False, use_int8_w8a8=False,
    use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
torch.cuda.synchronize()

# Run fused kernel
fused_cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3, "maxnreg": 96}
sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
    topk_ids, lora_ids_per_token, 64, E, 1)
out_fused = torch.zeros(M * TOP_K, N, device=device, dtype=DTYPE)
invoke_fused_moe_lora_kernel(
    hidden, w, out_fused, topk_weights,
    sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
    True, TOP_K, 1, fused_cfg, compute_type=tl.bfloat16)
torch.cuda.synchronize()


def analyze_kernel(kernel_obj, name):
    print(f"\n{'='*80}")
    print(f"Kernel: {name}")
    print(f"{'='*80}")
    for device_key, cache in kernel_obj.device_caches.items():
        compiled_dict = cache[0]
        for k, compiled in compiled_dict.items():
            print(f"  n_regs: {compiled.n_regs}, n_spills: {compiled.n_spills}")
            if hasattr(compiled, 'asm') and isinstance(compiled.asm, dict):
                ptx = compiled.asm.get('ptx', '')
                # Target
                for line in ptx.split('\n'):
                    if '.target' in line:
                        print(f"  Target: {line.strip()}")
                        break
                # Count instruction types
                tcgen05_mma = 0
                wgmma = 0
                mma_legacy = 0
                tcgen05_alloc = 0
                tcgen05_ld = 0
                tcgen05_st = 0
                cp_async = 0
                tma = 0
                for line in ptx.split('\n'):
                    l = line.strip().lower()
                    if 'tcgen05.mma' in l: tcgen05_mma += 1
                    elif 'wgmma' in l: wgmma += 1
                    elif 'mma.sync' in l: mma_legacy += 1
                    if 'tcgen05.alloc' in l: tcgen05_alloc += 1
                    if 'tcgen05.ld' in l: tcgen05_ld += 1
                    if 'tcgen05.st' in l: tcgen05_st += 1
                    if 'cp.async' in l: cp_async += 1
                    if 'cp.async.bulk' in l or 'tensormap' in l: tma += 1

                print(f"  Blackwell tcgen05.mma: {tcgen05_mma}")
                print(f"  Hopper wgmma: {wgmma}")
                print(f"  Legacy mma.sync: {mma_legacy}")
                print(f"  TMEM alloc/ld/st: {tcgen05_alloc}/{tcgen05_ld}/{tcgen05_st}")
                print(f"  cp.async (TMA): {cp_async}, bulk TMA: {tma}")

                if tcgen05_mma == 0 and wgmma == 0 and mma_legacy == 0:
                    print("  WARNING: No MMA instructions found!")
                if tcgen05_mma == 0 and wgmma > 0:
                    print("  WARNING: Using Hopper wgmma, NOT Blackwell tcgen05!")
                if tcgen05_mma == 0 and mma_legacy > 0:
                    print("  WARNING: Using legacy mma.sync, NOT Blackwell tcgen05!")
            break
        break


analyze_kernel(fused_moe_kernel, "Base MoE (fused_moe_kernel)")
analyze_kernel(fused_moe_with_lora_kernel, "Fused MoE+LoRA (fused_moe_with_lora_kernel)")
