"""Sweep maxnreg from low to very high to find optimal for fused kernel.
On Blackwell, TMEM holds accumulators off-registers. The base kernel
uses 254 regs/0 spills. Can we get the fused kernel closer to that?"""
import importlib
import time
import torch
import triton.language as tl

from vllm.lora.ops.triton_ops.moe_lora_align import moe_lora_align_block_size_fused
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel

E, N, K, RANK, TOP_K = 8, 4096, 4096, 16, 2
M = 65536
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

# Benchmark base for reference
base_cfg = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2, "SPLIT_K": 1}
sorted_b, expert_b, npost_b = moe_align_block_size(topk_ids, 128, E)
out_base = torch.zeros(M, TOP_K, N, device=device, dtype=DTYPE)
for _ in range(3):
    invoke_fused_moe_triton_kernel(
        hidden, w, out_base, None, None, topk_weights,
        sorted_b, expert_b, npost_b, True, TOP_K, base_cfg,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    invoke_fused_moe_triton_kernel(
        hidden, w, out_base, None, None, topk_weights,
        sorted_b, expert_b, npost_b, True, TOP_K, base_cfg,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
torch.cuda.synchronize()
base_ms = (time.perf_counter() - t0) / 10 * 1000
print(f"Base BM=128 NW=4: {base_ms:.3f} ms (reference)")

# Sweep maxnreg for fused kernel
import vllm.lora.ops.triton_ops.fused_moe_lora_fused_kernel as fmod

configs = [
    # (label, BM, NW, maxnreg_or_None)
    ("BM=64  NW=8 maxnreg=64",   64, 8, 64),
    ("BM=64  NW=8 maxnreg=80",   64, 8, 80),
    ("BM=64  NW=8 maxnreg=96",   64, 8, 96),
    ("BM=64  NW=8 maxnreg=128",  64, 8, 128),
    ("BM=64  NW=8 maxnreg=160",  64, 8, 160),
    ("BM=64  NW=8 maxnreg=192",  64, 8, 192),
    ("BM=64  NW=8 maxnreg=255",  64, 8, 255),
    ("BM=64  NW=8 no-maxnreg",   64, 8, None),
    ("BM=64  NW=4 maxnreg=128",  64, 4, 128),
    ("BM=64  NW=4 maxnreg=192",  64, 4, 192),
    ("BM=64  NW=4 maxnreg=255",  64, 4, 255),
    ("BM=64  NW=4 no-maxnreg",   64, 4, None),
    ("BM=128 NW=4 maxnreg=128",  128, 4, 128),
    ("BM=128 NW=4 maxnreg=192",  128, 4, 192),
    ("BM=128 NW=4 maxnreg=255",  128, 4, 255),
    ("BM=128 NW=4 no-maxnreg",   128, 4, None),
    ("BM=128 NW=8 maxnreg=128",  128, 8, 128),
    ("BM=128 NW=8 maxnreg=192",  128, 8, 192),
    ("BM=128 NW=8 maxnreg=255",  128, 8, 255),
    ("BM=128 NW=8 no-maxnreg",   128, 8, None),
]

print(f"\n{'Config':40s}  {'regs':>5s}  {'spills':>6s}  {'ms':>8s}  {'OH%':>7s}  {'tcgen05':>7s}")
print("-" * 90)

for label, bm, nw, maxnreg in configs:
    importlib.reload(fmod)
    cfg = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
           "GROUP_SIZE_M": 1, "num_warps": nw, "num_stages": 3}
    if maxnreg is not None:
        cfg["maxnreg"] = maxnreg

    sorted_f, expert_f, lora_f, npost_f = moe_lora_align_block_size_fused(
        topk_ids, lora_ids_per_token, bm, E, 1)
    out_fused = torch.zeros(M * TOP_K, N, device=device, dtype=DTYPE)

    try:
        for _ in range(3):
            fmod.invoke_fused_moe_lora_kernel(
                hidden, w, out_fused, topk_weights,
                sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                True, TOP_K, 1, cfg.copy(), compute_type=tl.bfloat16)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            fmod.invoke_fused_moe_lora_kernel(
                hidden, w, out_fused, topk_weights,
                sorted_f, expert_f, lora_f, npost_f, lora_a, lora_b,
                True, TOP_K, 1, cfg.copy(), compute_type=tl.bfloat16)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / 10 * 1000
        oh = (fused_ms - base_ms) / base_ms * 100

        # Get kernel stats
        kernel_fn = fmod.fused_moe_with_lora_kernel
        regs = spills = tcgen05 = "?"
        for device_key, cache in kernel_fn.device_caches.items():
            compiled_dict = cache[0]
            for k, compiled in compiled_dict.items():
                regs = compiled.n_regs
                spills = compiled.n_spills
                ptx = compiled.asm.get('ptx', '') if hasattr(compiled, 'asm') else ''
                tcgen05 = ptx.lower().count('tcgen05.mma')
                break
            break

        print(f"{label:40s}  {regs:>5}  {spills:>6}  {fused_ms:>7.3f}  {oh:>+6.1f}%  {tcgen05:>7}")

    except Exception as e:
        print(f"{label:40s}  ERROR: {e}")

    torch.cuda.empty_cache()
