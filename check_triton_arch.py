"""Check what MMA instructions and arch Triton generates for B200."""
import torch
import triton
import triton.language as tl


@triton.jit
def _test_matmul(a_ptr, b_ptr, c_ptr, K: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], c)


a = torch.randn(64, 32, device='cuda', dtype=torch.bfloat16)
b = torch.randn(32, 128, device='cuda', dtype=torch.bfloat16)
c = torch.zeros(64, 128, device='cuda', dtype=torch.float32)

_test_matmul[(1,)](a, b, c, 32, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
                   num_warps=4)
torch.cuda.synchronize()

# Inspect compiled kernel
for key, cache in _test_matmul.device_caches.items():
    compiled_dict = cache[0]
    for k, compiled in compiled_dict.items():
        print(f'n_regs: {compiled.n_regs}, n_spills: {compiled.n_spills}')
        if hasattr(compiled, 'asm'):
            asm = compiled.asm
            if isinstance(asm, dict):
                print(f'ASM keys: {list(asm.keys())}')
                if 'ptx' in asm:
                    ptx = asm['ptx']
                    # Find target arch
                    for line in ptx.split('\n'):
                        if '.target' in line or '.version' in line:
                            print(f'  {line.strip()}')
                    # Find MMA instructions
                    mma_found = set()
                    for line in ptx.split('\n'):
                        l = line.strip().lower()
                        if any(x in l for x in ['mma', 'wgmma', 'tcgen', 'tmem',
                                                  'cp.async', 'tma', 'ldmatrix']):
                            # Deduplicate by instruction prefix
                            instr = line.strip().split('//')[0].strip()
                            if instr and instr not in mma_found:
                                mma_found.add(instr)
                    print(f'\nKey instructions found ({len(mma_found)}):')
                    for instr in sorted(mma_found):
                        print(f'  {instr}')

                if 'ttgir' in asm:
                    ttgir = asm['ttgir']
                    for line in ttgir.split('\n'):
                        if 'mma' in line.lower() or 'dot' in line.lower() or 'warp' in line.lower():
                            print(f'  TTGIR: {line.strip()[:120]}')
        break
    break
