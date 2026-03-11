# Fused MoE + LoRA Kernel: Design & Continuation Plan

## Goal

Fuse the LoRA delta computation into the base MoE expert GEMM Triton kernel,
eliminating the separate LoRA kernel and recovering full arithmetic intensity.

## Motivation

Currently vLLM runs LoRA as a **separate kernel** after the base expert GEMM.
The standalone LoRA kernel has arithmetic intensity ~2r (where r is LoRA rank,
typically 8-64), making it deeply memory-bound regardless of batch size.

However, if fused into the base GEMM, the LoRA computation reuses the input
activation tile already in SRAM. The combined AI becomes:

    AI = (2SH^2 + 4SHr) / (2SH + H^2 + 2Hr)

For large S this approaches H + 2r — dominated by the base GEMM, not bounded
by r. The LoRA overhead becomes negligible.

## Architecture: How the Current Code Works

### Base MoE kernel (unquantized Triton path)

- **Kernel**: `fused_moe_kernel` at `vllm/model_executor/layers/fused_moe/fused_moe.py:314`
- **Launcher**: `invoke_fused_moe_triton_kernel` at same file, line 725
- **Orchestrator**: `TritonExperts.apply()` at line 1974
  - Calls w1 GEMM (line 2054), activation (line 2077), w2 GEMM (line 2091), moe_sum (line 2115)

The kernel:
1. Uses a 1-D grid mapped to (pid_m, pid_n) with grouped ordering for L2 reuse
2. `sorted_token_ids` (pre-sorted by expert via `moe_align_block_size`) maps
   each M-block to tokens sharing the same expert
3. `expert_ids[pid_m]` gives the expert index for weight lookup
4. K-reduction loop: loads `a` tile [BLOCK_M, BLOCK_K] and `w` tile [BLOCK_K, BLOCK_N],
   accumulates `tl.dot(a, w)` into [BLOCK_M, BLOCK_N] accumulator
5. Optional router weight multiply, then store

### Current LoRA MoE kernel (separate)

- **Kernel**: `_fused_moe_lora_kernel` at `vllm/lora/ops/triton_ops/fused_moe_lora_op.py:49`
- **Composed by**: `_fused_moe_lora` at line 429 (calls shrink then expand)
- **Called from**: `PunicaWrapperGPU.add_lora_fused_moe` at `vllm/lora/punica_wrapper/punica_gpu.py:360`
- **Injected via decorators**: `FusedMoEWithLoRA._inject_lora_into_fused_moe` at `vllm/lora/layers/fused_moe.py:126`
  - `act_decorator` (line 162): hooks after w1 GEMM to apply w13 LoRA
  - `moe_sum_decorator` (line 242): hooks after w2 GEMM to apply w2 LoRA

The separate kernel uses a 3-D grid: (M*N blocks, num_slices, max_loras+1).
It sorts tokens by (lora_id, expert_id) independently from the base kernel.
Uses PDL (Programmatic Dependent Launch) to overlap shrink→expand latency.

### Weight layouts

```
Base weights:   W          — (E, N, K)
LoRA-A:         lora_a     — (max_loras, E, rank, K)     contiguous in K
LoRA-B:         lora_b     — (max_loras, E, N, rank)     contiguous in rank
```

For gated MoE (w1/w3): lora_a and lora_b are tuples of 2 tensors (gate + up slices).
For w2: single tensor (1 slice).

## Fusion Design

### Core idea

During the K-reduction loop of the base GEMM, the input tile `a` [BLOCK_M, BLOCK_K]
is already loaded. We simultaneously accumulate `tl.dot(a, lora_a_tile)` into a
small [BLOCK_M, LORA_RANK] buffer. After the K-loop, we load lora_b [LORA_RANK, BLOCK_N]
(tiny) and add `tl.dot(lora_acc, lora_b_tile)` to the main accumulator before storing.

```
K-loop:
  a = load(a_ptrs)                         # [BLOCK_M, BLOCK_K]  — already needed
  w = load(w_ptrs)                         # [BLOCK_K, BLOCK_N]
  accumulator += dot(a, w)                 # base GEMM
  la = load(lora_a_ptrs)                   # [BLOCK_K, LORA_RANK]  — NEW, small
  lora_acc += dot(a, la)                   # shrink  — NEW, reuses 'a'

After K-loop:
  lb = load(lora_b_ptrs)                   # [LORA_RANK, BLOCK_N]  — NEW, tiny
  accumulator += dot(lora_acc, lb)         # expand  — NEW
  store(accumulator)
```

Extra memory traffic: only 2Hr bytes for LoRA weights (negligible vs H^2 for base).
Extra registers: BLOCK_M * LORA_RANK for lora_acc (~1024 for BM=64, r=16).

### Token sorting requirement

The base kernel sorts by expert_id only. The fused kernel must sort by
**(expert_id, lora_id)** so each M-block has a uniform (expert, lora) pair.
This produces slightly more padding but is necessary for correct LoRA weight lookup.

The prototype includes `moe_lora_align_block_size()` which does this sorting.
For production, this should be a CUDA op (similar to the existing `moe_align_block_size`).

### Multi-slice support (w1 gate+up)

For w1, the output has 2N columns (gate and up projections). Each LoRA slice
covers N columns. A thread block with `pid_n` can determine which slice it
belongs to via `slice_idx = pid_n * BLOCK_SIZE_N // N`. The prototype handles
single-slice only; this extension is straightforward.

## What Has Been Done

1. Full analysis of both kernels (base MoE and separate LoRA)
2. Arithmetic intensity analysis proving fusion recovers full AI
3. Prototype kernel written at `prototype_fused_moe_lora.py`:
   - `fused_moe_with_lora_kernel` — the fused Triton kernel
   - `moe_lora_align_block_size` — token sorting by (expert, lora)
   - `fused_moe_with_lora` — kernel launcher
   - `reference_moe_with_lora` — loop-based PyTorch reference for validation
   - `test_fused_moe_lora` — correctness test (small problem: 64 tokens, 8 experts, rank=16)
   - `bench_fused_vs_separate` — wall-clock benchmark (larger problem: 512 tokens, 64 experts)

## What Needs To Be Done (on GPU machine)

### Phase 1: Validate the prototype

1. **Run correctness test**: `python prototype_fused_moe_lora.py`
   - Expect bf16 tolerance: atol=1e-2, rtol=5e-2
   - If it fails, likely causes:
     - Pointer arithmetic off (check stride order for lora_a/lora_b)
     - LORA_RANK < 16 hitting tl.dot minimum dimension constraint
     - Token mask not applied to lora_acc (masked tokens contributing garbage)

2. **Debug if needed**: Add print statements in the reference to dump per-token
   results, compare against specific output positions from the kernel.

3. **Run benchmark** to get baseline fused kernel timing.

### Phase 2: Compare against existing separate kernels

1. Run the existing base `fused_moe_kernel` + `_fused_moe_lora` separately on
   the same inputs, compare outputs match the fused kernel.

2. Benchmark: fused vs (base + separate LoRA) to measure actual speedup.
   The fused kernel should show improvement from:
   - Eliminating one full read of the activation tensor (SH bytes saved)
   - Eliminating LoRA kernel launch overhead
   - Better SM utilization (no idle SMs between kernels)

### Phase 3: Production integration

1. **Multi-slice support**: Extend kernel for w1 (gate+up, 2 slices). Pass
   `slice_size` and `num_slices` as constexprs. Compute `slice_idx` from `pid_n`.

2. **No-LoRA tokens**: Add runtime check `if off_lora >= 0` to skip LoRA for
   tokens without an adapter. All threads in a block see the same lora_id
   (uniform branch).

3. **CUDA alignment op**: Replace the Python `moe_lora_align_block_size` with a
   CUDA kernel for production speed (or adapt the existing
   `ops.moe_lora_align_block_size`).

4. **Integration into FusedMoEWithLoRA**: Replace the decorator-based injection
   with a new `TritonExpertsWithLoRA` class (or modify `TritonExperts`) that
   calls the fused kernel directly instead of the base kernel + separate LoRA.

5. **Kernel config tuning**: The fused kernel has different register pressure
   than the base kernel (extra BLOCK_M * LORA_RANK accumulators). May need
   different BLOCK_SIZE_M/N/K configs. Use Triton autotuning.

6. **Rank < 16 support**: Pad LORA_RANK to 16 in the launcher, zero-fill the
   extra LoRA weight columns/rows.

### Phase 4: Testing

1. Run existing LoRA MoE tests with the fused kernel:
   - `tests/lora/test_fused_moe_lora_kernel.py`
   - `tests/lora/test_moe_lora_align_sum.py`
2. Test with real models (Mixtral + LoRA adapters) for end-to-end validation.

## Key Files Reference

| File | What's there |
|------|-------------|
| `prototype_fused_moe_lora.py` | The prototype (this work) |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:314` | Base `fused_moe_kernel` |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:725` | `invoke_fused_moe_triton_kernel` launcher |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1974` | `TritonExperts.apply()` |
| `vllm/lora/ops/triton_ops/fused_moe_lora_op.py:49` | Existing separate LoRA kernel |
| `vllm/lora/ops/triton_ops/fused_moe_lora_op.py:429` | `_fused_moe_lora` (shrink+expand) |
| `vllm/lora/layers/fused_moe.py:126` | `_inject_lora_into_fused_moe` (decorator injection) |
| `vllm/lora/punica_wrapper/punica_gpu.py:360` | `add_lora_fused_moe` (calls separate kernel) |
| `tests/lora/test_fused_moe_lora_kernel.py` | Existing LoRA MoE tests |
