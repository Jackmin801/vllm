"""
Prototype: Fused MoE + LoRA Triton Kernel

Fuses the LoRA delta computation (x @ A^T @ B^T) into the base expert GEMM
(x @ W^T) during the K-reduction loop, avoiding separate LoRA kernel launches
and redundant activation memory traffic.

The fused computation per (token, expert, lora):
    output = x @ W^T + x @ A^T @ B^T

Key insight: the input tile 'a' loaded for the base GEMM is reused for the
LoRA-A shrink at zero extra memory cost. After the K-loop, the tiny LoRA-B
expand ([BLOCK_M, r] @ [r, BLOCK_N]) is applied before the store.

Limitations of this prototype:
- Single LoRA slice (covers w2 / down projection). For w1 (gate+up) with
  2 slices, extend by computing slice_idx from pid_n.
- Requires LORA_RANK >= 16 (Triton tl.dot minimum dimension).
- No quantization support (unquantized bf16 path only).
- No expert parallelism (expert_id == -1 not handled).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton Kernel
# ---------------------------------------------------------------------------
@triton.jit
def fused_moe_with_lora_kernel(
    # Base MoE pointers
    a_ptr,  # input activations: (num_tokens, K)
    w_ptr,  # expert weights:    (E, N, K)
    c_ptr,  # output:            written at sorted token positions, width N
    topk_weights_ptr,  # router weights: (num_tokens, top_k)
    sorted_token_ids_ptr,  # (num_tokens_post_padded,)
    expert_ids_ptr,  # (num_m_blocks,)
    num_tokens_post_padded_ptr,  # scalar
    # LoRA pointers
    lora_a_ptr,  # (max_loras, E, LORA_RANK, K)
    lora_b_ptr,  # (max_loras, E, N, LORA_RANK)
    lora_ids_ptr,  # (num_m_blocks,) — per-block lora adapter id
    # Dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # A strides
    stride_am,
    stride_ak,
    # W strides  (W is (E, N, K))
    stride_we,
    stride_wn,
    stride_wk,
    # C strides
    stride_cm,
    stride_cn,
    # LoRA-A strides (max_loras, E, LORA_RANK, K)
    stride_la_l,
    stride_la_e,
    stride_la_r,
    stride_la_k,
    # LoRA-B strides (max_loras, E, N, LORA_RANK)
    stride_lb_l,
    stride_lb_e,
    stride_lb_n,
    stride_lb_r,
    # Constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LORA_RANK: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HAS_LORA: tl.constexpr,
):
    # --- Map 1-D pid to (pid_m, pid_n) with grouped ordering for L2 reuse ---
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Early exit for padding blocks ---
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # --- Load token indices and expert id for this M-block ---
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # --- Base GEMM pointer setup ---
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A tile pointers: [BLOCK_M, BLOCK_K]
    # offs_token // top_k recovers the original token index
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    # W tile pointers: [BLOCK_K, BLOCK_N]  (reading W^T from (E, N, K) layout)
    w_ptrs = (
        w_ptr
        + off_expert * stride_we
        + offs_k[:, None] * stride_wk
        + offs_bn[None, :] * stride_wn
    )

    # --- LoRA pointer setup ---
    if HAS_LORA:
        off_lora = tl.load(lora_ids_ptr + pid_m).to(tl.int64)
        offs_r = tl.arange(0, LORA_RANK)

        # LoRA-A base for this (lora, expert) pair
        lora_a_base = (
            lora_a_ptr + off_lora * stride_la_l + off_expert * stride_la_e
        )
        # Tile pointers: [BLOCK_K, LORA_RANK]  (reading A^T from (rank, K) layout)
        lora_a_ptrs = (
            lora_a_base
            + offs_k[:, None] * stride_la_k
            + offs_r[None, :] * stride_la_r
        )

        # Shrink accumulator: [BLOCK_M, LORA_RANK]
        lora_acc = tl.zeros((BLOCK_SIZE_M, LORA_RANK), dtype=tl.float32)

    # --- Main K-reduction loop ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining

        # Load input tile (shared between base GEMM and LoRA shrink)
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]

        # Load weight tile
        w = tl.load(
            w_ptrs, mask=k_mask[:, None], other=0.0
        )  # [BLOCK_K, BLOCK_N]

        # Base GEMM accumulation
        accumulator += tl.dot(a, w)

        # LoRA-A shrink: reuse 'a' tile already in registers
        if HAS_LORA:
            la = tl.load(
                lora_a_ptrs, mask=k_mask[:, None], other=0.0
            )  # [BLOCK_K, LORA_RANK]
            lora_acc += tl.dot(a, la)  # [BLOCK_M, LORA_RANK]
            lora_a_ptrs += BLOCK_SIZE_K * stride_la_k

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # --- LoRA-B expand: accumulator += lora_acc @ B^T ---
    if HAS_LORA:
        # Load full LoRA-B tile: [LORA_RANK, BLOCK_N]
        # (reading B^T from (N, rank) layout)
        lora_b_base = (
            lora_b_ptr + off_lora * stride_lb_l + off_expert * stride_lb_e
        )
        lora_b_tile = tl.load(
            lora_b_base
            + offs_r[:, None] * stride_lb_r
            + offs_bn[None, :] * stride_lb_n,
            mask=offs_bn[None, :] < N,
            other=0.0,
        )  # [LORA_RANK, BLOCK_N]

        # Expand: [BLOCK_M, LORA_RANK] @ [LORA_RANK, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        accumulator += tl.dot(lora_acc.to(compute_type), lora_b_tile)

    # --- Router weight multiplication ---
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # --- Store output ---
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Token alignment: sort by (expert_id, lora_id) and pad to BLOCK_SIZE_M
# ---------------------------------------------------------------------------
def moe_lora_align_block_size(
    topk_ids: torch.Tensor,  # (num_tokens, top_k)
    lora_ids: torch.Tensor,  # (num_tokens,)  — lora adapter per token
    block_size: int,
    num_experts: int,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort token-expert-lora triples so that each block of ``block_size``
    consecutive entries shares the same (expert, lora) pair.  Returns:
        sorted_token_ids   — (num_tokens_post_padded,)
        expert_ids_per_blk — (num_m_blocks,)
        lora_ids_per_blk   — (num_m_blocks,)
        num_tokens_post_padded — (1,) scalar tensor
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device

    # Flat index in [0, num_tokens * top_k): encodes (token, k) pair
    flat_token_ids = (
        torch.arange(num_tokens, device=device).unsqueeze(1) * top_k
        + torch.arange(top_k, device=device).unsqueeze(0)
    ).flatten()

    flat_expert_ids = topk_ids.flatten()
    flat_lora_ids = lora_ids.unsqueeze(1).expand(-1, top_k).flatten()

    # Combined sort key: (expert_id, lora_id)
    # +1 so that lora_id=-1 (no adapter) maps to 0
    sort_key = flat_expert_ids.long() * (max_loras + 1) + (flat_lora_ids.long() + 1)
    sorted_indices = sort_key.argsort(stable=True)

    sorted_token_ids = flat_token_ids[sorted_indices]
    sorted_experts = flat_expert_ids[sorted_indices]
    sorted_loras = flat_lora_ids[sorted_indices]

    # Group by (expert, lora) and pad each group to block_size
    num_valid = num_tokens * top_k
    padded_tokens: list[int] = []
    blk_experts: list[int] = []
    blk_loras: list[int] = []

    i = 0
    while i < len(sorted_token_ids):
        e = sorted_experts[i].item()
        l = sorted_loras[i].item()
        j = i
        while (
            j < len(sorted_token_ids)
            and sorted_experts[j].item() == e
            and sorted_loras[j].item() == l
        ):
            j += 1
        group_len = j - i
        padded_len = ((group_len + block_size - 1) // block_size) * block_size
        padded_tokens.extend(sorted_token_ids[i:j].tolist())
        padded_tokens.extend([num_valid] * (padded_len - group_len))
        n_blocks = padded_len // block_size
        blk_experts.extend([e] * n_blocks)
        blk_loras.extend([l] * n_blocks)
        i = j

    sorted_token_ids_t = torch.tensor(padded_tokens, device=device, dtype=torch.int64)
    expert_ids_t = torch.tensor(blk_experts, device=device, dtype=torch.int64)
    lora_ids_t = torch.tensor(blk_loras, device=device, dtype=torch.int64)
    num_post = torch.tensor([len(padded_tokens)], device=device, dtype=torch.int64)
    return sorted_token_ids_t, expert_ids_t, lora_ids_t, num_post


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
def fused_moe_with_lora(
    hidden_states: torch.Tensor,  # (num_tokens, K)
    w: torch.Tensor,  # (E, N, K)
    lora_a: torch.Tensor,  # (max_loras, E, rank, K)
    lora_b: torch.Tensor,  # (max_loras, E, N, rank)
    topk_weights: torch.Tensor,  # (num_tokens, top_k)
    topk_ids: torch.Tensor,  # (num_tokens, top_k)
    lora_ids: torch.Tensor,  # (num_tokens,)
    top_k: int,
    num_experts: int,
    max_loras: int,
    mul_routed_weight: bool = True,
    block_size_m: int = 64,
    block_size_n: int = 64,
    block_size_k: int = 32,
    group_size_m: int = 8,
) -> torch.Tensor:
    num_tokens, K = hidden_states.shape
    N = w.shape[1]
    rank = lora_a.shape[2]
    num_valid_tokens = num_tokens * top_k

    # Align tokens by (expert, lora)
    sorted_token_ids, expert_ids, blk_lora_ids, num_post = (
        moe_lora_align_block_size(
            topk_ids, lora_ids, block_size_m, num_experts, max_loras
        )
    )

    EM = sorted_token_ids.shape[0]

    # Output buffer: indexed by flat (token * top_k + k), width N
    output = torch.zeros(
        (num_valid_tokens, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    fused_moe_with_lora_kernel[grid](
        hidden_states,
        w,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_post,
        lora_a,
        lora_b,
        blk_lora_ids,
        # Dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # A strides
        hidden_states.stride(0),
        hidden_states.stride(1),
        # W strides (E, N, K)
        w.stride(0),
        w.stride(1),
        w.stride(2),
        # C strides
        output.stride(0),
        output.stride(1),
        # LoRA-A strides (max_loras, E, rank, K)
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        lora_a.stride(3),
        # LoRA-B strides (max_loras, E, N, rank)
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        lora_b.stride(3),
        # Constexprs
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        LORA_RANK=rank,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        HAS_LORA=True,
    )

    return output


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, float32 for precision)
# ---------------------------------------------------------------------------
def reference_moe_with_lora(
    hidden_states: torch.Tensor,  # (num_tokens, K)
    w: torch.Tensor,  # (E, N, K)
    lora_a: torch.Tensor,  # (max_loras, E, rank, K)
    lora_b: torch.Tensor,  # (max_loras, E, N, rank)
    topk_weights: torch.Tensor,  # (num_tokens, top_k)
    topk_ids: torch.Tensor,  # (num_tokens, top_k)
    lora_ids: torch.Tensor,  # (num_tokens,)
    mul_routed_weight: bool = True,
) -> torch.Tensor:
    """Compute base MoE + LoRA in float32, return per-(token, expert) results."""
    num_tokens, top_k = topk_ids.shape
    N = w.shape[1]

    x = hidden_states.float()
    W = w.float()
    A = lora_a.float()
    B = lora_b.float()
    tw = topk_weights.float()

    output = torch.zeros(
        (num_tokens * top_k, N), device=hidden_states.device, dtype=torch.float32
    )

    for t in range(num_tokens):
        for k in range(top_k):
            e = topk_ids[t, k].item()
            l = lora_ids[t].item()
            flat_idx = t * top_k + k

            # Base GEMM: W[e] @ x[t]  — W is (N, K), x is (K,)
            result = W[e] @ x[t]

            # LoRA delta: B[l, e] @ A[l, e] @ x[t]
            if l >= 0:
                intermediate = A[l, e] @ x[t]  # (rank,)
                result = result + B[l, e] @ intermediate  # (N,)

            if mul_routed_weight:
                result = result * tw[t, k]

            output[flat_idx] = result

    return output


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@torch.no_grad()
def test_fused_moe_lora():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Problem sizes
    num_tokens = 64
    num_experts = 8
    top_k = 2
    K = 256  # hidden size
    N = 512  # intermediate / output size
    max_loras = 4
    rank = 16  # must be >= 16 for tl.dot

    # Random data
    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01

    # Router: random top-k expert assignment
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )

    # LoRA assignment: each token gets a random adapter
    lora_ids = torch.randint(0, max_loras, (num_tokens,), device=device, dtype=torch.int64)

    # --- Fused kernel ---
    fused_output = fused_moe_with_lora(
        hidden_states,
        w,
        lora_a,
        lora_b,
        topk_weights,
        topk_ids,
        lora_ids,
        top_k=top_k,
        num_experts=num_experts,
        max_loras=max_loras,
        mul_routed_weight=True,
        block_size_m=32,
        block_size_n=64,
        block_size_k=32,
    )

    # --- Reference ---
    ref_output = reference_moe_with_lora(
        hidden_states,
        w,
        lora_a,
        lora_b,
        topk_weights,
        topk_ids,
        lora_ids,
        mul_routed_weight=True,
    )

    # Compare
    fused_f32 = fused_output.float()
    max_diff = (fused_f32 - ref_output).abs().max().item()
    mean_diff = (fused_f32 - ref_output).abs().mean().item()
    ref_norm = ref_output.abs().mean().item()
    rel_err = mean_diff / (ref_norm + 1e-8)

    print(f"Max absolute diff:  {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Reference mean abs: {ref_norm:.6f}")
    print(f"Relative error:     {rel_err:.6f}")

    # bf16 tolerance: accumulator is fp32, but inputs/outputs are bf16
    atol = 1e-2
    rtol = 5e-2
    if torch.allclose(fused_f32, ref_output, atol=atol, rtol=rtol):
        print(f"\nPASSED (atol={atol}, rtol={rtol})")
    else:
        # Find worst offenders
        diff = (fused_f32 - ref_output).abs()
        worst_idx = diff.argmax()
        row, col = worst_idx // N, worst_idx % N
        print(f"\nFAILED — worst at [{row}, {col}]: "
              f"fused={fused_f32[row, col]:.6f}, ref={ref_output[row, col]:.6f}")
        # Still print pass/fail ratio
        close_mask = (diff <= atol + rtol * ref_output.abs())
        pct = close_mask.float().mean().item() * 100
        print(f"  {pct:.1f}% of elements within tolerance")


# ---------------------------------------------------------------------------
# Benchmark: fused vs separate (base + standalone LoRA)
# ---------------------------------------------------------------------------
def bench_fused_vs_separate():
    """Quick wall-clock comparison."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    num_tokens = 512
    num_experts = 64
    top_k = 2
    K = 4096
    N = 4096
    max_loras = 4
    rank = 16

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.01
    lora_a = torch.randn(max_loras, num_experts, rank, K, device=device, dtype=dtype) * 0.01
    lora_b = torch.randn(max_loras, num_experts, N, rank, device=device, dtype=dtype) * 0.01

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
    )
    lora_ids = torch.randint(0, max_loras, (num_tokens,), device=device, dtype=torch.int64)

    # Warmup
    for _ in range(5):
        fused_moe_with_lora(
            hidden_states, w, lora_a, lora_b, topk_weights, topk_ids, lora_ids,
            top_k=top_k, num_experts=num_experts, max_loras=max_loras,
        )
    torch.cuda.synchronize()

    import time
    N_ITER = 50

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fused_moe_with_lora(
            hidden_states, w, lora_a, lora_b, topk_weights, topk_ids, lora_ids,
            top_k=top_k, num_experts=num_experts, max_loras=max_loras,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    fused_ms = (t1 - t0) / N_ITER * 1000
    print(f"\nBenchmark (num_tokens={num_tokens}, E={num_experts}, K={K}, N={N}, rank={rank}):")
    print(f"  Fused kernel: {fused_ms:.3f} ms")


if __name__ == "__main__":
    print("=" * 60)
    print("Correctness test")
    print("=" * 60)
    test_fused_moe_lora()

    print("\n" + "=" * 60)
    print("Benchmark")
    print("=" * 60)
    bench_fused_vs_separate()
