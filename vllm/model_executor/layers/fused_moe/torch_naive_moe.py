# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-torch reference MoE experts implementation with LoRA support.

This is a slow but correct implementation intended for testing and
validation of the EP+LoRA pipeline end-to-end. It implements the full
permute -> GEMM1 -> LoRA_w1 -> activation -> GEMM2 -> LoRA_w2 -> unpermute
pipeline using basic torch operations.

Based on meow_kernel.py reference implementation.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)


def _dequantize_fp8_block(
    w_fp8: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 weights with per-block scales to bfloat16."""
    shape = w_fp8.shape
    *batch, M, K = shape
    w = w_fp8.to(torch.bfloat16).reshape(
        *batch, M // block_size, block_size, K // block_size, block_size
    )
    s = scales.reshape(*batch, M // block_size, 1, K // block_size, 1)
    return (w * s).reshape(shape)


def _permute_by_expert(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_ids: torch.Tensor | None,
    num_experts: int,
) -> tuple[
    torch.Tensor,  # permuted_h
    list[int],  # expert_offsets (cumulative, len E+1)
    list[int],  # expert_counts
    torch.Tensor,  # token_indices (M_total,)
    torch.Tensor,  # k_indices (M_total,)
    torch.Tensor,  # weights_perm (M_total,)
    torch.Tensor | None,  # lora_ids_perm (M_total,) or None
]:
    """Permute tokens so each expert's assigned (token, k) pairs are
    contiguous."""
    device = hidden_states.device

    per_expert_tok = []
    per_expert_k = []
    expert_counts = []

    for e in range(num_experts):
        mask = topk_ids == e
        tok, k = torch.where(mask)
        per_expert_tok.append(tok)
        per_expert_k.append(k)
        expert_counts.append(len(tok))

    expert_offsets = [0]
    for cnt in expert_counts:
        expert_offsets.append(expert_offsets[-1] + cnt)

    M_total = expert_offsets[-1]

    if M_total == 0:
        K = hidden_states.shape[1]
        empty_h = torch.empty(0, K, dtype=hidden_states.dtype, device=device)
        empty_idx = torch.empty(0, dtype=torch.long, device=device)
        empty_w = torch.empty(0, dtype=topk_weights.dtype, device=device)
        empty_lora = None if lora_ids is None else torch.empty(
            0, dtype=torch.int32, device=device)
        return (empty_h, expert_offsets, expert_counts,
                empty_idx, empty_idx, empty_w, empty_lora)

    # Concatenate all expert token/k indices
    all_tok = torch.cat(per_expert_tok)
    all_k = torch.cat(per_expert_k)

    permuted_h = hidden_states[all_tok]
    weights_perm = topk_weights[all_tok, all_k]
    lora_ids_perm = lora_ids[all_tok] if lora_ids is not None else None

    return (permuted_h, expert_offsets, expert_counts,
            all_tok, all_k, weights_perm, lora_ids_perm)


def _grouped_gemm_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    expert_offsets: list[int],
    expert_counts: list[int],
    num_experts: int,
) -> torch.Tensor:
    """Per-expert matmul: A @ B[e].T for each expert e.

    A: (M_total, K) — permuted activations
    B: (E, N, K) — expert weights
    Returns: (M_total, N)
    """
    M_total = A.shape[0]
    N = B.shape[1]
    C = torch.zeros(M_total, N, dtype=torch.bfloat16, device=A.device)
    for e in range(num_experts):
        s = expert_offsets[e]
        cnt = expert_counts[e]
        if cnt == 0:
            continue
        C[s:s + cnt] = (
            A[s:s + cnt].float() @ B[e].float().T
        ).bfloat16()
    return C


def _apply_lora_w1(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_ids_perm: torch.Tensor,
    expert_offsets: list[int],
    expert_counts: list[int],
    w13_lora_a: torch.Tensor,
    w13_lora_b: torch.Tensor,
    num_experts: int,
) -> None:
    """Apply w1 (gate+up) LoRA corrections in-place on permuted GEMM1 output.

    output:       (M_total, 2*INTER) bf16
    hidden_states: (M_total, HIDDEN) bf16
    w13_lora_a:   (num_slices, max_loras, E, rank, hidden) bf16
    w13_lora_b:   (num_slices, max_loras, E, inter, rank) bf16
    """
    INTER = w13_lora_b.shape[3]
    num_slices = w13_lora_a.shape[0]

    for e in range(num_experts):
        s = expert_offsets[e]
        cnt = expert_counts[e]
        if cnt == 0:
            continue
        h_e = hidden_states[s:s + cnt]
        lids_e = lora_ids_perm[s:s + cnt]
        for lid in lids_e.unique():
            if lid.item() < 0:
                continue
            m = lids_e == lid
            h_l = h_e[m]
            idx = torch.arange(s, s + cnt, device=output.device)[m]
            if num_slices == 2:
                # Gate correction (slice 0)
                output[idx, :INTER] += (
                    h_l @ w13_lora_a[0, lid, e].T
                         @ w13_lora_b[0, lid, e].T
                )
                # Up correction (slice 1)
                output[idx, INTER:] += (
                    h_l @ w13_lora_a[1, lid, e].T
                         @ w13_lora_b[1, lid, e].T
                )
            else:
                # Single slice (non-gated MoE)
                output[idx] += (
                    h_l @ w13_lora_a[0, lid, e].T
                         @ w13_lora_b[0, lid, e].T
                )


def _apply_lora_w2(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_ids_perm: torch.Tensor,
    expert_offsets: list[int],
    expert_counts: list[int],
    w2_lora_a: torch.Tensor,
    w2_lora_b: torch.Tensor,
    num_experts: int,
) -> None:
    """Apply w2 (down) LoRA corrections in-place on permuted GEMM2 output.

    output:       (M_total, HIDDEN) bf16
    hidden_states: (M_total, INTER) bf16 — post-activation
    w2_lora_a:    (max_loras, E, rank, inter) bf16
    w2_lora_b:    (max_loras, E, hidden, rank) bf16
    """
    for e in range(num_experts):
        s = expert_offsets[e]
        cnt = expert_counts[e]
        if cnt == 0:
            continue
        h_e = hidden_states[s:s + cnt]
        lids_e = lora_ids_perm[s:s + cnt]
        for lid in lids_e.unique():
            if lid.item() < 0:
                continue
            m = lids_e == lid
            h_l = h_e[m]
            idx = torch.arange(s, s + cnt, device=output.device)[m]
            output[idx] += (
                h_l @ w2_lora_a[lid, e].T
                     @ w2_lora_b[lid, e].T
            )


class TorchNaiveFusedMoEExperts(mk.FusedMoEExpertsModular):
    """Pure-torch reference implementation of fused MoE experts with LoRA.

    Supports FP8 block-quantized weights and optional LoRA corrections.
    Slow but correct — intended for testing the EP+LoRA pipeline.

    LoRA weights are set as attributes by FusedMoEWithLoRA:
        self.w13_lora_a_stacked: tuple[torch.Tensor, ...]
        self.w13_lora_b_stacked: tuple[torch.Tensor, ...]
        self.w2_lora_a_stacked: tuple[torch.Tensor, ...]
        self.w2_lora_b_stacked: tuple[torch.Tensor, ...]
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Accept any scheme — we dequantize to bf16 anyway
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_lora(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        # We handle dequantization ourselves
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # We do weight application and reduction inside apply()
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # We don't use workspaces — all allocation is internal
        # But we need the output shape to be correct
        workspace13 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
        w13_lora_a: torch.Tensor | None = None,
        w13_lora_b: torch.Tensor | None = None,
        w2_lora_a: torch.Tensor | None = None,
        w2_lora_b: torch.Tensor | None = None,
    ) -> None:
        num_experts = w1.shape[0]
        N = w1.shape[1]  # 2*INTER for gated, INTER for non-gated

        # Get lora_ids from expert_tokens_meta if available
        lora_ids = (
            expert_tokens_meta.lora_ids
            if expert_tokens_meta is not None
            else None
        )

        has_lora = w13_lora_a is not None and lora_ids is not None

        # Apply expert_map to convert global expert IDs to local IDs.
        # Non-local experts become -1 and are skipped in permutation.
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        # 1. Permute tokens by expert
        (permuted_h, expert_offsets, expert_counts,
         token_indices, k_indices, weights_perm,
         lora_ids_perm) = _permute_by_expert(
            hidden_states, topk_ids, topk_weights, lora_ids, num_experts)

        M_total = permuted_h.shape[0]
        if M_total == 0:
            output.zero_()
            return

        # 2. Dequantize weights if FP8
        if w1.dtype == torch.float8_e4m3fn:
            assert self.w1_scale is not None and self.w2_scale is not None
            w1_deq = _dequantize_fp8_block(
                w1, self.w1_scale,
                block_size=self.block_shape[0] if self.block_shape else 128,
            )
            w2_deq = _dequantize_fp8_block(
                w2, self.w2_scale,
                block_size=self.block_shape[0] if self.block_shape else 128,
            )
        else:
            w1_deq = w1.to(torch.bfloat16)
            w2_deq = w2.to(torch.bfloat16)

        # 3. GEMM1: gate + up
        gemm1_out = _grouped_gemm_torch(
            permuted_h.to(torch.bfloat16),
            w1_deq, expert_offsets, expert_counts, num_experts)

        # 4. LoRA W1
        if has_lora and lora_ids_perm is not None:
            _apply_lora_w1(
                gemm1_out, permuted_h.to(torch.bfloat16),
                lora_ids_perm, expert_offsets, expert_counts,
                w13_lora_a, w13_lora_b,
                num_experts)

        # 5. Activation
        act_out_dim = self.adjust_N_for_activation(N, activation)
        act_out = torch.empty(
            M_total, act_out_dim,
            dtype=torch.bfloat16, device=gemm1_out.device)
        self.activation(activation, act_out, gemm1_out)

        # 6. GEMM2: down
        gemm2_out = _grouped_gemm_torch(
            act_out, w2_deq, expert_offsets, expert_counts, num_experts)

        # 7. LoRA W2
        if has_lora and lora_ids_perm is not None:
            _apply_lora_w2(
                gemm2_out, act_out,
                lora_ids_perm, expert_offsets, expert_counts,
                w2_lora_a, w2_lora_b,
                num_experts)

        # 8. Weighted unpermute + reduce
        output.zero_()
        weighted = (weights_perm.unsqueeze(1) * gemm2_out).to(output.dtype)
        output.index_add_(0, token_indices, weighted)
