# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Token alignment for fused MoE + LoRA kernel.

Sorts tokens by (expert_id, lora_id) and pads each group to block_size,
so that every block of BLOCK_SIZE_M consecutive tokens shares the same
(expert, lora) pair. This enables the fused kernel to use a single
per-block lora_id lookup.
"""

import torch


def moe_lora_align_block_size_fused(
    topk_ids: torch.Tensor,  # (num_tokens, top_k)
    token_lora_mapping: torch.Tensor,  # (num_tokens,) — lora adapter per token
    block_size: int,
    num_experts: int,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort token-expert-lora triples so that each block of ``block_size``
    consecutive entries shares the same (expert, lora) pair.

    Expert-first ordering provides better L2 reuse for the large base weight
    tiles.

    Args:
        topk_ids: (num_tokens, top_k) expert assignments per token.
        token_lora_mapping: (num_tokens,) lora adapter id per token.
            -1 means no LoRA adapter.
        block_size: BLOCK_SIZE_M for the Triton kernel.
        num_experts: total number of experts.
        max_loras: max number of LoRA adapters.

    Returns:
        sorted_token_ids: (num_tokens_post_padded,) flat sorted token ids.
            Padding entries have value == num_tokens * top_k.
        expert_ids_per_blk: (num_m_blocks,) expert id for each M-block.
        lora_ids_per_blk: (num_m_blocks,) lora id for each M-block.
            -1 means no LoRA.
        num_tokens_post_padded: (1,) scalar tensor with total padded length.
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device

    # Flat index in [0, num_tokens * top_k): encodes (token, k) pair
    flat_token_ids = (
        torch.arange(num_tokens, device=device).unsqueeze(1) * top_k
        + torch.arange(top_k, device=device).unsqueeze(0)
    ).flatten()

    flat_expert_ids = topk_ids.flatten()
    flat_lora_ids = token_lora_mapping.unsqueeze(1).expand(-1, top_k).flatten()

    # Combined sort key: (expert_id, lora_id)
    # +1 so that lora_id=-1 (no adapter) maps to 0
    sort_key = (
        flat_expert_ids.long() * (max_loras + 1) + (flat_lora_ids.long() + 1)
    )
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
    total = len(sorted_token_ids)
    while i < total:
        e = sorted_experts[i].item()
        lo = sorted_loras[i].item()
        j = i
        while (
            j < total
            and sorted_experts[j].item() == e
            and sorted_loras[j].item() == lo
        ):
            j += 1
        group_len = j - i
        padded_len = ((group_len + block_size - 1) // block_size) * block_size
        padded_tokens.extend(sorted_token_ids[i:j].tolist())
        padded_tokens.extend([num_valid] * (padded_len - group_len))
        n_blocks = padded_len // block_size
        blk_experts.extend([e] * n_blocks)
        blk_loras.extend([lo] * n_blocks)
        i = j

    sorted_token_ids_t = torch.tensor(
        padded_tokens, device=device, dtype=torch.int64
    )
    expert_ids_t = torch.tensor(blk_experts, device=device, dtype=torch.int64)
    lora_ids_t = torch.tensor(blk_loras, device=device, dtype=torch.int64)
    num_post = torch.tensor(
        [len(padded_tokens)], device=device, dtype=torch.int64
    )
    return sorted_token_ids_t, expert_ids_t, lora_ids_t, num_post
