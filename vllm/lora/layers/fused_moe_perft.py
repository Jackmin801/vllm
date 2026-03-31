# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import _get_lora_device


class FusedMoEWithPERFTE(BaseLayerWithLoRA):
    def __init__(self, base_layer: FusedMoE) -> None:
        super().__init__()
        self.base_layer = base_layer

        assert self.base_layer.tp_size == 1, "FusedMoEWithPERFTE does not support tensor parallelism"
        self.device = _get_lora_device(base_layer)

    def _create_lora_a_weights(self, max_loras: int, lora_config: LoRAConfig):
        self.lora_a = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def _create_lora_b_weights(self, max_loras: int, lora_config: LoRAConfig):
        self.lora_b = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                self.base_layer.hidden_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        self.max_loras = max_loras

        self.adapter_enabled = torch.zeros(
            max_loras, dtype=torch.bool, device=self.device
        )

        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        self.lora_a[index] = 0
        self.lora_b[index] = 0
        self.adapter_enabled[index] = False

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ):
        """Overwrites lora tensors at index."""
        assert len(lora_a.shape) == len(lora_b.shape) == 3, \
            "Lora should have shape (local_num_experts, r/H, H/r)"
        self.adapter_enabled[index] = True
        # Pad rank dimension if adapter rank < max_lora_rank
        rank_a = lora_a.shape[1]
        self.lora_a[index] = 0
        self.lora_b[index] = 0
        self.lora_a[index, :, :rank_a, :] = lora_a
        self.lora_b[index, :, :, :rank_a] = lora_b

    def set_mapping(self, punica_wrapper):
        self.punica_wrapper = punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""

        # source_layer is FusedMoE or SharedFusedMoE
        return isinstance(source_layer, FusedMoE) and envs.VLLM_MOE_LORA_USE_PERFTE

    #===============================================
    # Forward with parallel LoRA path
    #===============================================
    def forward(self, hidden_states, router_logits):
        shared_output, fused_output = self.base_layer.forward(
            hidden_states, router_logits)

        # Compute routing using the base layer's router
        topk_weights, topk_ids = self.base_layer.runner.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # Apply LoRA to the fused expert output.
        # Registered as a custom op so data-dependent branches are allowed
        # within torch.compile fullgraph mode.
        # token_lora_indices may be larger than the current batch;
        # slice to match the number of tokens in hidden_states.
        num_tokens = hidden_states.shape[0]
        token_lora_indices = self.punica_wrapper.token_lora_indices[:num_tokens]

        apply_perfte_lora(
            hidden_states, topk_ids, topk_weights, fused_output,
            self.lora_a, self.lora_b, self.adapter_enabled,
            token_lora_indices,
        )
        return shared_output, fused_output


def _apply_perfte_lora(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    adapter_enabled: torch.Tensor,
    token_lora_indices: torch.Tensor,
) -> None:
    """Apply LoRA as a parallel path respecting MoE routing and adapter IDs.

    Fully branchless — no data-dependent control flow — so it is safe under
    both torch.compile(fullgraph=True) and CUDA-graph capture.  Inactive
    tokens/adapters produce zero contribution via mask multiplication.

    For each adapter slot (loop count is a compile-time constant):
      1. Build a per-token mask from token_lora_indices & adapter_enabled
      2. Project ALL tokens through lora_a for all experts (einsum)
      3. Select the routed experts per token (gather)
      4. Project through lora_b per top-k slot (bmm)
      5. Mask and accumulate into output
    """
    top_k = topk_ids.shape[1]
    rank = lora_a.shape[2]
    num_adapters = lora_a.shape[0]
    x = hidden_states.to(lora_a.dtype)

    for adapter_idx in range(num_adapters):
        # Per-token mask: 1.0 for tokens using this adapter, 0.0 otherwise.
        # adapter_enabled[adapter_idx] is a 0-dim GPU tensor (no CPU sync).
        mask = (
            (token_lora_indices == adapter_idx) & adapter_enabled[adapter_idx]
        ).unsqueeze(1).to(lora_a.dtype)  # [N, 1]

        # Step 1: x @ lora_a for all experts via einsum
        # [N, H] x [E, r, H] -> [N, E, r]
        intermediate = torch.einsum('nh,erh->ner', x, lora_a[adapter_idx])

        # Step 2: select only routed experts
        # [N, E, r] -> [N, top_k, r]
        intermediate_selected = torch.gather(
            intermediate, 1,
            topk_ids.unsqueeze(-1).expand(-1, -1, rank),
        )

        # Step 3: project through lora_b per top-k slot and accumulate
        lora_out = torch.zeros_like(x)  # [N, H]
        for k in range(top_k):
            inter_k = intermediate_selected[:, k]       # [N, r]
            expert_ids_k = topk_ids[:, k]                # [N]
            weight_k = topk_weights[:, k:k + 1]          # [N, 1]

            # Gather lora_b for each token's expert
            lora_b_k = lora_b[adapter_idx][expert_ids_k]  # [N, H, r]

            # [N, 1, r] @ [N, r, H] -> [N, H]
            out_k = torch.bmm(
                inter_k.unsqueeze(1),
                lora_b_k.transpose(1, 2),
            ).squeeze(1)

            lora_out += out_k * weight_k

        output += (lora_out * mask).to(output.dtype)


def _apply_perfte_lora_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    adapter_enabled: torch.Tensor,
    token_lora_indices: torch.Tensor,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="apply_perfte_lora",
        op_func=_apply_perfte_lora,
        mutates_args=["output"],
        fake_impl=_apply_perfte_lora_fake,
    )
    apply_perfte_lora = torch.ops.vllm.apply_perfte_lora
except AttributeError:
    apply_perfte_lora = _apply_perfte_lora

    #===============================================
    # Passthrough methods
    #===============================================
    def maybe_all_reduce_tensor_model_parallel(self, *args, **kwargs):
        return self.base_layer.maybe_all_reduce_tensor_model_parallel(*args, **kwargs)

    @property
    def _shared_experts(self):
        return self.base_layer._shared_experts

    @property
    def quant_method(self):
        return self.base_layer.quant_method

    @property
    def is_internal_router(self) -> bool:
        return self.base_layer.is_internal_router
