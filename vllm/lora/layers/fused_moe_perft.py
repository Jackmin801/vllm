# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE
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
                max_loras + 1,
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
                max_loras + 1,
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
        assert lora_config.max_lora_rank % 8 == 0, (
            f"PERFT-E requires max_lora_rank to be a multiple of 8 for "
            f"grouped_mm alignment, got {lora_config.max_lora_rank}"
        )
        self.max_loras = max_loras

        self.adapter_enabled = torch.zeros(
            max_loras + 1, dtype=torch.bool, device=self.device
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
        index += 1
        self.adapter_enabled[index] = True
        # Pad rank dimension if adapter rank < max_lora_rank
        rank_a = lora_a.shape[1]
        self.lora_a[index] = 0
        self.lora_b[index] = 0
        self.lora_a[index, :, :rank_a, :] = lora_a
        self.lora_b[index, :, :, :rank_a] = lora_b

    def set_mapping(self, punica_wrapper):
        self.punica_wrapper = punica_wrapper

        # TODO: Idk why but the runner sees the base layer instead of this one
        self.base_layer.punica_wrapper = punica_wrapper
        self.base_layer.lora_a = self.lora_a
        self.base_layer.lora_b = self.lora_b

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
        is_moe = isinstance(source_layer, FusedMoE) or isinstance(source_layer, SharedFusedMoE)
        return is_moe and envs.VLLM_MOE_LORA_USE_PERFTE

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

    #===============================================
    # Forward with parallel LoRA path
    #===============================================
    def forward(self, *args, **kwargs):
        return self.base_layer.forward(*args, **kwargs)
