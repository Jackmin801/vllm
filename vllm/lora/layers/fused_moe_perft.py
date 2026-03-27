# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.layers.fused_moe import FusedMoE

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
        self.max_loras = lora_config.max_loras

        self.adapter_enabled = torch.tensor(
            [0] * (max_loras + 1), dtype=torch.int, device=self.device
        )

        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)

        # TODO I dont think this section is needed but something will explode
        if False:
            # They will be used by 'LoRALayerWeights.create_dummy_lora_weights'
            # to create a dummy LoRA weights.
            self.lora_a_stacked = []
            self.lora_b_stacked = []
            for lora_id in range(max_loras):
                for experts_id in range(self.base_layer.local_num_experts):
                    # For gated MoE: gate_proj (w1), down_proj (w2), up_proj (w3)
                    # For non-gated MoE: up_proj (w1), down_proj (w2)
                    self.lora_a_stacked.append(
                        self.w13_lora_a_stacked[0][lora_id][experts_id]
                    )
                    self.lora_a_stacked.append(
                        self.w2_lora_a_stacked[0][lora_id][experts_id]
                    )

                    self.lora_b_stacked.append(
                        self.w13_lora_b_stacked[0][lora_id][experts_id]
                    )
                    self.lora_b_stacked.append(
                        self.w2_lora_b_stacked[0][lora_id][experts_id]
                    )

                    # Only add w3 (up_proj) for gated MoE (_w13_slices == 2)
                    if self._w13_slices == 2:
                        self.lora_a_stacked.append(
                            self.w13_lora_a_stacked[1][lora_id][experts_id]
                        )
                        self.lora_b_stacked.append(
                            self.w13_lora_b_stacked[1][lora_id][experts_id]
                        )

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        self.lora_a[index] = 0
        self.lora_b[index] = 0
        self.adapter_enabled[index] = 0


    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ):
        """Overwrites lora tensors at index."""
        assert len(lora_a.shape) == len(lora_b.shape) == 3, "Lora should have shape (local_num_experts, r/H, H/r)"
        self.adapter_enabled[index] = 1
        self.lora_a[index] = lora_a
        self.lora_b[index] = lora_b


    def set_mapping(self, punica_wrapper):
        self.punica_wrapper = punica_wrapper
        moe_mk = self.base_layer.quant_method.moe_kernel.impl
        if moe_mk is not None:
            moe_mk.punica_wrapper = punica_wrapper
        else:
            raise ValueError("MoE LoRA requires a modular kernel.")
        self._attach_lora_weights()
    
    def _attach_lora_weights(self):
        """Store LoRA weight references on the modular kernel impl so they
        get passed through to experts.apply() at forward time."""
        moe_mk = self.base_layer.quant_method.moe_kernel.impl
        if not moe_mk.fused_experts.supports_lora():
            raise ValueError(
                f"{type(moe_mk.fused_experts).__name__} does not support "
                "LoRA. Set VLLM_MOE_USE_TORCH_NAIVE=1 for LoRA support."
            )
        moe_mk.lora_weights = {
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
        }

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
        return isinstance(source_layer, FusedMoE)

    #===============================================
    # Passthrough methods
    #===============================================
    def forward(self, *args, **kwargs):
        return self.base_layer.forward(*args, **kwargs)

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
