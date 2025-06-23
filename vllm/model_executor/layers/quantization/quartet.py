# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Supports Quartet compression, see https://arxiv.org/abs/2505.14669

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from qutlass import matmul_mxf4_bf16_tn, fusedQuantize
from qutlass.utils import to_blocked
import quartet

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class QuartetConfig(QuantizationConfig):
    """Config class for Quartet."""

    def __init__(
        self,
        group_size: int = 32,
        do_hadamard: bool = True,
        exponent_type: str = "e8m0",
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.do_hadamard = do_hadamard
        self.exponent_type = exponent_type

    def __repr__(self) -> str:
        return (f"QuartetConfig(group_size={self.group_size}, "
                f"do_hadamard={self.do_hadamard}, "
                f"exponent_type={self.exponent_type})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "quartet"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuartetConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        do_hadamard = cls.get_from_keys(config, ["do_hadamard"])
        exponent_type = cls.get_from_keys(config, ["exponent_type"])
        return cls(group_size, do_hadamard, exponent_type)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuartetLinearMethod"]:
        if isinstance(layer, LinearBase):
            return QuartetLinearMethod(self)
        return None


class QuartetLinearMethod(LinearMethodBase):
    """Linear method for Quartet.

    Args:
        quant_config: The Quartet quantization config.
    """

    def __init__(self, quant_config: QuartetConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del output_size  # Unused.
        del input_size  # Unused.

        if params_dtype != torch.bfloat16:
            raise ValueError("Only bfloat16 is currently supported by Quartet")
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size. Or other skill issues.")

        weight_q = Parameter(
            torch.empty(
                # There could actually be two pack factors, one along input and
                # one along output, but we don't currently support
                # out_group_size, and only the one along output needs to be
                # marked with "packed_dim" in order for QKVLinear to work.
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_q,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 2,
            },
        )

        assert self.quant_config.exponent_type == "e8m0", "Only e8m0 is supported for now"
        assert self.quant_config.group_size == 32, "Only group size of 32 is supported for now"
        shared_exponents = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e8m0fnu,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            shared_exponents,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.group_size,
            },
        )
        
        assert self.quant_config.do_hadamard, "Hadamard transform is required for now"
        forward_hadamard_matrix = Parameter(
            torch.empty(self.quant_config.group_size, self.quant_config.group_size, dtype=params_dtype),
            requires_grad=False,
        )

        layer.register_parameter("weight_q", weight_q)
        set_weight_attrs(weight_q, extra_weight_attrs)
        layer.register_parameter("shared_exponents", shared_exponents)
        set_weight_attrs(shared_exponents, extra_weight_attrs)
        layer.register_parameter("forward_hadamard_matrix", forward_hadamard_matrix)
        set_weight_attrs(forward_hadamard_matrix, extra_weight_attrs)
        
        
    def forward_quantize(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        match (self.quant_config.exponent_type, self.quant_config.group_size):
            case ("e8m0", 32):
                return fusedQuantize(x, self.forward_hadamard_matrix)
            case _:
                raise ValueError(f"Unsupported forward dtype: {self.quant_config.exponent_type} and group size: {self.quant_config.group_size}")


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_q = layer.weight_q
        weight_shared_exponents = layer.shared_exponents

        # Quartet forward quantization
        x_flat = x.contiguous().flatten(end_dim=-2)
        x_flat_q, x_flat_shared_exponents, _ = self.forward_quantize(x_flat)

        y = matmul_mxf4_bf16_tn(x_flat_q, weight_q, to_blocked(x_flat_shared_exponents), to_blocked(weight_shared_exponents), 1.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        return y
