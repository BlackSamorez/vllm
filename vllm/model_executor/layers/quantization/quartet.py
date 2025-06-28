# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Supports Quartet compression, see https://arxiv.org/abs/2505.14669

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from qutlass import matmul_mxf4_bf16_tn, fusedQuantizeMx
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
        hadamard_group_size: int = 32,
        forward_dtype: str = "mxfp4",
        forward_method: str = "abs_max",
    ) -> None:
        super().__init__()
        self.hadamard_group_size = hadamard_group_size
        self.forward_dtype = forward_dtype
        self.forward_method = forward_method

    def __repr__(self) -> str:
        return (f"QuartetConfig(hadamard_group_size={self.hadamard_group_size}, "
                f"forward_dtype={self.forward_dtype}, "
                f"forward_method={self.forward_method})")

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
        hadamard_group_size = cls.get_from_keys(config, ["hadamard_group_size"])
        forward_dtype = cls.get_from_keys(config, ["forward_dtype"])
        forward_method = cls.get_from_keys(config, ["forward_method"])
        return cls(hadamard_group_size, forward_dtype, forward_method)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuartetLinearMethod"]:
        if isinstance(layer, LinearBase):
            return QuartetLinearMethod(self)
        return None


@torch.library.custom_op("quartet::fused_quantize_op", mutates_args=())
def fused_quantize_mx_op(x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, forward_method: str) -> tuple[torch.Tensor, torch.Tensor]:
    return fusedQuantizeMx(x_flat, hadamard_matrix, method=forward_method)

@fused_quantize_mx_op.register_fake
def _(x_flat, hadamard_matrix, forward_method):
    return (
        torch.empty(x_flat.shape[0], x_flat.shape[1] // 2, dtype=torch.uint8, device=x_flat.device),
        torch.empty(x_flat.shape[0], x_flat.shape[1] // 32, dtype=torch.uint8, device=x_flat.device),
    )


@torch.library.custom_op("quartet::matmul_mxf4_bf16_tn", mutates_args=())
def matmul_mxf4_bf16_tn_op(x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: float) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(x, w, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu), alpha)


@matmul_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)


def quantized_forward(x: torch.Tensor, qweight: torch.Tensor, weight_scales: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, forward_method: str) -> torch.Tensor:
    x_flat = x.contiguous().flatten(end_dim=-2)
    x_flat_q, x_flat_scales = fused_quantize_mx_op(x_flat, forward_hadamard_matrix, forward_method)

    y = matmul_mxf4_bf16_tn_op(x_flat_q, qweight, to_blocked(x_flat_scales), to_blocked(weight_scales), 1. / 9.)
    
    y = y.view(*x.shape[:-1], y.shape[-1])
    if bias is not None:
        y += bias
    
    return y


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
        if input_size_per_partition % self.quant_config.hadamard_group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size. Or other skill issues.")

        assert self.quant_config.forward_dtype == "mxfp4", "Only mxfp4 is supported for now"
        qweight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 2,
            },
        )

        assert self.quant_config.hadamard_group_size == 32, "Only hadamard_group_size of 32 is supported for now"
        scales = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 32,
            },
        )

        forward_hadamard_matrix = Parameter(
            torch.empty(self.quant_config.hadamard_group_size, self.quant_config.hadamard_group_size, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(forward_hadamard_matrix, {"ignore_warning": True})
        backward_hadamard_matrix = Parameter(
            torch.empty(self.quant_config.hadamard_group_size, self.quant_config.hadamard_group_size, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(backward_hadamard_matrix, {"ignore_warning": True})

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("forward_hadamard_matrix", forward_hadamard_matrix)
        set_weight_attrs(forward_hadamard_matrix, extra_weight_attrs)
        layer.register_parameter("backward_hadamard_matrix", backward_hadamard_matrix)
        set_weight_attrs(backward_hadamard_matrix, extra_weight_attrs)


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return quantized_forward(x, layer.qweight, layer.scales, bias, layer.forward_hadamard_matrix, self.quant_config.forward_method)
