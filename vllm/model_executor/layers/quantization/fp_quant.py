# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Supports FP-Quant compression, see https://arxiv.org/abs/2505.14669

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from qutlass import matmul_mxf4_bf16_tn, matmul_ada_mxf4_bf16_tn, fusedQuantizeMx, matmul_nvf4_bf16_tn, fusedQuantizeNv
from qutlass.utils import to_blocked

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


class FPQuantConfig(QuantizationConfig):
    """Config class for FPQuant."""

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
        return (f"FPQuantConfig(hadamard_group_size={self.hadamard_group_size}, "
                f"forward_dtype={self.forward_dtype}, "
                f"forward_method={self.forward_method})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp_quant"

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
    def from_config(cls, config: dict[str, Any]) -> "FPQuantConfig":
        hadamard_group_size = cls.get_from_keys(config, ["hadamard_group_size"])
        forward_dtype = cls.get_from_keys(config, ["forward_dtype"])
        forward_method = cls.get_from_keys(config, ["forward_method"])
        return cls(hadamard_group_size, forward_dtype, forward_method)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["FPQuantLinearMethod"]:
        if isinstance(layer, LinearBase):
            return FPQuantLinearMethod(self)
        return None


class FPQuantLinearMethod(LinearMethodBase):
    """Linear method for FPQuant.

    Args:
        quant_config: The FPQuant quantization config.
    """

    def __init__(self, quant_config: FPQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del output_size  # Unused.
        del input_size  # Unused.

        if params_dtype != torch.bfloat16:
            raise ValueError("Only bfloat16 is currently supported by FPQuant")
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

        if self.quant_config.forward_dtype == "mxfp4":
            group_size = 32
        elif self.quant_config.forward_dtype == "nvfp4":
            group_size = 16
        else:
            raise ValueError(f"Unsupported forward_dtype: {self.quant_config.forward_dtype}")
        
        assert self.quant_config.hadamard_group_size == group_size, "Only hadamard_group_size of 32 is supported for now"
        scales = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // group_size,
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

        weight_global_scale = Parameter(
            torch.empty(1, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight_global_scale, {"ignore_warning": True})
        act_global_scale = Parameter(
            torch.empty(1, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(act_global_scale, {"ignore_warning": True})
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
        return quantized_forward(x, layer.qweight, layer.scales, layer.weight_global_scale, layer.act_global_scale, bias, layer.forward_hadamard_matrix, self.quant_config.forward_method)


def fused_quantize_mx(x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, forward_method: str) -> tuple[torch.Tensor, torch.Tensor]:
    return fusedQuantizeMx(x_flat, hadamard_matrix, method=forward_method)


def ceil_div(a, b):
    return (a + b - 1) // b


def fused_quantize_mx_fake(x_flat, hadamard_matrix, forward_method):
    rows, cols = x_flat.size(0), x_flat.size(1)//32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4
    
    xh_e2m1 = torch.empty(x_flat.size(0), x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device)
    xh_e8m0 = torch.empty(padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=x_flat.device)

    return xh_e2m1, xh_e8m0
    

direct_register_custom_op(
    op_name="fused_quantize_mx",
    op_func=fused_quantize_mx,
    mutates_args=[],
    fake_impl=fused_quantize_mx_fake,
    dispatch_key=current_platform.dispatch_key,
)


def matmul_mxf4_bf16(x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(x, w, to_blocked(xs).view(torch.float8_e8m0fnu), to_blocked(ws).view(torch.float8_e8m0fnu), alpha)


def matmul_mxf4_bf16_fake(x, w, xs, ws, alpha):
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name="matmul_mxf4_bf16",
    op_func=matmul_mxf4_bf16,
    mutates_args=[],
    fake_impl=matmul_mxf4_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


def matmul_ada_mxf4_bf16(x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return matmul_ada_mxf4_bf16_tn(x, w, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu), alpha)


def matmul_ada_mxf4_bf16_fake(x, w, xs, ws, alpha):
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name="matmul_ada_mxf4_bf16",
    op_func=matmul_ada_mxf4_bf16,
    mutates_args=[],
    fake_impl=matmul_ada_mxf4_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


def fused_quantize_nv(x_flat: torch.Tensor, hadamard_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return fusedQuantizeNv(x_flat, hadamard_matrix)


def fused_quantize_nv_fake(x_flat, hadamard_matrix):
    rows, cols = x_flat.size(0), x_flat.size(1)//16
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4
    
    xh_e2m1 = torch.empty(x_flat.size(0), x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device)
    xh_e8m0 = torch.empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=x_flat.device)

    return xh_e2m1, xh_e8m0


direct_register_custom_op(
    op_name="fused_quantize_nv",
    op_func=fused_quantize_nv,
    mutates_args=[],
    fake_impl=fused_quantize_nv_fake,
    dispatch_key=current_platform.dispatch_key,
)


def matmul_nvf4_bf16(x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return matmul_nvf4_bf16_tn(x, w, to_blocked(xs).view(torch.float8_e4m3fn), to_blocked(ws).view(torch.float8_e4m3fn), alpha)


def matmul_nvf4_bf16_fake(x, w, xs, ws, alpha):
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)


def quantized_forward(x: torch.Tensor, qweight: torch.Tensor, weight_scales: torch.Tensor, weight_global_scale: torch.Tensor, act_global_scale: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, forward_method: str, forward_dtype: str) -> torch.Tensor:
    x_flat = x.contiguous().flatten(end_dim=-2)
    if forward_dtype == "mxfp4":
        x_flat_q, x_flat_scales = torch.ops.vllm.fused_quantize_mx(x_flat, forward_hadamard_matrix, forward_method)
        if x_flat.shape[-1] <= 64:
            y = torch.ops.vllm.matmul_ada_mxf4_bf16(x_flat_q, qweight, x_flat_scales, weight_scales, weight_global_scale * act_global_scale)
        else:
            y = torch.ops.vllm.matmul_mxf4_bf16(x_flat_q, qweight, x_flat_scales, weight_scales, weight_global_scale * act_global_scale)
    elif forward_dtype == "nvfp4":
        x_flat_q, x_flat_scales = torch.ops.vllm.fused_quantize_nv(x_flat, forward_hadamard_matrix)
        y = torch.ops.vllm.matmul_nvf4_bf16(x_flat_q, qweight, x_flat_scales, weight_scales, weight_global_scale * act_global_scale)
    else:
        raise ValueError(f"Unsupported forward_dtype: {forward_dtype}")
    
    y = y.view(*x.shape[:-1], y.shape[-1])
    if bias is not None:
        y += bias
    
    return y
