# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.utils.int4_utils import dequant_quark_weight
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE, quant_dequant_mxfp4fp6fp8)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["QuarkW6MXFP6"]


class QuarkW6MXFP6(QuarkScheme):

    def __init__(self, weight_quant_spec: dict[str, Any],
                 input_quant_spec: dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.inp_dtype = None
        if self.input_quant_spec is not None:
            self.inp_dtype = self.input_quant_spec.get("dtype")
        self._custom_mode = "quark"
        self.w_qscheme = self.weight_quant_spec.get("qscheme")
        self.w_dtype= self.weight_quant_spec.get("dtype")
        self.w_group_size = self.weight_quant_spec.get("group_size")
        self.w_ch_axis = self.weight_quant_spec.get("ch_axis")
        self.reorder = True
        from quark.torch.utils.pack import create_pack_method
        self.pack_method = create_pack_method(qscheme=self.w_qscheme, dtype=self.w_dtype)   



    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                requires_grad=False)

        weight = self.pack_method.unpack(
            layer.weight, # [2880, 640]
            self.reorder,
            **({"origin_packed_axis_size": layer.weight_scale.shape[-1]} if layer.weight_scale.shape != torch.Size([]) else {}),
            )
        weight_zero_point = None
        # weight_scale = layer.weight_scale.data.t().contiguous() # only u-int8, u-int4, int2 are transposed when per_group
        float_dtype = torch.float32 # mx scale_format is e8m0, we should convert it to float32, mxf4 process this by hip kernel
        weight_scale = 2 ** (layer.weight_scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)
        dq_w = dequant_quark_weight(
            self.w_dtype,
            weight, # qkv[5120,2880]
            weight_scale, # [5120,90]
            weight_zero_point, # [5120, 90]
            self.w_ch_axis, # 1
            self.w_group_size, # 32
            self.w_qscheme,  #str
        )
        float_weight = Parameter(dq_w, requires_grad=False)
        layer.register_parameter("float_weight", float_weight)

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition, # 4096
                input_size_per_partition // 4 * 3, # 2880 / 4 * 3 = 2160
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition, # 4096
                input_size_per_partition // OCP_MX_BLOCK_SIZE, # 2880 // 32 = 90
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert hasattr(layer, "float_weight")
        x = quant_dequant_mxfp4fp6fp8(x, self.inp_dtype)
        dq_w = layer.float_weight.to(x.dtype)
        return F.linear(x, dq_w, bias)
