# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# TODO: this a fake implement

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.int4_utils import dequant_quark_weight
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform


logger = init_logger(__name__)

__all__ = ["QuarkW4Int4"]


class QuarkW4Int4(QuarkScheme):

    def __init__(self, weight_quant_spec: dict[str, Any],
                 input_quant_spec: dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec # this a dict
        self.input_quant_spec = input_quant_spec

        self.inp_dtype = None
        if self.input_quant_spec is not None:
            self.inp_dtype = self.input_quant_spec.get("dtype")
        self._custom_mode = "quark"
        self.w_qscheme = self.weight_quant_spec.get("qscheme")
        self.w_dtype= self.weight_quant_spec.get("dtype")
        self.w_group_size = self.weight_quant_spec.get("group_size")
        self.w_ch_axis = self.weight_quant_spec.get("ch_axis")

        from quark.torch.utils.pack import create_pack_method
        self.pack_method = create_pack_method(qscheme=self.w_qscheme, dtype=self.w_dtype)   
        self.reorder = True
        # self.static_input_scales = not input_quant_spec.get("is_dynamic")
        self.emulate=True



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
        weight_zero_point = self.pack_method.unpack(
            layer.weight_zero_point, # [90, 640]
            self.reorder,
            **({"origin_packed_axis_size": layer.weight_scale.shape[-1]} if layer.weight_scale.shape != torch.Size([]) else {}),
        )
        weight_scale = layer.weight_scale.data.t().contiguous() # [90, 5120]
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

        output_size_per_partition = sum(output_partition_sizes) # [4096, 512, 512]
        layer.logical_widths = output_partition_sizes
        # for q_proj [4096, 2880]; k_proj [512, 2880]; v_proj [512, 2880]
        '''
        qproj linear.weight: [out=4096, inp=2880]
        quantize_dim=-1

        safetensor of quark:
        weight [2880, 4096 // 8] for pack
        scale [90, 4096] transposed
        zero [90, 512] transpsoed and pack
        
        '''
        # WEIGHT
        # transposed because pergroup
        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,       # q_proj 2880 
                output_size_per_partition // 8, # q_proj sum([4096, 512, 512]) // 8 = [512, 64, 64] # 8 * int4 is packed to 
                dtype=torch.int32, # quark : self.qspec.dtype.to_torch_packed_dtype()
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=8,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # transposed because pergroup
        weight_scale = GroupQuantScaleParameter( # 
            data=torch.empty(
                input_size_per_partition // 32, # 2880 // 32 =90
                output_size_per_partition,  # sum [4096, 512, 512] 
                dtype=params_dtype, # bf16
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_zero_point = PackedvLLMParameter( # 
            data=torch.empty(
                input_size_per_partition // 32, 
                output_size_per_partition // 8, 
                dtype=torch.int32, # bf16
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=8,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_zero_point", weight_zero_point)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert hasattr(layer, "float_weight")
        dq_w = layer.float_weight.to(x.dtype)
        return F.linear(x, dq_w, bias)

                


