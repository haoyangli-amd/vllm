# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from torch.nn.parameter import Parameter, UninitializedParameter
logger = init_logger(__name__)

__all__ = [
    "QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod", "QuarkW4MXFp4MoEMethod"
]


class QuarkMoEMethod(FusedMoEMethodBase):
    def __init__(self):
        super().__init__()
        self.weight_dtype: str = None
        self.static_input_scales = None
        self.inp_dtype = None
        self._custom_mode = None
        self.w_qscheme = None
        self.w_group_size = None
        self.w_ch_axis = None
    @staticmethod
    def get_moe_method(
            quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
            module: torch.nn.Module,
            layer_name: str) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(
            layer_name, module)

        if (layer_quant_config.get("output_tensors")
                or layer_quant_config.get("bias")):
            raise NotImplementedError("Currently, Quark models with "
                                      "output_tensors and bias "
                                      "quantized are not supported")
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config)
        elif quant_config._is_mx_fp4(weight_config, input_config):
            return QuarkW4MXFp4MoEMethod(weight_config, input_config)
        elif quant_config._is_int4(weight_config, input_config):
            return QuarkW4Int4MoEMethod(weight_config, input_config)
        elif quant_config._is_mx_fp6(weight_config, input_config):
            return QuarkW6MXFp6MoEMethod(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.weight_dtype == "fp6_e2m3" or self.weight_dtype == "int4":
        # if self.weight_dtype == "int4":
            from quark.torch.utils.pack import create_pack_method
            pack_method = create_pack_method(qscheme="per_group", dtype=self.weight_dtype)   
            # w13 [32, 2880, 720]-> *8 = [32, 2880, 4096+512+512]

            w1=layer.w13_weight
            w2=layer.w2_weight
            E, N, _ = w1.size()
            w1_scale=layer.w13_weight_scale
            w2_scale=layer.w2_weight_scale
            if self.weight_dtype == "int4":
                w1_zero_point=layer.w13_weight_zero_point
                w2_zero_point=layer.w2_weight_zero_point

            dq_w1_list=[]
            dq_w2_list=[]
            for i in range(E): # quark only support 2dim
                _w1 = w1[i, ...] # [2880, 720]
                _w2 = w2[i, ...] # [2880, 360]
                _w1_scale = w1_scale[i, ...] # [90, 5760]
                _w2_scale = w2_scale[i, ...] # [90, 2880]

                _w1 = pack_method.unpack(
                    _w1, 
                    True, # self.reorder=True
                    **({"origin_packed_axis_size": _w1_scale.shape[-1]} if _w1_scale[i, ...].shape != torch.Size([]) else {}),
                )

                _w2 = pack_method.unpack(
                    _w2,
                    True, # self.reorder=True
                    **({"origin_packed_axis_size": _w2_scale.shape[-1]} if _w2_scale[i, ...].shape != torch.Size([]) else {}),
                )

                if self.weight_dtype == "int4":
                    _w1_zero_point = w1_zero_point[i, ...] # [90, 720]
                    _w2_zero_point = w2_zero_point[i, ...] # [90,360]
                    _w1_zero_point = pack_method.unpack(
                        _w1_zero_point, # [90, 640]
                        True, # self.reorder,
                        **({"origin_packed_axis_size": _w1_scale.shape[-1]} if _w1_scale[i, ...].shape != torch.Size([]) else {}),
                    )
                    _w2_zero_point = pack_method.unpack(
                        _w2_zero_point, 
                        True, # self.reorder,
                        **({"origin_packed_axis_size": _w2_scale.shape[-1]} if _w2_scale[i, ...].shape != torch.Size([]) else {}),
                    )
                    _w1_scale = _w1_scale.data.t().contiguous() # [90, 5120] # only u-int8, u-int4, int2 are transposed when per_group
                    _w2_scale = _w2_scale.data.t().contiguous() # [90, 5120] # only u-int8, u-int4, int2 are transposed when per_group

                else: # mxfp6
                    _w1_zero_point = None
                    _w2_zero_point = None

                    float_dtype = torch.float32 # mx scale_format is e8m0, we should convert it to float32, mxf4 process this by hip kernel
                    _w1_scale = 2 ** (_w1_scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)
                    _w2_scale = 2 ** (_w2_scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

                import quark
                dq_w1 = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                        self.weight_dtype, # self.weight_dtype,
                        _w1, # weight, # qkv[5120,2880]
                        _w1_scale, # weight_scale, # [5120,90]
                        _w1_zero_point, # weight_zero_point, # [5120, 90]
                        self.w_ch_axis, # 1
                        self.w_group_size, # 32
                        self.w_qscheme,  #str
                        )
                dq_w1_list.append(dq_w1)
                dq_w2 = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                        self.weight_dtype, # self.weight_dtype,
                        _w2, # weight, # qkv[5120,2880]
                        _w2_scale, # weight_scale, # [5120,90]
                        _w2_zero_point, # weight_zero_point, # [5120, 90]
                        self.w_ch_axis, # 1
                        self.w_group_size, # 32
                        self.w_qscheme,  #str
                        )
                dq_w2_list.append(dq_w2)
            w1 = torch.stack(dq_w1_list, dim=0).to(torch.bfloat16) # [32, 5760, 2880]
            w2 = torch.stack(dq_w2_list, dim=0).to(torch.bfloat16) # [32, 2880, 2880]
            

            delattr(layer, "w13_weight")
            delattr(layer, "w2_weight")
            delattr(layer, "w13_weight_scale")
            delattr(layer, "w2_weight_scale")

            tensor1 = torch.tensor([1], device="cuda")
            
            float_weight13 = Parameter(w1, requires_grad=False)
            float_weight2 = Parameter(w2, requires_grad=False)
            layer.register_parameter("float_weight13", float_weight13)
            layer.register_parameter("float_weight2", float_weight2)
            layer.register_parameter("w13_weight", Parameter(tensor1, requires_grad=False))
            layer.register_parameter("w2_weight", Parameter(tensor1, requires_grad=False))
            layer.register_parameter("w13_weight_scale", Parameter(tensor1, requires_grad=False))
            layer.register_parameter("w2_weight_scale", Parameter(tensor1, requires_grad=False))

            # decrease memory occupation


    def unpack_fp6(self, tensor):
        self.e_bits = 2
        self.m_bits=3
        input_shape = list(tensor.shape)
        tensor = tensor.reshape(*tensor.shape[:-1], -1, 3)

        # The packed `tensor` on 3 bytes (byte2, byte1, byte0):
        #
        # |_____'____________|_________'_________||___________'____|
        #   v1      v0           v2        v1         v3        v2
        #   2b      6b           4b        4b         6b        2b

        byte2 = tensor[..., 0]
        byte1 = tensor[..., 1]
        byte0 = tensor[..., 2]

        val0 = byte2 & 0b00111111

        val1_2b_low = byte2 >> 6
        val1_4b_high = byte1 & 0b00001111

        val2_4b_low = byte1 >> 4
        val2_2b_high = byte0 & 0b00000011

        val3 = byte0 >> 2

        val1 = (val1_4b_high << 2) + val1_2b_low
        val2 = (val2_2b_high << 4) + val2_4b_low

        unpacked = torch.stack((val0, val1, val2, val3), dim=-1)

        fp6_ebias = (1 << (self.e_bits - 1)) - 1
        fp6_mbias = 23 - self.m_bits
        unpacked = unpacked.to(torch.int32)
        unpacked = ((unpacked & 0x1F) << fp6_mbias) | ((unpacked & 0x20) << 26)
        unpacked = unpacked.view(torch.float32) * (2.0 ** (127 - fp6_ebias))

        # Apply scale and reshape
        input_shape[-1] = input_shape[-1] * 4 // 3
        return unpacked.reshape(input_shape)

    def fused_experts_impl_quark(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_mxfp4_w4a4: bool = False,
        per_channel_quant: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[list[int]] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        quark_act_dtype: Optional[str] = None,
        w1_zero_point: Optional[torch.Tensor] = None,
        w2_zero_point: Optional[torch.Tensor] = None,
        layer_for_quark: Optional[torch.nn.Module] = None,
        weight_dtype: str = "fp6_e2m3", # int4; fp6_e2m3; fp4
        ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
        from quark.torch.utils.pack import create_pack_method
        import quark

        if weight_dtype == "fp4": # oss qwqa # mxfp4
            E, N, _ = w1.size()
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            if inplace:
                out_hidden_states = hidden_states
            else:
                out_hidden_states = torch.empty_like(hidden_states)
            # hidden_states # [16384, 2880]
            # topk_weights # [16384, 4]
            # topk_ids # [16384, 4]
            # w1 # [32, 5760, 2880] # [32, out, inp]
            # w2 # [32, 2880, 2880] # [32, out, inp]
            qtype = quark_act_dtype
            qout_hidden_states, a1q_scale = moe_kernel_quantize_input(
                    A=out_hidden_states,
                    A_scale=None,
                    quant_dtype=qtype,
                    per_act_token_quant=False,
                    block_shape=None)

            expanded = qout_hidden_states.repeat(E, 1) # [32， 16384, 2880]
            expanded = expanded.view(E, -1, 2880) # [32， 16384, 2880]
            # [32, 16384, 2880] * [32, 5760, 2880] + [5760] = [32, 16384, 5760]
            w1 = w1.transpose(-2, -1)  # [32, 5760, 2880]-> # [32, 2880, 5760] # 转置前是按照linear weight的格式[32, out,inp]
            w2 = w2.transpose(-2, -1)# [32, 2880, 2880]-> # [32, 2880, 2880]# 转置前是按照linear weight的格式[32, out,inp],之后是符合bmm格式[m,n]*[n,k]
            gate_up = torch.bmm(expanded, w1) + w1_bias[..., None, :] 
            gate = gate_up[:,:,:2880]
            up = gate_up[:,:,2880:]
            gate = gate.clamp(min=None, max=7.0) # [32, 16384, 2880]
            up = up.clamp(min=-7.0, max=7.0) # [32, 16384, 2880]
            glu = gate * torch.sigmoid(gate * 1.702) # [32, 16384, 2880]
            act_after_glu = (up + 1) * glu
            qact_after_glu, a2q_scale = moe_kernel_quantize_input(
                    A=act_after_glu,
                    A_scale=None,
                    quant_dtype=qtype,
                    per_act_token_quant=False,
                    block_shape=None)
            next_states = torch.bmm(qact_after_glu, w2) # [32, 16384, 2880] * [32, 2880, 2880] # need transpose？
            next_states = next_states + w2_bias[..., None, :] # [32, 16384, 2880] + [32, 1 ,2880]
            full_weights = torch.zeros(topk_weights.shape[0], E, device=topk_weights.device)
            full_weights.scatter_(1, topk_ids.long(), topk_weights) # # [16384, 32]
            # [32, 16384, 2880] * 
            # [16384, 32]->[32, 16384]->[32, 16484, 1]
            next_states = next_states * full_weights.transpose(0, 1).unsqueeze(-1)
            final = next_states.sum(dim=0).to(hidden_states.dtype) # [32, 16384, 2880]
            out_hidden_states.copy_(final)
            return out_hidden_states
        elif weight_dtype == "fp6_e2m3" or weight_dtype == "int4": # int4
            # if weight_dtype == "fp6_e2m3":
            if 0:
                w1 = self.unpack_fp6(w1)
                w2 = self.unpack_fp6(w2)
                float_dtype = torch.float32
                _w1_scale = 2 ** (w1_scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)
                _w2_scale = 2 ** (w2_scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)
                w1 = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                    self.weight_dtype,
                    w1,
                    _w1_scale,
                    None, # zero_point,
                    self.w_ch_axis,
                    self.w_group_size,
                    self.w_qscheme,  # type: ignore[union-attr]
                ).to(hidden_states.dtype)
                w2 = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                    self.weight_dtype,
                    w2,
                    _w2_scale,
                    None, # zero_point,
                    self.w_ch_axis,
                    self.w_group_size,
                    self.w_qscheme,  # type: ignore[union-attr]
                ).to(hidden_states.dtype)


            else: # int4 
                assert layer_for_quark is not None
                assert hasattr(layer_for_quark, "float_weight13") 
                assert hasattr(layer_for_quark, "float_weight2")
                w1 = layer_for_quark.float_weight13.to(hidden_states.dtype)
                w2 = layer_for_quark.float_weight2.to(hidden_states.dtype)
            E, N, _ = w1.size()

            if inplace:
                out_hidden_states = hidden_states
            else:
                out_hidden_states = torch.empty_like(hidden_states)
            # hidden_states # [16384, 2880]
            # topk_weights # [16384, 4]
            # topk_ids # [16384, 4]
            # w1 # [32, 5760, 2880] # [32, out, inp]
            # w2 # [32, 2880, 2880] # [32, out, inp]
            qtype = quark_act_dtype
            qout_hidden_states, a1q_scale = moe_kernel_quantize_input(
                    A=out_hidden_states,
                    A_scale=None,
                    quant_dtype=qtype,
                    per_act_token_quant=False,
                    block_shape=None)

            expanded = qout_hidden_states.repeat(E, 1) # [32， 16384, 2880]
            expanded = expanded.view(E, -1, 2880) # [32， 16384, 2880]
            # [32, 16384, 2880] * [32, 5760, 2880] + [5760] = [32, 16384, 5760]
            w1 = w1.transpose(-2, -1)  # [32, 5760, 2880]-> # [32, 2880, 5760] # 转置前是按照linear weight的格式[32, out,inp]
            w2 = w2.transpose(-2, -1)# [32, 2880, 2880]-> # [32, 2880, 2880]# 转置前是按照linear weight的格式[32, out,inp],之后是符合bmm格式[m,n]*[n,k]
            gate_up = torch.bmm(expanded, w1) + w1_bias[..., None, :] 
            gate = gate_up[:,:,:2880]
            up = gate_up[:,:,2880:]
            gate = gate.clamp(min=None, max=7.0) # [32, 16384, 2880]
            up = up.clamp(min=-7.0, max=7.0) # [32, 16384, 2880]
            glu = gate * torch.sigmoid(gate * 1.702) # [32, 16384, 2880]
            act_after_glu = (up + 1) * glu
            qact_after_glu, a2q_scale = moe_kernel_quantize_input(
                            A=act_after_glu,
                            A_scale=None,
                            quant_dtype=qtype,
                            per_act_token_quant=False,
                            block_shape=None)
            next_states = torch.bmm(qact_after_glu, w2) # [32, 16384, 2880] * [32, 2880, 2880] # need transpose？
            next_states = next_states + w2_bias[..., None, :] # [32, 16384, 2880] + [32, 1 ,2880]
            full_weights = torch.zeros(topk_weights.shape[0], E, device=topk_weights.device)
            full_weights.scatter_(1, topk_ids.long(), topk_weights) # # [16384, 32]
            # [32, 16384, 2880] * 
            # [16384, 32]->[32, 16384]->[32, 16484, 1]
            next_states = next_states * full_weights.transpose(0, 1).unsqueeze(-1)
            final = next_states.sum(dim=0).to(hidden_states.dtype) # [32, 16384, 2880]
            out_hidden_states.copy_(final)
            return out_hidden_states
        elif 0: # for gptoss weight only
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
            if inplace:
                out_hidden_states = hidden_states
            else:
                out_hidden_states = torch.empty_like(hidden_states)
            # hidden_states # [16384, 2880]
            # topk_weights # [16384, 4]
            # topk_ids # [16384, 4]
            # w1 # [32, 5760, 2880] # [32, out, inp]
            # w2 # [32, 2880, 2880] # [32, out, inp]

            expanded = out_hidden_states.repeat(32, 1) # [32， 16384, 2880]
            expanded = expanded.view(32, -1, 2880) # [32， 16384, 2880]
            # [32, 16384, 2880] * [32, 5760, 2880] + [5760] = [32, 16384, 5760]
            w1 = w1.transpose(-2, -1)  # [32, 5760, 2880]-> # [32, 2880, 5760] # 转置前是按照linear weight的格式[32, out,inp]
            w2 = w2.transpose(-2, -1)# [32, 2880, 2880]-> # [32, 2880, 2880]# 转置前是按照linear weight的格式[32, out,inp],之后是符合bmm格式[m,n]*[n,k]
            gate_up = torch.bmm(expanded, w1) + w1_bias[..., None, :] 
            gate = gate_up[:,:,:2880]
            up = gate_up[:,:,2880:]
            gate = gate.clamp(min=None, max=7.0) # [32, 16384, 2880]
            up = up.clamp(min=-7.0, max=7.0) # [32, 16384, 2880]
            glu = gate * torch.sigmoid(gate * 1.702) # [32, 16384, 2880]
            next_states = torch.bmm(((up + 1) * glu), w2) # [32, 16384, 2880] * [32, 2880, 2880] # need transpose？
            next_states = next_states + w2_bias[..., None, :] # [32, 16384, 2880] + [32, 1 ,2880]
            full_weights = torch.zeros(topk_weights.shape[0], 32, device=topk_weights.device)
            full_weights.scatter_(1, topk_ids.long(), topk_weights) # # [16384, 32]
            # [32, 16384, 2880] * 
            # [16384, 32]->[32, 16384]->[32, 16484, 1]
            next_states = next_states * full_weights.transpose(0, 1).unsqueeze(-1)
            final = next_states.sum(dim=0).to(hidden_states.dtype) # [32, 16384, 2880]
            out_hidden_states.copy_(final)
            return out_hidden_states

class QuarkW4MXFp4MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str,
                                                                         Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config
        weight_qscheme = self.weight_quant.get("qscheme")
        self.weight_dtype = self.weight_quant.get("dtype")

        self.static_input_scales = False
        self.inp_dtype = None
        if self.input_quant is not None:
            self.inp_dtype = self.input_quant.get("dtype")
        self._custom_mode = "quark"
        self.w_qscheme = self.input_quant.get("qscheme")
        self.w_group_size = self.input_quant.get("group_size")
        self.w_ch_axis = self.input_quant.get("ch_axis")

        if not (weight_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}")  # noqa E501


    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # weight_bias
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)
       
        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW4MXFp4MoEMethod` yet.")

        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        # router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)  # (seq_len, top_k)
        # router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        # router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        '''
        out = fused_experts(
            x,
            layer.w13_weight, # [32, 5760, 1440] # [32, 5760, 2880]
            layer.w2_weight, # [32, 2880, 1440] # [32, 2880, 2880]
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_mxfp4_w4a4=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            activation=activation,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            quark_act_dtype=self.inp_dtype
        )
        '''
        out = self.fused_experts_impl_quark(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                use_mxfp4_w4a4=True,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation=activation,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                quark_act_dtype=self.inp_dtype,
                weight_dtype=self.weight_dtype)
        return out


class QuarkW6MXFp6MoEMethod(QuarkMoEMethod):
    def __init__(self, weight_config: dict[str, Any], input_config: dict[str,
                                                                         Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config
        weight_qscheme = self.weight_quant.get("qscheme")
        self.weight_dtype = self.weight_quant.get("dtype")

        self.static_input_scales = False
        self.inp_dtype = None
        if self.input_quant is not None:
            self.inp_dtype = self.input_quant.get("dtype")
        self._custom_mode = "quark"
        self.w_qscheme = self.input_quant.get("qscheme")
        self.w_group_size = self.input_quant.get("group_size")
        self.w_ch_axis = self.input_quant.get("ch_axis")

        if not (weight_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}")  # noqa E501


    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            # hidden_size // 2,
            hidden_size// 4 * 3,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            # intermediate_size_per_partition // 2,
            intermediate_size_per_partition // 4 * 3,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # weight_bias
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)
       
        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW4MXFp4MoEMethod` yet.")

        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        # router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)  # (seq_len, top_k)
        # router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        # router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        '''
        out = fused_experts(
            x,
            layer.w13_weight, # [32, 5760, 1440] # [32, 5760, 2880]
            layer.w2_weight, # [32, 2880, 1440] # [32, 2880, 2880]
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_mxfp4_w4a4=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            activation=activation,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            quark_act_dtype=self.inp_dtype
        )
        '''
        out = self.fused_experts_impl_quark(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                use_mxfp4_w4a4=True,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation=activation,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                quark_act_dtype=self.inp_dtype,
                layer_for_quark=layer,
                weight_dtype=self.weight_dtype)
        return out


class QuarkW4Int4MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str,
                                                                         Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config
        weight_qscheme = self.weight_quant.get("qscheme")
        self.weight_dtype = self.weight_quant.get("dtype")


        self.static_input_scales = False
        self.inp_dtype = None
        if self.input_quant is not None:
            self.inp_dtype = self.input_quant.get("dtype")
        self._custom_mode = "quark"
        self.w_qscheme = self.weight_quant.get("qscheme")
        self.w_group_size = self.weight_quant.get("group_size")
        self.w_ch_axis = self.weight_quant.get("ch_axis")

        if not (weight_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}")  # noqa E501


    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
       # WEIGHTS
        # transposed because per group
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size, # inp
            2 * intermediate_size_per_partition // 8, # out
            dtype=torch.int32),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition,
            hidden_size // 8,
            dtype=torch.int32),
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size // OCP_MX_BLOCK_SIZE,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # WEIGHT_ZEROPOINT
        w13_weight_zero_point = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size // OCP_MX_BLOCK_SIZE,
                2 * intermediate_size_per_partition // 8,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        w2_weight_zero_point = torch.nn.Parameter(
            torch.ones(
                num_experts,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                hidden_size // 8,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w13_weight_zero_point, extra_weight_attrs)
        set_weight_attrs(w2_weight_zero_point, extra_weight_attrs)

        layer.register_parameter("w13_weight_zero_point", w13_weight_zero_point)
        layer.register_parameter("w2_weight_zero_point", w2_weight_zero_point)
        # weight_bias
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)
       
        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW4MXFp4MoEMethod` yet.")

        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        '''
        out = fused_experts(
            x,
            layer.w13_weight, # [32, 5760, 1440] # [32, 5760, 2880]
            layer.w2_weight, # [32, 2880, 1440] # [32, 2880, 2880]
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_mxfp4_w4a4=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            activation=activation,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            quark_act_dtype=self.inp_dtype,
            w1_zero_point=layer.w13_weight_zero_point,
            w2_zero_point=layer.w2_weight_zero_point
        )
        '''

        out = self.fused_experts_impl_quark(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                use_mxfp4_w4a4=True,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation=activation,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                quark_act_dtype=self.inp_dtype,
                w1_zero_point=layer.w13_weight_zero_point,
                w2_zero_point=layer.w2_weight_zero_point,
                layer_for_quark=layer,
                weight_dtype=self.weight_dtype) # we attach float module for layer so that we can accelerate inference
        return out
