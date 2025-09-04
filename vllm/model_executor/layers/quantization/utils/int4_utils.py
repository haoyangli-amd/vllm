# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer

logger = init_logger(__name__)

OCP_MX_BLOCK_SIZE = 32




def _dequant_quark_weight(quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Union[torch.Tensor,None],
                axis: Union[int, None], group_size: Union[int, None], qscheme: Union[str, None]) -> torch.Tensor:
    import quark
    dq_w = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                quant_dtype,
                inputs, # qkv[5120,2880]
                scale, # [5120,90]
                zero_point, # [5120, 90]
                axis, # 1
                group_size, # 32
                qscheme,  #str
            )
    
    return dq_w


def _dequant_quark_weight_fake(quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Union[torch.Tensor,None],
                axis: Union[int, None], group_size: Union[int, None], qscheme: Union[str, None]) -> torch.Tensor:
    return torch.empty_like(inputs)



try:
    direct_register_custom_op(
        op_name="dequant_quark_weight",
        op_func=_dequant_quark_weight,
        mutates_args=[],
        fake_impl=_dequant_quark_weight_fake,
    )
    dequant_quark_weight = torch.ops.vllm.dequant_quark_weight
except AttributeError as error:
    raise error
