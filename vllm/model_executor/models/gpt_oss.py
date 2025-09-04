# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Optional
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig

from vllm import envs
from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_ep_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig
from typing import Any, Optional, cast
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)
from .utils import extract_layer_index, maybe_prefix


class OAIAttention(nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type":
                "yarn",
                "factor":
                config.rope_scaling["factor"],
                "original_max_position_embeddings":
                config.rope_scaling["original_max_position_embeddings"],
                "beta_fast":
                config.rope_scaling["beta_fast"],
                "beta_slow":
                config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )

        tp_size = get_tensor_model_parallel_world_size()

        attention_sink_dtype = (torch.float32 if envs.VLLM_USE_TRTLLM_ATTENTION
                                else torch.bfloat16)
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size,
                        dtype=attention_sink_dtype,
                        requires_grad=False))

        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.qkv = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        # Only apply sliding window to every other layer
        sliding_window = (config.sliding_window if self.layer_idx %
                          2 == 0 else None)
        self.attn = Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        t = self.norm(hidden_states)

        qkv, _ = self.qkv(t)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        v = v.contiguous()
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)

        return output + hidden_states


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.router = torch.nn.Linear(config.hidden_size,
                                      config.num_local_experts,
                                      dtype=torch.bfloat16)
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(num_experts=config.num_local_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                reduce_results=True,
                                renormalize=True,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts",
                                apply_router_weight_on_input=False,
                                has_bias=True,
                                activation="swiglu_oai")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.router(t)
        t = self.experts(hidden_states=t, router_logits=g)
        return x + t


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        if isinstance(quant_config, Mxfp4Config): # just for original gpt osss model
            self.attn = OAIAttention(config, quant_config=None, prefix=f"{prefix}.attn")
        elif isinstance(quant_config, QuarkConfig): #  # skip by checkout proj is quantized or not when just moe
            unquantized_attn = False
            exclude_layers = cast(list[str], quant_config.quant_config.get("exclude"))
            for exclude_layer in exclude_layers:
                if "k_proj" in exclude_layer:
                    unquantized_attn = True
                    break
            if unquantized_attn:
                self.attn = OAIAttention(config, quant_config=None, prefix=f"{prefix}.attn")
            else:
                self.attn = OAIAttention(config, quant_config=quant_config, prefix=f"{prefix}.attn")
        else:# or other
            self.attn = OAIAttention(config, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.mlp = MLPBlock(config,
                            self.layer_idx,
                            quant_config=quant_config,
                            prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(hidden_states, positions)
        output = self.mlp(attn_output)
        return output


@support_torch_compile
class GptOssModel(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            # quant_config=self.quant_config # quark don't quantize embeding, so unquantized
        )
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                self.config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"block.{layer_idx}"),
            ) for layer_idx in range(self.config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return x


class GptOssForCausalLM(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config.hf_config
        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        quant_config = vllm_config.quant_config
        if isinstance(quant_config, QuarkConfig): # just for original gpt osss model
            dtype = quant_config.quant_config.get("global_quant_config", {}).get("weight", {}).get("dtype")
            if dtype is None or dtype != "int4":
                quant_config = None

        self.lm_head = ParallelLMHead(
            self.model_config.vocab_size,
            self.model_config.hidden_size,
            quant_config=quant_config # quantized lmhead if int4

        )
        self.logits_processor = LogitsProcessor(self.model_config.vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert intermediate_tensors is None
        assert inputs_embeds is None
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def _load_weights_mxfp4(
            self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rename_mapping = {
            "self_attn": "attn",
            "input_layernorm.weight": "attn.norm.weight",
            "post_attention_layernorm.weight": "mlp.norm.weight",
            "embed_tokens": "embedding",
        }

        def maybe_rename(name: str) -> str:
            for remap_name, new_name in rename_mapping.items():
                if remap_name in name:
                    return name.replace(remap_name, new_name)
            return name

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        mxfp4_block = 32

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = self.model_config.intermediate_size
        intermediate_size_block = intermediate_size // mxfp4_block
        per_rank_intermediate_size_block = cdiv(intermediate_size_block,
                                                tp_size)
        per_rank_intermediate_size = (per_rank_intermediate_size_block *
                                      mxfp4_block)

        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size,
                          intermediate_size)

        # Attention heads per rank
        heads_per_rank = self.model_config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank
        num_experts = self.model_config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        for name, weight in weights:
            # FIXME(woosuk): Remove this after testing.
            weight = weight.cuda()

            if "gate_up_proj_blocks" in name:
                # Handle MLP gate and up projection weights
                new_name = name.replace("gate_up_proj_blocks", "w13_weight")

                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(num_experts, 2 * intermediate_size,
                                     -1).contiguous()

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end,
                                           ...]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "down_proj_blocks" in name:
                # Handle MLP down projection weights
                new_name = name.replace("down_proj_blocks", "w2_weight")
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(num_experts, -1,
                                     intermediate_size // 2).contiguous()
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[...,
                                           tp_rank_start // 2:tp_rank_end // 2]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "gate_up_proj_scales" in name:
                # Handle MLP gate and up projection weights scale
                new_name = name.replace("gate_up_proj_scales",
                                        "w13_weight_scale")
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end,
                                           ...]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "down_proj_scales" in name:
                # Handle MLP down projection weights
                new_name = name.replace("down_proj_scales", "w2_weight_scale")
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[..., tp_rank_start //
                                           mxfp4_block:tp_rank_end //
                                           mxfp4_block]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)
            elif "gate_up_proj_bias" in name:
                # Handle MLP gate and up projection biases
                new_name = name.replace("gate_up_proj_bias", "w13_bias")

                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "down_proj_bias" in name:
                # Handle MLP down projection bias
                new_name = name.replace("down_proj_bias", "w2_bias")
                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                weight_loader(param,
                              weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                name = name.replace("self_attn", "attn")
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                shard_id = ("q" if "q_proj" in name else
                            "k" if "k_proj" in name else "v")
                name = name.replace("self_attn", "attn")
                param_name = name.replace(f"{shard_id}_proj", "qkv")
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                weight_loader(param, weight, loaded_shard_id=shard_id)
                loaded_params.add(param_name)
            else:
                # Handle all other weights with potential renaming
                renamed_name = maybe_rename(name)
                if renamed_name not in params_dict:
                    continue
                param = params_dict[renamed_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(renamed_name)

        return loaded_params

    def _load_weights_other(
            self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rename_mapping = {
            "self_attn": "attn",
            "input_layernorm.weight": "attn.norm.weight",
            "post_attention_layernorm.weight": "mlp.norm.weight",
            "embed_tokens": "embedding",
        }

        def maybe_rename(name: str) -> str:
            for remap_name, new_name in rename_mapping.items():
                if remap_name in name:
                    return name.replace(remap_name, new_name)
            return name

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = self.model_config.intermediate_size

        per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size,
                          intermediate_size)

        # Attention heads per rank
        heads_per_rank = self.model_config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank
        num_experts = self.model_config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        for name, weight in weights:
            if ".experts.gate_up_proj" in name and "bias" not in name:
                # Handle MLP gate and up projection weights
                new_name = name.replace(".experts.gate_up_proj",
                                        ".experts.w13_weight")

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, :,
                                           2 * tp_rank_start:2 * tp_rank_end]

                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[new_name]

                param.copy_(narrow_weight)
                loaded_params.add(new_name)

            elif ".experts.down_proj" in name and "bias" not in name:
                # Handle MLP down projection weights
                new_name = name.replace(".experts.down_proj",
                                        ".experts.w2_weight")

                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[new_name]

                param.copy_(narrow_weight)
                loaded_params.add(new_name)

            elif "gate_up_proj_bias" in name:
                # Handle MLP gate and up projection biases
                new_name = name.replace("gate_up_proj_bias", "w13_bias")

                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end]

                param = params_dict[new_name]

                param.copy_(narrow_weight)
                loaded_params.add(new_name)

            elif "down_proj_bias" in name:
                # Handle MLP down projection bias
                new_name = name.replace("down_proj_bias", "w2_bias")

                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                param = params_dict[new_name]
                param.copy_(weight)
                loaded_params.add(new_name)
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                name = name.replace("self_attn", "attn")
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                shard_id = ("q" if "q_proj" in name else
                            "k" if "k_proj" in name else "v")
                name = name.replace("self_attn", "attn")
                param_name = name.replace(f"{shard_id}_proj", "qkv")
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                weight_loader(param, weight, loaded_shard_id=shard_id)
                loaded_params.add(param_name)
            else:
                # Handle all other weights with potential renaming

                renamed_name = maybe_rename(name)
                if renamed_name not in params_dict:
                    continue
                param = params_dict[renamed_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(renamed_name)

        return loaded_params

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        quant_method = (self.model_config.quantization_config['quant_method']
                        if hasattr(self.model_config, "quantization_config")
                        else None)
        if quant_method == "mxfp4":
            return self._load_weights_mxfp4(weights)
        elif quant_method == "quark":
            try:
                dtype = self.model_config.quantization_config["global_quant_config"]["weight"]["dtype"]
            except:
                dtype = None
            is_int4 = True if dtype == "int4" else False
            return self._load_quark_mxfp4_weights(weights, is_int4)
            # return self._load_weights_other(weights)
    def _load_quark_mxfp4_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]], is_int4=False) -> set[str]:
        rename_mapping = {
            "self_attn": "attn",
            "input_layernorm.weight": "attn.norm.weight",
            "post_attention_layernorm.weight": "mlp.norm.weight",
            "embed_tokens": "embedding",
        }

        def maybe_rename(name: str) -> str:
            for remap_name, new_name in rename_mapping.items():
                if remap_name in name:
                    return name.replace(remap_name, new_name)
            return name

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = self.model_config.intermediate_size

        per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size,
                          intermediate_size)

        # Attention heads per rank
        heads_per_rank = self.model_config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank
        num_experts = self.model_config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank
        # copy from qwen2_moe.py
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # ("qkv_proj", "q_proj", "q"),
            # ("qkv_proj", "k_proj", "k"),
            # ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            if "sinks" in name:
                weight = loaded_weight.cuda()
                name = name.replace("self_attn", "attn")
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                weight = loaded_weight.cuda()
                shard_id = ("q" if "q_proj" in name else
                            "k" if "k_proj" in name else "v")
                name = name.replace("self_attn", "attn")
                param_name = name.replace(f"{shard_id}_proj", "qkv")
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                weight_loader(param, weight, loaded_shard_id=shard_id) # qkv_weight and qkv_bias
                loaded_params.add(param_name)
            elif "mlp.experts" in name:
                weight = loaded_weight.cuda()
                for mapping in expert_params_mapping: # from 0 to 32
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, # [32, 2880] for bias
                                  loaded_weight, # [2880] for bias
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id,
                                  is_int4=is_int4)
            else:
                weight = loaded_weight.cuda()
                renamed_name = maybe_rename(name)
                if renamed_name not in params_dict:
                    continue
                param = params_dict[renamed_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(renamed_name)
                name = renamed_name


            if 0:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    # ("gate_up_proj", "gate_proj", 0),
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if "mlp.experts" in name:
                        continue
                    name = name.replace(weight_name, param_name) # model.layers.10.self_attn.k_proj.weight->'model.layers.10.self_attn.qkv_proj.weight'
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)

                        # Skip layers on other devices.
                        if is_pp_missing_parameter(name, self):
                            continue
                        # Skip loading extra bias for GPTQ models.
                        if ((name.endswith(".bias") or name.endswith("_bias"))
                                and name not in params_dict):
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param,
                                    loaded_weight,
                                    name,
                                    shard_id=shard_id,
                                    expert_id=expert_id)
                        break
                # 如果 for 循环里执行了 break，就不会进入 else。
                # 如果 for 循环正常执行完（即没有被 break 中断），就会执行 else 里的内容
                    else:
                        # # Skip loading extra bias for GPTQ models.
                        # if ((name.endswith(".bias") or name.endswith("_bias"))
                        #         and name not in params_dict):
                        #     print(f"{name} not in pamras_dict")
                        #     continue
                        # Skip layers on other devices.
                        if is_pp_missing_parameter(name, self):
                            continue
                        # Remapping the name of FP8 kv-scale.
                        if name.endswith("kv_scale"):
                            remapped_kv_scale_name = name.replace(
                                ".kv_scale", ".attn.kv_scale")
                            if remapped_kv_scale_name not in params_dict:
                                logger.warning_once(
                                    "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                                    name,
                                    remapped_kv_scale_name,
                                )
                                continue
                            else:
                                name = remapped_kv_scale_name
                        renamed_name = maybe_rename(name) # Embedding, LN, and other generic names need to be renamed.
                        if renamed_name not in params_dict:
                            print(f"{renamed_name} not in pamras_dict6")
                            continue
                        name = renamed_name
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.model_config.num_local_experts)