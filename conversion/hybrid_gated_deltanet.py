from __future__ import annotations

import math

from collections.abc import Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("HybridGatedDeltaNetForCausalLM")
@ModelBase.example("Idiap/gated-deltanet-nsa-0.4B-10B")
class HybridGatedDeltaNetModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HYBRID_GATED_DELTANET

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gdn_qkv: dict[int, dict[str, Tensor]] = {}
        self._gdn_conv: dict[int, dict[str, Tensor]] = {}

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_vocab_base_pre(self, tokenizer) -> str:
        return "gemma4"

    def set_gguf_parameters(self):
        hp = self.hparams
        attn = hp["attn"]
        n_layer = hp["num_hidden_layers"]
        gdn_head_dim = hp["head_dim"]
        intermediate_size = math.ceil(hp["hidden_size"] * hp["hidden_ratio"] * 2 / 3 / 256) * 256

        hp["num_attention_heads"] = [hp["num_heads"] if il % 2 == 0 else attn["num_heads"] for il in range(n_layer)]
        hp["num_key_value_heads"] = [hp["num_heads"] if il % 2 == 0 else attn["num_kv_heads"] for il in range(n_layer)]
        hp["head_dim"] = attn["head_dim"]
        hp["intermediate_size"] = intermediate_size
        super().set_gguf_parameters()

        self.gguf_writer.add_rope_dimension_count(attn["head_dim"])
        self.gguf_writer.add_rope_freq_base(attn["rope_theta"])
        self.gguf_writer.add_recurrent_layers([il % 2 == 0 for il in range(n_layer)])
        self.gguf_writer.add_ssm_conv_kernel(hp["conv_size"])
        self.gguf_writer.add_ssm_inner_size(hp["hidden_size"] * hp["expand_v"])
        self.gguf_writer.add_ssm_state_size(gdn_head_dim)
        self.gguf_writer.add_ssm_time_step_rank(hp["num_heads"])
        self.gguf_writer.add_ssm_group_count(hp["num_heads"])
        self.gguf_writer.add_indexer_head_count(attn["num_kv_heads"])
        self.gguf_writer.add_indexer_key_length(attn["head_dim"])
        self.gguf_writer.add_indexer_top_k(attn["block_counts"])
        self.gguf_writer.add_indexer_block_size(attn["block_size"])
        self.gguf_writer.add_indexer_local_blocks(attn["window_size"] // attn["block_size"])
        self.gguf_writer.add_indexer_types([il in attn["layers"] for il in range(n_layer)])

    def _name(self, tensor: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        return self.format_tensor_name(tensor, bid, suffix)

    def _fuse_gdn(self, bid: int, kind: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        buffers = self._gdn_qkv if kind.endswith("proj") else self._gdn_conv
        buffer = buffers.setdefault(bid, {})
        buffer[kind[0]] = data_torch
        if len(buffer) != 3:
            return ()

        q, k, v = buffer.pop("q"), buffer.pop("k"), buffer.pop("v")
        del buffers[bid]
        if kind.endswith("proj"):
            return ((self._name(gguf.MODEL_TENSOR.ATTN_QKV, bid), torch.cat((q, k, v), dim=0)),)
        return ((self._name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), torch.cat((q, k, v), dim=0).squeeze(1)),)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embeddings.weight":
            return ((self._name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch),)
        if name == "model.norm.weight":
            return ((self._name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch),)
        if name == "lm_head.weight":
            return ((self._name(gguf.MODEL_TENSOR.OUTPUT), data_torch),)
        if bid is None or not name.startswith(f"model.layers.{bid}."):
            raise ValueError(f"Unexpected tensor {name!r}")

        suffix = name.removeprefix(f"model.layers.{bid}.")
        direct = {
            "attn_norm.weight": gguf.MODEL_TENSOR.ATTN_NORM,
            "mlp_norm.weight": gguf.MODEL_TENSOR.ATTN_POST_NORM,
            "mlp.gate_proj.weight": gguf.MODEL_TENSOR.FFN_GATE,
            "mlp.down_proj.weight": gguf.MODEL_TENSOR.FFN_DOWN,
            "mlp.up_proj.weight": gguf.MODEL_TENSOR.FFN_UP,
        }
        if suffix in direct:
            return ((self._name(direct[suffix], bid), data_torch),)

        if bid % 2 == 1:
            nsa = {
                "attn.q_proj.weight": gguf.MODEL_TENSOR.ATTN_Q,
                "attn.k_proj.weight": gguf.MODEL_TENSOR.ATTN_K,
                "attn.v_proj.weight": gguf.MODEL_TENSOR.ATTN_V,
                "attn.g_proj.weight": gguf.MODEL_TENSOR.ATTN_GATE,
                "attn.o_proj.weight": gguf.MODEL_TENSOR.ATTN_OUT,
            }
            if suffix in nsa:
                return ((self._name(nsa[suffix], bid), data_torch),)
            raise ValueError(f"Unexpected NSA tensor {name!r}")

        if suffix in ("attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"):
            return self._fuse_gdn(bid, suffix.removesuffix(".weight").removeprefix("attn."), data_torch)
        if suffix in ("attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight"):
            return self._fuse_gdn(bid, suffix.removesuffix(".weight").removeprefix("attn."), data_torch)

        gdn = {
            "attn.g_proj.weight": gguf.MODEL_TENSOR.ATTN_GATE,
            "attn.A_log": gguf.MODEL_TENSOR.SSM_A,
            "attn.dt_bias": gguf.MODEL_TENSOR.SSM_DT,
            "attn.a_proj.weight": gguf.MODEL_TENSOR.SSM_ALPHA,
            "attn.b_proj.weight": gguf.MODEL_TENSOR.SSM_BETA,
            "attn.o_norm.weight": gguf.MODEL_TENSOR.SSM_NORM,
            "attn.o_proj.weight": gguf.MODEL_TENSOR.SSM_OUT,
        }
        if suffix not in gdn:
            raise ValueError(f"Unexpected Gated DeltaNet tensor {name!r}")
        if suffix == "attn.A_log":
            data_torch = -torch.exp(data_torch)
        tensor_suffix = "" if suffix == "attn.A_log" else ".bias" if suffix == "attn.dt_bias" else ".weight"
        return ((self._name(gdn[suffix], bid, tensor_suffix), data_torch),)
