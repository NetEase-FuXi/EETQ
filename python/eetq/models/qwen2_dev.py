import torch.nn as nn
import torch

from eetq.utils import *
from eetq.modules import W8A16Linear
from .base import BaseEETQForCausalLM


class Qwen2EETQForCausalLM(BaseEETQForCausalLM):

    def fuse_layers(self, tp):
        self.tp = tp
        self.fuser = Qwen2Handler(self.model, tp)
        import pdb
        pdb.set_trace()
        self.fuser.hf2vllm()

        import pdb

        pdb.set_trace()

    def split_layers(self):
        if self.tp > 1:
            print("[EET][INFO] merging tp ...")
            self.fuser.merge_tp()
        print("[EET][INFO] spliting qkv and gateup ...")
        self.fuser.split_qkv_gateup()


class Qwen2Handler:
    def __init__(self, model, tp):

        self.model = model
        self.num_attention_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        assert self.num_attention_heads != 0, "num_attention_heads not equal to 0!"
        self.num_key_value_heads = self.model.config.num_key_value_heads
        if self.num_key_value_heads == None:
            self.num_key_value_heads = num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads

        self.tp = tp
        assert (
            self.num_attention_heads % self.tp == 0
        ), "num_attention_heads must be divisible by tp"
        self.num_heads = self.num_attention_heads // self.tp
        self.num_kv_heads = self.num_key_value_heads // self.tp

        self.intermediate_size = self.model.config.intermediate_size
        assert (
            self.intermediate_size % self.tp == 0
        ), "intermediate_size must be divisible by tp"

        self.qkv_shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            (0, self.num_heads * self.head_size),
            (self.num_heads * self.head_size, self.num_kv_heads * self.head_size),
            (
                (self.num_heads + self.num_kv_heads) * self.head_size,
                self.num_kv_heads * self.head_size,
            ),
        ]
        self.qkv_weights_offsets = [
            [
                (self.num_heads * self.head_size * i, self.num_heads * self.head_size),
                (
                    self.num_kv_heads * self.head_size * i,
                    self.num_kv_heads * self.head_size,
                ),
                (
                    self.num_kv_heads * self.head_size * i,
                    self.num_kv_heads * self.head_size,
                ),
            ]
            for i in range(self.tp)
        ]

        self.gateup_shard_offsets = [
            (0, self.intermediate_size // self.tp),
            (self.intermediate_size // self.tp, self.intermediate_size // self.tp),
        ]
        self.gateup_weights_offsets = [
            [
                (
                    self.intermediate_size // self.tp * i,
                    self.intermediate_size // self.tp,
                ),
                (
                    self.intermediate_size // self.tp * i,
                    self.intermediate_size // self.tp,
                ),
            ]
            for i in range(self.tp)
        ]
        self.qkv_index_map = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
        self.gateup_index_map = {"gate_proj": 0, "up_proj": 1}
        self.dtype = self.model.dtype

    def hf2vllm(self):

        qkv = [
            [
                nn.Linear(
                    self.hidden_size,
                    (self.num_heads + 2 * self.num_key_value_heads)
                    * self.head_size
                    // self.tp,
                    dtype=self.dtype,
                )
                for j in range(self.num_hidden_layers)
            ]
            for i in range(self.tp)
        ]
        gateup = [
            [
                nn.Linear(
                    self.hidden_size,
                    self.intermediate_size // self.tp * 2,
                    dtype=self.dtype,
                )
                for j in range(self.num_hidden_layers)
            ]
            for i in range(self.tp)
        ]
        if self.tp > 1:
            assert self.tp == 1, "TP > 1 to do"
        for name, m in self.model.named_modules():
            if type(m) in [nn.Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name in ["q_proj", "k_proj", "v_proj"]:
                    for i in range(self.tp):
                        idx = self.qkv_index_map[linear_name]
                        weight_loader = qkv[i][num_layers].weight
                        weight_loader = weight_loader.narrow(
                            0,
                            self.qkv_shard_offsets[idx][0],
                            self.qkv_shard_offsets[idx][1],
                        )
                        weight_toload = m.weight.narrow(
                            0,
                            self.qkv_weights_offsets[i][idx][0],
                            self.qkv_weights_offsets[i][idx][1],
                        )
                        weight_loader.copy_(weight_toload)

                elif linear_name in ["gate_proj", "up_proj"]:
                    for i in range(self.tp):
                        idx = self.gateup_index_map[linear_name]
                        weight_loader = gateup[i][num_layers].weight
                        weight_loader = weight_loader.narrow(
                            0,
                            self.gateup_shard_offsets[idx][0],
                            self.gateup_shard_offsets[idx][1],
                        )
                        weight_toload = m.weight.narrow(
                            0,
                            self.gateup_weights_offsets[i][idx][0],
                            self.gateup_weights_offsets[i][idx][1],
                        )
                        weight_loader.copy_(weight_toload)
        for i in range(self.tp):
            for j in range(self.num_hidden_layers):
                qkv_module = qkv[i][j]
                gateup_module = gateup[i][j]
                replace_fused_qkv(
                    self.model, f"model.layers.{j}.self_attn.qkv_proj_{i}", qkv_module
                )
                replace_fused_gateup(
                    self.model, f"model.layers.{j}.mlp.gateup_proj_{i}", gateup_module
                )

    def vllm2hf(self):
        for i in range(self.tp):
            for j in range(self.num_hidden_layers):
                qkv_module = qkv[i][j]
                gateup_module = gateup[i][j]
                replace_fused_qkv(
                    self.model, f"model.layers.{j}.self_attn.qkv_proj", qkv_module
                )
                replace_fused_gateup(
                    self.model, f"model.layers.{j}.mlp.gateup_proj", gateup_module
                )