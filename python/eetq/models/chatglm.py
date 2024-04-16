from typing_extensions import Annotated, Doc
import torch.nn as nn

from eetq.utils import split_tp_column, split_tp_row, merge_tp_handler
from eetq.modules import W8A16Linear
from transformers import PreTrainedModel, PretrainedConfig
from .base import BaseEETQForCausalLM


"""
under developement
"""
class ChatGLMEETQForCausalLM(BaseEETQForCausalLM):
    def __init__(self, model: PreTrainedModel, model_type: str, is_quantized: bool, config: PretrainedConfig, quant_config: dict, tp: int):
        self.tp = tp
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (config.multi_query_group_num
                                   if config.multi_query_attention else
                                   config.num_attention_heads)
        super().__init__(model, model_type, is_quantized, config, quant_config)
        
    
    def fuse_layers(self, tp):
        self.fuser = ChatGLMFuser(self.model)
        if self.tp > 1:
            print("[EET][INFO] spliting tp ...")
            self.fuser.split_tp(self.tp)
        
    def split_layers(self):
        if self.tp > 1:
            print("[EET][INFO] merging tp ...")
            self.fuser.merge_tp()

class ChatGLMFuser:
    def __init__(self, model):
        self.model = model
        

    def split_tp(self, tp=2):
        self.tp = tp
        modules = [(name, m) for name, m in self.model.named_modules()]
        for name, m in modules:
            if type(m) in [nn.Linear] and "lm_head" not in name and "output_layer" not in name:
                levels = name.split(".")
                num_layers = int(levels[3])
                linear_name = levels[-1]
                if linear_name == "query_key_value" or linear_name == "dense_h_to_4h":
                    split_tp_column(self.model, name, m, tp)
                elif linear_name == "dense" or linear_name == "dense_4h_to_h":
                    split_tp_row(self.model, name, m, tp)
                    
    def merge_tp(self):
        all_qkv_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_layers)]
        all_o_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_layers)]
        all_gateup_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_layers)]
        all_down_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_layers)]
        
        for name, m in self.model.named_modules():
            if type(m) in [W8A16Linear] and "lm_head" not in name and "output_layer" not in name:
                print(name)
                levels = name.split(".")
                num_layers = int(levels[3])
                linear_name = levels[-1]
                linear_name_levels = linear_name.split("_")
                tp_num = int(linear_name_levels[-1][2:])
                linear_name_levels.pop()
                name = "_".join(linear_name_levels)
                
                if name == "query_key_value":
                    all_qkv_tp[num_layers][tp_num] = m
                elif name == "dense":
                    all_o_tp[num_layers][tp_num] = m
                elif name == "dense_h_to_4h":
                    all_gateup_tp[num_layers][tp_num] = m
                elif name == "dense_4h_to_h":
                    all_down_tp[num_layers][tp_num] = m
        
        merge_tp_handler(self.model, all_qkv_tp, "transformer.encoder.layers.{}.self_attention.query_key_value", True)
        merge_tp_handler(self.model, all_o_tp, "transformer.encoder.layers.{}.self_attention.dense", False)
        merge_tp_handler(self.model, all_gateup_tp, "transformer.encoder.layers.{}.mlp.dense_h_to_4h", True)
        merge_tp_handler(self.model, all_down_tp, "transformer.encoder.layers.{}.mlp.dense_4h_to_h", False)