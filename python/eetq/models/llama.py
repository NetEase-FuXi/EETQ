import torch.nn as nn
import torch

from eetq.utils import (replace_fused_qkv, 
                        replace_fused_gateup, 
                        replace_split_qkv, 
                        replace_split_gateup,
                        split_tp_row,
                        split_tp_column,
                        merge_tp_handler)

from eetq.modules import W8A16Linear
from .base import BaseEETQForCausalLM

class LlamaEETQForCausalLM(BaseEETQForCausalLM):
    
    def fuse_layers(self, tp=1):
        self.tp = tp
        self.fuser = LlamaFuser(self.model)
        print("[EET][INFO] fusing qkv and gateup ...")
        self.fuser.fuse_qkv_gateup()
        if self.tp > 1:
            print("[EET][INFO] spliting tp ...")
            self.fuser.split_tp(self.tp)

        
    def split_layers(self):
        if self.tp > 1:
            print("[EET][INFO] merging tp ...")
            self.fuser.merge_tp()
        print("[EET][INFO] spliting qkv and gateup ...")
        self.fuser.split_qkv_gateup()

class LlamaFuser:
    def __init__(self, model):
        self.model = model


    def fuse_qkv_gateup(self):
        device = self.model.device
        
        all_qkv = [[None, None, None] for i in range(self.model.config.num_hidden_layers)]
        all_gateup = [[None, None] for i in range(self.model.config.num_hidden_layers)]
        
        qkv_index_map = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
        gateup_index_map = {"gate_proj": 0, "up_proj": 1}
        
        self.qkv_index_map = [[0, 0, 0] for i in range(self.model.config.num_hidden_layers)]
        self.gateup_index_map = [[0, 0] for i in range(self.model.config.num_hidden_layers)]
        for name, m in self.model.named_modules():
            if type(m) in [nn.Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name in ["q_proj", "k_proj", "v_proj"]:
                    all_qkv[num_layers][qkv_index_map[linear_name]] = m
                elif linear_name in ["gate_proj", "up_proj"]:
                    all_gateup[num_layers][gateup_index_map[linear_name]] = m
        
        for _, qkv in enumerate(all_qkv):
            self.qkv_index_map[_][1] = qkv[0].weight.shape[0]
            self.qkv_index_map[_][2] = self.qkv_index_map[_][1] + qkv[1].weight.shape[0]
            name = f"model.layers.{_}.self_attn.qkv_proj"
            qkv_weight = [x.weight for x in qkv]
            qkv_weight = torch.cat(qkv_weight, dim=0)
            fused_qkv = nn.Linear(qkv_weight.shape[1], qkv_weight.shape[0])
            fused_qkv.weight = nn.Parameter(qkv_weight.to(device), requires_grad=False)
            replace_fused_qkv(self.model, name, fused_qkv)
        
        for _, gateup in enumerate(all_gateup):
            self.gateup_index_map[_][1] = gateup[0].weight.shape[0]
            name = f"model.layers.{_}.mlp.gateup_proj"
            gateup_weight = [x.weight for x in gateup]
            gateup_weight = torch.cat(gateup_weight, dim=0)
            fused_gateup = nn.Linear(gateup_weight.shape[1], gateup_weight.shape[0])
            fused_gateup.weight = nn.Parameter(gateup_weight.to(device), requires_grad=False)
            replace_fused_gateup(self.model, name, fused_gateup)
        
    def split_qkv_gateup(self):
        modules = [(name, m) for name, m in self.model.named_modules()]
        for name, m in modules:
            if type(m) in [W8A16Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name == "qkv_proj":
                    replace_split_qkv(self.model, name, m, self.qkv_index_map[num_layers])
                elif linear_name == "gateup_proj":
                    replace_split_gateup(self.model, name, m, self.gateup_index_map[num_layers])
        
        
    def split_tp(self, tp=2):
        self.tp = tp
        modules = [(name, m) for name, m in self.model.named_modules()]
        for name, m in modules:
            if type(m) in [nn.Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name == "qkv_proj" or linear_name == "gateup_proj":
                    split_tp_column(self.model, name, m, tp)
                elif linear_name == "o_proj" or linear_name == "down_proj":
                    split_tp_row(self.model, name, m, tp)
    
    def merge_tp(self):
        all_qkv_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_hidden_layers)]
        all_o_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_hidden_layers)]
        all_gateup_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_hidden_layers)]
        all_down_tp = [[None for j in range(self.tp)] for i in range(self.model.config.num_hidden_layers)]
        
        for name, m in self.model.named_modules():
            if type(m) in [W8A16Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                linear_name_levels = linear_name.split("_")
                tp_num = int(linear_name_levels[-1][2:])
                name = linear_name_levels[0]
                
                if name == "qkv":
                    all_qkv_tp[num_layers][tp_num] = m
                elif name == "o":
                    all_o_tp[num_layers][tp_num] = m
                elif name == "gateup":
                    all_gateup_tp[num_layers][tp_num] = m
                elif name == "down":
                    all_down_tp[num_layers][tp_num] = m
        
        merge_tp_handler(self.model, all_qkv_tp, "model.layers.{}.self_attn.qkv_proj", True)
        merge_tp_handler(self.model, all_o_tp, "model.layers.{}.self_attn.o_proj", False)
        merge_tp_handler(self.model, all_gateup_tp, "model.layers.{}.mlp.gateup_proj", True)
        merge_tp_handler(self.model, all_down_tp, "model.layers.{}.mlp.down_proj", False)
