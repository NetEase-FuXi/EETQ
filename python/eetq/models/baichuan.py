import tqdm
from typing import List, Tuple
import torch.nn as nn
import torch

from eetq.utils import replace_fused_qkv, replace_fused_gateup , replace_split_qkv, replace_split_gateup, eet_quantize
from eetq.modules import W8A16Linear
from .base import BaseEETQForCausalLM

class BaichuanEETQForCausalLM(BaseEETQForCausalLM):
    
    def fuse_layers(self):
        self.fuser = BaichuanFuser(self.model)
        self.fuser.fuse_gateup()
        
    def split_layers(self):
        self.fuser.split_gateup()

class BaichuanFuser:
    def __init__(self, model):
        self.model = model


    def fuse_gateup(self):
        device = self.model.device
        
        all_gateup = [[None, None] for i in range(self.model.config.num_hidden_layers)]
        
        gateup_index_map = {"gate_proj": 0, "up_proj": 1}
        
        self.gateup_index_map = [[0, 0] for i in range(self.model.config.num_hidden_layers)]
        for name, m in self.model.named_modules():
            if type(m) in [nn.Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name in ["gate_proj", "up_proj"]:
                    all_gateup[num_layers][gateup_index_map[linear_name]] = m

        
        for _, gateup in enumerate(all_gateup):
            self.gateup_index_map[_][1] = gateup[0].weight.shape[0]
            name = f"model.layers.{_}.mlp.gateup_proj"                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            gateup_weight = [x.weight for x in gateup]
            gateup_weight = torch.cat(gateup_weight, dim=0)
            fused_gateup = nn.Linear(gateup_weight.shape[1], gateup_weight.shape[0])
            fused_gateup.weight = nn.Parameter(gateup_weight.to(device), requires_grad=False)
            replace_fused_gateup(self.model, name, fused_gateup)
        
    def split_gateup(self):
        for name, m in self.model.named_modules():
            if type(m) in [W8A16Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name == "gateup_proj":
                    replace_split_gateup(self.model, name, m, self.gateup_index_map[num_layers])
