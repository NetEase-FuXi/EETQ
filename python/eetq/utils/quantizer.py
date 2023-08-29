import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from ..modules.qlinear import W8A16Linear
from .base import set_op_by_name, get_named_linears


def make_eet_qlinear(module, names, device):
    if isinstance(module, W8A16Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)



def replace_with_eet_qlinear(model, device="cuda:0"):
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="replace with eet weight quantize only linear..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)            # linear dict
        for name, linear in named_linears.items():
            if linear.weight.dtype == torch.float16:
                q_linear = W8A16Linear.from_torch(linear, scales=None, init_only=False)
            elif linear.weight.dtype == torch.int8:
                scales = torch.div(linear.state_dict()["SCB"], 127.0)
                q_linear = W8A16Linear.from_torch(linear, scales=scales, init_only=False)
            set_op_by_name(layer, name, q_linear)
            
            # del linear
            linear.cpu()
            del linear
            torch.cuda.empty_cache()
            gc.collect()

