import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from ..modules.qlinear import W8A16Linear
from .base import set_op_by_name, get_named_linears
from .mapping import get_submodule_name


def replace_with_eet_qlinear(model, init_only=False, target_model="llama", device="cuda:0"):
    layers = eval(get_submodule_name(model, name=target_model, sub_name="decoder"))
    for i in tqdm(range(len(layers)), desc="replace with eet weight quantize only linear..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)            # linear dict
        for name, linear in named_linears.items():
            if linear.weight.dtype == torch.float16:
                q_linear = W8A16Linear.from_torch(linear, scales=None, init_only=init_only)
            elif linear.weight.dtype == torch.int8:         # bitsandbytes int8
                scales = torch.div(linear.state_dict()["SCB"], 127.0)
                q_linear = W8A16Linear.from_torch(linear, scales=scales, init_only=init_only)
            set_op_by_name(layer, name, q_linear)

            if not init_only:
                # del linear
                linear.cpu()
                del linear
                torch.cuda.empty_cache()
                gc.collect()
            else:
                del linear
                gc.collect()

