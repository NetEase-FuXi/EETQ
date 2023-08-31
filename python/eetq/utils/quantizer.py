import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from transformers import AutoModelForCausalLM
import peft.tuners.lora as lora
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from ..modules.qlinear import W8A16Linear, W8A16LoraLinear
from .base import set_op_by_name, get_named_layers, find_submodule
from .mapping import structure_mapping


def replace_with_eet_qlinear(model, init_only=False, target_model="llama", device="cuda:0"):
    sub_name = structure_mapping(model, target_model)['decoder']
    layers = find_submodule(model, sub_name)
    for i in tqdm(range(len(layers)), desc="replace with eet weight quantize only linear..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_layers(layer, layers=[nn.Linear])            # linear dict
        for name, linear in named_linears.items():
            # print("name:{}, type(linear):{}".format(name, type(linear)))
            if linear.weight.dtype == torch.float16:
                q_linear = W8A16Linear.from_torch(linear, scales=None, init_only=init_only)
            elif linear.weight.dtype == torch.int8:         # bitsandbytes int8
                scales = torch.div(linear.state_dict()["SCB"], 127.0)
                q_linear = W8A16Linear.from_torch(linear, scales=scales, init_only=init_only)
            else:
                raise ValueError("Unsupported data type: {}".format(linear.weight.dtype))
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


def replace_with_eet_lora_qlinear(model, init_only=False, target_model="llama", device="cuda:0"):
    sub_name = structure_mapping(model, target_model)['decoder']
    layers = find_submodule(model, sub_name)
    for i in tqdm(range(len(layers)), desc="replace with eet weight quantize only linear..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_layers(layer, layers=[lora.Linear])            # linear dict
        for name, linear in named_linears.items():
            # print("name:{}, type(linear):{}".format(name, type(linear)))
            if linear.weight.dtype == torch.float16:
                q_linear = W8A16LoraLinear.from_torch(linear, scales=None, init_only=init_only)
            elif linear.weight.dtype == torch.int8:         # bitsandbytes int8
                scales = torch.div(linear.state_dict()["SCB"], 127.0)
                q_linear = W8A16LoraLinear.from_torch(linear, scales=scales, init_only=init_only)
            else:
                raise ValueError("Unsupported data type: {}".format(linear.weight.dtype))
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
