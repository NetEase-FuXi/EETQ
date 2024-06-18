import psutil
import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
import math
from torch.autograd import Function

from EETQ import quant_weights, preprocess_weights, w8_a16_gemm


def quantize_and_preprocess_weights(weight, scales=None):
    data_type = weight.dtype
    int8_weight = torch.t(weight).contiguous().cpu()
    if data_type == torch.int8:
        assert scales is not None   # need scales for real quantization
        int8_weight = preprocess_weights(int8_weight)
    elif data_type == torch.float16:
        int8_weight, scales = quant_weights(int8_weight, torch.int8, False)
    else:
        raise ValueError("Unsupported data type: {}".format(data_type))
    return int8_weight, scales


class W8A16Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dev="cuda:0"):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.int8, device=dev))
        self.register_buffer("weight_scales", torch.zeros((out_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_torch(cls, linear, scales=None, init_only=False):
        eet_qlinear = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, dev=linear.weight.device)
        if init_only:   # just prepare for loading weights
            return eet_qlinear

        if linear.bias is not None:
            eet_qlinear.bias = linear.bias.clone().half()

        int8_weight, scales = quantize_and_preprocess_weights(linear.weight, scales)

        eet_qlinear.qweight = int8_weight.to(linear.weight.device)
        eet_qlinear.weight_scales = scales.half().to(linear.weight.device)
        return eet_qlinear

    @torch.no_grad()
    def forward(self, input):
        output = w8_a16_gemm(input, self.qweight, self.weight_scales)
        output = output + self.bias if self.bias is not None else output
        return output

class EetqLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        weight,
        scales,
        bias=None
    ):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, weight, scales, bias)
        output = w8_a16_gemm(x, weight, scales)
        output = output + bias if bias is not None else output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, scales, bias = ctx.saved_tensors
        identity = torch.eye(weight.shape[0]).to(weight.device).to(input.dtype)

        # Dequantize the weight
        weight = w8_a16_gemm(identity, weight, scales)
        
        if ctx.needs_input_grad[0]:
            # 2D matrix multiplication, unsqueeze to 3D
            grad_input = grad_output.squeeze(0).matmul(
                weight.transpose(0, 1)
            ).unsqueeze(0)

        return grad_input, None, None, None

class EetqLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device="cuda:0"):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((in_features, out_features), dtype=torch.int8, device=device))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=torch.float16, device=device))
        else:
            self.bias = None

    def register(self, buffer_name, tensor):
        self.register_buffer(buffer_name, tensor)

    def register_scale(self, device):
        out_features = self.weight.shape[-1]
        weight_scale = torch.zeros((out_features), dtype=torch.float16, device=device)
        self.register_buffer("weight_scales", weight_scale)

    def forward(self, input):
        if self.training:
            output = EetqLinearMMFunction.apply(input, self.weight, self.weight_scales, self.bias)
        else:
            with torch.no_grad():
                output = EetqLinearMMFunction.apply(input, self.weight, self.weight_scales, self.bias)
        return output


class W8A16LoraLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_A=None,
        lora_B=None,
        scaling=0,
        dev="cuda:0"
    ):
        self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.int8, device=dev))
        self.register_buffer("weight_scales", torch.zeros((out_features), dtype=torch.float16, device=dev))
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.scaling = scaling

    @classmethod
    def from_torch(cls, linear, scales=None, init_only=False):
        print("fan_in_fan_out: ", linear.fan_in_fan_out)
        is_lora = hasattr(linear, "lora_A") and not linear.merged
        if is_lora:
            eet_qlinear = cls(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                lora_A=linear.lora_A[linear.active_adapter],
                lora_B=linear.lora_B[linear.active_adapter],
                scaling=linear.scaling[linear.active_adapter],
            )
        else:
            eet_qlinear = cls(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
            )

        if init_only:   # just prepare for loading weights
            return eet_qlinear

        if linear.bias is not None:
            eet_qlinear.bias = linear.bias.clone().half()

        int8_weight, scales = preprocess_weights(linear.weight, scales)

        eet_qlinear.qweight = int8_weight.cuda()
        eet_qlinear.weight_scales = scales.half().cuda()

        return eet_qlinear

    @torch.no_grad()
    def forward(self, input):
        output = w8_a16_gemm(input, self.qweight, self.weight_scales)
        if self.lora_A is not None:
            output += (
                self.lora_B(self.lora_A(input)) * self.scaling
            )
        output = output + self.bias if self.bias is not None else output

        return output

