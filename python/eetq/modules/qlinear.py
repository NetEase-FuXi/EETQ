import psutil
import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
import math

from EETQ import quant_weights, preprocess_weights, w8_a16_gemm


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
    def from_linear(cls, linear, scales=None, init_only=False):
        eet_qlinear = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        if init_only:   # just prepare for loading weights
            return eet_qlinear

        if linear.bias is not None:
            eet_qlinear.bias = linear.bias.clone().half()
        
        data_type = linear.weight.dtype
        int8_weight = torch.t(linear.weight).contiguous()
        if data_type == torch.int8:
            assert scales is not None   # need scales for real quantization
            int8_weight = preprocess_weights(int8_weight)
        elif data_type == torch.float16:
            int8_weight, scales = quant_weights(int8_weight, torch.int8, False)
        else:
            raise ValueError("Unsupported data type: {}".format(data_type))
        eet_qlinear.qweight = int8_weight.cuda()
        eet_qlinear.weight_scales = scales.half().cuda()

        return eet_qlinear

    @classmethod
    def from_pretrained(cls, model_dict, bias=True, dev="cuda:0"):
        self.qweight = torch.t(model_dict["qweight"]).cuda().contiguous()
        self.weight_scales = model_dict["weight_scales"].cuda()
        self.bias = model_dict["bias"].cuda() if bias else None
        return eet_qlinear

    @torch.no_grad()
    def forward(self, input, output):
        w8_a16_gemm(input, self.qweight, self.weight_scales, output)
        output = output + self.bias if self.bias is not None else output
        return output

