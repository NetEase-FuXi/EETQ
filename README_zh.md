# EETQ

EETQ(Easy & Efficient Quantization for Transformers)是一款针对transformer模型的量化工具

## 目录

- [EETQ](#eetq)
  - [目录](#目录)
  - [特点](#特点)
  - [快速开始](#快速开始)
    - [环境](#环境)
    - [安装](#安装)
    - [使用](#使用)

## 特点

- 高性能的INT8权重训练后量化算子

* 提取自[FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/main/src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm)的高性能GEMM内核，可以更加方便集成至您的项目中

* 无需量化感知训练

- 使用[Flash-Attention V2](https://github.com/Dao-AILab/flash-attention)优化attention的推理性能

- 简单易用，只需一行代码即可适配您的PyTorch模型
## 快速开始

### 环境

* cuda:>=11.1
* python:>=3.8 
* gcc:>= 7.4.0 
* torch:>=1.14.0 
* transformers:>=4.27.0

### 安装
推荐使用Dockerfile.
```bash
$ git clone https://github.com/NetEase-FuXi/EETQ.git
$ git submodule update --init --recursive
$ pip install .
```

### 使用

1. 将huggingface的float16的模型量化并加速推理
```python
...
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
from eetq.utils import eet_accelerator
eet_accelerator(model, quantize=True, fused_attn=True, dev="cuda:0")
model.to("cuda:0")

# 推理
res = model.generate(...)

```

2. 支持保存和加载EETQ优化后的模型
```python
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
from eetq.utils import eet_accelerator
eet_accelerator(model, quantize=True, fused_attn=True, dev="cuda:0")
# 保存
torch.save(model, "eetq_llama13B_model.pt")

#加载
model = torch.load("eetq_llama13B_model_fused_attn_v2.pt",map_location="cuda:0")
res = model.generate(...)

```