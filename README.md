# EETQ
<div align='right' ><font size="1"><b><a href="./README_zh.md">中文README</a></b> </font></div>
Easy & Efficient Quantization for Transformers

## Table of Contents
- [EETQ](#EETQ)
- [Features](#features)
- [Getting started](#getting-started)
    - [Environment](#environment)
    - [Installation](#installation)
    - [Usage](#usage)
- [Examples](#examples)
- [Performance](#performance)

## Features
- W8A16 Quantization
- Flash Attention V2
- Cutlass gemm/kernels Acceleration
- Save and load EETQ quantized model


## Getting started

### Environment

* cuda:>=11.1
* python:>=3.8 
* gcc:>= 7.4.0 
* torch:>=1.14.0 
* transformers:>=4.27.0

Mirror: hub.fuxi.netease.com/danlu-modelserving/eet:eet_llama_v2

### Installation
https://gitlab.fuxi.netease.com:8081/zhaosida/eetq.git

```bash
$ git clone https://github.com/NetEase-FuXi/EET.git
$ pip install .

```

### Usage

1. Quantize float16 model from huggingface.co and speed up inference
```python
...
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
from eetq.utils import eet_accelerator
eet_accelerator(model, quantize=True, fused_attn=True, dev="cuda:0")
model.to("cuda:0")

# inference
res = model.generate(...)

```

2. Support save and load quantized EETQ model
```python
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
from eetq.utils import eet_accelerator
eet_accelerator(model, quantize=True, fused_attn=True, dev="cuda:0")
# save
torch.save(model, "eetq_llama13B_model.pt")

# load
model = torch.load("eetq_llama13B_model_fused_attn_v2.pt",map_location="cuda:0")
res = model.generate(...)

```

## Examples

Model:
- [examples/models/llama_transformers_example.py](examples/models/llama_transformers_example.py)

Layer:
- [examples/layers/test_qlinear.py](examples/layers/test_qlinear.py)
- [examples/layers/test_w8a16_gemm.py](examples/layers/test_w8a16_gemm.py)

## Performance

- llama-13b (test on 3090)

| Methods | Sequence length (tokens) | Latency (ms) | GPU Memory (GB) |
| :----: | :----: | :----: | :----: |
| original | input=7,output=32 |  |  |
| quantize=False, fused_attn=True | input=7,output=32 |  |  |
| quantize=True, fused_attn=False | input=7,output=32 | 0.996 | 14.061 |
| quantize=True, fused_attn=True | input=7,output=32 | 0.721 | 13.956 |
