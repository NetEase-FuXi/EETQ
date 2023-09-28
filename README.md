# EETQ
<div align='right' ><font size="1"><b><a href="./README_zh.md">中文README</a></b> </font></div>
Easy & Efficient Quantization for Transformers

## Table of Contents
- [EETQ](#eetq)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting started](#getting-started)
    - [Environment](#environment)
    - [Installation](#installation)
    - [Usage](#usage)
  - [Examples](#examples)
  - [Performance](#performance)

## Features
- INT8 weight only PTQ
  * High-performance GEMM kernels from FasterTransformer, [original code](https://github.com/NVIDIA/FasterTransformer/tree/main/src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm)
  * No need for quantization training
- Optimized attention layer using [Flash-Attention V2](https://github.com/Dao-AILab/flash-attention)
- Easy to use, adapt to your pytorch model with one line of code


## Getting started

### Environment

* cuda:>=11.1
* python:>=3.8 
* gcc:>= 7.4.0 
* torch:>=1.14.0 
* transformers:>=4.27.0

The above environment is the minimum configuration, and it is best to use a newer version.

### Installation
Recommend using Dockerfile.


```bash
$ git clone https://github.com/NetEase-FuXi/EETQ.git
$ cd EETQ/
$ git submodule update --init --recursive
$ pip install .
```
If your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might
run too many parallel compilation jobs that could exhaust the amount of RAM. To
limit the number of parallel compilation jobs, you can set the environment
variable `MAX_JOBS`:
```bash
$ MAX_JOBS=4 pip install .
```

### Usage
1. Quantize torch model
```python
from eetq.utils import eet_quantize
eet_quantize(torch_model)
```

2. Quantize torch model and optimize with flash attention
```python
...
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
from eetq.utils import eet_accelerator
eet_accelerator(model, quantize=True, fused_attn=True, dev="cuda:0")
model.to("cuda:0")

# inference
res = model.generate(...)
```

3. Use EETQ in TGI([text-generation-inference](https://github.com/huggingface/text-generation-inference))
see [this](https://github.com/huggingface/text-generation-inference/pull/1018)
```bash
--quantize eetq
```

## Examples

Model:
- [examples/models/llama_transformers_example.py](examples/models/llama_transformers_example.py)

## Performance

- llama-13b (test on 3090)

<img src="./docs/images/benchmark.png" style="zoom:50%;" />