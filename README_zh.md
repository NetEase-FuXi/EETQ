# EETQ

EETQ(Easy & Efficient Quantization for Transformers)是一款针对transformer模型的量化工具

## 快速开始

### 环境

* cuda:>=11.1
* python:>=3.8 
* gcc:>= 7.4.0 
* torch:>=1.14.0 
* transformers:>=4.27.0

镜像：hub.fuxi.netease.com/danlu-modelserving/eet:eet_llama_v2

### 安装
```bash
$ git clone https://gitlab.fuxi.netease.com:8081/zhaosida/eetq.git
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