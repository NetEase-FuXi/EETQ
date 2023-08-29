import os
import time
import psutil
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizer,AutoConfig,AutoModelForCausalLM

# model_dir = "decapoda-research/llama-7b-hf"
model_dir = "/root/project/huggingface/llama-7b-hf/"

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def torch_demo():
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)

    model_name = '/root/data/models/2023/llama-13B-v1/'
    MAX_NEW_TOKENS = 32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    
    config.num_hidden_layers = 1
    
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-1}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}    

    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map=0,
      # torch_dtype=torch.float16,
      load_in_8bit=True,
      # load_in_4bit=True,
      max_memory=max_memory,
      # config=config,
    )
    model.eval()
    # for k, v in model.state_dict().items():
    #     if isinstance(v, torch.Tensor):
    #         print(k, v.shape, v.dtype, v.device)

    from eetq.utils import replace_with_eet_qlinear
    replace_with_eet_qlinear(model)
    model.cuda()

    text = '中国的首都在'
    kwargs = {
        "input_text": str(text),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "do_sample": bool(False),
        # "num_beams": 8,
        "temperature": float(0.75),
        "top_k": int(1),
        "top_p": float(0.7),
        "use_cache": bool(True),
    }

    inputs = tokenizer(kwargs["input_text"], return_tensors='pt')
    # input_ids = torch.randint(1000, 8000, (1, 332), dtype=torch.long, device='cuda')
    kwargs["inputs"] = inputs.input_ids.to('cuda')
    del kwargs["input_text"]

    for i in range(1):
        print('i:', i)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            generate_ids = model.generate(**kwargs)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        print("time:", t2 - t1)

    outputs_str = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    print("ori_str", outputs_str)

    print("***********************************")
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))

def eet_demo():
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)

    model_name = '/root/data/models/2023/llama-13B-v1/'
    MAX_NEW_TOKENS = 32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    #config.num_hidden_layers = 1

    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-1}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}    

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    from eetq.utils import replace_with_eet_qlinear
    replace_with_eet_qlinear(model)
    model.to("cuda:0")
    # for k, v in model.state_dict().items():
    #     print(k, v.shape, v.dtype, v.device)


    text = '中国的首都在'
    kwargs = {
        "input_text": str(text),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "do_sample": bool(False),
        # "num_beams": 1,
        "temperature": float(0.75),
        "top_k": int(1),
        "top_p": float(0.7),
        "use_cache": bool(True),
    }

    inputs = tokenizer(kwargs["input_text"], return_tensors='pt')
    kwargs["inputs"] = inputs.input_ids.to('cuda:0')

    # input_ids = torch.randint(1000, 8000, (1, 332), dtype=torch.long, device='cuda')
    # kwargs["inputs"] = input_ids.cuda()
    del kwargs["input_text"]

    # warm up
    generate_ids = model.generate(**kwargs)

    for i in range(1):
        print('i:', i)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            generate_ids = model.generate(**kwargs)
            # generate_ids = model.generate(input_ids, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        print("time:", t2 - t1)

    outputs_str = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    print("ori_str", outputs_str)

    print("***********************************")
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))


if __name__ == "__main__":
    # torch_demo()
    eet_demo()
