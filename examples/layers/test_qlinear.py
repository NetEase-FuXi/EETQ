import psutil
import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
import math

from eetq.modules.qlinear import W8A16Linear

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    set_random_seed(1)
    M = 128
    N = 4096
    K = 1024
    torch_linear = nn.Linear(K, N, bias=False, dtype=torch.float16)
    eet_linear = W8A16Linear.from_linear(torch_linear, scales=None, init_only=False)
    input = torch.rand(M, K, dtype=torch.float16).cuda()
    output = torch.zeros(M, N, dtype=torch.float16).cuda()
    eet_linear(input, output)
    print("eet out: ", output)

    # test torch matmul
    torch_linear = torch_linear.cuda().to(torch.float16)
    output_torch = torch_linear(input)
    print("out torch: ", output_torch)

