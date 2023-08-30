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
    eet_linear = W8A16Linear.from_torch(torch_linear, scales=None, init_only=False)
    input = torch.rand(M, K, dtype=torch.float16).cuda()
    # output = torch.zeros(M, N, dtype=torch.float16).cuda()
    output = eet_linear(input)
    print("eet out: ", output)

    # test torch matmul
    torch_linear = torch_linear.cuda().to(torch.float16)
    output_torch = torch_linear(input)
    print("out torch: ", output_torch)

    print(torch.allclose(output, output_torch, atol=1e-2))

    print("torch state_dict: ", torch_linear.state_dict())
    print("eet state_dict: ", eet_linear.state_dict())

    torch.save(torch_linear.state_dict(), "/root/project/eetq/examples/tests/torch_linear.pt")
    torch.save(torch_linear, "/root/project/eetq/examples/tests/torch_linear_model.pt")
    torch.save(eet_linear.state_dict(), "/root/project/eetq/examples/tests/eet_linear.pt")
    torch.save(eet_linear, "/root/project/eetq/examples/tests/eet_linear_model.pt")

