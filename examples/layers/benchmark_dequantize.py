import itertools
from typing import Optional, Tuple, Union


from regex import F
from sympy import RisingFactorial
import torch
import triton
from torch import nn
# from vllm import _custom_ops as vllm_ops
from EETQ import quant_weights, w8_a16_gemm, dequantize_weight


def w8a16_dequant(qweight, scale):
    dtype = torch.float16
    device = qweight.device
    I = torch.eye(qweight.shape[0], dtype=dtype, device=device)
    gemm_dequantized_weight = w8_a16_gemm(I, qweight, scale)
    return gemm_dequantized_weight


def dequant_weight(qweight, scale):
    dtype = torch.float16
    device = qweight.device
    dequantized_weight = torch.zeros_like(qweight, dtype=dtype, device=device)
    dequantize_weight(dequantized_weight, qweight, scale)
    return dequantized_weight


def calculate_diff(m, n):
    dtype = torch.float16
    device = "cuda"
    weight = torch.randn(m, n, dtype=dtype)
    qweight, scale = quant_weights(weight, torch.int8, False)
    qweight = qweight.to(device)
    scale = scale.to(device)


    # Calculate the dequantized weight using w8_a16_gemm
    gemm_dequantized_weight = w8a16_dequant(qweight, scale)


    # Calculate the dequantized weight using dequantize_weight
    dequantized_weight = dequant_weight(qweight, scale)


    if torch.allclose(
        gemm_dequantized_weight, dequantized_weight, atol=1e-2, rtol=1e-2
    ):
        print(f"✅ ({m}, {n}) implementations match")
    else:
        print(f"❌ ({m}, {n}) Implementations differ")
    del gemm_dequantized_weight
    del dequantized_weight



M = [2048 + i * 1024 for i in range(0, 11, 2)]
N = [2048 + i * 1024 for i in range(0, 11, 2)]


configs = list(itertools.product(M, N))


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["w8a16", "dequant"],
            line_names=["w8a16", "dequant"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name=f"dequantized_performance",
            args={},
        )
    )
    def benchmark(m, n, provider):
        dtype = torch.float16
        device = "cuda"
        weight = torch.randn(m, n, dtype=dtype)
        qweight, scale = quant_weights(weight, torch.int8, False)
        qweight = qweight.to(device)
        scale = scale.to(device)


        quantiles = [0.5, 0.2, 0.8]


        if provider == "w8a16":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: w8a16_dequant(
                    qweight,
                    scale
                ),
                quantiles=quantiles,
            )
        elif provider == "dequant":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: dequant_weight(
                    qweight,
                    scale
                ),
                quantiles=quantiles,
            )



        return 1000 * ms, 1000 * max_ms, 1000 * min_ms


    return benchmark



if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/",
        help="Path to save rmsnorm benchmark results",
    )
    args = parser.parse_args()



    # M = [i * 256 for i in range(1, 50, 2)]
    # N = [i * 256 for i in range(1, 50, 2)]


    # for m in M:
    #     for n in N:
    #         calculate_diff(
    #             m, n
    #         )


    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)