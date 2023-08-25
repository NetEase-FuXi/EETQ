#include <torch/extension.h>
#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include "fpA_intB_gemm_wrapper.h"
#include "fpA_intB_gemm.h"
#include "cuda_utils.h"

int getWorkspaceSize(const int m, const int n, const int k)
{
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = (m + 31) / 32;
    const int max_grid_n = (n + 127) / 128;
    const int split_k_limit = 7;
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return max_grid_m * max_grid_n * split_k_limit * 4;
}

torch::Tensor preprocess_weights_cuda(torch::Tensor &origin_weight,
                                      bool is_int4)
{
    // guarantee the weight is cpu tensor
    if (origin_weight.device().type() != torch::kCPU)
    {
        std::string err_msg = "preprocess_weights expects weight to be CPU tensor";
        throw std::runtime_error("[EET Error][preprocess_weights_cuda] "+ err_msg);
        // torch::Tensor origin_weight_cpu = origin_weight.to(torch::kCPU);
    }
    torch::Tensor preprocessed_quantized_weight = torch::empty_like(origin_weight);
    int8_t *preprocessed_quantized_weight_ptr = reinterpret_cast<int8_t *>(preprocessed_quantized_weight.data_ptr());
    const int8_t *row_major_quantized_weight_ptr = reinterpret_cast<const int8_t *>(origin_weight.data_ptr());
    size_t rows = origin_weight.size(-2);
    size_t cols = origin_weight.size(-1);
    int arch = fastertransformer::getSMVersion();
    fastertransformer::preprocess_weights(preprocessed_quantized_weight_ptr,
                                          row_major_quantized_weight_ptr,
                                          rows,
                                          cols,
                                          is_int4,
                                          arch);
    return preprocessed_quantized_weight;
}

void fpA_intB_gemm_forward_cuda(torch::Tensor &input,
                                torch::Tensor &weight,
                                torch::Tensor &scale,
                                torch::Tensor &output,
                                int m, int n, int k)
{
    c10::cuda::CUDAGuard device_guard(input.device());
    const fastertransformer::half *input_ptr = reinterpret_cast<fastertransformer::half *>(input.data_ptr());
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight.data_ptr());
    const fastertransformer::half *scale_ptr = reinterpret_cast<fastertransformer::half *>(scale.data_ptr());
    fastertransformer::half *output_ptr = reinterpret_cast<fastertransformer::half *>(output.data_ptr());
    // const int max_size = std::max(n, k);
    // size_t workspace_size = getWorkspaceSize(m, max_size, max_size);
    // void *ptr = nullptr;
    // char *workspace_ptr = workspace_size > 0 ? (char *)cudaMalloc((void **)&ptr, workspace_size) : nullptr;

    fastertransformer::gemm_fp16_int_bias_act(
        input_ptr,
        weight_ptr,
        scale_ptr,
        nullptr,
        output_ptr,
        std::nullopt,
        m, n, k,
        0,
        nullptr,
        0,
        0);
}