#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/array.h"
#include "dequantize_kernel.h"
#include <iostream>

template<typename result_type = cutlass::Array<half, 4>, typename source_type = cutlass::Array<uint8_t, 4>>
__device__ static result_type convert(source_type const& source)
{
    result_type result;
    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t mask_for_elt_01     = 0x5250;
    static constexpr uint32_t mask_for_elt_23     = 0x5351;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
    return result;
}


__device__ const int reverse_permutation[32] = {0, 1, 4, 5, 8, 9, 12, 13,
                                 2, 3, 6, 7, 10, 11, 14, 15,
                                 16, 17, 20, 21, 24, 25, 28, 29,
                                 18, 19, 22, 23, 26, 27, 30, 31};


template<const int THREAD_NUM=256, const int TILE_DIM = 32>
__global__ void transpose_row_interleave_scale(const half *A, half *B, const half *scale, int m, int n)
{
    __shared__ half S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    const int NUM_ITERS = TILE_DIM * TILE_DIM / THREAD_NUM;
    const int NUM_ROWS = THREAD_NUM / TILE_DIM;

    #pragma unroll
    for(int i = 0; i < NUM_ITERS; i++){
        int thread_row = threadIdx.x / TILE_DIM + i * NUM_ROWS;
        int thread_col = threadIdx.x % TILE_DIM;

        int nx1 = bx + thread_col;
        int ny1 = by + thread_row;
        if(nx1 < n && ny1 < m)
            S[thread_row][thread_col] = A[ny1 * n + nx1];
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < NUM_ITERS; i++){
        int thread_row = threadIdx.x / TILE_DIM + i * NUM_ROWS;
        int permuted_row = reverse_permutation[thread_row];
        int thread_col = threadIdx.x % TILE_DIM;

        int nx2 = by + thread_col;
        int ny2 = bx + thread_row;
        if (nx2 < m && ny2 < n)
        {
            B[ny2 * m + nx2] = S[thread_col][permuted_row] * scale[nx2];
        }
    }
}


__global__ void permute_cast(half* dequantized_weight, const uint8_t* weight, int m, int n){
    half* out = &dequantized_weight[blockIdx.x * n];
    const uint8_t* in = &weight[blockIdx.x * n];
    const int interleave_block_size = 64;
    const int block_per_row = n / interleave_block_size;


    for(int i = threadIdx.x; i * 4 < n; i += blockDim.x){
        cutlass::Array<half, 4> output;
        
        int col_offset_global = i * 4;
        int col_offset_local  = col_offset_global % interleave_block_size;
        int col_index         = col_offset_global / interleave_block_size;
        int global_index      = blockIdx.x * block_per_row + col_index;
        int is_second         = global_index % 2;

        int origin_row        = global_index / (block_per_row * 2) * 2 + is_second;
        int origin_col        = (global_index / 2) % block_per_row;

        output = convert(reinterpret_cast<const uint32_t*>(&in[col_offset_global])[0]);
        uint64_t* t = reinterpret_cast<uint64_t*>(&output);
        
        half* out = &dequantized_weight[origin_row * n];
        reinterpret_cast<uint64_t*>(&out[col_offset_local + origin_col * interleave_block_size])[0] = t[0];
    }
}



void invoke_dequantize(half* dequantized_weight,
                       const uint8_t*     weight,
                       const half*        scale,
                       int                    m,
                       int                    n)
{
    half* tmp;
    cudaMalloc(&tmp, m * n * sizeof(half));
    dim3 block(std::min(256, m / 4));
    dim3 grid(n);
    permute_cast<<<grid, block>>>(tmp, weight, n, m);

    constexpr int BLOCK_SZ = 32;
    dim3 block_0(256);
    dim3 grid_0((m + BLOCK_SZ - 1) / BLOCK_SZ, (n + BLOCK_SZ - 1) / BLOCK_SZ);
    transpose_row_interleave_scale<<<grid_0, block_0>>>(tmp, dequantized_weight, scale, n, m);
    // cudaMemcpy(dequantized_weight, tmp, m * n * sizeof(half), cudaMemcpyDeviceToDevice);
    cudaFree(tmp);
}