#include <torch/extension.h>

torch::Tensor preprocess_weights_cuda(torch::Tensor &ori_weight,
                                      bool is_int4);

void fpA_intB_gemm_forward_cuda(torch::Tensor &input,
                                torch::Tensor &weight,
                                torch::Tensor &scale,
                                torch::Tensor &output,
                                int m, int n, int k);