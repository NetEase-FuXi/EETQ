#include <torch/extension.h>

void dequantize_weight_cuda(torch::Tensor _dequantized_weight, torch::Tensor _weight, torch::Tensor _scale);
