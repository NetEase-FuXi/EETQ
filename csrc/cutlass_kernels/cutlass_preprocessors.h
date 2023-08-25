#pragma once
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include <cstddef>
#include <cstdint>

namespace fastertransformer {

void preprocess_weights(int8_t *preprocessed_quantized_weight,
                        const int8_t *row_major_quantized_weight, size_t rows,
                        size_t cols, bool is_int4, int arch);

} // namespace fastertransformer
