/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cutlass_preprocessors.h"
#include "cuda_utils.h"
#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

#include <vector>

namespace fastertransformer {

enum class QuantType { INT8_WEIGHT_ONLY, PACKED_INT4_WEIGHT_ONLY };

int get_bits_in_quant_type(QuantType quant_type) {
  switch (quant_type) {
  case QuantType::INT8_WEIGHT_ONLY:
    return 8;
  case QuantType::PACKED_INT4_WEIGHT_ONLY:
    return 4;
  default:
    return -1;
  }
}

struct LayoutDetails {
    enum class Layout {
        UNKNOWN,
        ROW_MAJOR,
        COLUMN_MAJOR
    };

    Layout layoutB              = Layout::UNKNOWN;
    int    rows_per_column_tile = 1;
    int    columns_interleaved  = 1;

    bool uses_imma_ldsm = false;
};

template<typename Layout>
struct getLayoutDetails {
};

template<>
struct getLayoutDetails<cutlass::layout::RowMajor> {
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::ROW_MAJOR;
        return layout_details;
    }
};

template<>
struct getLayoutDetails<cutlass::layout::ColumnMajor> {
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
        return layout_details;
    }
};

template<int RowsPerTile, int ColumnsInterleaved>
struct getLayoutDetails<cutlass::layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>> {
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB              = LayoutDetails::Layout::COLUMN_MAJOR;
        layout_details.rows_per_column_tile = RowsPerTile;
        layout_details.columns_interleaved  = ColumnsInterleaved;
        return layout_details;
    }
};

template<typename cutlassArch, typename TypeB>
LayoutDetails getLayoutDetailsForArchAndQuantType()
{

    using CompileTraits    = cutlass::gemm::kernel::LayoutDetailsB<TypeB, cutlassArch>;
    using LayoutB          = typename CompileTraits::Layout;
    using MmaOperator      = typename CompileTraits::Operator;
    LayoutDetails details  = getLayoutDetails<LayoutB>()();
    details.uses_imma_ldsm = std::is_same<MmaOperator, cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA>::value;
    return details;
}

template<typename cutlassArch>
LayoutDetails getLayoutDetailsForArch(QuantType quant_type)
{
    LayoutDetails details;
    if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, uint8_t>();
    }
    else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::uint4b_t>();
    }
    else {
        FT_CHECK_WITH_INFO(false, "Unsupported quantization type");
    }
    return details;
}

LayoutDetails getLayoutDetailsForTransform(QuantType quant_type, int arch)
{
    if (arch >= 70 && arch < 75) {
        return getLayoutDetailsForArch<cutlass::arch::Sm70>(quant_type);
    }
    else if (arch >= 75 && arch < 80) {
        return getLayoutDetailsForArch<cutlass::arch::Sm75>(quant_type);
    }
    else if (arch >= 80 && arch < 90) {
        return getLayoutDetailsForArch<cutlass::arch::Sm80>(quant_type);
    }
    else {
        FT_CHECK_WITH_INFO(false, "Unsupported Arch");
        return LayoutDetails();
    }
}

// Permutes the rows of B for Turing and Ampere. Throws an error for other
// architectures. The data is permuted such that: For int8, each group of 16
// rows is permuted using the map below:
//  0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
// For int4, each group of 32 rows is permuted using the map below:
//  0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22
//  23 30 31
void permute_B_rows_for_mixed_gemm(int8_t *permuted_quantized_tensor,
                                   const int8_t *quantized_tensor,
                                   const std::vector<size_t> &shape,
                                   QuantType quant_type,
                                   const int64_t arch_version) {
  const size_t num_rows = shape[0];
  const size_t num_cols = shape[1];

  const int BITS_PER_ELT = get_bits_in_quant_type(quant_type);
  const int K = 16 / BITS_PER_ELT;
  const int ELTS_PER_REG = 32 / BITS_PER_ELT;

  const uint32_t *input_byte_ptr =
      reinterpret_cast<const uint32_t *>(quantized_tensor);
  uint32_t *output_byte_ptr =
      reinterpret_cast<uint32_t *>(permuted_quantized_tensor);

  int MMA_SHAPE_N = 8;
  int B_ROWS_PER_MMA = 8 * K;
  const int elts_in_int32 = 32 / BITS_PER_ELT;

  const int num_vec_cols = num_cols / elts_in_int32;

  FT_CHECK_WITH_INFO(arch_version >= 75,
                     "Unsupported Arch. Pre-volta not supported. Column "
                     "interleave not needed on Volta.");

  FT_CHECK_WITH_INFO(num_rows % B_ROWS_PER_MMA == 0,
                     fmtstr("Invalid shape for quantized tensor. Number of "
                            "rows of quantized matrix must be a multiple of %d",
                            B_ROWS_PER_MMA));

  FT_CHECK_WITH_INFO(
      num_cols % MMA_SHAPE_N == 0,
      fmtstr("Invalid shape for quantized tensor. On turing/Ampere, the number "
             "of cols must be a multiple of %d.",
             MMA_SHAPE_N));

  // The code is written as below so it works for both int8
  // and packed int4.
  for (size_t base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
    for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {

      for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
        const int write_row = base_row + tile_row;
        const int tile_read_row = 8 * (((tile_row % ELTS_PER_REG) / 2)) +
                                  tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
        const int read_row = base_row + tile_read_row;
        const int read_col = write_col;

        const int64_t read_offset = int64_t(read_row) * num_vec_cols + read_col;
        const int64_t write_offset =
            int64_t(write_row) * num_vec_cols + write_col;

        output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
      }
    }
  }
}

// We need to use this transpose to correctly handle packed int4 and int8 data
// The reason this code is relatively complex is that the "trivial" loops took a
// substantial amount of time to transpose leading to long preprocessing times.
// This seemed to be a big issue for relatively large models.
template <QuantType quant_type>
void subbyte_transpose_impl(int8_t *transposed_quantized_tensor,
                            const int8_t *quantized_tensor,
                            const std::vector<size_t> &shape) {
  const int bits_per_elt = get_bits_in_quant_type(quant_type);
  const size_t num_rows = shape[0];
  const size_t num_cols = shape[1];

  const size_t col_bytes = num_cols * bits_per_elt / 8;
  const size_t col_bytes_trans = num_rows * bits_per_elt / 8;

  const uint8_t *input_byte_ptr =
      reinterpret_cast<const uint8_t *>(quantized_tensor);
  uint8_t *output_byte_ptr =
      reinterpret_cast<uint8_t *>(transposed_quantized_tensor);

  static constexpr int ELTS_PER_BYTE =
      quant_type == QuantType::INT8_WEIGHT_ONLY ? 1 : 2;

  static constexpr int M_TILE_L1 = 64;
  static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
  uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

  static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

  // We assume the dims are a multiple of vector width. Our kernels only handle
  // dims which are multiples of 64 for weight-only quantization. As a result,
  // this seemed like a reasonable tradeoff because it allows GCC to emit vector
  // instructions.
  FT_CHECK_WITH_INFO(
      !(col_bytes_trans % VECTOR_WIDTH) && !(col_bytes % VECTOR_WIDTH),
      fmtstr("Number of bytes for rows and cols must be a multiple of %d. "
             "However, num_rows_bytes = %ld and num_col_bytes = %d.",
             VECTOR_WIDTH, col_bytes_trans, col_bytes));

  for (size_t row_tile_start = 0; row_tile_start < num_rows;
       row_tile_start += M_TILE_L1) {
    for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes;
         col_tile_start_byte += N_TILE_L1) {

      const int row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
      const int col_limit =
          std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

      for (int ii = 0; ii < M_TILE_L1; ++ii) {
        const int row = row_tile_start + ii;

        for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
          const int col = col_tile_start_byte + jj;

          const size_t logical_src_offset = row * col_bytes + col;

          if (row < row_limit && col < col_limit) {
            for (int v = 0; v < VECTOR_WIDTH; ++v) {
              cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
            }
          }
        }
      }

      if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          for (int jj = ii + 1; jj < N_TILE_L1; ++jj) {
            std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
          }
        }
      } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {

        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          // Using M_TILE_L1 here is deliberate since we assume that the cache
          // tile is square in the number of elements (not necessarily the
          // number of bytes).
          for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
            const int ii_byte = ii / ELTS_PER_BYTE;
            const int ii_bit_offset = ii % ELTS_PER_BYTE;

            const int jj_byte = jj / ELTS_PER_BYTE;
            const int jj_bit_offset = jj % ELTS_PER_BYTE;

            uint8_t src_elt =
                0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
            uint8_t tgt_elt =
                0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

            cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
            cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

            cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
            cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
          }
        }
      } else {
        FT_CHECK_WITH_INFO(false, "Unsupported quantization type.");
      }

      const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
      const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

      const int row_limit_trans =
          std::min(row_tile_start_trans + M_TILE_L1, num_cols);
      const int col_limit_trans =
          std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

      for (int ii = 0; ii < M_TILE_L1; ++ii) {
        const int row = row_tile_start_trans + ii;
        for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
          const int col = col_tile_start_byte_trans + jj;

          const size_t logical_tgt_offset = row * col_bytes_trans + col;

          if (row < row_limit_trans && col < col_limit_trans) {
            for (int v = 0; v < VECTOR_WIDTH; ++v) {
              output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
            }
          }
        }
      }
    }
  }
}

void subbyte_transpose(int8_t *transposed_quantized_tensor,
                       const int8_t *quantized_tensor,
                       const std::vector<size_t> &shape, QuantType quant_type) {

  if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
    subbyte_transpose_impl<QuantType::INT8_WEIGHT_ONLY>(
        transposed_quantized_tensor, quantized_tensor, shape);
  } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
    subbyte_transpose_impl<QuantType::PACKED_INT4_WEIGHT_ONLY>(
        transposed_quantized_tensor, quantized_tensor, shape);
  } else {
    FT_CHECK_WITH_INFO(false, "Invalid quant_tye");
  }
}

void add_bias_and_interleave_int8s_inplace(int8_t *int8_tensor,
                                           const size_t num_elts) {
  for (size_t ii = 0; ii < num_elts; ++ii) {
    int8_tensor[ii] = int8_t(int(int8_tensor[ii]) + 128);
  }

  // Step 2 will transform the layout of a 32-bit register in CUDA in order to
  // match the int4 layout. This has no performance benefit and is purely so
  // that int4 and int8 have the same layout. Pictorially, this does the
  // following: bit 32                                                      0
  //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
  //
  // And it will rearrange the output 32 bit register to be the following:
  // bit 32                                                      0
  //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

  FT_CHECK_WITH_INFO(num_elts % 4 == 0, "Dimensions of int8 tensor must be a "
                                        "multiple of 4 for register relayout");
  for (size_t base = 0; base < num_elts; base += 4) {
    std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
  }
}

void add_bias_and_interleave_int4s_inplace(int8_t *packed_int4_tensor,
                                           const size_t num_elts) {
  const size_t num_bytes = num_elts / 2;

  // Step 1 will be to transform all the int4s to unsigned in order to make the
  // dequantize take as little instructions as possible in the CUDA code.
  for (size_t ii = 0; ii < num_bytes; ++ii) {
    int8_t transformed_packed_int4s = 0;
    int8_t transformed_first_elt =
        (int8_t(packed_int4_tensor[ii] << 4) >> 4) +
        8; // The double shift here is to ensure sign extension
    int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4) + 8;

    FT_CHECK_WITH_INFO(transformed_first_elt >= 0 &&
                           transformed_first_elt <= 15,
                       "Illegal result for int4 transform (first elt)");
    FT_CHECK_WITH_INFO(transformed_second_elt >= 0 &&
                           transformed_second_elt <= 15,
                       "Illegal result for int4 transform (second elt)");

    // We don't need to mask in these ops since everything should be in the
    // range 0-15
    transformed_packed_int4s |= transformed_first_elt;
    transformed_packed_int4s |= (transformed_second_elt << 4);
    packed_int4_tensor[ii] = transformed_packed_int4s;
  }

  // Step 2 will transform the layout of a 32-bit register in CUDA in order to
  // minimize the number of shift & logical instructions That are needed to
  // extract the int4s in the GEMM main loop. Pictorially, the loop below will
  // do the following: Take as input a 32 bit register with layout: bit 32 0
  //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt
  //      occupies 4 bits)
  //
  // And it will rearrange the output 32 bit register to be the following:
  // bit 32                                                      0
  //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt
  //      occupies 4 bits)

  FT_CHECK_WITH_INFO(num_bytes % 4 == 0, "Dimensions of int4 tensor must be a "
                                         "multiple of 8 for register relayout");
  const size_t num_registers = num_bytes / 4;

  uint32_t *register_ptr = reinterpret_cast<uint32_t *>(packed_int4_tensor);
  for (size_t ii = 0; ii < num_registers; ++ii) {
    const uint32_t current_register = register_ptr[ii];
    uint32_t transformed_register = 0;

    for (int dest_idx = 0; dest_idx < 8; ++dest_idx) {
      const int src_idx = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
      const int src_shift = 4 * src_idx;
      const int dest_shift = 4 * dest_idx;

      const uint32_t src_bits = (current_register >> src_shift) & 0xF;
      transformed_register |= (src_bits << dest_shift);
    }
    register_ptr[ii] = transformed_register;
  }
}

void add_bias_and_interleave_quantized_tensor_inplace(int8_t *tensor,
                                                      const size_t num_elts,
                                                      QuantType quant_type) {
  if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
    add_bias_and_interleave_int8s_inplace(tensor, num_elts);
  } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
    add_bias_and_interleave_int4s_inplace(tensor, num_elts);
  } else {
    FT_CHECK_WITH_INFO(false, "Invalid quantization type for interleaving.");
  }
}

void interleave_column_major_tensor(int8_t *interleaved_quantized_tensor,
                                    const int8_t *quantized_tensor,
                                    const std::vector<size_t> &shape,
                                    QuantType quant_type,
                                    LayoutDetails details) {
  // We only want to run this step for weight only quant.
  FT_CHECK(quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY ||
           quant_type == QuantType::INT8_WEIGHT_ONLY);
  FT_CHECK_WITH_INFO(shape.size() == 2, "Shape must be 2-D");

  const size_t num_rows = shape[0];
  const size_t num_cols = shape[1];

  const int BITS_PER_ELT = get_bits_in_quant_type(quant_type);
  const int elts_in_int32 = 32 / BITS_PER_ELT;

  const int rows_per_tile = details.rows_per_column_tile;

  FT_CHECK_WITH_INFO(!(num_rows % elts_in_int32),
                     fmtstr("The number of rows must be a multiple of %d but "
                            "the number of rows is %d.",
                            elts_in_int32, num_rows));

  FT_CHECK_WITH_INFO(!(num_cols % rows_per_tile),
                     fmtstr("The number of columns must be a multiple of %d "
                            "but the number of columns is %ld",
                            rows_per_tile, num_cols));

  const uint32_t *input_byte_ptr =
      reinterpret_cast<const uint32_t *>(quantized_tensor);
  uint32_t *output_byte_ptr =
      reinterpret_cast<uint32_t *>(interleaved_quantized_tensor);

  FT_CHECK_WITH_INFO(!(num_cols % rows_per_tile),
                     fmtstr("The number of columns must be a multiple of %d "
                            "but the number of columns is %d.",
                            rows_per_tile, num_cols));

  const int num_vec_rows = num_rows / elts_in_int32;
  const int vec_rows_per_tile = rows_per_tile / elts_in_int32;
  const int interleave = details.columns_interleaved;

  for (size_t read_col = 0; read_col < num_cols; ++read_col) {
    const auto write_col = read_col / interleave;
    for (int base_vec_row = 0; base_vec_row < num_vec_rows;
         base_vec_row += vec_rows_per_tile) {
      for (int vec_read_row = base_vec_row;
           vec_read_row <
           std::min(num_vec_rows, base_vec_row + vec_rows_per_tile);
           ++vec_read_row) {
        const int64_t vec_write_row =
            interleave * base_vec_row +
            vec_rows_per_tile * (read_col % interleave) +
            vec_read_row % vec_rows_per_tile;

        const int64_t read_offset =
            int64_t(read_col) * num_vec_rows + vec_read_row;
        const int64_t write_offset =
            int64_t(write_col) * num_vec_rows * interleave + vec_write_row;
        output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
      }
    }
  }
}

void preprocess_weights_for_mixed_gemm(int8_t *preprocessed_quantized_weight,
                                       const int8_t *row_major_quantized_weight,
                                       const std::vector<size_t> &shape,
                                       QuantType quant_type, int arch) {
  LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

  FT_CHECK_WITH_INFO(shape.size() == 2, "Shape must be 2-D");

  size_t num_elts = 1;
  for (const auto &dim : shape) {
    num_elts *= dim;
  }

  const size_t num_bytes = num_elts * get_bits_in_quant_type(quant_type) / 8;

  std::vector<int8_t> src_buf(num_bytes);
  std::vector<int8_t> dst_buf(num_bytes);
  std::copy(row_major_quantized_weight, row_major_quantized_weight + num_bytes,
            src_buf.begin());

  // Works on row major data, so issue this permutation first.
  if (details.uses_imma_ldsm) {
    permute_B_rows_for_mixed_gemm(dst_buf.data(), src_buf.data(), shape,
                                  quant_type, arch);
    src_buf.swap(dst_buf);
  }

  if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR) {
    subbyte_transpose(dst_buf.data(), src_buf.data(), shape, quant_type);
    src_buf.swap(dst_buf);
  }

  if (details.columns_interleaved > 1) {
    interleave_column_major_tensor(dst_buf.data(), src_buf.data(), shape,
                                   quant_type, details);
    src_buf.swap(dst_buf);
  }

  add_bias_and_interleave_quantized_tensor_inplace(src_buf.data(), num_elts,
                                                   quant_type);
  std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
}

void preprocess_weights(int8_t *preprocessed_quantized_weight,
                        const int8_t *row_major_quantized_weight, size_t rows,
                        size_t cols, bool is_int4, int arch) {
  QuantType qtype = is_int4 ? QuantType::PACKED_INT4_WEIGHT_ONLY
                            : QuantType::INT8_WEIGHT_ONLY;
  preprocess_weights_for_mixed_gemm(preprocessed_quantized_weight,
                                    row_major_quantized_weight, {rows, cols},
                                    qtype, arch);
}

} // namespace fastertransformer
