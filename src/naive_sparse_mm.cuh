#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <torch/extension.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#include "naive_reducer.cuh"

template <typename scalar_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_kernel(const int *rowptr_data, const int *col_data,
                            const scalar_t *value_data,
                            const scalar_t *mat_data, scalar_t *out_data,
                            int64_t *arg_out_data, int B, int M, int N, int K);

void naive_spmm_wrapper(float *dA_values, int *dA_columns, int *dA_csrOffsets,
				int nnzA, int A_rows, int A_cols,
				torch::Tensor B, int B_rows, int B_cols,
				torch::Tensor C);

void naive_batched_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
                            int a_rows, int a_cols, int b_rows, int b_cols, int batch_dim);

