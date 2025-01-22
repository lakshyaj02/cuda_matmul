#include <iostream>
#include <array>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <sys/time.h>

#define NUM_THREADS (64)

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp ();

__global__ void dummyKernel();
__global__ void check_equal(float *d_Arr, float *h_Arr, size_t rows, size_t cols);
__global__ void copySharedMem(float *odata, const float *idata);

void dummy_kernel_launch();

void dense_to_sparse_blockedell(cusparseHandle_t handle, torch::Tensor A, float *h_ell_values, int *hA_columns,
                            int A_num_rows, int A_num_cols, int A_ell_blocksize, int A_ell_cols, int num_blocks);

void blocksparse_mm_wrapper(cusparseHandle_t handle,
                         float *hA_values, int *hA_columns,
                         int A_num_rows, int A_num_cols, int A_ell_blocksize, int A_ell_cols, int num_blocks,
                         torch::Tensor B, int B_rows, int B_cols,
                         torch::Tensor C);

void dense_to_blocksparse(cusparseHandle_t handle,  float *dA_values, int *dA_columns,
                  const int num_rows, const int num_cols, const int ell_blk_size, const int ell_width);

void dense_to_csr(cusparseHandle_t handle, 
                  torch::Tensor dense, const int num_rows, const int num_cols,
                  float **d_csr_values, int **d_csr_columns, int **d_csr_offsets, int *nnz);

void free_csr(float *d_csr_values, int *d_csr_columns, int *d_csr_offsets);
