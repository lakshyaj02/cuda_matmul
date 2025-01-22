/*
 Intermediate connection between python front-end and CUDA back-end
 Use this file mostly for forward declaration ands calling CUDA functions.
 The pybind11 linker is at the bottom of the file.

 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/
#include <iostream>
#include <assert.h>
#include <unordered_map>
#include <string>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iterator>

#include <cusparse_v2.h>
#include <ATen/cuda/CUDABlas.h>

#include "baseline_mm.cuh"
#include "naive_sparse_mm.cuh"

// dummy kernel forward declaration
void dummy_kernel_launch();

// blocksparse mm forward declaration
// void blocksparse_mm_wrapper(cusparseHandle_t handle, float *dA_values, int *dA_columns, 
                        // int A_num_rows, int A_num_cols, int A_ell_blocksize, int A_ell_cols, int A_num_blocks,
                        //  torch::Tensor B, int B_rows, int B_cols,
                        //  torch::Tensor C);

// void dense_to_blocksparse(cusparseHandle_t handle, torch::Tensor dense, const int num_rows, const int num_cols, const int ell_blk_size, const int ell_width);

// manual handles in case none exist.
// initialization/destruction functions are near the bottom
cusparseHandle_t g_cusparse_handle = nullptr;

// BlockSparse HANDLE STRUCTURE
struct BlockSparse_handle {
  int M;
  int N;
  int K;

  int *colind;
  float *values;

  int ell_blocksize;
  int a_num_blocks;
  int ell_cols;
  };

int blocksparse_handle_len = 0;
BlockSparse_handle *blocksparse_handle = nullptr;
std::unordered_map<std::string, int> blocksparse_layer_lookup;

void blocksparse_inspect(torch::Tensor values, torch::Tensor colindex, int M, int N, int K, int block_size, int ell_cols, int num_blocks, std::string layer)
{
  int *_colindex = colindex.data_ptr<int>();
  float *_value = values.data_ptr<float>();

  BlockSparse_handle *cur_handle = new BlockSparse_handle;

  cur_handle->M = M;
  cur_handle->N = N;
  cur_handle->K = K;    

  cur_handle->colind = _colindex;
  cur_handle->values = _value;
  
  cur_handle->ell_blocksize = block_size;
  cur_handle->a_num_blocks = num_blocks;
  cur_handle->ell_cols = ell_cols;

  blocksparse_handle_len++;
  blocksparse_handle = (BlockSparse_handle*) realloc(blocksparse_handle, blocksparse_handle_len * sizeof(BlockSparse_handle));
  blocksparse_handle[blocksparse_handle_len - 1] = *cur_handle;
  blocksparse_layer_lookup[layer] = blocksparse_handle_len - 1;
}

torch::Tensor blocksparse_mmul(torch::Tensor A, torch::Tensor A_columns, int A_rows, int A_cols, int block_size, int ell_cols, int num_blocks, torch::Tensor B_val, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes a sparse A and a dense B.
  int B_rows = B.size(0);
  int B_cols = B.size(1);
  
  int *hA_columns = A_columns.data_ptr<int>();

  blocksparse_inspect(A, A_columns, A_rows, A_cols, B_cols, block_size, ell_cols, num_blocks, "blocksparse_mmul");

  const int nnz = ell_cols * A_rows;
  float h_ell_values[nnz] = {};

  auto handle_conv = at::cuda::getCurrentCUDASparseHandle();

  dense_to_sparse_blockedell(handle_conv, A, h_ell_values, hA_columns,
                            A_rows, A_cols, block_size, ell_cols, num_blocks);

  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // cusparse_mm_wrapper(g_cusparse_handle, d_A_values, d_A_columns, d_A_offsets, nnzA, A_rows, A_cols, B, B_rows, B_cols, C);
  blocksparse_mm_wrapper(handle, h_ell_values, hA_columns, A_rows, A_cols, block_size, ell_cols, num_blocks, B_val, B_rows, B_cols, C);
  // blocksparse_mm_wrapper(handle, A_values, A_columns, A_rows, A_cols, block_size, ell_cols, num_blocks, B, B_cols, B_rows, C);  

  return C;
}

torch::Tensor naive_spmm(torch::Tensor A_values, torch::Tensor A_columns, torch::Tensor A_offsets, int nnzA, int A_rows, int A_cols, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes two dense matrices, and sparsifies A.
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  float *d_A_values = A_values.data_ptr<float>();
  int *d_A_columns = A_columns.data_ptr<int>();
  int *d_A_offsets = A_offsets.data_ptr<int>();

  naive_spmm_wrapper(d_A_values, d_A_columns, d_A_offsets, nnzA, A_rows, A_cols, B, B_rows, B_cols, C);
  return C;
}

torch::Tensor naive_spmm_csr_conversion(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes two dense matrices, and sparsifies A.
  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  float *d_A_values = nullptr;
  int *d_A_columns = nullptr;
  int *d_A_offsets = nullptr;
  int nnzA = 0;

  dense_to_csr(g_cusparse_handle, A, A_rows, A_cols, &d_A_values, &d_A_columns, &d_A_offsets, &nnzA);

  naive_spmm_wrapper(d_A_values, d_A_columns, d_A_offsets, nnzA, A_rows, A_cols, B, B_rows, B_cols, C);

  free_csr(d_A_values, d_A_columns, d_A_offsets);
  return C;
}

void blocksparse_free() {
  for (int i = 0; i < blocksparse_handle_len; i++) {
    cudaFree(blocksparse_handle[i].colind);
    cudaFree(blocksparse_handle[i].values);
  }
  free(blocksparse_handle);
  blocksparse_handle_len = 0;
  blocksparse_handle = nullptr;
}    

void init_blocksparse_handle() {
  cusparseStatus_t status = cusparseCreate(&g_cusparse_handle);
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    std::cerr << "blockSPARSE initialization error.";
  }
}

void destroy_blocksparse_handle() {
  cusparseStatus_t status = cusparseDestroy(g_cusparse_handle);
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    std::cerr << "Shutdown error!";
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("blocksparse_mmul", &blocksparse_mmul, "BlockSPARSE Torch Matrix Multiplication");

  m.def("dummy_kernel", &dummy_kernel_launch, "Launch dummy kernel.");

  m.def("blocksparse_inspect", &blocksparse_inspect, "Inspect function for BLOCKSPARSE with CSR input");

  m.def("naive_spmm", &naive_spmm, "A naive implementation of Sparse Matrix Multiplication");

  m.def("init_blocksparse", &init_blocksparse_handle, "Create cuSPARSE handle.");
  m.def("destroy_blocksparse", &destroy_blocksparse_handle, "Destroy cuSPARSE handle.");
}
