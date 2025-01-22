#ifndef __BASELINE_MM_H_
#define __BASELINE_MM_H_

#include "baseline_mm.cuh"
#include "utils.cuh"
#include <stdlib.h>           // EXIT_FAILURE
#include <stdio.h>            // printf

// #include <iostream>
// #include <array>
// #include <torch/extension.h>
// #include "utils.cuh"

// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <cusparse_v2.h>

// #include <sys/time.h>
// typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp () {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

// #define NUM_THREADS (64)
// TILE_DIM is a multiple of 16
#define TILE_DIM 16
#define BLOCK_ROWS 8 

__global__ void dummyKernel()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n", tid);
}

// __global__ void copySharedMem(float *odata, const float *idata, const int TILE_DIM, const int BLOCK_ROWS, const int BLOCK_COLS)
// {
//   __shared__ float tile[TILE_DIM * TILE_DIM];

//   int x = blockIdx.x * TILE_DIM + threadIdx.x;
//   int y = blockIdx.y * TILE_DIM + threadIdx.y;
//   int width = gridDim.x * TILE_DIM;

//   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//      tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

//   __syncthreads();

//   for (int j = 0; j < TILE_DIM; j += BLOCK_COLS)
//      odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
// }

void dummy_kernel_launch() {
    dim3 threads_per_block(NUM_THREADS);
    dim3 blocks_per_grid(NUM_THREADS);
    dummyKernel<<<blocks_per_grid, threads_per_block>>>();
    checkCudaStatus(cudaDeviceSynchronize());
}

__global__ void check_equal(float *d_Arr, float *h_Arr, size_t rows, size_t cols)
{
    // checks two values are equal between two arrays
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= rows * cols) {
        return;
    }

    if (d_Arr[tid] == h_Arr[tid]) {
        printf("Equal %d\n", tid);
    } else {
        printf("Not Equal %d\n", tid);
    }
}

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        tile[threadIdx.y+i][threadIdx.x] =
        idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        odata[index_out+i*height] =
        tile[threadIdx.x][threadIdx.y+i];
    }
} 

void dense_to_csr(cusparseHandle_t handle, 
                  torch::Tensor dense, const int num_rows, const int num_cols,
                  float **d_csr_values, int **d_csr_columns, int **d_csr_offsets, int *nnz)
{
    int ld = num_cols;
    float *d_dense = dense.data_ptr<float>();
    // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Allocate memory for offsets
    checkCudaStatus(cudaMalloc((void**) &(*d_csr_offsets), (num_rows + 1) * sizeof(int)));

    // Create dense matrix A
    cusparseSafeCall(cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create sparse matrix B in CSR format
    cusparseSafeCall(cusparseCreateCsr(&matB, num_rows, num_cols, 0, *d_csr_offsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // allocate an external buffer if needed
    cusparseSafeCall(cusparseDenseToSparse_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
    checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

    // execute Sparse to Dense conversion
    cusparseSafeCall(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz_tmp;
    cusparseSafeCall(cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz_tmp));
    *nnz = nnz_tmp;

    // allocate CSR column indices and values
    checkCudaStatus(cudaMalloc((void**) &(*d_csr_columns), nnz_tmp * sizeof(int)));
    checkCudaStatus(cudaMalloc((void**) &(*d_csr_values),  nnz_tmp * sizeof(float)));
    // reset offsets, column indices, and values pointers
    cusparseSafeCall(cusparseCsrSetPointers(matB, *d_csr_offsets, *d_csr_columns, *d_csr_values));
    // execute Sparse to Dense conversion
    cusparseSafeCall(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
    // destroy matrix/vector descriptors
    cusparseSafeCall(cusparseDestroyDnMat(matA));
    cusparseSafeCall(cusparseDestroySpMat(matB));
    // device memory deallocation
    checkCudaStatus(cudaFree(dBuffer));
}


void dense_to_sparse_blockedell(cusparseHandle_t handle, torch::Tensor A, float *h_ell_values, int *hA_columns,
                            int A_num_rows, int A_num_cols, int A_ell_blocksize, int A_ell_cols, int num_blocks)
{
    // For the dense ell values representation 
    // you need the ell_columns, number of blocks, nnz and you can get the values representation
    int   ld           = A_num_cols;
    int   dense_size   = ld * A_num_rows;
    int   ell_width    = A_ell_cols;
    int   nnz          = ell_width * A_num_rows;
    float *hA_values = A.data_ptr<float>();

    // printf("n_ell_cols: %d\n", nnz / (A_ell_blocksize * A_ell_blocksize));

    // Find the ell_values float vector
    int   *d_ell_columns;
    float *d_ell_values,  *d_dense;
    checkCudaStatus( cudaMalloc((void**) &d_dense, dense_size * sizeof(float)));
    checkCudaStatus( cudaMalloc((void**) &d_ell_columns, nnz / (A_ell_blocksize * A_ell_blocksize) * sizeof(int)));
    checkCudaStatus( cudaMalloc((void**) &d_ell_values, nnz * sizeof(float)));
    checkCudaStatus( cudaMemcpy(d_dense, hA_values, dense_size * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaStatus( cudaMemcpy(d_ell_columns, hA_columns, nnz / (A_ell_blocksize * A_ell_blocksize) * sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaStatus( cudaMemcpy(d_ell_values, h_ell_values, nnz * sizeof(float), cudaMemcpyHostToDevice) );

     // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    
    cusparseSafeCall( cusparseCreate(&handle) );
    // Create dense matrix A
    cusparseSafeCall( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) );

    // Create sparse matrix B in Blocked ELL format
    cusparseSafeCall( cusparseCreateBlockedEll(&matB, A_num_rows, A_num_cols,
                                             A_ell_blocksize, ell_width,
                                             d_ell_columns, d_ell_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_32F) );

    // allocate an external buffer if needed
    cusparseSafeCall( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) );
    checkCudaStatus( cudaMalloc(&dBuffer, bufferSize) );

    // execute Sparse to Dense conversion
    cusparseSafeCall( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );

    // execute Sparse to Dense conversion
    cusparseSafeCall( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );

    // destroy matrix/vector descriptors
    cusparseSafeCall( cusparseDestroyDnMat(matA) );
    cusparseSafeCall( cusparseDestroySpMat(matB) );
    cusparseSafeCall( cusparseDestroy(handle) );
    checkCudaStatus( cudaMemcpy(h_ell_values, d_ell_values,
                           nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) );
}

void free_csr(float *d_csr_values, int *d_csr_columns, int *d_csr_offsets) {
    checkCudaStatus(cudaFree(d_csr_values));
    checkCudaStatus(cudaFree(d_csr_columns));
    checkCudaStatus(cudaFree(d_csr_offsets));
}

void blocksparse_mm_wrapper(cusparseHandle_t handle,
                            float* hA_values, int* hA_columns,
                            int A_num_rows, int A_num_cols, int A_ell_blocksize, int A_ell_cols, int num_blocks,
                            torch::Tensor B, int B_rows, int B_cols,
                            torch::Tensor C)
{
    float alpha           = 1.0f;
    float beta            = 0.0f;

    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = B_cols;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   A_num_blocks    = A_ell_cols * A_num_rows / (A_ell_blocksize * A_ell_blocksize);
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;

    size_t               bufferSize = 0;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    void* dBuffer = NULL;

    // Define float pointers for the A dense tenor and the col values
    float *hB = B.data_ptr<float>();
    float *hC = C.data_ptr<float>();

    int *dA_columns;
    float *dA_values, *dB, *dC;


    checkCudaStatus( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) );
    checkCudaStatus( cudaMalloc((void**) &dA_values, A_ell_cols * A_num_rows * sizeof(float)) );
    checkCudaStatus( cudaMalloc((void**) &dB, B_size * sizeof(float)) );
    checkCudaStatus( cudaMalloc((void**) &dC, C_size * sizeof(float)) );
    

    checkCudaStatus( cudaMemcpy(dA_columns, hA_columns, A_num_blocks * sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaStatus( cudaMemcpy(dA_values, hA_values, A_num_rows * A_ell_cols * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaStatus( cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaStatus( cudaMemcpy(dC, hC, C_size * sizeof(float), cudaMemcpyHostToDevice) );


    cusparseSafeCall( cusparseCreate(&handle) );
    // Create sparse matrix A in blocked ELL format
    cusparseSafeCall( cusparseCreateBlockedEll(
                                        &matA,
                                        A_num_rows, A_num_cols, A_ell_blocksize,
                                        A_ell_cols, dA_columns, dA_values,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

    // Create dense matrix B
    cusparseSafeCall( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) );
    // Create dense matrix C
    cusparseSafeCall( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, B_num_cols, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) );
    // allocate an external buffer if needed
    cusparseSafeCall( cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    checkCudaStatus( cudaMalloc(&dBuffer, bufferSize) );
    // execute SpMM
    //cusparseSafeCall( cusparseSpMM(handle,
    cusparseStatus_t status = cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

    if(status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("SpMM failed\n");
    }

    checkCudaStatus( cudaMemcpy(hC, dC, C_size * sizeof(float), cudaMemcpyDeviceToHost) );

    // destroy matrix/vector descriptors
    cusparseSafeCall( cusparseDestroySpMat(matA) );
    cusparseSafeCall( cusparseDestroyDnMat(matB) );
    cusparseSafeCall( cusparseDestroyDnMat(matC) );
    cusparseSafeCall( cusparseDestroy(handle) );

    return;

}

void dense_to_blocksparse(cusparseHandle_t handle,  float *dA_values, int *dA_columns,
                  const int num_rows, const int num_cols, const int ell_blk_size, const int ell_width)
{

    // int nnz = ell_width * num_rows;
    int ld = num_cols;
    // int dense_size = ld * num_rows;
    // float *d_dense = dense.data_ptr<float>();

    // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Create dense matrix A
    cusparseSafeCall(cusparseCreateDnMat(&matA, num_rows, num_cols, ld, dA_values, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create sparse matrix B in CSR format
    cusparseSafeCall(cusparseCreateBlockedEll(&matB, num_rows, num_cols,
                                             ell_blk_size, ell_width,
                                             dA_columns, dA_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_32F));

    // allocate an external buffer if needed
    // cusparseSafeCall(cusparseSpMM_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
    checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

    // analyze dense to sparse conversion
    cusparseSafeCall(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    // execute the dense to sparse conversion
    cusparseSafeCall(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    // destroy matrix/vector descriptors
    cusparseSafeCall(cusparseDestroyDnMat(matA));
    cusparseSafeCall(cusparseDestroySpMat(matB));
    cusparseSafeCall(cusparseDestroy(handle));
    // device memory deallocation
    checkCudaStatus(cudaFree(dBuffer));
    checkCudaStatus(cudaFree(dA_values));
    checkCudaStatus(cudaFree(dA_columns));

}

#endif // __BASELINE_MM_H_