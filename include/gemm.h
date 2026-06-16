#ifndef GEMM_H
#define GEMM_H

/**
 * @brief Computes the baseline naive matrix multiplication (C = A * B).
 *
 * This function implements the standard i-j-k nested scalar loop architecture.
 * It purposefully lacks spatial locality and vectorization to establish a 
 * worst-case execution time and L1 cache miss baseline for the Mini-BLAS library.
 *
 * @param N The dimension of the matrices (assumes N x N square matrices).
 * @param A Pointer to the first input matrix, stored as a contiguous 1D array.
 * @param B Pointer to the second input matrix, stored as a contiguous 1D array.
 * @param C Pointer to the output matrix, stored as a contiguous 1D array.
 * WARNING: This memory block must be zero-initialized before calling,
 * as the function accumulates values directly into it.
 */
void gemm_naive(int N, float *A, float *B, float *C);

#endif