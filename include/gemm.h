#ifndef GEMM_H
#define GEMM_H

#include "matrix.h"

#define BLOCK_SIZE 64

/**
 * @brief Computes the baseline naive matrix multiplication (C = A * B).
 *
 * This function implements the standard i-j-k nested scalar loop architecture.
 * It purposefully lacks spatial locality and vectorization to establish a 
 * worst-case execution time and L1 cache miss baseline for the Mini-BLAS library.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct.
 */
void gemm_naive(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Computes matrix multiplication (C = A * B) using AVX2 SIMD intrinsics.
 *
 * Processes 8 single-precision floats per clock cycle using 256-bit ymm registers.
 * Requires all matrix data pointers to be 32-byte aligned (guaranteed by matrix_create).
 * Matrix dimensions must be a multiple of 8 (guaranteed by matrix_create padding).
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct. Must be zero-initialised before call.
 */
void gemm_avx2(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Computes matrix multiplication using a 64x64 Cache-Tiled Architecture.
 *
 * This function divides the matrices into 16KB sub-blocks to ensure 
 * the working set remains completely resident within the L1 hardware cache 
 * during computation. It heavily relies on the zero-padded memory allocations 
 * in the matrix_t struct to bypass edge-case boundary checks.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct.
 */
void gemm_tiled(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Computes matrix multiplication by fusing a 64x64 Cache-Tiled Architecture with 256-bit AVX2 SIMD intrinsics.
 *
 * This function achieves peak CPU throughput by locking sub-blocks into the L1 
 * hardware cache while vectorizing the innermost computations. It utilizes an 
 * i-j-k loop reordering to accumulate fused multiply-adds (FMA) entirely within 
 * hardware registers, eliminating store-forwarding stalls. Furthermore, it strictly 
 * relies on the 32-byte aligned, zero-padded memory in the matrix_t struct to safely 
 * execute 256-bit loads and stores without edge-case boundary checks or segmentation faults.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct.
 */
void gemm_tiled_simd(const matrix_t *A, const matrix_t *B, matrix_t *C);


#endif