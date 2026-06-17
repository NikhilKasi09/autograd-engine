#ifndef GEMM_H
#define GEMM_H

#include "matrix.h"

/**
 * @brief Computes the baseline naive matrix multiplication (C = A * B).
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


#endif