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

#endif