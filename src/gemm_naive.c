#include "gemm.h"

// Private internal kernel
static void gemm_naive_kernel(size_t N, 
                              size_t stride,
                              const float * restrict A, 
                              const float * restrict B, 
                              float * restrict C) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                C[i * stride + j] += A[i * stride + k] * B[k * stride + j];
            }
        }
    }
}

// Public wrapper
void gemm_naive(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Safety check: ensure matrices are the same size before doing math
    if (A->size != B->size || A->size != C->size) {
        return; 
    }

    // Extract the size and raw data pointers
    gemm_naive_kernel(A->size, A->padded_size, A->data, B->data, C->data);
}