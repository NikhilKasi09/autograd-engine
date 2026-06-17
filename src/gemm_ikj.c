#include "gemm_ikj.h"
#include <stdio.h>

// Private internal kernel
static void gemm_ikj_kernel(size_t N,
                             const float * restrict A,
                             const float * restrict B,
                             float * restrict C) {
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (size_t j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Public wrapper
void gemm_ikj(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Safety check: ensure matrices are the same size before doing math
    if (A->size != B->size || A->size != C->size) {
        return;
    }

    // Extract the size and raw data pointers
    gemm_ikj_kernel(A->size, A->data, B->data, C->data);
}