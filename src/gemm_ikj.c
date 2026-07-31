#include <stdio.h>
#include "gemm.h"
#include "gemm_internal.h"

// Private internal kernel
static void gemm_ikj_kernel(size_t N,
                            size_t stride,
                            const float * restrict A,
                            const float * restrict B,
                            float * restrict C) {
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < N; k++) {
            float a_ik = A[i * stride + k];
            for (size_t j = 0; j < N; j++) {
                C[i * stride + j] += a_ik * B[k * stride + j];
            }
        }
    }
}

// Public wrapper
void gemm_ikj(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Safety check: ensure the shapes work before doing math
    if (!gemm_shapes_square_ok("gemm_ikj", A, B, C)) {
        return;
    }

    // Extract the size and raw data pointers
    gemm_ikj_kernel(A->rows, A->stride, A->data, B->data, C->data);
}