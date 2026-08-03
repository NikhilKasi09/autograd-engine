#include <stdio.h>
#include "gemm.hpp"
#include "gemm_internal.hpp"

// Private internal kernel. C(MxN) += A(MxK) * B(KxN), row-major.
//
// Same arithmetic as gemm_naive with j and k swapped. Fixing row i of A and
// row k of B lets the innermost loop walk B and C along their rows, in the
// order they sit in memory.
//
// k is bounded by K and j by N. Those were the same number while everything
// was square, and swapping them is the easy mistake here.
static void gemm_ikj_kernel(size_t M, size_t N, size_t K,
                            const float * GEMM_RESTRICT A, size_t lda,
                            const float * GEMM_RESTRICT B, size_t ldb,
                            float * GEMM_RESTRICT C, size_t ldc) {
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float a_ik = A[i * lda + k];
            for (size_t j = 0; j < N; j++) {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

// Public wrapper
void gemm_ikj(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Safety check: ensure the shapes work before doing math
    if (!gemm_check_shapes("gemm_ikj", A, B, C)) {
        return;
    }

    // M and N come from C, K is the inner dimension the two operands share
    gemm_ikj_kernel(C->rows, C->cols, A->cols,
                    A->data, A->stride,
                    B->data, B->stride,
                    C->data, C->stride);
}
