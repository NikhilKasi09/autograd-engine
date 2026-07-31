#include <stdio.h>
#include "gemm.h"
#include "gemm_internal.h"

/*
 Private internal kernel. C(MxN) += A(MxK) * B(KxN), row-major throughout.

 Same arithmetic as gemm_naive, with the j and k loops swapped. Fixing row i of
 A and row k of B means the innermost loop walks B and C along their rows, so
 both are read and written in the order they sit in memory.

 The k loop bound is K and the j loop bound is N, which were the same number
 back when everything was square. Getting these the wrong way round is the
 easiest mistake to make here and shows up immediately on any non-square shape.
*/
static void gemm_ikj_kernel(size_t M, size_t N, size_t K,
                            const float * restrict A, size_t lda,
                            const float * restrict B, size_t ldb,
                            float * restrict C, size_t ldc) {
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
