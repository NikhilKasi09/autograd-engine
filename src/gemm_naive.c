#include "gemm.h"
#include "gemm_internal.h"

/*
 Private internal kernel. C(MxN) += A(MxK) * B(KxN), row-major throughout.

 The leading dimensions are what make this work on anything other than a whole
 matrix. lda is the distance in floats from one row of A to the next, which is
 not the same as the number of columns being read: a caller wanting a
 sub-rectangle passes a pointer into the middle of a bigger matrix along with
 the extents it wants, and the leading dimension keeps the row arithmetic
 pointing at the right places. Every sub-block of work in the later kernels is
 a full GEMM on offset pointers, and this is the mechanism that allows it.

 Note that the pointer and its leading dimension are adjacent in the parameter
 list, so a mismatched pair is visible at the call site rather than hidden
 among six size_t arguments.
*/
static void gemm_naive_kernel(size_t M, size_t N, size_t K,
                              const float * restrict A, size_t lda,
                              const float * restrict B, size_t ldb,
                              float * restrict C, size_t ldc) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < K; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// Public wrapper
void gemm_naive(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Safety check: ensure the shapes work before doing math
    if (!gemm_check_shapes("gemm_naive", A, B, C)) {
        return;
    }

    // M and N come from C, K is the inner dimension the two operands share
    gemm_naive_kernel(C->rows, C->cols, A->cols,
                      A->data, A->stride,
                      B->data, B->stride,
                      C->data, C->stride);
}
