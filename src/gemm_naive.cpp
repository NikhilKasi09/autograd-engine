#include "gemm.hpp"
#include "gemm_internal.hpp"

// Private internal kernel. C(MxN) += A(MxK) * B(KxN), row-major.
//
// lda is the distance from one row of A to the next, which is not the same as
// the number of columns being read. That is what lets a caller pass a pointer
// into the middle of a bigger matrix with the extents it wants and still get
// correct row arithmetic. Every sub-block in the later kernels works this way.
//
// Each pointer sits next to its own leading dimension in the parameter list,
// so a mismatched pair is visible at the call site.
static void gemm_naive_kernel(size_t M, size_t N, size_t K,
                              const float * GEMM_RESTRICT A, size_t lda,
                              const float * GEMM_RESTRICT B, size_t ldb,
                              float * GEMM_RESTRICT C, size_t ldc) {
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
