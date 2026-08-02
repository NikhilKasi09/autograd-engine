#include "gemm.h"
#include "gemm_internal.h"
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Cache-tiled scalar kernel. Declared in gemm_internal.h because the vector
// kernels call it for their ragged edges.
//
// The three tile loops are independent and each works out its own extent, so a
// trailing tile is just smaller in whichever dimensions are ragged. M, N and K
// need no relationship to each other or to BLOCK_SIZE.
//
// Extents are counts, not clamped absolute indices. block_i < M is the loop
// condition, so M - block_i cannot underflow. Clamping block_i + BLOCK_SIZE
// against M and subtracting is what goes wrong in the SIMD kernels.
void gemm_tiled_kernel(size_t M, size_t N, size_t K,
                       const float * restrict A, size_t lda,
                       const float * restrict B, size_t ldb,
                       float * restrict C, size_t ldc) {

    // Outer loops slide the 64x64 window across the matrices
    for (size_t block_i = 0; block_i < M; block_i += BLOCK_SIZE) {
        const size_t mt = MIN(BLOCK_SIZE, M - block_i);

        for (size_t block_k = 0; block_k < K; block_k += BLOCK_SIZE) {
            const size_t kt = MIN(BLOCK_SIZE, K - block_k);

            for (size_t block_j = 0; block_j < N; block_j += BLOCK_SIZE) {
                const size_t nt = MIN(BLOCK_SIZE, N - block_j);

                // Top left corner of this tile. Leading dimensions stay those
                // of the full matrices, so the inner loops drop block_i and
                // block_j out of their address arithmetic entirely.
                const float *A_tile = A + block_i * lda + block_k;
                const float *B_tile = B + block_k * ldb + block_j;
                float       *C_tile = C + block_i * ldc + block_j;

                // ikj order, so the innermost loop walks B and C along rows
                for (size_t i = 0; i < mt; i++) {
                    for (size_t k = 0; k < kt; k++) {

                        float a_ik = A_tile[i * lda + k];

                        for (size_t j = 0; j < nt; j++) {
                            C_tile[i * ldc + j] += a_ik * B_tile[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_tiled(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (!gemm_check_shapes("gemm_tiled", A, B, C)) {
        return;
    }

    // M and N come from C, K is the inner dimension the two operands share
    gemm_tiled_kernel(C->rows, C->cols, A->cols,
                      A->data, A->stride,
                      B->data, B->stride,
                      C->data, C->stride);
}
