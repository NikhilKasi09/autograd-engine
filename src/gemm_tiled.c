#include "gemm.h"
#include "gemm_internal.h"
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 Cache-tiled scalar kernel. Declared in gemm_internal.h rather than being
 private here, because the vector kernels call it for their ragged edges.

 The three tile loops are independent now. Each one walks its own dimension and
 works out its own extent, so a trailing tile is simply smaller than the rest
 in whichever dimensions happen to be ragged. There is no requirement that M, N
 and K relate to each other or to BLOCK_SIZE in any way.

 Note the extents are worked out as counts rather than as clamped absolute
 indices:

     const size_t mt = MIN(BLOCK_SIZE, M - block_i);

 Since block_i < M is the loop condition, M - block_i cannot underflow. The
 alternative, clamping block_i + BLOCK_SIZE against M and then subtracting,
 is where unsigned arithmetic goes wrong in this file's neighbours.

 Working inside the tile on offset pointers keeps the inner loops free of
 block_i and block_j entirely, so what is left reads as a small dense GEMM,
 which is exactly what it is.
*/
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

                // Top left corner of this tile in each matrix. The leading
                // dimensions stay those of the full matrices.
                const float *A_tile = A + block_i * lda + block_k;
                const float *B_tile = B + block_k * ldb + block_j;
                float       *C_tile = C + block_i * ldc + block_j;

                // Inner loops, in ikj order so the innermost one walks B and C
                // along their rows
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
