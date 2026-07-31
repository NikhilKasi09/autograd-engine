#include "gemm.h"
#include "gemm_internal.h"
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// logical_size is the matrix dimension and stride is the distance between
// rows. They were the same thing back when every allocation was padded, which
// is why the old comment here claimed this parameter was the padded size.
static void gemm_tiled_kernel(size_t logical_size, size_t stride,
                              const float * restrict A, const float * restrict B, 
                              float * restrict C) {

    // Outer loops (slide the 64x64 window across the matrices)
    for (size_t block_i = 0; block_i < logical_size; block_i += BLOCK_SIZE) {
        for (size_t block_k = 0; block_k < logical_size; block_k += BLOCK_SIZE) {
            for (size_t block_j = 0; block_j < logical_size; block_j += BLOCK_SIZE) {

                // Clamp the inner loops to ensure we stop at the logical edge
                size_t end_i = MIN(block_i + BLOCK_SIZE, logical_size);
                size_t end_k = MIN(block_k + BLOCK_SIZE, logical_size);
                size_t end_j = MIN(block_j + BLOCK_SIZE, logical_size);

                // Inner loops (calculate inside the current window)
                for (size_t i = block_i; i < end_i; i++) {
                    for (size_t k = block_k; k < end_k; k++) {
                        
                        float a_ik = A[i * stride + k];

                        for (size_t j = block_j; j < end_j; j++) {
                            C[i * stride + j] += a_ik * B[k * stride + j];
                        }
                    }
                }
            }
        }
    }
}


void gemm_tiled(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (!gemm_shapes_square_ok("gemm_tiled", A, B, C)) {
        return;
    }
    gemm_tiled_kernel(A->rows, A->stride, A->data, B->data, C->data);
}
