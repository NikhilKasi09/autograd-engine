#include "gemm.h"
#include <stdio.h>

// Note: 'N' represents the physical padded size, not the logical matrix size
static void gemm_tiled_kernel(size_t N, const float * restrict A, const float * restrict B, float * restrict C) {

    // Outer loops (slide the 64x64 window across the matrices)
    for (size_t block_i = 0; block_i < N; block_i += BLOCK_SIZE) {
        for (size_t block_k = 0; block_k < N; block_k += BLOCK_SIZE) {
            for (size_t block_j = 0; block_j < N; block_j += BLOCK_SIZE) {

                // Inner loops (calculate inside the current window)
                for (size_t i = block_i; i < block_i + BLOCK_SIZE; i++) {
                    for (size_t k = block_k; k < block_k + BLOCK_SIZE; k++) {
                        
                        float a_ik = A[i * N + k];

                        for (size_t j = block_j; j < block_j + BLOCK_SIZE; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
                
            }
        }
    }
}


void gemm_tiled(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (A->size != B->size || A->size != C->size) {
        fprintf(stderr, "gemm_tiled: matrix size mismatch (%zu, %zu, %zu)\n", A->size, B->size, C->size);
        return; 
    }
    gemm_tiled_kernel(A->padded_size, A->data, B->data, C->data);
}
