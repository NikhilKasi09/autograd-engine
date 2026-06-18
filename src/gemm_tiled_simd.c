#include "gemm.h"
#include <immintrin.h>
#include <stdio.h>

// Macro to clamp loop boundaries so we don't compute the massive zero-padding blocks
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void gemm_tiled_simd_kernel(size_t logical_size, size_t stride, 
                                   const float * restrict A, 
                                   const float * restrict B, 
                                   float * restrict C) {
    
    // OUTER LOOPS: L1 Cache Blocking
    // Slide the 64x64 window across the matrices up to the logical boundary
    for (size_t block_i = 0; block_i < logical_size; block_i += BLOCK_SIZE) {
        for (size_t block_k = 0; block_k < logical_size; block_k += BLOCK_SIZE) {
            for (size_t block_j = 0; block_j < logical_size; block_j += BLOCK_SIZE) {

                // Clamp boundaries to prevent computing empty padded blocks
                size_t end_i = MIN(block_i + BLOCK_SIZE, logical_size);
                size_t end_k = MIN(block_k + BLOCK_SIZE, logical_size);
                size_t end_j = MIN(block_j + BLOCK_SIZE, logical_size);

                // INNER LOOPS: Hardware SIMD Execution (i-j-k order to prevent store stalls)
                for (size_t i = block_i; i < end_i; i++) {
                    
                    // j steps by 8 floats (256-bit AVX2 vectors)
                    for (size_t j = block_j; j < end_j; j += 8) {
                        
                        // Load 8 floats from C into a register exactly once per block
                        __m256 c_vec = _mm256_load_ps(&C[i * stride + j]);

                        // Accumulate the K-dimension Fused-Multiply Adds inside the silicon
                        for (size_t k = block_k; k < end_k; k++) {
                            __m256 a_broadcast = _mm256_set1_ps(A[i * stride + k]);
                            __m256 b_vec       = _mm256_load_ps(&B[k * stride + j]);
                            
                            c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                        }

                        // Write back the accumulated 8 floats to C exactly once
                        _mm256_store_ps(&C[i * stride + j], c_vec);
                    }
                }
            }
        }
    }
}

// Public Wrapper
void gemm_tiled_simd(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (A->size != B->size || A->size != C->size) {
        fprintf(stderr, "gemm_tiled_simd: matrix size mismatch (%zu, %zu, %zu)\n",
                A->size, B->size, C->size);
        return;
    }
    
    gemm_tiled_simd_kernel(A->size, A->padded_size, A->data, B->data, C->data);
}