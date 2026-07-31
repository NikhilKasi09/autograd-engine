#include "gemm.h"
#include <immintrin.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Every load and store below is the unaligned form. posix_memalign only aligns
// the base pointer, whereas each access here addresses row i at
// data + i * stride, which is 32-byte aligned only when stride % 8 == 0. That
// holds today because matrix_create rounds every dimension up to a multiple of
// BLOCK_SIZE, but step B drops the padding and makes stride equal to cols, and
// then no row past the first can be assumed aligned. vmovups on an aligned
// address runs at the same speed as vmovaps on AVX2, so this costs nothing.
static void gemm_tiled_simd_kernel(size_t logical_size, size_t stride,
                                   const float * restrict A, 
                                   const float * restrict B, 
                                   float * restrict C) {
    
    for (size_t block_i = 0; block_i < logical_size; block_i += BLOCK_SIZE) {
        for (size_t block_k = 0; block_k < logical_size; block_k += BLOCK_SIZE) {
            for (size_t block_j = 0; block_j < logical_size; block_j += BLOCK_SIZE) {

                size_t end_i = MIN(block_i + BLOCK_SIZE, logical_size);
                size_t end_k = MIN(block_k + BLOCK_SIZE, logical_size);
                size_t end_j = MIN(block_j + BLOCK_SIZE, logical_size);

                size_t i = block_i;

                // 2D Register Block: Unroll 'i' by 4, and 'j' by 16 (2 vectors).
                // This uses 8 accumulators for C, maximizing arithmetic intensity.
                for (; i <= end_i - 4; i += 4) {
                    size_t j = block_j;
                    
                    for (; j <= end_j - 16; j += 16) {
                        
                        // Load 8 contiguous vectors of C into YMM registers
                        __m256 c00 = _mm256_loadu_ps(&C[(i + 0) * stride + j + 0]);
                        __m256 c01 = _mm256_loadu_ps(&C[(i + 0) * stride + j + 8]);
                        __m256 c10 = _mm256_loadu_ps(&C[(i + 1) * stride + j + 0]);
                        __m256 c11 = _mm256_loadu_ps(&C[(i + 1) * stride + j + 8]);
                        __m256 c20 = _mm256_loadu_ps(&C[(i + 2) * stride + j + 0]);
                        __m256 c21 = _mm256_loadu_ps(&C[(i + 2) * stride + j + 8]);
                        __m256 c30 = _mm256_loadu_ps(&C[(i + 3) * stride + j + 0]);
                        __m256 c31 = _mm256_loadu_ps(&C[(i + 3) * stride + j + 8]);
                        
                        for (size_t k = block_k; k < end_k; k++) {
                            // Load B twice. We use these EXACT SAME vectors 4 times! (Bandwidth saver)
                            __m256 b0 = _mm256_loadu_ps(&B[k * stride + j + 0]);
                            __m256 b1 = _mm256_loadu_ps(&B[k * stride + j + 8]);
                            
                            __m256 a0 = _mm256_set1_ps(A[(i + 0) * stride + k]);
                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);
                            
                            __m256 a1 = _mm256_set1_ps(A[(i + 1) * stride + k]);
                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);
                            
                            __m256 a2 = _mm256_set1_ps(A[(i + 2) * stride + k]);
                            c20 = _mm256_fmadd_ps(a2, b0, c20);
                            c21 = _mm256_fmadd_ps(a2, b1, c21);
                            
                            __m256 a3 = _mm256_set1_ps(A[(i + 3) * stride + k]);
                            c30 = _mm256_fmadd_ps(a3, b0, c30);
                            c31 = _mm256_fmadd_ps(a3, b1, c31);
                        }
                        
                        // Store the 8 calculated vectors back to C
                        _mm256_storeu_ps(&C[(i + 0) * stride + j + 0], c00);
                        _mm256_storeu_ps(&C[(i + 0) * stride + j + 8], c01);
                        _mm256_storeu_ps(&C[(i + 1) * stride + j + 0], c10);
                        _mm256_storeu_ps(&C[(i + 1) * stride + j + 8], c11);
                        _mm256_storeu_ps(&C[(i + 2) * stride + j + 0], c20);
                        _mm256_storeu_ps(&C[(i + 2) * stride + j + 8], c21);
                        _mm256_storeu_ps(&C[(i + 3) * stride + j + 0], c30);
                        _mm256_storeu_ps(&C[(i + 3) * stride + j + 8], c31);
                    }
                    
                    // j clean-up loop for the remaining 8 floats if end_j isn't perfectly divisible by 16
                    for (; j < end_j; j += 8) {
                        __m256 c0 = _mm256_loadu_ps(&C[(i + 0) * stride + j]);
                        __m256 c1 = _mm256_loadu_ps(&C[(i + 1) * stride + j]);
                        __m256 c2 = _mm256_loadu_ps(&C[(i + 2) * stride + j]);
                        __m256 c3 = _mm256_loadu_ps(&C[(i + 3) * stride + j]);
                        
                        for (size_t k = block_k; k < end_k; k++) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * stride + j]);
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 0) * stride + k]), b_vec, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 1) * stride + k]), b_vec, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 2) * stride + k]), b_vec, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 3) * stride + k]), b_vec, c3);
                        }
                        
                        _mm256_storeu_ps(&C[(i + 0) * stride + j], c0);
                        _mm256_storeu_ps(&C[(i + 1) * stride + j], c1);
                        _mm256_storeu_ps(&C[(i + 2) * stride + j], c2);
                        _mm256_storeu_ps(&C[(i + 3) * stride + j], c3);
                    }
                }

                // i clean-up loop for the remaining 1-3 rows if end_i isn't perfectly divisible by 4
                for (; i < end_i; i++) {
                    for (size_t j = block_j; j < end_j; j += 8) {
                        __m256 c_vec = _mm256_loadu_ps(&C[i * stride + j]);
                        for (size_t k = block_k; k < end_k; k++) {
                            __m256 a_broadcast = _mm256_set1_ps(A[i * stride + k]);
                            __m256 b_vec       = _mm256_loadu_ps(&B[k * stride + j]);
                            c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                        }
                        _mm256_storeu_ps(&C[i * stride + j], c_vec);
                    }
                }
            }
        }
    }
}

void gemm_tiled_simd(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (A->size != B->size || A->size != C->size) {
        fprintf(stderr, "gemm_tiled_simd: matrix size mismatch (%zu, %zu, %zu)\n",
                A->size, B->size, C->size);
        return;
    }
    gemm_tiled_simd_kernel(A->size, A->padded_size, A->data, B->data, C->data);
}