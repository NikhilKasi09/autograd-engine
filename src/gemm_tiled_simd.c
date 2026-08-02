#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 Vectorised core. C(MxN) += A(MxK) * B(KxN), row-major.

 PRECONDITION: M % GEMM_MR == 0 and N % GEMM_VEC == 0. The caller guarantees
 this by peeling the ragged rows and columns off first, which is why there is
 no scalar cleanup anywhere in this function. K is unconstrained.
*/
static void gemm_tiled_simd_core(size_t M, size_t N, size_t K,
                                 const float * restrict A, size_t lda,
                                 const float * restrict B, size_t ldb,
                                 float * restrict C, size_t ldc) {

    for (size_t block_i = 0; block_i < M; block_i += BLOCK_SIZE) {

        // Extent as a count, never as a clamped absolute index: block_i < M is
        // the loop condition, so M - block_i cannot underflow.
        const size_t mt = MIN(BLOCK_SIZE, M - block_i);

        for (size_t block_k = 0; block_k < K; block_k += BLOCK_SIZE) {

            // K has no constraints on it, so kt is whatever is left, 1 to 64
            const size_t kt = MIN(BLOCK_SIZE, K - block_k);

            for (size_t block_j = 0; block_j < N; block_j += BLOCK_SIZE) {

                const size_t nt = MIN(BLOCK_SIZE, N - block_j);

                // Top left corner of this tile in each matrix. Everything below
                // is indexed relative to these, so the inner loops never carry
                // block_i or block_j through their address arithmetic.
                const float *A_tile = A + block_i * lda + block_k;
                const float *B_tile = B + block_k * ldb + block_j;
                float       *C_tile = C + block_i * ldc + block_j;


                // mt inherits the caller's guarantee: M and BLOCK_SIZE are both
                // multiples of GEMM_MR, so mt is too, and this loop lands
                // exactly on mt with nothing left over. Written as i + 4 <= mt
                // rather than i <= mt - 4, which would wrap when mt < 4.
                for (size_t i = 0; i + GEMM_MR <= mt; i += GEMM_MR) {

                    size_t j = 0;

                    // Same reasoning for nt: it is a multiple of GEMM_VEC, so
                    // this walks 16 columns at a time and leaves either 0 or 8.
                    for (; j + GEMM_NR <= nt; j += GEMM_NR) {
                        
                        // Load 8 values into c00 to c31
                        __m256 c00 = _mm256_loadu_ps(&C_tile[(i + 0) * ldc + j + 0]);
                        __m256 c01 = _mm256_loadu_ps(&C_tile[(i + 0) * ldc + j + 8]);
                        __m256 c10 = _mm256_loadu_ps(&C_tile[(i + 1) * ldc + j + 0]);
                        __m256 c11 = _mm256_loadu_ps(&C_tile[(i + 1) * ldc + j + 8]);
                        __m256 c20 = _mm256_loadu_ps(&C_tile[(i + 2) * ldc + j + 0]);
                        __m256 c21 = _mm256_loadu_ps(&C_tile[(i + 2) * ldc + j + 8]);
                        __m256 c30 = _mm256_loadu_ps(&C_tile[(i + 3) * ldc + j + 0]);
                        __m256 c31 = _mm256_loadu_ps(&C_tile[(i + 3) * ldc + j + 8]);

                        
                        for (size_t k = 0; k < kt; k++){

                            // Load the 16 values into groups of 2x8 b vectors
                            __m256 b0 = _mm256_loadu_ps(&B_tile[k * ldb + j + 0]); 
                            __m256 b1 = _mm256_loadu_ps(&B_tile[k * ldb + j + 8]);


                            // Calculate row a0 lane product  b0 and then a0 lane product b1
                            __m256 a0 = _mm256_set1_ps(A_tile[(i + 0) * lda + k]);  
                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);
                            
                            // Calculate a1 lane product b0 and then a1 lane product b1
                            __m256 a1 = _mm256_set1_ps(A_tile[(i + 1) * lda + k]);
                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);

                            // Calculate a2 lane product b0 and then a2 lane product b1
                            __m256 a2 = _mm256_set1_ps(A_tile[(i + 2) * lda + k]);
                            c20 = _mm256_fmadd_ps(a2, b0, c20);
                            c21 = _mm256_fmadd_ps(a2, b1, c21);

                            // Calculate a3 lane product b0 and then a3 lane product b1
                            __m256 a3 = _mm256_set1_ps(A_tile[(i + 3) * lda + k]);
                            c30 = _mm256_fmadd_ps(a3, b0, c30);
                            c31 = _mm256_fmadd_ps(a3, b1, c31);

                        }

                        _mm256_storeu_ps(&C_tile[(i + 0) * ldc + j + 0], c00);
                        _mm256_storeu_ps(&C_tile[(i + 0) * ldc + j + 8], c01);
                        _mm256_storeu_ps(&C_tile[(i + 1) * ldc + j + 0], c10);
                        _mm256_storeu_ps(&C_tile[(i + 1) * ldc + j + 8], c11);
                        _mm256_storeu_ps(&C_tile[(i + 2) * ldc + j + 0], c20);
                        _mm256_storeu_ps(&C_tile[(i + 2) * ldc + j + 8], c21);
                        _mm256_storeu_ps(&C_tile[(i + 3) * ldc + j + 0], c30);
                        _mm256_storeu_ps(&C_tile[(i + 3) * ldc + j + 8], c31);
                    }

                    if (j < nt) { // If it not a multiple of 16 but a multiple of 8 we need to calculate the rest

                        // Load 8 values into c0 to c3 
                        __m256 c0 = _mm256_loadu_ps(&C_tile[(i + 0) * ldc + j]);
                        __m256 c1 = _mm256_loadu_ps(&C_tile[(i + 1) * ldc + j]);
                        __m256 c2 = _mm256_loadu_ps(&C_tile[(i + 2) * ldc + j]);
                        __m256 c3 = _mm256_loadu_ps(&C_tile[(i + 3) * ldc + j]);


                        
                        for (size_t k = 0; k < kt; k++){

                            // Load the 8 balues into a b vector
                            __m256 b0 = _mm256_loadu_ps(&B_tile[k * ldb + j + 0]); 


                            // Calculate row a0 lane product b0 
                            __m256 a0 = _mm256_set1_ps(A_tile[(i + 0) * lda + k]);  
                            c0 = _mm256_fmadd_ps(a0, b0, c0);
                            
                            // Calculate a1 lane product b0 
                            __m256 a1 = _mm256_set1_ps(A_tile[(i + 1) * lda + k]);
                            c1 = _mm256_fmadd_ps(a1, b0, c1);
   

                            // Calculate a2 lane product b0 
                            __m256 a2 = _mm256_set1_ps(A_tile[(i + 2) * lda + k]);
                            c2 = _mm256_fmadd_ps(a2, b0, c2);


                            // Calculate a3 lane product b0 
                            __m256 a3 = _mm256_set1_ps(A_tile[(i + 3) * lda + k]);
                            c3 = _mm256_fmadd_ps(a3, b0, c3);

                        }

                        _mm256_storeu_ps(&C_tile[(i + 0) * ldc + j], c0);
                        _mm256_storeu_ps(&C_tile[(i + 1) * ldc + j], c1);
                        _mm256_storeu_ps(&C_tile[(i + 2) * ldc + j], c2);
                        _mm256_storeu_ps(&C_tile[(i + 3) * ldc + j], c3);
                    }
                }
            }
        }
    }
}

/*
 Full kernel: peels the ragged strips, then calls the core on what is left.

 The decomposition, with M4 = M - M % GEMM_MR and N8 = N - N % GEMM_VEC:

     core    [0, M4) x [0, N8)     vectorised, no edge cases
     bottom  [M4, M) x [0,  N)     up to 3 rows, full width
     right   [0, M4) x [N8, N)     up to 7 columns

 Their union is the whole of M x N and no two of them overlap, so every element
 of C is written by exactly one path. That matters twice over: nothing is
 counted twice or missed, and the fact that the paths sum over k in different
 orders raises no consistency question, because no single element is ever split
 across two of them.

 Peeling once here rather than handling edges inside every tile is what removes
 the 1-row cleanup path and the scalar corner from the core entirely.

 Cost: the strips run scalar, and they cost far more than their share of the
 arithmetic. The FLOP fraction is 1 - (M4 * N8) / (M * N), which comes to 3.9%
 at 255x255 and 1.0% at 1023x1023. Measured throughput:

     256x256   130.4 GFLOP/s      255x255    86.4 GFLOP/s   (-34%)
     1024x1024  69.7 GFLOP/s     1023x1023   64.4 GFLOP/s   (-7.5%)

 So 3.9% of the arithmetic takes about a third of the runtime. The strips are
 scalar rather than vectorised, and a 3-row sliver has almost no reuse to give
 the cache, so it is slower per FLOP than the scalar tiled kernel working on a
 square block. Quote the measured number for the shape in question and never
 the FLOP fraction, which understates the time cost by roughly ten times.

 What this means in practice: keep N a multiple of 8 and M a multiple of 4 and
 none of this runs. An MLP layer is (batch, in) @ (in, out), so padding layer
 widths to multiples of 8 and batch sizes to multiples of 4 stays on the fast
 path for free. The exception is the final classification layer, where out is
 however many classes there are: 128 -> 10 puts 2 of its 10 columns on the
 scalar path.

 _mm256_maskload_ps would remove the strips entirely by letting the vector path
 handle a partial column block directly, and is the obvious next optimisation.
*/
static void gemm_tiled_simd_kernel(size_t M, size_t N, size_t K,
                                   const float * restrict A, size_t lda,
                                   const float * restrict B, size_t ldb,
                                   float * restrict C, size_t ldc) {

    // Largest multiples of the register block and the vector width that fit.
    // Subtracting the remainder means neither can underflow.
    const size_t M4 = M - (M % GEMM_MR);
    const size_t N8 = N - (N % GEMM_VEC);

    // The bulk of the work, on the part that needs no edge handling
    if (M4 > 0 && N8 > 0) {
        gemm_tiled_simd_core(M4, N8, K, A, lda, B, ldb, C, ldc);
    }

    // Bottom strip: the last 1 to 3 rows, across the full width of C. Offset
    // A and C down by M4 rows; B is untouched because the strip needs all of it.
    if (M4 < M) {
        gemm_tiled_kernel(M - M4, N, K,
                          A + M4 * lda, lda,
                          B, ldb,
                          C + M4 * ldc, ldc);
    }

    // Right strip: the last 1 to 7 columns, but only down to row M4, since the
    // bottom strip already covered everything below that for the full width.
    // Offset B and C right by N8 columns; A is untouched.
    if (N8 < N) {
        gemm_tiled_kernel(M4, N - N8, K,
                          A, lda,
                          B + N8, ldb,
                          C + N8, ldc);
    }
}

void gemm_tiled_simd(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (!gemm_check_shapes("gemm_tiled_simd", A, B, C)) {
        return;
    }


    // M and N come from C, K is the inner dimension the two operands share
    gemm_tiled_simd_kernel(C->rows, C->cols, A->cols,
                           A->data, A->stride,
                           B->data, B->stride,
                           C->data, C->stride);
}
