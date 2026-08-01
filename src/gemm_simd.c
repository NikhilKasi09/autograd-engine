#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdio.h>

/*
 Private AVX2 kernel. C(MxN) += A(MxK) * B(KxN), row-major throughout.

 This is the "just SIMD" rung of the ladder, so it stays self-contained: no
 tiling, and its own scalar tail rather than a call out to gemm_tiled_kernel.
 The job here is to establish the vector-then-remainder pattern on the simplest
 possible body, before step G has to do the same thing inside a tile loop.

 The columns split in two:

     [0, N8)   handled 8 at a time by the vector loop, where N8 = N - N % 8
     [N8, N)   the 0 to 7 left over, handled by the scalar tail

 Every element of C is written by exactly one of those two paths, so there is
 no question of one being counted twice or missed. The two paths sum over k in
 the same order, but they will not always agree to the last bit, because FMA
 does not round the intermediate product while the scalar path does. That is
 expected and the tolerance in the test harness accounts for it.

 Loads and stores are the unaligned forms. Only the base pointer of a matrix is
 32-byte aligned; row i starts at data + i * stride, which is aligned only when
 stride % 8 == 0, and nothing guarantees that. vmovups on an aligned address
 costs the same as vmovaps on AVX2, so nothing is lost.
*/
static void gemm_avx2_kernel(size_t M, size_t N, size_t K,
                             const float * restrict A, size_t lda,
                             const float * restrict B, size_t ldb,
                             float * restrict C, size_t ldc) {

    // Largest multiple of the vector width that fits in N. Written as a
    // subtraction of the remainder, so there is nothing to underflow: N8 <= N
    // always, and N8 == 0 when N < 8.
    const size_t N8 = N - (N % GEMM_VEC);

    for (size_t i = 0; i < M; i++) {

        // Vector part: columns [0, N8), eight at a time:

        // C[i][j+0] += A[i][k] * B[k][j+0]
        // C[i][j+1] += A[i][k] * B[k][j+1]
        // ...
        // C[i][j+7] += A[i][k] * B[k][j+7]
        
        for (size_t j = 0; j < N8; j += GEMM_VEC) {

            __m256 acc = _mm256_loadu_ps(&C[i * ldc + j]); // Load the previous 8 C values into the accumulator

            for (size_t k = 0; k < K; k++){ // Loop for the number of columns in A
                
                __m256 a_bcast = _mm256_set1_ps(A[i * lda + k]); // The A[i][k] that is the same throughout 
                __m256 b_vec = _mm256_loadu_ps(&B[k * ldb + j]); // Load the 8 B values 

                acc =  _mm256_fmadd_ps(a_bcast, b_vec, acc); // a_broadcast * b_vec + acc for all 8 values

            }
            _mm256_storeu_ps(&C[i * ldc + j], acc); // Store the accumulated result back into the correct 8 slots in C 
        }

        /*
         Scalar tail: columns [N8, N), which is 0 to 7 columns wide.

         Reading the running total into acc, adding to it and writing it back
         keeps the += contract that the vector path above also has to keep. It
         is not a plain assignment.
        */
        for (size_t j = N8; j < N; j++) {
            float acc = C[i * ldc + j];

            for (size_t k = 0; k < K; k++) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }

            C[i * ldc + j] = acc;
        }
    }
}

// Public wrapper
void gemm_avx2(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (!gemm_check_shapes("gemm_avx2", A, B, C)) {
        return;
    }

    // M and N come from C, K is the inner dimension the two operands share
    gemm_avx2_kernel(C->rows, C->cols, A->cols,
                     A->data, A->stride,
                     B->data, B->stride,
                     C->data, C->stride);
}
