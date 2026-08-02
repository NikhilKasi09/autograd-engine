#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdio.h>

// Private AVX2 kernel. C(MxN) += A(MxK) * B(KxN), row-major.
//
// The plain SIMD rung, so it stays self-contained: no tiling, and its own
// scalar tail rather than calling gemm_tiled_kernel.
//
// Columns split in two, with N8 = N - N % 8:
//     [0, N8)   eight at a time in the vector loop
//     [N8, N)   the 0 to 7 left over, in the scalar tail
// Each element of C is written by exactly one of them, so nothing is missed or
// counted twice. They will not agree to the last bit, since FMA does not round
// the intermediate product and the scalar path does.
//
// Loads and stores are unaligned: only the base pointer is 32-byte aligned,
// and row i sits at data + i * stride. vmovups on an aligned address costs the
// same as vmovaps on AVX2.
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

        // Scalar tail: columns [N8, N), so 0 to 7 wide. Reads the running
        // total, adds to it and writes it back, keeping the += contract.
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
