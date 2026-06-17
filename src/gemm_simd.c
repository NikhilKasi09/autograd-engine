#include "gemm.h"
#include <immintrin.h>
#include <stdio.h>


//Private AVX2 kernel.
static void gemm_avx2_kernel(size_t N,
                             const float * restrict A,
                             const float * restrict B,
                             float * restrict C) {
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < N; k++) {
            //Broadcast A[i][k] into all 8 lanes of a 256-bit register.
            __m256 a_broadcast = _mm256_set1_ps(A[i * N + k]);

            for (size_t j = 0; j < N; j += 8) {
                //Load 8 contiguous floats from row k of B
                __m256 b_vec = _mm256_load_ps(&B[k * N + j]);

                //Load the corresponding 8 floats from row i of C
                __m256 c_vec = _mm256_load_ps(&C[i * N + j]);

                //Fused multiply-add: c_vec = (a_broadcast * b_vec) + c_vec
                c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);

                //Write the 8 results back to C
                _mm256_store_ps(&C[i * N + j], c_vec);
            }
        }
    }                
}

//Public wrapper
void gemm_avx2(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (A->size != B->size || A->size != C->size) {
        fprintf(stderr, "gemm_avx2: matrix size mismatch (%zu, %zu, %zu)\n",
                A->size, B->size, C->size);
        return;
    }
    gemm_avx2_kernel(A->size, A->data, B->data, C->data);
}