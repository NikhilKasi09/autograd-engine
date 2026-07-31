#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdio.h>

// Private AVX2 kernel.
//
// The loads and stores are the unaligned forms. posix_memalign only aligns the
// base pointer, whereas row i sits at data + i * stride, which is 32-byte
// aligned only when stride % 8 == 0. That is true today because matrix_create
// rounds every dimension up to a multiple of BLOCK_SIZE, but step B drops the
// padding and makes stride equal to cols. vmovups on an aligned address runs at
// the same speed as vmovaps on AVX2, so this costs nothing.
static void gemm_avx2_kernel(size_t logical_size,
                             size_t stride,
                             const float * restrict A,
                             const float * restrict B,
                             float * restrict C) {
    for (size_t i = 0; i < logical_size; i++) {
        for (size_t j = 0; j < logical_size; j += 8) {
            
            __m256 c_vec = _mm256_loadu_ps(&C[i * stride + j]);
            
            for (size_t k = 0; k < logical_size; k++) {
                // Broadcast A[i][k] into all 8 lanes of a 256-bit register
                __m256 a_broadcast = _mm256_set1_ps(A[i * stride + k]);

                // Load 8 contiguous floats from row k of B
                __m256 b_vec = _mm256_loadu_ps(&B[k * stride + j]);

                // Fused multiply-add: c_vec = (a_broadcast * b_vec) + c_vec
                c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
            }
            
            // Write the 8 accumulated results back to C ONCE
            _mm256_storeu_ps(&C[i * stride + j], c_vec);
        }
    }                
}

// Public wrapper
void gemm_avx2(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    if (!gemm_shapes_square_ok("gemm_avx2", A, B, C)) {
        return;
    }

    // The j loop steps 8 at a time with no scalar tail, so anything that is
    // not a whole number of vectors wide would walk off the end of the row.
    // The padding used to hide this. Lifted at step F.
    if (A->cols % GEMM_VEC != 0) {
        fprintf(stderr, "gemm_avx2: N must be a multiple of %d for now, got %zu\n", GEMM_VEC,
                A->cols);
        return;
    }

    gemm_avx2_kernel(A->rows, A->stride, A->data, B->data, C->data);
}