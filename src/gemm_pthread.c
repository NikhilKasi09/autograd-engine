#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Macro to clamp loop boundaries so we don't compute the zero-padding blocks
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Every load and store below is the unaligned form. posix_memalign only aligns
// the base pointer, whereas each access addresses row i at data + i * stride,
// which is 32-byte aligned only when stride % 8 == 0. That holds today because
// matrix_create rounds every dimension up to a multiple of BLOCK_SIZE, but step
// B drops the padding. A worker also starts at whatever row its slice begins
// on, so even a padded stride would not make the alignment obvious by
// inspection. vmovups on an aligned address costs the same as vmovaps on AVX2.
static void gemm_worker_kernel(size_t start_row, size_t end_row,
                               size_t logical_size, size_t stride,
                               const float * restrict A,
                               const float * restrict B,
                               float * restrict C) {

    for (size_t block_i = start_row; block_i < end_row; block_i += BLOCK_SIZE) {
        for (size_t block_k = 0; block_k < logical_size; block_k += BLOCK_SIZE) {
            for (size_t block_j = 0; block_j < logical_size; block_j += BLOCK_SIZE) {

                // Clamp to this thread's row boundary
                size_t end_i = MIN(block_i + BLOCK_SIZE, end_row);
                size_t end_k = MIN(block_k + BLOCK_SIZE, logical_size);
                size_t end_j = MIN(block_j + BLOCK_SIZE, logical_size);

                size_t i = block_i;

                // 2D Register Block: Unroll 'i' by 4, and 'j' by 16 (2 vectors)
                for (; i <= end_i - 4; i += 4) {
                    size_t j = block_j;
                    
                    for (; j <= end_j - 16; j += 16) {
                        
                        __m256 c00 = _mm256_loadu_ps(&C[(i + 0) * stride + j + 0]);
                        __m256 c01 = _mm256_loadu_ps(&C[(i + 0) * stride + j + 8]);
                        __m256 c10 = _mm256_loadu_ps(&C[(i + 1) * stride + j + 0]);
                        __m256 c11 = _mm256_loadu_ps(&C[(i + 1) * stride + j + 8]);
                        __m256 c20 = _mm256_loadu_ps(&C[(i + 2) * stride + j + 0]);
                        __m256 c21 = _mm256_loadu_ps(&C[(i + 2) * stride + j + 8]);
                        __m256 c30 = _mm256_loadu_ps(&C[(i + 3) * stride + j + 0]);
                        __m256 c31 = _mm256_loadu_ps(&C[(i + 3) * stride + j + 8]);
                        
                        for (size_t k = block_k; k < end_k; k++) {
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
                        
                        _mm256_storeu_ps(&C[(i + 0) * stride + j + 0], c00);
                        _mm256_storeu_ps(&C[(i + 0) * stride + j + 8], c01);
                        _mm256_storeu_ps(&C[(i + 1) * stride + j + 0], c10);
                        _mm256_storeu_ps(&C[(i + 1) * stride + j + 8], c11);
                        _mm256_storeu_ps(&C[(i + 2) * stride + j + 0], c20);
                        _mm256_storeu_ps(&C[(i + 2) * stride + j + 8], c21);
                        _mm256_storeu_ps(&C[(i + 3) * stride + j + 0], c30);
                        _mm256_storeu_ps(&C[(i + 3) * stride + j + 8], c31);
                    }
                    
                    // j clean-up loop
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

                // i clean-up loop
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

// pthread entry point: unpacks a thread_args_t and runs that slice's kernel.
static void *gemm_worker(void *raw_args) {
    const thread_args_t *args = (const thread_args_t *)raw_args;

    gemm_worker_kernel(args->start_row, args->end_row,
                        args->logical_size, args->stride,
                        args->A, args->B, args->C);

    return NULL;
}

// Public Wrapper
void gemm_multithreaded(const matrix_t *A, const matrix_t *B, matrix_t *C, int num_threads) {
    if (!gemm_shapes_square_ok("gemm_multithreaded", A, B, C)) {
        return;
    }

    // Same whole-vector requirement as the kernel it shares its body with.
    // Lifted at step H.
    if (A->cols % GEMM_VEC != 0) {
        fprintf(stderr, "gemm_multithreaded: N must be a multiple of %d for now, got %zu\n",
                GEMM_VEC, A->cols);
        return;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "gemm_multithreaded: num_threads must be positive\n");
        return;
    }

    const size_t logical_size = A->rows;
    const size_t stride       = A->stride;

    // Never spin up more threads than there are rows to hand out.
    size_t nthreads = (size_t)num_threads;
    if (nthreads > logical_size) {
        nthreads = logical_size;
    }
    if (nthreads == 0) {
        return;
    }

    pthread_t      *threads = malloc(nthreads * sizeof(pthread_t));
    thread_args_t  *args    = malloc(nthreads * sizeof(thread_args_t));

    if (!threads || !args) {
        fprintf(stderr, "gemm_multithreaded: allocation failed\n");
        free(threads);
        free(args);
        return;
    }

    // Slice C horizontally by row, evening out the remainder 
    size_t base_rows   = logical_size / nthreads;
    size_t remainder   = logical_size % nthreads;
    size_t current_row = 0;

    for (size_t t = 0; t < nthreads; t++) {
        args[t].start_row = current_row;

        size_t thread_rows = base_rows + (t < remainder ? 1 : 0);
        args[t].end_row    = current_row + thread_rows;

        args[t].logical_size = logical_size;
        args[t].stride       = stride;
        args[t].A            = A->data;
        args[t].B            = B->data;
        args[t].C            = C->data;

        current_row = args[t].end_row;
    }

    // Spin up the worker pool
    size_t spawned = 0;
    for (size_t t = 0; t < nthreads; t++) {
        int rc = pthread_create(&threads[t], NULL, gemm_worker, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "gemm_multithreaded: pthread_create failed on thread %zu (%d)\n", t, rc);
            break;
        }
        spawned++;
    }

    // Join whichever threads actually got created
    for (size_t t = 0; t < spawned; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
}