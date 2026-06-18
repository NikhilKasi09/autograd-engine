#include "gemm.h"
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Macro to clamp loop boundaries so we don't compute the zero-padding blocks
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void gemm_worker_kernel(size_t start_row, size_t end_row,
                                size_t logical_size, size_t stride,
                                const float * restrict A,
                                const float * restrict B,
                                float * restrict C) {

    for (size_t block_i = start_row; block_i < end_row; block_i += BLOCK_SIZE) {
        for (size_t block_k = 0; block_k < logical_size; block_k += BLOCK_SIZE) {
            for (size_t block_j = 0; block_j < logical_size; block_j += BLOCK_SIZE) {

                // Clamp to this thread's row boundary (not just the matrix's), since start_row/end_row need not be aligned to BLOCK_SIZE.
                size_t end_i = MIN(block_i + BLOCK_SIZE, end_row);
                size_t end_k = MIN(block_k + BLOCK_SIZE, logical_size);
                size_t end_j = MIN(block_j + BLOCK_SIZE, logical_size);

                for (size_t i = block_i; i < end_i; i++) {
                    for (size_t j = block_j; j < end_j; j += 8) {

                        __m256 c_vec = _mm256_load_ps(&C[i * stride + j]);

                        for (size_t k = block_k; k < end_k; k++) {
                            __m256 a_broadcast = _mm256_set1_ps(A[i * stride + k]);
                            __m256 b_vec       = _mm256_load_ps(&B[k * stride + j]);

                            c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                        }

                        _mm256_store_ps(&C[i * stride + j], c_vec);
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
    if (A->size != B->size || A->size != C->size) {
        fprintf(stderr, "gemm_multithreaded: matrix size mismatch (%zu, %zu, %zu)\n",
                A->size, B->size, C->size);
        return;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "gemm_multithreaded: num_threads must be positive\n");
        return;
    }

    const size_t logical_size = A->size;
    const size_t stride       = A->padded_size;

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

    size_t base_rows   = logical_size / nthreads;
    size_t remainder    = logical_size % nthreads;
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

    size_t spawned = 0;
    for (size_t t = 0; t < nthreads; t++) {
        int rc = pthread_create(&threads[t], NULL, gemm_worker, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "gemm_multithreaded: pthread_create failed on thread %zu (%d)\n", t, rc);
            break;
        }
        spawned++;
    }

    // Join whichever threads actually got created, even if one failed midway,
    // so we never leak running threads or join uninitialised pthread_t values.
    for (size_t t = 0; t < spawned; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
}