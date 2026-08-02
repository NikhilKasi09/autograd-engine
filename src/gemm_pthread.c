#include "gemm.h"
#include "gemm_internal.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// pthread entry point: unpacks a thread_args_t and runs that slice's kernel.
// The pointers are already offset, so this is just a GEMM of m_rows by N.
static void *gemm_worker(void *raw_args) {
    const thread_args_t *args = (const thread_args_t *)raw_args;

    gemm_tiled_simd_kernel(args->m_rows, args->N, args->K,
                           args->A, args->lda,
                           args->B, args->ldb,
                           args->C, args->ldc);

    return NULL;
}

// Public Wrapper
void gemm_multithreaded(const matrix_t *A, const matrix_t *B, matrix_t *C, int num_threads) {
    if (!gemm_check_shapes("gemm_multithreaded", A, B, C)) {
        return;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "gemm_multithreaded: num_threads must be positive\n");
        return;
    }

    // Slicing is over the rows of C, so M is what gets divided up
    const size_t M = C->rows;
    const size_t N = C->cols;
    const size_t K = A->cols;

    // Never spin up more threads than there are rows to hand out
    size_t nthreads = (size_t)num_threads;
    if (nthreads > M) {
        nthreads = M;
    }

    pthread_t     *threads = malloc(nthreads * sizeof(pthread_t));
    thread_args_t *args    = malloc(nthreads * sizeof(thread_args_t));

    if (!threads || !args) {
        fprintf(stderr, "gemm_multithreaded: allocation failed\n");
        free(threads);
        free(args);
        return;
    }

    // Hand out whole 4-row blocks rather than rounding each slice down to a
    // multiple of 4. Rounding down looks tidier but wrecks the load balance:
    // at M=1024 across 24 threads it gives everyone 40 rows and dumps the
    // leftover 104 on the last one, so the critical path is 2.4x the ideal.
    const size_t nblocks = M / GEMM_MR; // whole 4-row blocks available
    const size_t tail    = M % GEMM_MR; // 0 to 3 rows left over at the bottom

    size_t row = 0;

    for (size_t t = 0; t < nthreads; t++) {
        size_t thread_rows;

        if (nblocks < nthreads) {
            // Not enough blocks to give every worker one, so just split the
            // rows evenly. Slices are not multiples of 4 here and each worker
            // peels its own bottom strip, which is fine because M is tiny.
            thread_rows = M / nthreads + (t < M % nthreads ? 1 : 0);
        } else {
            // Spread the blocks out, remainder first-come, so the biggest and
            // smallest slices differ by at most one block
            const size_t base  = nblocks / nthreads;
            const size_t extra = nblocks % nthreads;

            thread_rows = (base + (t < extra ? 1 : 0)) * GEMM_MR;

            // The odd rows at the bottom go to the last worker, which is the
            // only one that ends up peeling a scalar strip
            if (t == nthreads - 1) {
                thread_rows += tail;
            }
        }

        args[t].m_rows = thread_rows;
        args[t].N      = N;
        args[t].K      = K;

        // Offset A and C down to this worker's first row. B is left alone,
        // since every worker reads all of it.
        args[t].A   = A->data + row * A->stride;
        args[t].lda = A->stride;
        args[t].B   = B->data;
        args[t].ldb = B->stride;
        args[t].C   = C->data + row * C->stride;
        args[t].ldc = C->stride;

        row += thread_rows;
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

    // Any thread that never got created still has rows to compute, so run its
    // slice here rather than silently leaving part of C unwritten
    for (size_t t = spawned; t < nthreads; t++) {
        gemm_tiled_simd_kernel(args[t].m_rows, args[t].N, args[t].K,
                               args[t].A, args[t].lda,
                               args[t].B, args[t].ldb,
                               args[t].C, args[t].ldc);
    }

    // Join whichever threads actually got created
    for (size_t t = 0; t < spawned; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
}
