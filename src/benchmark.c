#include "benchmark.h"
#include <time.h>
#include <string.h>

#define TIMED_RUNS    5

benchmark_result_t run_benchmark(gemm_kernel_ptr kernel, const matrix_t *A, const matrix_t *B, matrix_t *C) {
    
    struct timespec start, end;
    benchmark_result_t result;
    double times[TIMED_RUNS];

    // Scrub C clean before the warmup run
    matrix_zero(C);

    // Warmup run for cold cache
    kernel(A, B, C);

    // Volatile sink forcing GCC to actually execute the kernel
    volatile float sink_warmup = C->data[0];
    (void)sink_warmup; // Suppresses the "unused variable" warning

    // Timed hot runs 
    for (int i = 0; i < TIMED_RUNS; i++) {

        // Scrub C clean BEFORE starting the hardware clock
        matrix_zero(C);
        
        // Start clock
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Execute kernel
        kernel(A, B, C);
        
        // Stop clock
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Force compiler materialization for every loop iteration
        volatile float sink = C->data[0];
        (void)sink;

        // Stitch together results and put it into the array 
        times[i] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    // Calculate the median using a simple bubble sort
    for (int i = 0; i < TIMED_RUNS - 1; i++) {
        for (int j = i + 1; j < TIMED_RUNS; j++) {
            if (times[i] > times[j]) {
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }
        }
    }

    // Median is at index 2
    double median_seconds = times[2];
    result.elapsed_seconds = median_seconds;

    // Calculate GigaFLOP/s. Every factor is cast to double before being
    // multiplied, otherwise M * N * K overflows in size_t first.
    double total_flops = 2.0 * (double)C->rows * (double)C->cols * (double)A->cols;

    // Prevent division by zero if the clock was too fast
    if (median_seconds > 0.0) {
        result.gigaflops = total_flops / (median_seconds * 1e9);
    } else {
        result.gigaflops = 0.0; 
    }

    return result;
}