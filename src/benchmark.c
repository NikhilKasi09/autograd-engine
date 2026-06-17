#include "benchmark.h"
#include <time.h>

benchmark_result_t run_benchmark(gemm_kernel_ptr kernel, const matrix_t *A, const matrix_t *B, matrix_t *C){
    
    struct timespec start, end;

    benchmark_result_t result;

    // Start the hardware clock
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Execute the kernel 
    (*kernel)(A, B, C);

    // Stop the hardware clock
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Stitch together the seconds and nanoseconds
    double seconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    result.elapsed_seconds = seconds;

    // Calculate GigaFLOP/s
    double n_double = (double)A -> size;
    double total_flops = 2.0 * n_double * n_double * n_double;

    // Prevent division by zero if the clock was too fast
    if (seconds > 0.0) {
        result.gigaflops = total_flops / (seconds * 1e9);
    } else {
        result.gigaflops = 0.0; 
    }

    return result;
}