#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "gemm.h"
#include "benchmark.h"
#include "validate.h"

void gemm_multithreaded_wrapper(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Spawning 8 threads for the benchmark run
    gemm_multithreaded(A, B, C, 8); 
}

// Struct to hold kernel metadata for our testing loop
typedef struct {
    const char *name;
    gemm_kernel_ptr func;
} kernel_info_t;

int main() {
    size_t sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Registry tailored to your specific output format
    kernel_info_t kernels[] = {
        {"Naive (ijk)", gemm_naive},
        {"Local (ikj)", gemm_ikj},
        {"Tiled + AVX2", gemm_tiled_simd},
        {"Multithreaded", gemm_multithreaded_wrapper}
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t N = sizes[s];
        
        matrix_t *A = matrix_create(N, N);
        matrix_t *B = matrix_create(N, N);
        matrix_t *C = matrix_create(N, N);
        matrix_t *expected_C = matrix_create(N, N);

        if (!A || !B || !C || !expected_C) {
            fprintf(stderr, "Fatal: Memory allocation failed for size %zu\n", N);
            return 1;
        }

        matrix_randomize(A);
        matrix_randomize(B);

        printf("Matrix Size: %zu x %zu\n", N, N);
        printf("--------------------------------------------------\n");
        printf("%-18s | %-9s | %s\n", "Implementation", "Time (ms)", "Performance");
        printf("--------------------------------------------------\n");

        double naive_time = 0.0;
        double final_time = 0.0;

        for (int k = 0; k < num_kernels; k++) {
            // Run the benchmark
            benchmark_result_t res = run_benchmark(kernels[k].func, A, B, C);

            if (k == 0) {
                // If this is the Naive run, save its output as the ultimate source of truth
                matrix_copy(expected_C, C);
                naive_time = res.elapsed_seconds;
            } else {
                // For all other kernels, prove they match the Naive output.
                // matrices_match now reports and returns 0 rather than calling
                // exit(1) itself, so the bail-out lives here.
                if (!matrices_match(expected_C, C)) {
                    fprintf(stderr, "Validation failed for %s at size %zu\n",
                            kernels[k].name, N);
                    return 1;
                }
            }

            if (k == num_kernels - 1) {
                final_time = res.elapsed_seconds;
            }

            // Print formatted row
            printf("%d. %-15s | %-9.2f | %.2f GFLOP/s\n", 
                   k + 1,
                   kernels[k].name, 
                   res.elapsed_seconds * 1000.0, 
                   res.gigaflops);
        }
        
        printf("--------------------------------------------------\n");
        
        double total_speedup = (final_time > 0.0) ? (naive_time / final_time) : 0.0;
        printf("Total Speedup: %.1fx\n\n", total_speedup);

        // Free all dynamically allocated memory
        matrix_free(A);
        matrix_free(B);
        matrix_free(C);
        matrix_free(expected_C); 
    }

    return 0;
}