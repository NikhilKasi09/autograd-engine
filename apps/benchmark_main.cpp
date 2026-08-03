#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.hpp"
#include "gemm.hpp"
#include "benchmark.hpp"
#include "validate.hpp"

void gemm_multithreaded_wrapper(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    // Spawning 8 threads for the benchmark run
    gemm_multithreaded(A, B, C, 8);
}

// Struct to hold kernel metadata for our testing loop
typedef struct {
    const char *name;
    gemm_kernel_ptr func;
} kernel_info_t;

// C is M x N, A is M x K, B is K x N
typedef struct {
    size_t M, N, K;
    const char *note;
} bench_shape_t;

int main(void) {
    bench_shape_t shapes[] = {
        { 256,  256,  256, "square"},
        { 512,  512,  512, "square"},
        {1024, 1024, 1024, "square"},
        { 128,  256,  512, "non-square"},
        { 512, 1024,  256, "non-square"},
        {1023, 1023, 1023, "ragged, worst case for the scalar strips"}
    };
    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);

    // Full ladder, slowest first. The slower rungs stay in on purpose.
    kernel_info_t kernels[] = {
        {"Naive (ijk)",    gemm_naive},
        {"Local (ikj)",    gemm_ikj},
        {"Tiled",          gemm_tiled},
        {"AVX2",           gemm_avx2},
        {"Tiled + AVX2",   gemm_tiled_simd},
        {"Multithreaded",  gemm_multithreaded_wrapper}
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    for (int s = 0; s < num_shapes; s++) {
        size_t M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;

        matrix_t *A = matrix_create(M, K);
        matrix_t *B = matrix_create(K, N);
        matrix_t *C = matrix_create(M, N);
        matrix_t *expected_C = matrix_create(M, N);

        if (!A || !B || !C || !expected_C) {
            fprintf(stderr, "Fatal: Memory allocation failed for %zux%zux%zu\n", M, N, K);
            return 1;
        }

        matrix_randomize(A);
        matrix_randomize(B);

        printf("%zu x %zu x %zu  (%s)\n", M, N, K, shapes[s].note);
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
                // For all other kernels, prove they match the Naive output
                if (!matrices_match(expected_C, C)) {
                    fprintf(stderr, "Validation failed for %s at %zux%zux%zu\n",
                            kernels[k].name, M, N, K);
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
