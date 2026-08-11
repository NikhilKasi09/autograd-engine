#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.hpp"
#include "gemm.hpp"
#include "benchmark.hpp"
#include "validate.hpp"
#include <stdexcept>
#include <thread>

// run_benchmark takes a plain function pointer, so the thread count cannot be
// captured and has to live here. Defaults to every hardware thread; override
// with a single argument, e.g. ./gemm_benchmark 8.
namespace {
int bench_threads = 8;
}

void gemm_multithreaded_wrapper(const Matrix &A, const Matrix &B, Matrix &C) {
    gemm_multithreaded(A, B, C, bench_threads);
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

int main(int argc, char **argv) {
    // hardware_concurrency reports 0 when it cannot tell, so fall back rather
    // than asking gemm_multithreaded for zero threads.
    const unsigned detected = std::thread::hardware_concurrency();
    bench_threads = (detected > 0) ? (int)detected : 8;

    if (argc > 1) {
        const int requested = atoi(argv[1]);
        if (requested <= 0) {
            fprintf(stderr, "usage: %s [num_threads]\n", argv[0]);
            return 1;
        }
        bench_threads = requested;
    }

    printf("Multithreaded rung uses %d threads\n\n", bench_threads);

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

    try{

        for (int s = 0; s < num_shapes; s++) {
            size_t M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;

            Matrix A(M, K);
            Matrix B(K, N);
            Matrix C(M, N);
            Matrix expected_C(M, N);

            A.randomize();
            B.randomize();

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
                    expected_C = C.clone();
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
        }
    } catch (const std::exception &e) {
    fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
}
 
    return 0;
}
