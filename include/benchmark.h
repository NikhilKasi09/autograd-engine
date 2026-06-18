#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "matrix.h"

typedef struct {
    double elapsed_seconds;
    double gigaflops;
} benchmark_result_t;

typedef void (*gemm_kernel_ptr)(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Wraps a hardware timer around a math kernel and calculates throughput.
 *
 * @param kernel A function pointer to the specific GEMM implementation.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param C Pointer to the output matrix.
 * @return A struct containing the exact execution time and GigaFLOP/s.
 */
benchmark_result_t run_benchmark(gemm_kernel_ptr kernel, const matrix_t *A, const matrix_t *B, matrix_t *C);

#endif