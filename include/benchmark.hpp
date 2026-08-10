#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "matrix.hpp"

typedef struct {
    double elapsed_seconds;
    double gigaflops;
} benchmark_result_t;

typedef void (*gemm_kernel_ptr)(const Matrix &A, const Matrix &B, Matrix &C);

/**
 * @brief Wraps a hardware timer around a math kernel and calculates throughput.
 *
 * @param kernel A function pointer to the specific GEMM implementation.
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param C The output matrix, accumulated into.
 * @return A struct containing the exact execution time and GigaFLOP/s.
 */
benchmark_result_t run_benchmark(gemm_kernel_ptr kernel, const Matrix &A, const Matrix &B, Matrix &C);

#endif