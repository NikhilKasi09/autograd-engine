#ifndef GEMM_H
#define GEMM_H

#include "matrix.h"

/*
 All matrices are row-major, and every kernel computes C += A * B rather than
 C = A * B. C has to be zeroed by the caller if the plain product is what is
 wanted, and accumulating is what the backward pass will rely on later.

 None of these kernels may be called with C aliasing A or B. The inner loops
 are written with restrict pointers, so gemm(A, B, A) is undefined behaviour
 rather than a slow path.

 Kernels are being generalised from square-only to arbitrary M, N and K one at
 a time. Each one below says what it currently accepts, and its wrapper
 reports on stderr and returns without touching C if given anything else.
*/

/*
 Packages the matrix pointers, dimensions, and row boundaries required 
 for a single worker thread to compute its assigned horizontal slice
*/
typedef struct {
    size_t start_row;
    size_t end_row;
    size_t logical_size;
    size_t stride;
    
    const float *A;
    const float *B;
    float *C;
} thread_args_t;

/**
 * @brief Computes the baseline naive matrix multiplication (C = A * B).
 *
 * This function implements the standard i-j-k nested scalar loop architecture.
 * It purposefully lacks spatial locality and vectorization to establish a
 * worst-case execution time and L1 cache miss baseline for the Mini-BLAS library.
 *
 * Accepts any M, N and K, including degenerate ones such as 1xNx1.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct.
 */
void gemm_naive(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Computes matrix multiplication (C = A * B) using AVX2 SIMD intrinsics.
 *
 * Processes 8 single-precision floats per clock cycle using 256-bit ymm registers.
 * Uses unaligned loads, since only the base pointer of a matrix is 32-byte
 * aligned and row i sits at data + i * stride.
 *
 * The j loop has no scalar tail yet, so N must currently be a multiple of 8.
 * The wrapper rejects anything else. Lifted at step F.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct. Must be zero-initialised before call.
 */
void gemm_avx2(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Computes matrix multiplication using a 64x64 Cache-Tiled Architecture.
 *
 * This function divides the matrices into 16KB sub-blocks to ensure
 * the working set remains completely resident within the L1 hardware cache
 * during computation. The tile loops clamp against the matrix dimensions, so
 * a trailing tile is simply smaller than the rest.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct.
 */
void gemm_tiled(const matrix_t *A, const matrix_t *B, matrix_t *C);

/**
 * @brief Computes matrix multiplication by fusing a 64x64 Cache-Tiled Architecture with 256-bit AVX2 SIMD intrinsics.
 *
 * This function achieves peak CPU throughput by locking sub-blocks into the L1 
 * hardware cache while vectorizing the innermost computations. It utilizes an 
 * i-j-k loop reordering to accumulate fused multiply-adds (FMA) entirely within
 * hardware registers, eliminating store-forwarding stalls.
 *
 * The 256-bit loads and stores are the unaligned forms, since only the base
 * pointer of a matrix is 32-byte aligned. Neither j loop has a scalar tail
 * yet, so N must currently be a multiple of 8 and the wrapper rejects anything
 * else. Lifted at step G.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct.
 */
void gemm_tiled_simd(const matrix_t *A, const matrix_t *B, matrix_t *C);


/**
 * @brief Computes matrix multiplication (C = A * B) by horizontally partitioning
 *        the output matrix across a pool of POSIX threads.
 *
 * Splits C into num_threads horizontal row-slices (see thread_args_t), so that
 * each worker thread owns a disjoint, contiguous range of rows and never writes
 * to a cache line touched by another thread (avoiding false sharing). Each
 * worker computes its slice using the same 64x64 cache-tiled, AVX2-vectorized
 * kernel as gemm_tiled_simd, restricted to its assigned row boundaries.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct. Must be zero-initialised before call.
 * @param num_threads Number of worker threads to spawn (clamped to matrix size).
 */
void gemm_multithreaded(const matrix_t *A, const matrix_t *B, matrix_t *C, int num_threads);

/**
 * @brief Computes matrix multiplication (C = A * B) using cache-aware loop reordering.
 *
 * This function implements the i-k-j nested loop order, swapping the position of the
 * j and k loops relative to the naive i-j-k implementation. By fixing row i of A and
 * row k of B for each middle-loop iteration, the innermost j loop walks through B and
 * C sequentially in row-major order. This basically eliminates the cache misses by exploting spatial locality.
 *
 * Accepts any M, N and K.
 *
 * @param A Pointer to the first input matrix struct (read-only).
 * @param B Pointer to the second input matrix struct (read-only).
 * @param C Pointer to the output matrix struct. Must be zero-initialised before call,
 *          since this function accumulates into existing values rather than overwriting.
 */
void gemm_ikj(const matrix_t *A, const matrix_t *B, matrix_t *C);


#endif