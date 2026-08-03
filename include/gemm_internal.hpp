#ifndef GEMM_INTERNAL_H
#define GEMM_INTERNAL_H

#include "matrix.hpp"

// Cs `restrict` was never adopted into C++. __restrict is the compiler
// extension with identical semantics, spelled the same way by gcc, clang and
// MSVC. Load-bearing for performance, not cosmetic — see docs/cpp-notes.md.
#define GEMM_RESTRICT __restrict

// Kernel tuning parameters. These used to live in matrix.h, which is how the
// allocator ended up padding every matrix to a multiple of BLOCK_SIZE.
#define BLOCK_SIZE 64 /* side of the square cache tile            */
#define GEMM_MR    4  /* rows in one register block               */
#define GEMM_NR    16 /* columns in one register block, 2 vectors */
#define GEMM_VEC   8  /* floats in one AVX2 vector                */

// Cache-tiled scalar kernel. Any extents, no alignment requirements.
// The vector kernels call this on offset pointers for their ragged strips, so
// the edges reuse tested code instead of a second set of edge loops. Tiled
// rather than a plain triple loop because an Mx7xK strip at K=1024 walks B
// down a column and is slow enough to matter.
void gemm_tiled_kernel(size_t M, size_t N, size_t K,
                       const float * GEMM_RESTRICT A, size_t lda,
                       const float * GEMM_RESTRICT B, size_t ldb,
                       float * GEMM_RESTRICT C, size_t ldc);

// Tiled + AVX2 kernel, shared so the threaded wrapper can call it too.
// Peels its own ragged rows and columns, so any sub-rectangle is fine.
void gemm_tiled_simd_kernel(size_t M, size_t N, size_t K,
                            const float * GEMM_RESTRICT A, size_t lda,
                            const float * GEMM_RESTRICT B, size_t ldb,
                            float * GEMM_RESTRICT C, size_t ldc);

// What one worker thread needs to compute its slice of C. The pointers are
// pre-offset, so a worker just runs a GEMM of m_rows by N and never needs to
// know where its slice sits in the full matrix.
typedef struct {
    size_t m_rows; // rows of C this worker owns
    size_t N;
    size_t K;

    const float *A;   // offset to the worker's first row
    size_t       lda;
    const float *B;   // not offset, every worker reads all of B
    size_t       ldb;
    float       *C;   // offset to the worker's first row
    size_t       ldc;
} thread_args_t;

// Checks A(MxK) * B(KxN) is defined and C is the right shape to hold it.
// Returns 1 if usable, 0 otherwise, reporting on stderr. Every wrapper calls it.
int gemm_check_shapes(const char *who, const matrix_t *A, const matrix_t *B,
                      const matrix_t *C);

#endif
