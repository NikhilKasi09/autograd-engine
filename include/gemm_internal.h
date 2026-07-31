#ifndef GEMM_INTERNAL_H
#define GEMM_INTERNAL_H

#include "matrix.h"

/*
 Kernel tuning parameters, shared between the kernel implementations but not
 part of the public interface.

 These used to live in matrix.h, which is how the allocator ended up padding
 every matrix out to a multiple of BLOCK_SIZE: a number describing the shape of
 a cache tile had no business being visible to the thing that allocates memory.
*/

#define BLOCK_SIZE 64 /* side of the square cache tile          */
#define GEMM_MR    4  /* rows in one register block             */
#define GEMM_NR    16 /* columns in one register block, 2 vectors */
#define GEMM_VEC   8  /* floats in one AVX2 vector              */

/*
 The cache-tiled scalar kernel, shared rather than private to gemm_tiled.c.

 C(MxN) += A(MxK) * B(KxN), row-major, any extents, no alignment or multiple-of
 requirements at all. The vector kernels call this for the ragged strips they
 cannot cover with whole vectors, so those edges reuse code that has already
 been tested on its own instead of growing a second set of hand-written edge
 loops.

 It is called on offset pointers for that: a caller wanting the sub-rectangle
 of C starting at (i0, j0) passes A + i0*lda + k0, B + k0*ldb + j0 and
 C + i0*ldc + j0 with the extents of the piece it wants. The leading dimensions
 stay those of the full matrices.

 Being tiled rather than a plain triple loop matters for those strips: an
 untiled Mx7xK strip at M = K = 1024 walks B down a column and can cost
 milliseconds against a main kernel that takes about twenty.
*/
void gemm_tiled_kernel(size_t M, size_t N, size_t K,
                       const float * restrict A, size_t lda,
                       const float * restrict B, size_t ldb,
                       float * restrict C, size_t ldc);

/*
 The shape contract for C(MxN) += A(MxK) * B(KxN): the product has to be
 defined and C has to be the right shape to hold it. Every wrapper calls this.

 Returns 1 if the shapes are usable, 0 otherwise, and reports on stderr.
*/
int gemm_check_shapes(const char *who, const matrix_t *A, const matrix_t *B,
                      const matrix_t *C);

/*
 The stronger check the kernels that are still square only need. They take a
 single N and a single stride for all three matrices, so anything rectangular
 has to be turned away rather than quietly miscomputed.

 Each kernel drops this for plain gemm_check_shapes as it is generalised, and
 the function goes away entirely at step H.
*/
int gemm_shapes_square_ok(const char *who, const matrix_t *A, const matrix_t *B,
                          const matrix_t *C);

#endif
