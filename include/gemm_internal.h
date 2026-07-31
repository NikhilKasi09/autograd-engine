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
