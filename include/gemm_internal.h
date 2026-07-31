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
 Shape check shared by all six wrappers.

 The kernel bodies are still square only, so they take a single N and a single
 stride for all three matrices, and the wrappers have to reject anything else
 rather than compute nonsense. This gets replaced by the general
 gemm_check_shapes at step C, as each kernel learns to handle M, N and K
 separately.

 Returns 1 if the shapes are usable, 0 otherwise, and reports on stderr.
*/
int gemm_shapes_square_ok(const char *who, const matrix_t *A, const matrix_t *B,
                          const matrix_t *C);

#endif
