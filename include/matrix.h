#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Alignment requirement for AVX2 registers. This aligns the base pointer only,
// and says nothing about where any individual row starts (see stride below).
#define ALIGNMENT_REQ 32

typedef struct {
    size_t rows;
    size_t cols;

    /*
     Floats from the start of one row to the start of the next.

     matrix_create always sets stride == cols, so there is no padding. The
     field is kept separate anyway because the invariant is stride >= cols,
     which is the seam a padded allocation or a view onto a bigger matrix can
     come through later without a single kernel having to change.
    */
    size_t stride;

    float *data;
} matrix_t;

/*
 Allocates and zero-initialises a rows x cols matrix, row-major.

 The data block is 32-byte aligned, but row i starts at data + i * stride and
 is only 32-byte aligned when stride % 8 == 0, which nothing guarantees. The
 kernels use unaligned loads for that reason.

 Returns NULL if either dimension is zero or the allocation fails.
*/
matrix_t *matrix_create(size_t rows, size_t cols);

// Safely frees the aligned memory block and the struct
void matrix_free(matrix_t *mat);

// Populates the matrix with random floating-point values between 0.0 and 1.0
void matrix_randomize(matrix_t *mat);

// Sets every element to zero, skipping anything between cols and stride
void matrix_zero(matrix_t *mat);

// Copies src into dst element by element. The two are allowed to have
// different strides, but must agree on rows and cols.
void matrix_copy(matrix_t *dst, const matrix_t *src);

#endif
