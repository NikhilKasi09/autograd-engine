#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Alignment for AVX2 registers. Applies to the base pointer only, not to
// where any individual row starts.
#define ALIGNMENT_REQ 32

typedef struct {
    size_t rows;
    size_t cols;

    // Floats from the start of one row to the start of the next.
    // matrix_create always sets this to cols, so there is no padding. It stays
    // a separate field because the invariant is stride >= cols, which is how a
    // view onto a bigger matrix arrives later without changing any kernel.
    size_t stride;

    float *data;
} matrix_t;

// Allocates and zero-initialises a rows x cols matrix, row-major.
// The block is 32-byte aligned but row i sits at data + i * stride, which is
// only aligned when stride % 8 == 0, hence the unaligned loads in the kernels.
// Returns NULL on a zero dimension or a failed allocation.
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
