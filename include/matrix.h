#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Alignment requirement for AVX2 registers
#define ALIGNMENT_REQ 32
#define BLOCK_SIZE 64

typedef struct {
    size_t size;
    size_t padded_size;
    float *data;
} matrix_t;

// Allocates and zero-initializes a 32-byte aligned N x N matrix (where 'size' is N)
matrix_t *matrix_create(size_t size);

// Safely frees the aligned memory block and the struct
void matrix_free(matrix_t *mat);

// Populates matrix with random floating-point values between 0.0 and 1.0
void matrix_randomize(matrix_t *mat);

#endif