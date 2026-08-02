// posix_memalign is POSIX rather than ISO C, and -pedantic only lets it
// through today because gcc defaults to gnu17. Ask for it explicitly.
#define _POSIX_C_SOURCE 200112L

#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h> /* SIZE_MAX */

matrix_t *matrix_create(size_t rows, size_t cols) {
    // Both dimensions are size_t, so the old 'size <= 0' could never fire
    if (rows == 0 || cols == 0) {
        fprintf(stderr, "Error: Matrix dimensions must be greater than 0.\n");
        return NULL;
    }

    // rows * cols has to fit in a size_t before it can be multiplied by
    // sizeof(float), otherwise the allocation silently comes back too small
    if (rows > SIZE_MAX / cols) {
        fprintf(stderr, "Error: Matrix dimensions %zu x %zu overflow size_t.\n", rows, cols);
        return NULL;
    }

    // Allocate standard memory for the struct
    matrix_t *mat = malloc(sizeof(matrix_t));
    // Ensure malloc succeeded
    if (mat == NULL) {
        fprintf(stderr, "Error: malloc failed to allocate matrix struct.\n");
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;

    // No padding: rows are packed back to back. The kernels handle ragged
    // edges themselves, with a scalar loop for whatever the vector loop
    // cannot cover.
    mat->stride = cols;

    size_t num_elements = rows * cols;
    size_t num_bytes    = num_elements * sizeof(float);

    // Allocate the 32-byte aligned memory for the matrix data. This aligns the
    // base pointer only: row i sits at data + i * stride, aligned only when
    // stride % 8 == 0, which is why the kernels use unaligned loads. Keeping
    // it costs nothing and leaves room for a padded-stride experiment later.
    void *aligned_ptr = NULL;
    int align_status = posix_memalign(&aligned_ptr, ALIGNMENT_REQ, num_bytes);

    if (align_status != 0) {
        fprintf(stderr, "Error: posix_memalign failed to allocate 32-byte aligned memory.\n");
        // Prevent memory leak by freeing the allocated struct
        free(mat);
        return NULL;
    }

    mat->data = (float *)aligned_ptr;

    // Zero out the memory using memset to fill the array with 0s
    memset(mat->data, 0, num_bytes);

    return mat;
}

void matrix_free(matrix_t *mat) {
    // Defensive check to prevent double-freeing / segfaults
    if (mat == NULL) {
        return;
    }

    // Free data block first
    if (mat->data != NULL) {
        free(mat->data);
    }

    // Free struct itself
    free(mat);
}

void matrix_randomize(matrix_t *mat) {
    if (mat == NULL || mat->data == NULL) {
        return;
    }

    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            mat->data[i * mat->stride + j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void matrix_zero(matrix_t *mat) {
    if (mat == NULL || mat->data == NULL) {
        return;
    }

    // One memset per row, not one for rows * stride. Same thing while
    // stride == cols, but stays correct once a matrix can be a view.
    for (size_t i = 0; i < mat->rows; i++) {
        memset(&mat->data[i * mat->stride], 0, mat->cols * sizeof(float));
    }
}

void matrix_copy(matrix_t *dst, const matrix_t *src) {
    if (dst == NULL || src == NULL || dst->data == NULL || src->data == NULL) {
        return;
    }

    if (dst->rows != src->rows || dst->cols != src->cols) {
        fprintf(stderr, "Error: matrix_copy shape mismatch (%zux%zu into %zux%zu).\n",
                src->rows, src->cols, dst->rows, dst->cols);
        return;
    }

    // Copied row by row, since the two matrices can have different strides
    for (size_t i = 0; i < dst->rows; i++) {
        memcpy(&dst->data[i * dst->stride], &src->data[i * src->stride],
               dst->cols * sizeof(float));
    }
}
