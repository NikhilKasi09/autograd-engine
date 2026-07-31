#include "gemm_internal.h"
#include <stdio.h>

int gemm_shapes_square_ok(const char *who, const matrix_t *A, const matrix_t *B,
                          const matrix_t *C) {
    // A * B has to be defined and C has to be the right shape to hold it
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "%s: shape mismatch (%zux%zu * %zux%zu into %zux%zu)\n", who, A->rows,
                A->cols, B->rows, B->cols, C->rows, C->cols);
        return 0;
    }

    // The kernel bodies still take one N for all three matrices
    if (A->rows != A->cols || B->rows != B->cols || C->rows != C->cols) {
        fprintf(stderr, "%s: kernel is square only for now, got %zux%zu * %zux%zu\n", who,
                A->rows, A->cols, B->rows, B->cols);
        return 0;
    }

    // ...and one stride, which holds while nothing hands out views
    if (A->stride != B->stride || A->stride != C->stride) {
        fprintf(stderr, "%s: kernel needs one stride for all three, got %zu, %zu, %zu\n", who,
                A->stride, B->stride, C->stride);
        return 0;
    }

    return 1;
}
