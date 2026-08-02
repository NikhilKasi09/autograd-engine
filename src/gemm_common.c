#include "gemm_internal.h"
#include <stdio.h>

int gemm_check_shapes(const char *who, const matrix_t *A, const matrix_t *B,
                      const matrix_t *C) {
    // A * B has to be defined and C has to be the right shape to hold it
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "%s: shape mismatch (%zux%zu * %zux%zu into %zux%zu)\n", who, A->rows,
                A->cols, B->rows, B->cols, C->rows, C->cols);
        return 0;
    }

    return 1;
}
