#include "validate.h"
#include <math.h>
#include <stdio.h>

//tolerance to check if two floats the same
#define TOLERANCE 1e-4f

int matrices_match(const matrix_t *expected, const matrix_t *actual) {
    //see if the shape is actually the same
    if (expected->rows != actual->rows || expected->cols != actual->cols) {
        fprintf(stderr,
                "Shape mismatch (expected=%zux%zu, actual=%zux%zu)\n",
                expected->rows, expected->cols, actual->rows, actual->cols);
        return 0;
    }

    // Two index expressions, not one. The old code indexed both matrices with
    // the expected matrix's stride, which is wrong the moment the two differ.
    for (size_t i = 0; i < expected->rows; i++) {
        for (size_t j = 0; j < expected->cols; j++) {
            size_t e_index = i * expected->stride + j;
            size_t a_index = i * actual->stride + j;

            float diff = fabsf(expected->data[e_index] - actual->data[a_index]);

            if (diff > TOLERANCE) {
                fprintf(stderr,
                        "Mismatch at [%zu][%zu]: expected=%f, actual=%f, diff=%f\n",
                        i, j, expected->data[e_index], actual->data[a_index], diff);
                return 0;
            }
        }
    }

    return 1;
}