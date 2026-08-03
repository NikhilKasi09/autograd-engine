#include "validate.hpp"
#include <math.h>
#include <stdio.h>

// Tolerance scaled by the value, since a flat 1e-4 breaks down at large K:
// results near 256 have a float ULP of 3e-5, and FMA and tiling reorder the
// summation so the kernels legitimately differ in the last bits.
#define ATOL 1e-5f
#define RTOL 1e-5f

int matrices_match(const matrix_t *expected, const matrix_t *actual) {
    //see if the shape is actually the same
    if (expected->rows != actual->rows || expected->cols != actual->cols) {
        fprintf(stderr,
                "Shape mismatch (expected=%zux%zu, actual=%zux%zu)\n",
                expected->rows, expected->cols, actual->rows, actual->cols);
        return 0;
    }

    // Two index expressions, one per matrix. Using the expected matrix's
    // stride for both goes wrong the moment the two differ.
    for (size_t i = 0; i < expected->rows; i++) {
        for (size_t j = 0; j < expected->cols; j++) {
            size_t e_index = i * expected->stride + j;
            size_t a_index = i * actual->stride + j;

            float diff = fabsf(expected->data[e_index] - actual->data[a_index]);

            if (diff > ATOL + RTOL * fabsf(expected->data[e_index])) {
                fprintf(stderr,
                        "Mismatch at [%zu][%zu]: expected=%f, actual=%f, diff=%f\n",
                        i, j, expected->data[e_index], actual->data[a_index], diff);
                return 0;
            }
        }
    }

    return 1;
}