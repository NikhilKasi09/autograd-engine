#include "validate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//tolerance to check if two floats the same
#define TOLERANCE 1e-4f

int matrices_match(const matrix_t *expected, const matrix_t *actual) {
    //see if the size is actually the same
    if (expected->size != actual->size) {
        fprintf(stderr,
                "Size mismatch (expected=%zu, actual=%zu)\n",
                expected->size, actual->size);
        exit(1);
    }

    size_t N = expected->size;
    size_t total_elements = N * N;

    //actually compare values now
    for (size_t i = 0; i < total_elements; i++) {
        float diff = fabsf(expected->data[i] - actual->data[i]);

        if (diff > TOLERANCE) {
            fprintf(stderr,
                    "Mismatch at index %zu: expected=%f, actual=%f, diff=%f\n",
                    i, expected->data[i], actual->data[i], diff);
            exit(1);
        }
    }

    return 1;
}