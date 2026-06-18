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

    // Actually compare values now using a 2D loop to skip the padding buffer
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            // Calculate the physical memory index using the padded stride
            size_t index = i * expected->padded_size + j;
            
            float diff = fabsf(expected->data[index] - actual->data[index]);

            if (diff > TOLERANCE) {
                fprintf(stderr,
                        "Mismatch at logical [%zu][%zu] (physical index %zu):\n"
                        "expected=%f, actual=%f, diff=%f\n",
                        i, j, index, expected->data[index], actual->data[index], diff);
                exit(1);
            }
        }
    }

    return 1;
}