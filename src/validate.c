#include "validate.h"
#include <math.h>
#include <stdio.h>

//commented out code gives a detailed report on the mismatches 
#define TOLERANCE 1e-4f
// #define MAX_PRINTED_MISMATCHES 10

int matrices_match(const matrix_t *expected, const matrix_t *actual) {

    //make sure that both sizes are the same 
    if (expected->size != actual->size) {
        return 0;
    }

    //initiliasers for the tests
    size_t N = expected->size;
    size_t total_elements = N * N;
    int ok = 1;
    // int mismatches_printed = 0;
    // size_t mismatch_count = 0;

    //lop through each elements and check that they are within tolerance
    for (size_t i = 0; i < total_elements; i++) {
        float diff = fabsf(expected->data[i] - actual->data[i]);

        if (diff > TOLERANCE) {
            ok = 0;
            // mismatch_count++;

            //below will print max 10 mismatches because if there is a lot of mismatches (example 100) then it will be harder to debug.
            // if (mismatches_printed < MAX_PRINTED_MISMATCHES) {
            //     fprintf(stderr,
            //             "Mismatch at index %zu: expected=%f, actual=%f",
            //             i, expected->data[i], actual->data[i], diff);
            //     mismatches_printed++;
            // }
        }
    }


    return ok;
}