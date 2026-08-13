#include "validate.hpp"
#include <math.h>
#include <stdio.h>

// Tolerance scaled by the value, since a flat 1e-4 breaks down at large K:
// results near 256 have a float ULP of 3e-5, and FMA and tiling reorder the
// summation so the kernels legitimately differ in the last bits.
#define ATOL 1e-5f
#define RTOL 1e-5f

bool tensors_match(const Tensor &expected, const Tensor &actual) {
    //see if the shape is actually the same
    if (expected.shape(0) != actual.shape(0) || expected.shape(1) != actual.shape(1)) {
        fprintf(stderr,
                "Shape mismatch (expected=%zux%zu, actual=%zux%zu)\n",
                expected.shape(0), expected.shape(1), actual.shape(0), actual.shape(1));
        return false;
    }

    // Two index expressions, one per tensor. Using the expected tensor's
    // stride for both goes wrong the moment the two differ.
    for (size_t i = 0; i < expected.shape(0); i++) {
        for (size_t j = 0; j < expected.shape(1); j++) {
            size_t e_index = i * expected.stride(0) + j;
            size_t a_index = i * actual.stride(0) + j;

            float diff = fabsf(expected.data()[e_index] - actual.data()[a_index]);

            if (diff > ATOL + RTOL * fabsf(expected.data()[e_index])) {
                fprintf(stderr,
                        "Mismatch at [%zu][%zu]: expected=%f, actual=%f, diff=%f\n",
                        i, j, expected.data()[e_index], actual.data()[a_index], diff);
                return false;
            }
        }
    }

    return true;
}