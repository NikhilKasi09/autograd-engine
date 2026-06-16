#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

matrix_t *matrix_create(size_t size) {
    // Allocate standard memory for the struct
    matrix_t *mat = malloc(sizeof(matrix_t));
    // Ensure malloc succeeded
    if (mat == NULL) {
        fprintf(stderr, "Error: malloc failed to allocate matrix struct.\n");
        return NULL;
    }
    
    mat->size = size;

    // Calculate required size in bytes for N x N float array
    size_t num_bytes = size * size * sizeof(float);

    // Allocate the 32-byte aligned memory for the matrix data
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

    size_t total_elements = mat->size * mat->size;

    // Populate array with floats between 0.0 and 1.0
    for (size_t i = 0; i < total_elements; i++) {
        mat->data[i] = (float)rand() / (float)RAND_MAX;
    }
}