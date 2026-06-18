#ifndef VALIDATE_H
#define VALIDATE_H

#include "matrix.h"

// Compares two matrices element-by-element. Returns 1 if they match, 0 otherwise.
int matrices_match(const matrix_t *expected, const matrix_t *actual);

#endif