#ifndef VALIDATE_H
#define VALIDATE_H

#include "matrix.hpp"

// Compares two matrices element-by-element. Returns true if they match.
bool matrices_match(const Matrix &expected, const Matrix &actual);

#endif
