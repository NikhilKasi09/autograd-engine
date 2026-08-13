#ifndef VALIDATE_H
#define VALIDATE_H

#include "tensor.hpp"

// Compares two rank-2 tensors element-by-element. Returns true if they match.
bool tensors_match(const Tensor &expected, const Tensor &actual);

#endif
