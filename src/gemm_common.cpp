#include "gemm_internal.hpp"
#include <stdio.h>

bool gemm_check_shapes(const char *who, const Tensor &A, const Tensor &B,
                       const Tensor &C) {
    // A * B has to be defined and C has to be the right shape to hold it
    if (A.shape(1) != B.shape(0) || C.shape(0) != A.shape(0) || C.shape(1) != B.shape(1)) {
        fprintf(stderr, "%s: shape mismatch (%zux%zu * %zux%zu into %zux%zu)\n", who, A.shape(0),
                A.shape(1), B.shape(0), B.shape(1), C.shape(0), C.shape(1));
        return false;
    }

    return true;
}
