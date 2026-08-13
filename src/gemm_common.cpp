#include "gemm_internal.hpp"
#include <stdio.h>

bool gemm_check_shapes(const char *who, const Tensor &A, const Tensor &B,
                       const Tensor &C) {
                        
    if (A.rank() != 2 || B.rank() != 2 || C.rank() != 2) {
        fprintf(stderr, "%s: all operands must be rank 2 (got %zu, %zu, %zu)\n",
                who, A.rank(), B.rank(), C.rank());
        return false;
    }

    if (A.stride(1) != 1 || B.stride(1) != 1 || C.stride(1) != 1) {
        fprintf(stderr, "%s: all operands must have unit inner stride (got %zu, %zu, %zu)\n",
                who, A.stride(1), B.stride(1), C.stride(1));
        return false;
    }
    
    if (A.shape(1) != B.shape(0) || C.shape(0) != A.shape(0) || C.shape(1) != B.shape(1)) {
        fprintf(stderr, "%s: shape mismatch (%zux%zu * %zux%zu into %zux%zu)\n", who, A.shape(0),
                A.shape(1), B.shape(0), B.shape(1), C.shape(0), C.shape(1));
        return false;
    }

    return true;
}
