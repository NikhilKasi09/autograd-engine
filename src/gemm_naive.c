#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


void gemm_naive(int N, float *A, float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}
