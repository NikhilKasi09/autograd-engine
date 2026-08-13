#include "gemm.hpp"
#include "gemm_internal.hpp"
#include <thread> 
#include <vector>
#include <cstdio>
#include <system_error>


void gemm_multithreaded(const Tensor &A, const Tensor &B, Tensor &C, int num_threads) {
    if (!gemm_check_shapes("gemm_multithreaded", A, B, C)) {
        return;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "gemm_multithreaded: num_threads must be positive\n");
        return;
    }

    // Slicing is over the rows of C, so M is what gets divided up
    const size_t M = C.shape(0);
    const size_t N = C.shape(1);
    const size_t K = A.shape(1);

    // Never spin up more threads than there are rows to hand out
    size_t nthreads = (size_t)num_threads;
    if (nthreads > M) {
        nthreads = M;
    }

    
    std::vector<std::jthread> workers; // Create a mutable container of workers
    workers.reserve(nthreads);
    
    size_t nblocks = M / GEMM_MR; // How many 4 row blocks fit in the matrix
    size_t tail = M % GEMM_MR; // How many left over
    size_t row = 0;

    for (size_t t = 0; t < nthreads; t++){

        size_t thread_rows;
        
        if (nblocks < nthreads){ // Forget blocks, divide rows evenly
            thread_rows = M / nthreads + (t < M % nthreads? 1 : 0);
        } else { // Plenty of blocks to go around
            size_t base = nblocks / nthreads;
            size_t extra = nblocks % nthreads;
            thread_rows = (base + (t < extra ? 1 : 0)) * GEMM_MR;

            if (t == nthreads - 1){
                thread_rows += tail;
            }
        }
        
        // Captures the loop variables by value: row and thread_rows change on
        // the next iteration, so a worker holding references to them would read
        // whatever they became. A, B and C are safe by reference because the
        // jthread destructors join before this scope ends.
        auto run_slice = [&A, &B, &C, row, thread_rows, N, K]() {
            gemm_tiled_simd_kernel(
                thread_rows, N, K,
                A.data() + row * A.stride(0), A.stride(0),
                B.data(),                    B.stride(0),
                C.data() + row * C.stride(0), C.stride(0)
            );
        };

        try {
            workers.emplace_back(run_slice);
        } catch (const std::system_error& e) {
            fprintf(stderr, "gemm_multithreaded: thread %zu failed to spawn (%s), running inline\n", t, e.what());
            run_slice();
        }

        row += thread_rows;
    }
}
