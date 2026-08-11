#ifndef ALIGNED_H
#define ALIGNED_H

#include <cstddef>
#include <new> // std::align_val_t

// Alignment for AVX2 registers. Applies to the base pointer only, not to
// where any individual row starts.
#define ALIGNMENT_REQ 32

// Releases a block obtained from the matching aligned operator new. Aligned
// allocation and deallocation must be paired: handing an over-aligned pointer
// to plain operator delete is undefined behaviour, which is exactly the bug
// this type exists to make unrepresentable.
struct AlignedDeleter {
    // Defined in-class, so implicitly inline and needing no .cpp of its own.
    // The body used to live in matrix.cpp, which Tensor outlives.
    void operator()(float *p) const noexcept {
        ::operator delete(p, std::align_val_t{ALIGNMENT_REQ});
    }
};

// The kernels read 8 floats per AVX2 vector; ALIGNMENT_REQ is 32 bytes for
// that reason and not by coincidence.
static_assert(sizeof(float) == 4, "AVX2 kernels assume 4-byte floats");
static_assert(ALIGNMENT_REQ == 8 * sizeof(float),
              "ALIGNMENT_REQ should be one AVX2 vector wide");

#endif
