// Shape-driven correctness harness for the GEMM ladder.
//
// Three properties are load-bearing and survive from the hand-written version:
//
//   1. Every kernel is compared against an independent dense reference, naive
//      included, rather than using naive as the oracle. A wrong reference would
//      otherwise silently bless every kernel at once.
//   2. Every case runs twice, with C zeroed and with C prefilled. The contract
//      is C += A*B; zeroing C before every call makes = and += indistinguishable.
//   3. Restriction bitmasks let a half-finished kernel report SKIP with a reason
//      instead of failing the build. All masks are empty now; the mechanism
//      stays for the next kernel that lands incrementally.
//
// Filtering is Catch2's now: `gemm_tests "[gemm]"` for the table, and the fuzz
// run is hidden behind a dot tag - `gemm_tests "[.fuzz]"` - so it costs nothing
// unless asked for.

#include "gemm.hpp"
#include "tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <vector>

namespace {

// Fixed seed so a failure is reproducible. Inputs are reseeded per shape from
// the dimensions, so one shape's data never depends on what ran before it.
constexpr unsigned SEED = 0xC0FFEE;

// Tolerance scaled by the value. A flat 1e-4 breaks down at large K: results
// near 256 have a float ULP of 3e-5, and FMA and tiling reorder the summation
// so the kernels legitimately differ in the last bits.
constexpr float ATOL = 1e-5f;
constexpr float RTOL = 1e-5f;

constexpr int FUZZ_ITERS   = 300;
constexpr int FUZZ_MAX_DIM = 80;

/* ------------------------------------------------------------------------ */
/* Shape restrictions                                                        */
/* ------------------------------------------------------------------------ */

enum KernelRestrict : unsigned {
    RESTRICT_SQUARE   = 1u << 0, /* requires M == N == K                     */
    RESTRICT_N_MULT8  = 1u << 1, /* requires N % 8 == 0 (vector width)       */
    RESTRICT_N_MIN16  = 1u << 2, /* requires N >= 16    (j-underflow, bug 2) */
    RESTRICT_M_MIN4   = 1u << 3, /* requires M >= 4     (i-underflow, bug 2) */
    RESTRICT_MT_4ROWS = 1u << 4  /* requires M/nthreads >= 4        (bug 1)  */
};

// Returns the name of the first violated restriction, or nullptr if this kernel
// can legally be run on this shape
const char *restriction_violated(unsigned mask, size_t M, size_t N, size_t K, int nthreads) {
    if ((mask & RESTRICT_SQUARE) && !(M == N && N == K)) {
        return "SQUARE";
    }
    if ((mask & RESTRICT_N_MULT8) && (N % 8u) != 0u) {
        return "N_MULT8";
    }
    if ((mask & RESTRICT_N_MIN16) && N < 16u) {
        return "N_MIN16";
    }
    if ((mask & RESTRICT_M_MIN4) && M < 4u) {
        return "M_MIN4";
    }
    if (mask & RESTRICT_MT_4ROWS) {
        // Mirror the clamp in gemm_multithreaded: never more threads than rows
        size_t nt = (nthreads > 0) ? (size_t)nthreads : 1u;
        if (nt > M) {
            nt = M;
        }
        if (nt == 0u || M / nt < 4u) {
            return "MT_4ROWS";
        }
    }
    return nullptr;
}

/* ------------------------------------------------------------------------ */
/* Reference implementation                                                  */
/* ------------------------------------------------------------------------ */

// C(MxN) += A(MxK) * B(KxN), all three packed with no leading dimensions.
// The simplest loop nest that could work, on purpose.
//
// The accumulator is a double. This is what every kernel is judged against, so
// it should carry less rounding error than they do, not the same amount in a
// different order.
void reference_gemm(size_t M, size_t N, size_t K, const float *A, const float *B, float *C) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double acc = (double)C[i * N + j];
            for (size_t k = 0; k < K; k++) {
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)acc;
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Test data                                                                 */
/* ------------------------------------------------------------------------ */

struct Shape {
    size_t M, N, K;
};

// Catch2 prints generated values in the subcase name, so these three streaming
// operators are what makes a failure say "17x31x13 tiled_simd accum" instead of
// three copies of {?}.
std::ostream &operator<<(std::ostream &os, const Shape &s) {
    return os << s.M << "x" << s.N << "x" << s.K;
}

enum class Prefill { Zero, Pattern };

std::ostream &operator<<(std::ostream &os, Prefill p) {
    return os << (p == Prefill::Zero ? "zero" : "accum");
}

// Declared here rather than pulling in benchmark.hpp for its gemm_kernel_ptr:
// a correctness harness has no business depending on the timing header.
using GemmFn = void (*)(const Tensor &, const Tensor &, Tensor &);

struct KernelEntry {
    const char *name;
    GemmFn      func;
    unsigned    restrictions;
    int         nthreads; /* 0 for the single-threaded kernels */
};

std::ostream &operator<<(std::ostream &os, const KernelEntry &k) {
    return os << k.name;
}

// gemm_multithreaded takes a thread count, so it needs one gemm_kernel_ptr per
// count under test. Captureless lambdas convert to a plain function pointer,
// which is what retired the five named mt* shims.
const KernelEntry kernels[] = {
    {"naive",      gemm_naive,      0,  0},
    {"ikj",        gemm_ikj,        0,  0},
    {"tiled",      gemm_tiled,      0,  0},
    {"avx2",       gemm_avx2,       0,  0},
    {"tiled_simd", gemm_tiled_simd, 0,  0},
    {"mt1",  [](const Tensor &A, const Tensor &B, Tensor &C) { gemm_multithreaded(A, B, C, 1); },  0,  1},
    {"mt2",  [](const Tensor &A, const Tensor &B, Tensor &C) { gemm_multithreaded(A, B, C, 2); },  0,  2},
    {"mt3",  [](const Tensor &A, const Tensor &B, Tensor &C) { gemm_multithreaded(A, B, C, 3); },  0,  3},
    {"mt8",  [](const Tensor &A, const Tensor &B, Tensor &C) { gemm_multithreaded(A, B, C, 8); },  0,  8},
    // 64 threads is well past the row count of most shapes in the table, so
    // this is the one that exercises the clamp
    {"mt64", [](const Tensor &A, const Tensor &B, Tensor &C) { gemm_multithreaded(A, B, C, 64); }, 0, 64},
};

// Each dimension independently crosses the vector width (8), the register
// block (4 and 16) and the tile boundary (64). A bug that only appears when
// two dimensions are ragged at once is the main risk, so the table has to
// cross them independently rather than scale one number.
const Shape shapes[] = {
    {  1,   1,   1}, /* degenerate                                */
    {  1, 512,   1}, /* single row, long N                        */
    {  1, 128,  64}, /* M below the register block                */
    { 64,  96,   1}, /* K = 1, a rank-1 update                    */
    {  3,   5,   7}, /* every dimension below every block size    */
    {  4,  16,   8}, /* exactly one register block                */
    {  5,  17,   9}, /* one above each of 4, 16 and 8             */
    {  7,   7,   7}, /* below one vector                          */
    {  8,   8,   8}, /* exactly one vector                        */
    {  9,  15,  33}, /* mixed                                     */
    { 16,  16,  16},
    { 17,  31,  13}, /* N < 16, the j underflow probe             */
    { 63,  63,  63}, /* one below the tile                        */
    { 63,  65,  64}, /* mixed around the tile                     */
    { 64,  64,  64}, /* exactly one tile                          */
    { 65,  65,  65}, /* one above, so the last tile is 1x1x1      */
    {127, 127, 127},
    {128, 128, 128},
    {129, 130, 131}, /* ragged in all three dimensions            */
    {128, 256, 512}, /* non-square, tile aligned                  */
    {256, 256, 256}  /* square, tile aligned                      */
};

/* ------------------------------------------------------------------------ */
/* Fill and compare helpers                                                  */
/* ------------------------------------------------------------------------ */

// Seed derived from the dimensions, so one shape's inputs do not depend on
// which shapes ran before it or on whether the fuzz run is what invoked it
void seed_for_shape(const Shape &s) {
    srand((unsigned)(SEED ^ (s.M * 73856093u) ^ (s.N * 19349663u) ^ (s.K * 83492791u)));
}

// Inputs from [0.5, 1.5) rather than [0, 1), so no expected value can drift
// near zero and hide a missing region. A region no path wrote stays 0 and one
// two paths wrote is roughly 2x; both need non-zero expected values to show up.
void fill_random(std::vector<float> &buf) {
    for (float &v : buf) {
        v = 0.5f + ((float)rand() / (float)RAND_MAX);
    }
}

// Prefill pattern, bounded to [1.0, 2.5]. The bound matters: an unbounded ramp
// hits ~8192 at 128x256, which dwarfs the ~512 product term and waters the
// relative tolerance down until the accumulate check stops discriminating.
void make_prefill(std::vector<float> &buf, size_t M, size_t N, Prefill mode) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            buf[i * N + j] =
                (mode == Prefill::Zero) ? 0.0f : 1.0f + (float)((i * N + j) % 7u) * 0.25f;
        }
    }
}

// Copies a dense MxN buffer into a Tensor, honouring the tensor's row stride
void scatter_to_tensor(Tensor &dst, const float *src, size_t rows, size_t cols) {
    const size_t stride = dst.stride(0);

    // Scrub the whole allocation first so nothing carries over between runs
    memset(dst.data(), 0, dst.shape(0) * stride * sizeof(float));

    for (size_t i = 0; i < rows; i++) {
        memcpy(&dst.data()[i * stride], &src[i * cols], cols * sizeof(float));
    }
}

struct CompareResult {
    bool   ok;
    double max_abs_err;
    double max_rel_err;
    size_t bad_i, bad_j;
    float  expected, actual;
};

CompareResult compare_to_reference(const std::vector<float> &ref, const Tensor &C, size_t M,
                                   size_t N) {
    CompareResult r{true, 0.0, 0.0, 0, 0, 0.0f, 0.0f};
    const size_t  stride = C.stride(0);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            const float  expected = ref[i * N + j];
            const float  actual   = C.data()[i * stride + j];
            const double diff     = std::fabs((double)expected - (double)actual);
            const double rel      = (expected != 0.0f) ? diff / std::fabs((double)expected) : diff;

            if (diff > r.max_abs_err) {
                r.max_abs_err = diff;
            }
            if (rel > r.max_rel_err) {
                r.max_rel_err = rel;
            }

            if (diff > (double)ATOL + (double)RTOL * std::fabs((double)expected)) {
                // Keep the first failure only, but let the maxima keep updating
                if (r.ok) {
                    r.bad_i    = i;
                    r.bad_j    = j;
                    r.expected = expected;
                    r.actual   = actual;
                }
                r.ok = false;
            }
        }
    }
    return r;
}

/* ------------------------------------------------------------------------ */
/* Fixture                                                                   */
/* ------------------------------------------------------------------------ */

// Everything one (shape, prefill) pair needs: the two inputs, the C seed, and
// the reference answer for that seed. Built once and reused across kernels in
// the fuzz loop, where recomputing the reference ten times over would dominate.
struct Fixture {
    std::vector<float> a, b, c_seed, ref;
};

Fixture build_fixture(const Shape &s, Prefill mode) {
    Fixture f;
    f.a.resize(s.M * s.K);
    f.b.resize(s.K * s.N);
    f.c_seed.resize(s.M * s.N);

    seed_for_shape(s);
    fill_random(f.a);
    fill_random(f.b);
    make_prefill(f.c_seed, s.M, s.N, mode);

    f.ref = f.c_seed;
    reference_gemm(s.M, s.N, s.K, f.a.data(), f.b.data(), f.ref.data());
    return f;
}

// A wrapper that rejects its input leaves C untouched, so comparing zero
// against zero would report a pass. Insisting the reference is not identically
// zero is what stops that being silent.
double reference_magnitude(const std::vector<float> &ref) {
    double sum = 0.0;
    for (float v : ref) {
        sum += std::fabs((double)v);
    }
    return sum;
}

void check_kernel(const Shape &s, const KernelEntry &ke, Prefill mode, const Fixture &f) {
    Tensor A({s.M, s.K}), B({s.K, s.N}), C({s.M, s.N});
    scatter_to_tensor(A, f.a.data(), s.M, s.K);
    scatter_to_tensor(B, f.b.data(), s.K, s.N);
    scatter_to_tensor(C, f.c_seed.data(), s.M, s.N);

    ke.func(A, B, C);

    const CompareResult r = compare_to_reference(f.ref, C, s.M, s.N);

    INFO("shape " << s << "  kernel " << ke.name << "  prefill " << mode);
    INFO("max_abs_err=" << r.max_abs_err << "  max_rel_err=" << r.max_rel_err);
    if (!r.ok) {
        INFO("first mismatch at [" << r.bad_i << "][" << r.bad_j << "]  expected=" << r.expected
                                   << "  actual=" << r.actual);
    }
    CHECK(r.ok);
}

} // namespace

/* ------------------------------------------------------------------------ */
/* Test cases                                                                */
/* ------------------------------------------------------------------------ */

TEST_CASE("gemm kernels match an independent reference", "[gemm]") {
    const auto s    = GENERATE(from_range(shapes));
    const auto ke   = GENERATE(from_range(kernels));
    const auto mode = GENERATE(Prefill::Zero, Prefill::Pattern);

    const char *why = restriction_violated(ke.restrictions, s.M, s.N, s.K, ke.nthreads);
    if (why != nullptr) {
        SKIP(ke.name << " restricted on " << s << ": " << why);
    }

    const Fixture f = build_fixture(s, mode);
    REQUIRE(reference_magnitude(f.ref) > 0.0);

    check_kernel(s, ke, mode, f);
}

// One case checked against values worked out by hand, so the reference itself
// is pinned to something rather than trusted because it looks obvious.
//
//   A = [ 1  2  3  4 ]      B = [ 1  2 ]      C = [  50   60 ]
//       [ 5  6  7  8 ]          [ 3  4 ]          [ 114  140 ]
//       [ 9 10 11 12 ]          [ 5  6 ]          [ 178  220 ]
//                               [ 7  8 ]
//
// Every value is exactly representable, so this compares with == rather than a
// tolerance.
TEST_CASE("gemm kernels reproduce a hand-computed product exactly", "[gemm][exact]") {
    const auto ke = GENERATE(from_range(kernels));

    constexpr size_t M = 3, N = 2, K = 4;

    static const float a_vals[M * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    static const float b_vals[K * N] = {1, 2, 3, 4, 5, 6, 7, 8};
    static const float expected[M * N] = {50, 60, 114, 140, 178, 220};

    const char *why = restriction_violated(ke.restrictions, M, N, K, ke.nthreads);
    if (why != nullptr) {
        SKIP(ke.name << " restricted on the exact case: " << why);
    }

    Tensor A({M, K}), B({K, N}), C({M, N});
    scatter_to_tensor(A, a_vals, M, K);
    scatter_to_tensor(B, b_vals, K, N);
    C.zero();

    ke.func(A, B, C);

    INFO("kernel " << ke.name);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            INFO("at [" << i << "][" << j << "]");
            CHECK(C.data()[i * C.stride(0) + j] == expected[i * N + j]);
        }
    }
}

// Randomised shapes against the reference. A fixed table always misses some
// combination of edges, and each iteration is at most 2*80^3 = 1.0 MFLOP.
//
// Hidden behind a dot tag, so it runs only when asked for: `gemm_tests [.fuzz]`.
// The reference is built once per (shape, prefill) and reused across all ten
// kernels - rebuilding it per kernel would dominate the run.
TEST_CASE("gemm kernels match the reference on random shapes", "[gemm][.fuzz]") {
    srand(SEED);

    for (int it = 0; it < FUZZ_ITERS; it++) {
        Shape s;
        s.M = (size_t)(rand() % FUZZ_MAX_DIM) + 1u;
        s.N = (size_t)(rand() % FUZZ_MAX_DIM) + 1u;
        s.K = (size_t)(rand() % FUZZ_MAX_DIM) + 1u;

        for (const Prefill mode : {Prefill::Zero, Prefill::Pattern}) {
            // build_fixture reseeds from the dimensions, so a fuzz failure is
            // reproducible from the shape alone
            const Fixture f = build_fixture(s, mode);
            REQUIRE(reference_magnitude(f.ref) > 0.0);

            for (const KernelEntry &ke : kernels) {
                if (restriction_violated(ke.restrictions, s.M, s.N, s.K, ke.nthreads) != nullptr) {
                    continue;
                }
                check_kernel(s, ke, mode, f);
            }
        }
    }
}
