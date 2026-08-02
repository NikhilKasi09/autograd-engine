/*
 Correctness harness for the GEMM kernel ladder.

 Every kernel is checked against reference_gemm() below, which is a plain
 triple loop over densely packed buffers with no leading-dimension arithmetic.
 gemm_naive gets tested the same way as everything else instead of being used
 as the source of truth, so a bug in it cannot quietly bless the other five.

 The kernels do not all become general at the same time, and two of them are
 already broken on shapes this file has to cover eventually (see PLAN.md).
 So each registry entry carries a mask of the shapes it cannot handle yet, and
 a violated restriction prints SKIP with a reason instead of FAIL. Phase 1 is
 finished when every mask is empty and the SKIP count reaches zero.

 Every case runs twice, once with C zeroed and once with C holding a known
 pattern. Zeroing C before every call makes 'C = sum' and 'C += sum' look
 identical, and the backward pass in roadmap phase 6 accumulates into C.

 Usage:
   ./gemm_tests                 run the shape table
   ./gemm_tests tiled_simd      run only kernels whose name contains this
   ./gemm_tests --fuzz          also run randomised sizes
*/

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark.h" /* gemm_kernel_ptr */
#include "gemm.h"
#include "matrix.h"

// Printed on every run so that a failure can be reproduced exactly
#define SEED 0xC0FFEE

/*
 Mixed tolerance: diff > ATOL + RTOL * |expected|.

 A purely absolute tolerance gets marginal at large K. Inputs in [0.5, 1.5)
 with K = 256 give results near 256, where one float ULP is already about
 1.5e-5. The SIMD kernels also differ from the scalar ones for legitimate
 reasons: FMA does not round the intermediate product, and tiling reorders the
 k-summation. max_abs_err is printed for every case so the real margin is
 visible rather than guessed at.
*/
#define ATOL 1e-5f
#define RTOL 1e-5f

// Randomised mode settings (--fuzz)
#define FUZZ_ITERS   300
#define FUZZ_MAX_DIM 80

/* ------------------------------------------------------------------------ */
/* Shape restrictions                                                        */
/* ------------------------------------------------------------------------ */

typedef enum {
    RESTRICT_SQUARE   = 1u << 0, /* requires M == N == K                     */
    RESTRICT_N_MULT8  = 1u << 1, /* requires N % 8 == 0 (vector width)       */
    RESTRICT_N_MIN16  = 1u << 2, /* requires N >= 16    (j-underflow, bug 2) */
    RESTRICT_M_MIN4   = 1u << 3, /* requires M >= 4     (i-underflow, bug 2) */
    RESTRICT_MT_4ROWS = 1u << 4  /* requires M/nthreads >= 4        (bug 1)  */
} kernel_restrict_t;

// Returns the name of the first violated restriction, or NULL if this kernel
// can legally be run on this shape
static const char *restriction_violated(unsigned mask, size_t M, size_t N, size_t K,
                                        int nthreads) {
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
    return NULL;
}

/* ------------------------------------------------------------------------ */
/* matrix_t access helpers                                                   */
/* ------------------------------------------------------------------------ */

/*
 Every access to a matrix_t goes through these three helpers, so that a change
 to the struct is a small edit here instead of a rewrite of the file. They
 earned their keep at step B, when size and padded_size became rows, cols and
 stride.

 mat_new can build any shape now that the allocator takes two dimensions, so
 the alloc-nonsquare path in run_shape no longer fires. It stays until the
 shape table actually goes non-square at step C.
*/
static matrix_t *mat_new(size_t rows, size_t cols) {
    return matrix_create(rows, cols);
}

static size_t mat_stride(const matrix_t *m) {
    return m->stride;
}

// Floats in the whole allocation. There is no padding any more, so this is
// just rows * stride, but scrubbing still goes through one name.
static size_t mat_alloc_floats(const matrix_t *m) {
    return m->rows * m->stride;
}

/* ------------------------------------------------------------------------ */
/* Reference implementation                                                  */
/* ------------------------------------------------------------------------ */

/*
 C(MxN) += A(MxK) * B(KxN), all three buffers packed row-major with no
 leading dimensions. Deliberately the simplest loop nest that could work.

 The accumulator is a double rather than a float. This is the thing every
 kernel gets judged against, so it should carry less rounding error than they
 do, not the same amount in a different order.
*/
static void reference_gemm(size_t M, size_t N, size_t K, const float *A, const float *B,
                           float *C) {
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
/* Fill and compare helpers                                                  */
/* ------------------------------------------------------------------------ */

/*
 Inputs come from [0.5, 1.5) rather than [0, 1) so that no expected value can
 drift near zero and hide a missing region. A region that no kernel path wrote
 stays at 0 and a region two paths wrote is roughly 2x, and both of those need
 reliably non-zero expected values to show up.
*/
static void fill_random(float *buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        buf[i] = 0.5f + ((float)rand() / (float)RAND_MAX);
    }
}

/*
 Bounded prefill pattern in [1.0, 2.5]. The bound matters: an unbounded ramp
 reaches about 8192 at 128x256, which dwarfs the ~512 product term and waters
 the relative tolerance down until the accumulate check stops discriminating.
*/
static float prefill_value(size_t i, size_t j, size_t N) {
    return 1.0f + (float)((i * N + j) % 7u) * 0.25f;
}

typedef enum { PREFILL_ZERO = 0, PREFILL_PATTERN = 1 } prefill_mode_t;

static const char *prefill_name(prefill_mode_t mode) {
    return (mode == PREFILL_ZERO) ? "zero" : "accum";
}

// Builds the dense M*N buffer used to seed both C and the reference
static void make_prefill(float *buf, size_t M, size_t N, prefill_mode_t mode) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            buf[i * N + j] = (mode == PREFILL_ZERO) ? 0.0f : prefill_value(i, j, N);
        }
    }
}

// Copies a dense MxN buffer into a matrix_t, honouring the matrix's stride
static void scatter_to_matrix(matrix_t *dst, const float *src, size_t rows, size_t cols) {
    const size_t stride = mat_stride(dst);

    // Scrub the whole allocation first, padding included. The SIMD kernels are
    // allowed to accumulate into padding columns today, and stale values there
    // would otherwise carry over from the previous run.
    memset(dst->data, 0, mat_alloc_floats(dst) * sizeof(float));

    for (size_t i = 0; i < rows; i++) {
        memcpy(&dst->data[i * stride], &src[i * cols], cols * sizeof(float));
    }
}

typedef struct {
    bool   ok;
    double max_abs_err;
    double max_rel_err;
    size_t bad_i, bad_j;
    float  expected, actual;
} compare_result_t;

static compare_result_t compare_to_reference(const float *ref, const matrix_t *C, size_t M,
                                             size_t N) {
    compare_result_t r;
    r.ok          = true;
    r.max_abs_err = 0.0;
    r.max_rel_err = 0.0;
    r.bad_i = r.bad_j = 0;
    r.expected = r.actual = 0.0f;

    const size_t stride = mat_stride(C);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            const float expected = ref[i * N + j];
            const float actual   = C->data[i * stride + j];
            const double diff    = fabs((double)expected - (double)actual);
            const double rel     = (expected != 0.0f) ? diff / fabs((double)expected) : diff;

            if (diff > r.max_abs_err) {
                r.max_abs_err = diff;
            }
            if (rel > r.max_rel_err) {
                r.max_rel_err = rel;
            }

            if (diff > (double)ATOL + (double)RTOL * fabs((double)expected)) {
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
/* Kernel registry                                                           */
/* ------------------------------------------------------------------------ */

// gemm_multithreaded takes a thread count, so it needs one gemm_kernel_ptr
// wrapper per count under test. Counts that hand a worker fewer than 4 rows
// are the path that is broken today (RESTRICT_MT_4ROWS).
static void mt1(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    gemm_multithreaded(A, B, C, 1);
}
static void mt2(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    gemm_multithreaded(A, B, C, 2);
}
static void mt8(const matrix_t *A, const matrix_t *B, matrix_t *C) {
    gemm_multithreaded(A, B, C, 8);
}

typedef struct {
    const char      *name;
    gemm_kernel_ptr  func;
    unsigned         restrictions;
    int              nthreads; /* 0 for the single-threaded kernels */
} kernel_entry_t;

/*
 Masks follow the capability table in PLAN.md, currently the step C column.
 They burn down to zero over steps D to H, so this array doubles as the
 progress tracker.

 naive is first out, and is now the only kernel that runs the whole table.

 N_MULT8 arrived for the three SIMD kernels at step B. Those kernels step the
 j loop in whole vectors with no scalar tail, and 63, 65 and 127 only ever
 worked because the allocator quietly rounded the stride up to 64 or 128. With
 stride == cols the vector loop would run off the end of the row, so the
 wrappers reject those shapes until steps F and G put the tails in.
*/
static const kernel_entry_t kernels[] = {
    {"naive",      gemm_naive,      0,                                                      0},
    {"ikj",        gemm_ikj,        0,                                                      0},
    {"tiled",      gemm_tiled,      0,                                                      0},
    {"avx2",       gemm_avx2,       0,                                                      0},
    {"tiled_simd", gemm_tiled_simd, 0,                                                      0},
    {"mt1",        mt1,             RESTRICT_SQUARE | RESTRICT_N_MULT8 | RESTRICT_N_MIN16 |
                                    RESTRICT_M_MIN4 | RESTRICT_MT_4ROWS,                    1},
    {"mt2",        mt2,             RESTRICT_SQUARE | RESTRICT_N_MULT8 | RESTRICT_N_MIN16 |
                                    RESTRICT_M_MIN4 | RESTRICT_MT_4ROWS,                    2},
    {"mt8",        mt8,             RESTRICT_SQUARE | RESTRICT_N_MULT8 | RESTRICT_N_MIN16 |
                                    RESTRICT_M_MIN4 | RESTRICT_MT_4ROWS,                    8}
};
static const size_t num_kernels = sizeof(kernels) / sizeof(kernels[0]);

/* ------------------------------------------------------------------------ */
/* Shape table                                                               */
/* ------------------------------------------------------------------------ */

typedef struct {
    size_t M, N, K;
} shape_t;

/*
 Each dimension independently crosses the vector width (8), the register block
 (4 and 16) and the tile boundary (64), because a bug that only shows up when
 two dimensions are ragged at once is the whole risk here.

 The shapes that would send a kernel off unbounded, such as anything with
 N < 16 for gemm_tiled_simd, are in the table from now on and held back by the
 restriction masks rather than by being left out. That way the day a mask is
 cleared, the shape it was hiding gets run.
*/
static const shape_t shapes[] = {
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
static const size_t num_shapes = sizeof(shapes) / sizeof(shapes[0]);

/* ------------------------------------------------------------------------ */
/* Test driver                                                               */
/* ------------------------------------------------------------------------ */

typedef struct {
    size_t passed;
    size_t failed;
    size_t skipped;
    bool   quiet; /* fuzz mode prints failures only */
} tally_t;

// Seed derived from the dimensions, so one shape's inputs do not depend on
// which shapes ran before it or on whether --fuzz was passed
static void seed_for_shape(size_t M, size_t N, size_t K) {
    srand((unsigned)(SEED ^ (M * 73856093u) ^ (N * 19349663u) ^ (K * 83492791u)));
}

// Runs one shape against every registered kernel in both prefill modes and
// returns the number of failures
static size_t run_shape(shape_t s, const char *filter, tally_t *tally) {
    const size_t M = s.M, N = s.N, K = s.K;
    size_t failures = 0;

    float *a_buf   = malloc(M * K * sizeof(float));
    float *b_buf   = malloc(K * N * sizeof(float));
    float *c_seed  = malloc(M * N * sizeof(float)); /* prefill pattern */
    float *ref     = malloc(M * N * sizeof(float));

    matrix_t *A = mat_new(M, K);
    matrix_t *B = mat_new(K, N);

    if (!a_buf || !b_buf || !c_seed || !ref) {
        fprintf(stderr, "FATAL: harness allocation failed at %zux%zux%zu\n", M, N, K);
        exit(2);
    }

    if (!A || !B) {
        printf("SKIP %-11s %4zux%4zux%4zu [%-5s] alloc-failed\n", "*all*", M, N, K, "-");
        tally->skipped += num_kernels * 2;
        matrix_free(A);
        matrix_free(B);
        free(a_buf);
        free(b_buf);
        free(c_seed);
        free(ref);
        return 0;
    }

    seed_for_shape(M, N, K);
    fill_random(a_buf, M * K);
    fill_random(b_buf, K * N);
    scatter_to_matrix(A, a_buf, M, K);
    scatter_to_matrix(B, b_buf, K, N);

    for (int mode_i = 0; mode_i < 2; mode_i++) {
        const prefill_mode_t mode = (prefill_mode_t)mode_i;

        make_prefill(c_seed, M, N, mode);

        // The reference accumulates too, so seeding it with the same pattern
        // keeps the accumulate check to one code path instead of two
        memcpy(ref, c_seed, M * N * sizeof(float));
        reference_gemm(M, N, K, a_buf, b_buf, ref);

        // Latent bug 6: a wrapper that rejects its input leaves C untouched, so
        // comparing zero against zero would report PASS. Check the reference is
        // actually non-trivial before trusting anything compared against it.
        double checksum = 0.0;
        for (size_t idx = 0; idx < M * N; idx++) {
            checksum += fabs((double)ref[idx]);
        }
        if (!(checksum > 0.0)) {
            printf("FAIL %-11s %4zux%4zux%4zu [%-5s] reference checksum is zero\n",
                   "reference", M, N, K, prefill_name(mode));
            failures++;
            tally->failed++;
            continue;
        }

        for (size_t k = 0; k < num_kernels; k++) {
            const kernel_entry_t *ke = &kernels[k];

            if (filter && !strstr(ke->name, filter)) {
                continue;
            }

            const char *why =
                restriction_violated(ke->restrictions, M, N, K, ke->nthreads);
            if (why) {
                if (!tally->quiet) {
                    printf("SKIP %-11s %4zux%4zux%4zu [%-5s] %s\n", ke->name, M, N, K,
                           prefill_name(mode), why);
                }
                tally->skipped++;
                continue;
            }

            matrix_t *C = mat_new(M, N);
            if (!C) {
                fprintf(stderr, "FATAL: could not allocate C at %zux%zu\n", M, N);
                exit(2);
            }
            scatter_to_matrix(C, c_seed, M, N);

            ke->func(A, B, C);

            const compare_result_t r = compare_to_reference(ref, C, M, N);

            if (r.ok) {
                if (!tally->quiet) {
                    printf("PASS %-11s %4zux%4zux%4zu [%-5s] max_abs_err=%.2e rel=%.2e\n",
                           ke->name, M, N, K, prefill_name(mode), r.max_abs_err,
                           r.max_rel_err);
                }
                tally->passed++;
            } else {
                printf("FAIL %-11s %4zux%4zux%4zu [%-5s] at [%zu][%zu] "
                       "expected=%.6f actual=%.6f max_abs_err=%.2e rel=%.2e\n",
                       ke->name, M, N, K, prefill_name(mode), r.bad_i, r.bad_j,
                       (double)r.expected, (double)r.actual, r.max_abs_err, r.max_rel_err);
                failures++;
                tally->failed++;
            }

            matrix_free(C);
        }
    }

    matrix_free(A);
    matrix_free(B);
    free(a_buf);
    free(b_buf);
    free(c_seed);
    free(ref);

    return failures;
}

/*
 Randomised shapes against the reference. A fixed table always misses some
 combination of edges, which is exactly what this phase is about, and each
 iteration is at most 2*80^3 = 1.0 MFLOP so the whole run costs about a second.

 M, N and K are drawn independently, so the ragged combinations that no
 handwritten table would think to include get covered.
*/
static size_t run_fuzz(const char *filter, tally_t *tally) {
    size_t failures = 0;

    printf("\n--- fuzz: %d random shapes with M, N, K in [1,%d] ---\n", FUZZ_ITERS,
           FUZZ_MAX_DIM);

    srand((unsigned)SEED);
    for (int it = 0; it < FUZZ_ITERS; it++) {
        shape_t s;
        s.M = (size_t)(rand() % FUZZ_MAX_DIM) + 1u;
        s.N = (size_t)(rand() % FUZZ_MAX_DIM) + 1u;
        s.K = (size_t)(rand() % FUZZ_MAX_DIM) + 1u;
        // run_shape reseeds from the dimensions, so a fuzz failure can be
        // reproduced from the shape alone
        failures += run_shape(s, filter, tally);
    }
    return failures;
}

/*
 One case checked against values worked out by hand rather than against
 reference_gemm, so that the reference itself is pinned to something.

   A = [ 1  2  3  4 ]      B = [ 1  2 ]      C = [  50   60 ]
       [ 5  6  7  8 ]          [ 3  4 ]          [ 114  140 ]
       [ 9 10 11 12 ]          [ 5  6 ]          [ 178  220 ]
                               [ 7  8 ]

 3x2x4 is non-square in all three dimensions, and every value here is exactly
 representable in a float, so this compares exactly rather than with a
 tolerance. Runs against every kernel the restrictions allow, which is naive
 alone at step C and grows as the masks clear.
*/
static size_t run_known_values(const char *filter, tally_t *tally) {
    static const float a_vals[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    static const float b_vals[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
    static const float expected[6] = {50, 60, 114, 140, 178, 220};

    const size_t M = 3, N = 2, K = 4;
    size_t failures = 0;

    printf("\n--- hand-checked 3x2x4 ---\n");

    matrix_t *A = mat_new(M, K);
    matrix_t *B = mat_new(K, N);
    if (!A || !B) {
        fprintf(stderr, "FATAL: allocation failed in run_known_values\n");
        exit(2);
    }
    scatter_to_matrix(A, a_vals, M, K);
    scatter_to_matrix(B, b_vals, K, N);

    for (size_t k = 0; k < num_kernels; k++) {
        const kernel_entry_t *ke = &kernels[k];

        if (filter && !strstr(ke->name, filter)) {
            continue;
        }

        const char *why = restriction_violated(ke->restrictions, M, N, K, ke->nthreads);
        if (why) {
            printf("SKIP %-11s %4zux%4zux%4zu [%-5s] %s\n", ke->name, M, N, K, "exact", why);
            tally->skipped++;
            continue;
        }

        matrix_t *C = mat_new(M, N);
        if (!C) {
            fprintf(stderr, "FATAL: allocation failed in run_known_values\n");
            exit(2);
        }
        matrix_zero(C);

        ke->func(A, B, C);

        bool ok = true;
        for (size_t i = 0; i < M && ok; i++) {
            for (size_t j = 0; j < N && ok; j++) {
                const float got = C->data[i * mat_stride(C) + j];
                if (got != expected[i * N + j]) {
                    printf("FAIL %-11s %4zux%4zux%4zu [%-5s] at [%zu][%zu] "
                           "expected=%.1f actual=%.1f\n",
                           ke->name, M, N, K, "exact", i, j, (double)expected[i * N + j],
                           (double)got);
                    ok = false;
                }
            }
        }

        if (ok) {
            printf("PASS %-11s %4zux%4zux%4zu [%-5s] exact match\n", ke->name, M, N, K,
                   "exact");
            tally->passed++;
        } else {
            failures++;
            tally->failed++;
        }

        matrix_free(C);
    }

    matrix_free(A);
    matrix_free(B);
    return failures;
}

int main(int argc, char **argv) {
    const char *filter = NULL;
    bool fuzz = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fuzz") == 0) {
            fuzz = true;
        } else {
            filter = argv[i];
        }
    }

    printf("GEMM correctness harness (seed=0x%X, atol=%.1e, rtol=%.1e)\n", SEED, (double)ATOL,
           (double)RTOL);
    if (filter) {
        printf("Filter: kernels containing \"%s\"\n", filter);
    }
    printf("--------------------------------------------------------------------------\n");

    tally_t tally;
    tally.passed = tally.failed = tally.skipped = 0;
    tally.quiet  = false;

    size_t failures = 0;
    for (size_t i = 0; i < num_shapes; i++) {
        failures += run_shape(shapes[i], filter, &tally);
    }

    failures += run_known_values(filter, &tally);

    if (fuzz) {
        tally.quiet = true; // one line per shape and kernel would be 5000 lines
        failures += run_fuzz(filter, &tally);
    }

    printf("--------------------------------------------------------------------------\n");
    printf("passed=%zu failed=%zu skipped=%zu\n", tally.passed, tally.failed, tally.skipped);
    // The SKIP count is the phase 1 progress tracker and has to reach zero
    printf("phase-1 progress: %zu restricted (kernel, shape) pairs remaining\n",
           tally.skipped);

    return failures != 0;
}
