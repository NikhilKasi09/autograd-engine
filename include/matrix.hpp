#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Alignment for AVX2 registers. Applies to the base pointer only, not to
// where any individual row starts.
#define ALIGNMENT_REQ 32

typedef struct {
    size_t rows;
    size_t cols;

    // Floats from the start of one row to the start of the next.
    // matrix_create always sets this to cols, so there is no padding. It stays
    // a separate field because the invariant is stride >= cols, which is how a
    // view onto a bigger matrix arrives later without changing any kernel.
    size_t stride;

    float *data;
} matrix_t;

// Allocates and zero-initialises a rows x cols matrix, row-major.
// The block is 32-byte aligned but row i sits at data + i * stride, which is
// only aligned when stride % 8 == 0, hence the unaligned loads in the kernels.
// Returns NULL on a zero dimension or a failed allocation.
matrix_t *matrix_create(size_t rows, size_t cols);

// Safely frees the aligned memory block and the struct
void matrix_free(matrix_t *mat);

// Populates the matrix with random floating-point values between 0.0 and 1.0
void matrix_randomize(matrix_t *mat);

// Sets every element to zero, skipping anything between cols and stride
void matrix_zero(matrix_t *mat);

// Copies src into dst element by element. The two are allowed to have
// different strides, but must agree on rows and cols.
void matrix_copy(matrix_t *dst, const matrix_t *src);

// ---------------------------------------------------------------------------
// Matrix - the RAII replacement for matrix_t.
//
// Lives alongside matrix_t while the kernels are migrated over one at a time,
// so the harness stays green throughout. matrix_t and its five free functions
// are deleted once nothing calls them.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <memory>
#include <type_traits>

// Releases a block obtained from the matching aligned operator new. Aligned
// allocation and deallocation must be paired: handing an over-aligned pointer
// to plain operator delete is undefined behaviour, which is exactly the bug
// this type exists to make unrepresentable.
struct AlignedDeleter {
    void operator()(float *p) const noexcept;
};

class Matrix {
public:
    // Allocates a rows x cols row-major matrix, zero-initialised, with the
    // base pointer aligned to ALIGNMENT_REQ. stride is set to cols.
    //
    // Throws std::invalid_argument if either dimension is zero or if
    // rows * cols would overflow size_t; std::bad_alloc if allocation fails.
    // Note this is a behaviour change from matrix_create, which returned NULL:
    // a constructor has no return value, so failure has to be an exception.
    Matrix(std::size_t rows, std::size_t cols);

    // Destructor, move constructor and move assignment are all defaulted:
    // unique_ptr already does exactly the right thing for each. Declaring
    // them yourself would be the classic rule-of-five mistake here.
    ~Matrix() = default;
    Matrix(Matrix &&) noexcept = default;
    Matrix &operator=(Matrix &&) noexcept = default;

    // Copying is deleted rather than defaulted. A 1024x1024 matrix is 4 MB,
    // and an accidental copy in a benchmark loop would be invisible in the
    // results and ruinous to them. Deep copies must say so.
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    // The explicit deep copy. Same shape, same contents, independent storage.
    Matrix clone() const;

    // Row-major element access. Element (i, j) lives at data()[i * stride + j].
    float *data() noexcept;
    const float *data() const noexcept;

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t stride() const noexcept;

    // Sets every element to zero, skipping any padding between cols and
    // stride - same contract as matrix_zero.
    void zero();

    // Fills with values in [0, 1] - inclusive at the top, because rand() can
    // return RAND_MAX. Same contract as matrix_randomize, including using
    // rand(), so the harness's reproducibility guarantee (seed derived from
    // the shape) is unaffected.
    void randomize();

private:
    std::size_t rows_   = 0;
    std::size_t cols_   = 0;
    std::size_t stride_ = 0;

    // Kept separate from cols_ for the same reason matrix_t did: the invariant
    // is stride >= cols, which is how a view onto a larger matrix arrives in
    // phase 3 without touching a single kernel.
    std::unique_ptr<float[], AlignedDeleter> data_;
};

// These catch the two mistakes that are easy to make above and silent to miss.
// If you defaulted the copy operations instead of deleting them, or wrote a
// move constructor that can throw, the build stops here rather than at some
// benchmark number you cannot explain three steps later.
static_assert(!std::is_copy_constructible_v<Matrix>,
              "Matrix must not be copyable - use clone() for a deep copy");
static_assert(!std::is_copy_assignable_v<Matrix>,
              "Matrix must not be copy-assignable - use clone() for a deep copy");
static_assert(std::is_nothrow_move_constructible_v<Matrix>,
              "Matrix must be nothrow-movable so containers can relocate it");
static_assert(std::is_nothrow_move_assignable_v<Matrix>,
              "Matrix must be nothrow-move-assignable");

// The kernels read 8 floats per AVX2 vector; ALIGNMENT_REQ is 32 bytes for
// that reason and not by coincidence.
static_assert(sizeof(float) == 4, "AVX2 kernels assume 4-byte floats");
static_assert(ALIGNMENT_REQ == 8 * sizeof(float),
              "ALIGNMENT_REQ should be one AVX2 vector wide");

#endif
