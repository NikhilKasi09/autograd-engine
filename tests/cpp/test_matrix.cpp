// Unit tests for the RAII Matrix class. Written before the implementation, so
// they all fail on a fresh scaffold. Work down them in order; each one pins a
// property that a plausible-looking wrong implementation would break.

#include "matrix.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>

TEST_CASE("Matrix reports the shape it was constructed with", "[matrix]") {
    const Matrix m(3, 5);

    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 5);

    // stride == cols today. It is a separate field so a view onto a larger
    // matrix can arrive later without any kernel changing.
    REQUIRE(m.stride() == 5);
}

TEST_CASE("Matrix allocates storage aligned for AVX2", "[matrix]") {
    const Matrix m(7, 13); // deliberately not a multiple of anything

    REQUIRE(m.data() != nullptr);

    // The BASE pointer is aligned. Row i sits at data + i * stride and is only
    // aligned when stride % 8 == 0, which is why the kernels use loadu. An
    // aligned allocator is not permission to switch to _mm256_load_ps.
    const auto address = reinterpret_cast<std::uintptr_t>(m.data());
    REQUIRE(address % ALIGNMENT_REQ == 0);
}

TEST_CASE("Matrix is zero-initialised on construction", "[matrix]") {
    // matrix_create memsets, and the harness relies on a freshly created C
    // being zero. Same contract here.
    const Matrix m(4, 6);

    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            REQUIRE(m.data()[i * m.stride() + j] == 0.0f);
        }
    }
}

TEST_CASE("zero() clears every element", "[matrix]") {
    Matrix m(3, 3);

    for (std::size_t i = 0; i < 9; i++) {
        m.data()[i] = 1.5f;
    }
    m.zero();

    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            REQUIRE(m.data()[i * m.stride() + j] == 0.0f);
        }
    }
}

TEST_CASE("randomize() fills the range and touches every element", "[matrix]") {
    Matrix m(8, 8);
    m.randomize();

    bool any_nonzero = false;
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            const float v = m.data()[i * m.stride() + j];
            // Inclusive at the top: rand() can return RAND_MAX, and no float
            // divisor makes that land strictly below 1.0 anyway.
            REQUIRE(v >= 0.0f);
            REQUIRE(v <= 1.0f);
            if (v != 0.0f) {
                any_nonzero = true;
            }
        }
    }
    // A do-nothing randomize() would pass the range checks on a zeroed buffer.
    REQUIRE(any_nonzero);
}

TEST_CASE("Matrix rejects degenerate and overflowing shapes", "[matrix]") {
    REQUIRE_THROWS_AS(Matrix(0, 5), std::invalid_argument);
    REQUIRE_THROWS_AS(Matrix(5, 0), std::invalid_argument);

    // rows * cols must be checked BEFORE the multiply, or it wraps and the
    // allocation silently comes back too small. matrix_create guards this the
    // same way.
    const std::size_t huge = SIZE_MAX / 2 + 1;
    REQUIRE_THROWS_AS(Matrix(huge, 4), std::invalid_argument);
}

TEST_CASE("moving a Matrix transfers ownership", "[matrix]") {
    Matrix source(2, 3);
    source.data()[0] = 42.0f;
    const float *original = source.data();

    Matrix moved(std::move(source));

    // The buffer is transferred, not copied - same address, no reallocation.
    REQUIRE(moved.data() == original);
    REQUIRE(moved.rows() == 2);
    REQUIRE(moved.cols() == 3);
    REQUIRE(moved.data()[0] == 42.0f);

    // A moved-from Matrix must be destructible and must not free the buffer
    // its contents were taken from. unique_ptr guarantees this if you let it.
    REQUIRE(source.data() == nullptr); // NOLINT(bugprone-use-after-move)
}

TEST_CASE("clone() produces an independent deep copy", "[matrix]") {
    Matrix original(3, 4);
    original.data()[5] = 9.0f;

    Matrix copy = original.clone();

    REQUIRE(copy.rows() == original.rows());
    REQUIRE(copy.cols() == original.cols());
    REQUIRE(copy.stride() == original.stride());

    // Separate storage, not an aliased view.
    REQUIRE(copy.data() != original.data());
    REQUIRE(copy.data()[5] == 9.0f);

    // Mutating the copy must leave the original alone.
    copy.data()[5] = 1.0f;
    REQUIRE(original.data()[5] == 9.0f);
}

TEST_CASE("Matrix matches matrix_t element layout", "[matrix]") {
    // The kernels take raw pointers plus lda/ldb/ldc, so Matrix has to hand
    // out storage the existing micro-kernels can already walk. Row-major,
    // element (i, j) at data[i * stride + j], contiguous within a row.
    Matrix m(3, 4);
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            m.data()[i * m.stride() + j] = static_cast<float>(i * 10 + j);
        }
    }

    REQUIRE(m.data()[0] == 0.0f);
    REQUIRE(m.data()[1] == 1.0f);
    REQUIRE(m.data()[4] == 10.0f); // start of row 1, since stride == 4
    REQUIRE(m.data()[11] == 23.0f); // (i=2, j=3) -> 2*10 + 3
}
