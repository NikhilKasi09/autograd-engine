// Unit tests for Tensor construction and element access. Written before the
// implementation, so they all fail on a fresh scaffold. Work down them in
// order; each one pins a property that a plausible-looking wrong
// implementation would break.
//
// Views (transpose, permute, slice, expand, reshape, contiguous) are step 4
// and are not tested here.

#include "tensor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>

// A brace-enclosed initialiser list inside a macro argument: the preprocessor
// splits on the commas because braces do not protect them, so every one of
// these needs the extra parentheses. Leaving them off gives an error about the
// macro taking 2 arguments, not about the tensor.
TEST_CASE("Tensor reports the shape it was constructed with", "[tensor]") {
    const Tensor a({5});
    REQUIRE(a.rank() == 1);
    REQUIRE(a.shape(0) == 5);

    const Tensor b({3, 5});
    REQUIRE(b.rank() == 2);
    REQUIRE(b.shape(0) == 3);
    REQUIRE(b.shape(1) == 5);

    const Tensor d({2, 3, 4, 5});
    REQUIRE(d.rank() == 4);
    REQUIRE(d.shape(0) == 2);
    REQUIRE(d.shape(3) == 5);
}

TEST_CASE("Tensor computes row-major strides", "[tensor]") {
    // Innermost dimension is unit stride; each outer one is the product of
    // everything inside it.
    const Tensor a({7});
    REQUIRE(a.stride(0) == 1);

    const Tensor b({3, 5});
    REQUIRE(b.stride(0) == 5);
    REQUIRE(b.stride(1) == 1);

    // Deliberately not powers of two, so a stride computed from the wrong
    // dimension cannot coincidentally match.
    const Tensor c({2, 3, 7});
    REQUIRE(c.stride(0) == 21);
    REQUIRE(c.stride(1) == 7);
    REQUIRE(c.stride(2) == 1);

    const Tensor d({2, 3, 5, 7});
    REQUIRE(d.stride(0) == 105);
    REQUIRE(d.stride(1) == 35);
    REQUIRE(d.stride(2) == 7);
    REQUIRE(d.stride(3) == 1);
}

TEST_CASE("a freshly constructed Tensor is contiguous", "[tensor]") {
    REQUIRE(Tensor({4}).is_contiguous());
    REQUIRE(Tensor({3, 5}).is_contiguous());
    REQUIRE(Tensor({2, 3, 5, 7}).is_contiguous());

    // A length-1 dimension has an unobservable stride. Whatever the
    // implementation puts there, this has to come out contiguous.
    REQUIRE(Tensor({1, 5}).is_contiguous());
    REQUIRE(Tensor({5, 1}).is_contiguous());
}

TEST_CASE("numel is the product of the shape", "[tensor]") {
    REQUIRE(Tensor({7}).numel() == 7);
    REQUIRE(Tensor({3, 5}).numel() == 15);
    REQUIRE(Tensor({2, 3, 7}).numel() == 42);
    REQUIRE(Tensor({2, 3, 5, 7}).numel() == 210);
}

TEST_CASE("Tensor rejects degenerate and overflowing shapes", "[tensor]") {
    // Rank 0: a scalar is shape {1} in this engine, deliberately unlike torch.
    REQUIRE_THROWS_AS((Tensor({})), std::invalid_argument);

    // Above MAX_RANK.
    REQUIRE_THROWS_AS((Tensor({1, 2, 3, 4, 5})), std::invalid_argument);

    // No empty tensors, in any position.
    REQUIRE_THROWS_AS((Tensor({0})), std::invalid_argument);
    REQUIRE_THROWS_AS((Tensor({3, 0})), std::invalid_argument);
    REQUIRE_THROWS_AS((Tensor({0, 3})), std::invalid_argument);

    // The product must be checked BEFORE the multiply, or it wraps and the
    // allocation silently comes back too small.
    const std::size_t huge = SIZE_MAX / 2 + 1;
    REQUIRE_THROWS_AS((Tensor({huge, 4})), std::invalid_argument);

    // Fits as a count but not once scaled to bytes. A guard written against
    // the element count alone passes the case above and fails this one.
    REQUIRE_THROWS_AS((Tensor({SIZE_MAX / 2, 1})), std::invalid_argument);
}

TEST_CASE("Tensor allocates storage aligned for AVX2", "[tensor]") {
    const Tensor t({7, 13}); // deliberately not a multiple of anything

    REQUIRE(t.data() != nullptr);

    // The BASE pointer is aligned. Row i sits at data + i * stride(0) and is
    // only aligned when stride(0) % 8 == 0, which is why the kernels use
    // loadu. An aligned allocator is not permission to switch to load_ps.
    const auto address = reinterpret_cast<std::uintptr_t>(t.data());
    REQUIRE(address % ALIGNMENT_REQ == 0);
}

TEST_CASE("Tensor is zero-initialised on construction", "[tensor]") {
    // The GEMM harness relies on a freshly created C being zero. Same
    // contract here as Matrix had.
    const Tensor t({4, 6});

    for (std::size_t i = 0; i < t.numel(); i++) {
        REQUIRE(t.data()[i] == 0.0f);
    }
}

TEST_CASE("indexing agrees with hand-computed row-major offsets", "[tensor]") {
    Tensor t({2, 3, 4});

    // strides are {12, 4, 1}, so (i, j, k) lands at 12i + 4j + k.
    t(0, 0, 0) = 1.0f;
    t(1, 2, 3) = 2.0f;
    t(0, 1, 2) = 3.0f;

    REQUIRE(t.data()[0] == 1.0f);
    REQUIRE(t.data()[23] == 2.0f);
    REQUIRE(t.data()[6] == 3.0f);

    // And the read path has to agree with the write path.
    REQUIRE(t(1, 2, 3) == 2.0f);
    REQUIRE(t(0, 1, 2) == 3.0f);

    // Const overload reaches the same element.
    const Tensor &ct = t;
    REQUIRE(ct(1, 2, 3) == 2.0f);
}

TEST_CASE("copying a Tensor shares storage", "[tensor]") {
    Tensor original({3, 4});
    original(1, 1) = 7.0f;

    Tensor copy = original; // shallow, unlike Matrix, which deleted this

    REQUIRE(copy.shares_storage_with(original));
    REQUIRE(copy.data() == original.data());
    REQUIRE(copy.shape(0) == 3);
    REQUIRE(copy.shape(1) == 4);

    // The mutation is visible through both handles, because there is one buffer.
    copy(1, 1) = 9.0f;
    REQUIRE(original(1, 1) == 9.0f);
}

TEST_CASE("clone() produces an independent deep copy", "[tensor]") {
    Tensor original({3, 4});
    original(1, 1) = 9.0f;

    Tensor copy = original.clone();

    REQUIRE(copy.rank() == original.rank());
    REQUIRE(copy.shape(0) == original.shape(0));
    REQUIRE(copy.shape(1) == original.shape(1));
    REQUIRE(copy(1, 1) == 9.0f);

    // Separate storage, not an aliased view - the distinction copy assignment
    // no longer makes for you.
    REQUIRE_FALSE(copy.shares_storage_with(original));
    REQUIRE(copy.data() != original.data());

    copy(1, 1) = 1.0f;
    REQUIRE(original(1, 1) == 9.0f);
}

TEST_CASE("moving a Tensor transfers ownership", "[tensor]") {
    Tensor source({2, 3});
    source(0, 0) = 42.0f;
    const float *original = source.data();

    Tensor moved(std::move(source));

    // The buffer is transferred, not copied - same address, no reallocation.
    REQUIRE(moved.data() == original);
    REQUIRE(moved.shape(0) == 2);
    REQUIRE(moved.shape(1) == 3);
    REQUIRE(moved(0, 0) == 42.0f);

    // A moved-from Tensor must be destructible and must not free the buffer
    // its contents were taken from. shared_ptr guarantees this if you let it.
    REQUIRE(source.data() == nullptr); // NOLINT(bugprone-use-after-move)
}

TEST_CASE("zero() clears every element", "[tensor]") {
    Tensor t({3, 3});

    for (std::size_t i = 0; i < t.numel(); i++) {
        t.data()[i] = 1.5f;
    }
    t.zero();

    for (std::size_t i = 0; i < t.numel(); i++) {
        REQUIRE(t.data()[i] == 0.0f);
    }
}

TEST_CASE("randomize() fills the range and touches every element", "[tensor]") {
    Tensor t({8, 8});
    t.randomize();

    bool any_nonzero = false;
    for (std::size_t i = 0; i < t.numel(); i++) {
        const float v = t.data()[i];
        // Inclusive at the top: rand() can return RAND_MAX, and no float
        // divisor makes that land strictly below 1.0 anyway.
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
        if (v != 0.0f) {
            any_nonzero = true;
        }
    }
    // A do-nothing randomize() would pass the range checks on a zeroed buffer.
    REQUIRE(any_nonzero);
}
