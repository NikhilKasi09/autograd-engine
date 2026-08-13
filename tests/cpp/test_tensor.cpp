// Unit tests for Tensor: construction and element access first, then the six
// view operations. Written before the implementation, so they fail on a fresh
// scaffold. Work down them in order; each one pins a property that a
// plausible-looking wrong implementation would break.

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
    // contract here as the matrix type this replaced.
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

    Tensor copy = original; // shallow, where the old matrix type deleted copy

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

/* ------------------------------------------------------------------------ */
/* Views                                                                     */
/*                                                                           */
/* Three detectors have to agree here and none of them is sufficient alone:  */
/* assert_within_storage() catches an offset that leaves the buffer, the     */
/* hand-computed values below catch arithmetic that stays in bounds and      */
/* reads the wrong element, and ASan catches what escapes the allocation.    */
/* A view bug that reads a neighbouring row is invisible to two of the       */
/* three, so the hand-computed expectations are load-bearing.                */
/* ------------------------------------------------------------------------ */

// Fills a contiguous tensor with 0, 1, 2, ... in memory order, so any wrong
// element read is immediately identifiable as the position it came from.
namespace {
void fill_iota(Tensor &t) {
    for (std::size_t i = 0; i < t.numel(); i++) {
        t.data()[i] = static_cast<float>(i);
    }
}
} // namespace

TEST_CASE("transpose swaps shape and strides without moving data", "[tensor][view]") {
    Tensor t({2, 3});
    fill_iota(t); // rows are {0 1 2} and {3 4 5}

    const Tensor tr = t.transpose(0, 1);

    REQUIRE(tr.rank() == 2);
    REQUIRE(tr.shape(0) == 3);
    REQUIRE(tr.shape(1) == 2);
    REQUIRE(tr.stride(0) == 1);
    REQUIRE(tr.stride(1) == 3);

    // Same buffer, no allocation.
    REQUIRE(tr.shares_storage_with(t));
    REQUIRE(tr.data() == t.data());

    // The transpose is only in the indexing: element (i, j) of the view is
    // element (j, i) of the original.
    REQUIRE(tr(0, 0) == 0.0f);
    REQUIRE(tr(0, 1) == 3.0f);
    REQUIRE(tr(1, 0) == 1.0f);
    REQUIRE(tr(2, 1) == 5.0f);
}

TEST_CASE("a transposed view is not contiguous", "[tensor][view]") {
    const Tensor t({2, 3});
    REQUIRE(t.is_contiguous());
    REQUIRE_FALSE(t.transpose(0, 1).is_contiguous());

    // But transposing a vector changes nothing observable, so it stays
    // contiguous - the length-1 dimension has no stride worth checking.
    const Tensor v({1, 5});
    REQUIRE(v.transpose(0, 1).is_contiguous());
}

TEST_CASE("transposing twice returns the original layout", "[tensor][view]") {
    const Tensor t({3, 7});
    const Tensor back = t.transpose(0, 1).transpose(0, 1);

    REQUIRE(back.shape(0) == 3);
    REQUIRE(back.shape(1) == 7);
    REQUIRE(back.stride(0) == 7);
    REQUIRE(back.stride(1) == 1);
    REQUIRE(back.is_contiguous());
    REQUIRE(back.shares_storage_with(t));
}

TEST_CASE("transpose rejects an out-of-range dimension", "[tensor][view]") {
    const Tensor t({2, 3});
    REQUIRE_THROWS_AS(t.transpose(0, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(t.transpose(5, 0), std::invalid_argument);
}

TEST_CASE("permute reorders dimensions", "[tensor][view]") {
    const Tensor t({2, 3, 4}); // strides {12, 4, 1}
    const Tensor p = t.permute({2, 0, 1});

    REQUIRE(p.shape(0) == 4);
    REQUIRE(p.shape(1) == 2);
    REQUIRE(p.shape(2) == 3);
    REQUIRE(p.stride(0) == 1);
    REQUIRE(p.stride(1) == 12);
    REQUIRE(p.stride(2) == 4);
    REQUIRE(p.shares_storage_with(t));
}

TEST_CASE("permute rejects anything that is not a permutation", "[tensor][view]") {
    const Tensor t({2, 3, 4});

    REQUIRE_THROWS_AS(t.permute({0, 1}), std::invalid_argument);       // too short
    REQUIRE_THROWS_AS(t.permute({0, 1, 2, 0}), std::invalid_argument); // too long
    REQUIRE_THROWS_AS(t.permute({0, 1, 3}), std::invalid_argument);    // out of range

    // The one a length check alone lets through: right length, every entry in
    // range, but dimension 1 appears twice and dimension 2 not at all.
    REQUIRE_THROWS_AS(t.permute({0, 1, 1}), std::invalid_argument);
}

TEST_CASE("slice narrows one dimension and moves the offset", "[tensor][view]") {
    Tensor t({4, 3});
    fill_iota(t); // row r holds {3r, 3r+1, 3r+2}

    // Rows 1 and 2. offset moves by 1 * stride(0) = 3 floats.
    const Tensor rows = t.slice(0, 1, 2);

    REQUIRE(rows.shape(0) == 2);
    REQUIRE(rows.shape(1) == 3);
    REQUIRE(rows.stride(0) == 3); // strides are untouched by a slice
    REQUIRE(rows.stride(1) == 1);
    REQUIRE(rows.shares_storage_with(t));

    // Shares storage but NOT the same base pointer - this is the case that
    // makes comparing data() the wrong test for sharing.
    REQUIRE(rows.data() == t.data() + 3);
    REQUIRE(rows(0, 0) == 3.0f);
    REQUIRE(rows(1, 2) == 8.0f);

    // Slicing the inner dimension leaves a row-major-looking shape whose
    // stride no longer matches its extent: exactly the stride > cols case the
    // project has carried since phase 1, and still contiguous-in-rows.
    const Tensor cols = t.slice(1, 1, 2);
    REQUIRE(cols.shape(0) == 4);
    REQUIRE(cols.shape(1) == 2);
    REQUIRE(cols.stride(0) == 3);
    REQUIRE(cols.data() == t.data() + 1);
    REQUIRE(cols(0, 0) == 1.0f);
    REQUIRE(cols(3, 1) == 11.0f);
    REQUIRE_FALSE(cols.is_contiguous());
}

TEST_CASE("slice rejects a range that leaves the dimension", "[tensor][view]") {
    const Tensor t({4, 3});

    REQUIRE_THROWS_AS(t.slice(2, 0, 1), std::invalid_argument); // no such dim
    REQUIRE_THROWS_AS(t.slice(0, 0, 0), std::invalid_argument); // no empty tensors
    REQUIRE_THROWS_AS(t.slice(0, 3, 2), std::invalid_argument); // runs one past
    REQUIRE_THROWS_AS(t.slice(0, 4, 1), std::invalid_argument); // starts past the end

    // A guard written as start > shape - count wraps instead of rejecting.
    REQUIRE_THROWS_AS(t.slice(0, 1, SIZE_MAX), std::invalid_argument);
}

TEST_CASE("expand broadcasts a length-1 dimension with stride 0", "[tensor][view]") {
    Tensor bias({1, 3});
    bias(0, 0) = 7.0f;
    bias(0, 1) = 8.0f;
    bias(0, 2) = 9.0f;

    const Tensor rows = bias.expand({4, 3});

    REQUIRE(rows.shape(0) == 4);
    REQUIRE(rows.shape(1) == 3);
    REQUIRE(rows.stride(0) == 0); // the whole trick
    REQUIRE(rows.stride(1) == 1);
    REQUIRE(rows.shares_storage_with(bias));
    REQUIRE_FALSE(rows.is_contiguous());

    // Every row reads the same three floats - no MxN temporary anywhere.
    for (std::size_t i = 0; i < 4; i++) {
        REQUIRE(rows(i, 0) == 7.0f);
        REQUIRE(rows(i, 2) == 9.0f);
    }
}

TEST_CASE("expand rejects stretching a dimension that is not 1", "[tensor][view]") {
    const Tensor t({2, 3});

    REQUIRE_THROWS_AS(t.expand({4, 3}), std::invalid_argument); // 2 is not 1
    REQUIRE_THROWS_AS(t.expand({2, 3, 1}), std::invalid_argument); // rank changed
    REQUIRE_NOTHROW(t.expand({2, 3}));                            // identity is fine
}

TEST_CASE("reshape reinterprets a contiguous tensor", "[tensor][view]") {
    Tensor t({2, 6});
    fill_iota(t);

    const Tensor r = t.reshape({3, 4});

    REQUIRE(r.rank() == 2);
    REQUIRE(r.shape(0) == 3);
    REQUIRE(r.shape(1) == 4);
    REQUIRE(r.stride(0) == 4); // fresh row-major strides for the new shape
    REQUIRE(r.stride(1) == 1);
    REQUIRE(r.shares_storage_with(t));
    REQUIRE(r.is_contiguous());

    // Same elements in the same memory order, read through a new shape.
    REQUIRE(r(0, 0) == 0.0f);
    REQUIRE(r(1, 0) == 4.0f);
    REQUIRE(r(2, 3) == 11.0f);

    // Rank may change too, as long as the element count does not.
    const Tensor flat = t.reshape({12});
    REQUIRE(flat.rank() == 1);
    REQUIRE(flat.shape(0) == 12);
    REQUIRE(flat(11) == 11.0f);
}

TEST_CASE("reshape refuses a non-contiguous tensor and a size change", "[tensor][view]") {
    const Tensor t({2, 6});

    // The deliberate divergence from torch: no silent fallback to a copy.
    REQUIRE_THROWS_AS(t.transpose(0, 1).reshape({12}), std::invalid_argument);

    // Element count must be preserved.
    REQUIRE_THROWS_AS(t.reshape({3, 5}), std::invalid_argument);
    REQUIRE_THROWS_AS(t.reshape({0, 12}), std::invalid_argument);
    REQUIRE_THROWS_AS(t.reshape({1, 2, 3, 4, 5}), std::invalid_argument);

    // And the documented way round it still works.
    REQUIRE_NOTHROW(t.transpose(0, 1).contiguous().reshape({12}));
}

TEST_CASE("contiguous() shares storage when it can and copies when it must",
          "[tensor][view]") {
    Tensor t({2, 3});
    fill_iota(t);

    // Fast path: already dense, so this must not allocate. Every kernel
    // boundary calls it, which is why the cheap case has to stay cheap.
    const Tensor same = t.contiguous();
    REQUIRE(same.shares_storage_with(t));
    REQUIRE(same.data() == t.data());

    // Slow path: a transpose has to be materialised.
    const Tensor tr = t.transpose(0, 1);
    const Tensor dense = tr.contiguous();

    REQUIRE_FALSE(dense.shares_storage_with(t));
    REQUIRE(dense.is_contiguous());
    REQUIRE(dense.shape(0) == 3);
    REQUIRE(dense.shape(1) == 2);

    // The values land transposed in memory: {0 3 1 4 2 5}, not {0 1 2 3 4 5}.
    // A contiguous() that memcpy'd the buffer would produce the latter and
    // pass every shape assertion above it.
    const float expected[6] = {0, 3, 1, 4, 2, 5};
    for (std::size_t i = 0; i < 6; i++) {
        REQUIRE(dense.data()[i] == expected[i]);
    }
}

TEST_CASE("a view of a view composes", "[tensor][view]") {
    Tensor t({4, 6});
    fill_iota(t);

    // Slice rows 1..2, then columns 2..4, then transpose the result. Offsets
    // accumulate; strides survive the slices and swap on the transpose.
    const Tensor v = t.slice(0, 1, 2).slice(1, 2, 3).transpose(0, 1);

    REQUIRE(v.shape(0) == 3);
    REQUIRE(v.shape(1) == 2);
    REQUIRE(v.stride(0) == 1);
    REQUIRE(v.stride(1) == 6);
    REQUIRE(v.shares_storage_with(t));

    // offset is 1*6 + 2 = 8, so v(0, 0) is t(1, 2) = 8.
    REQUIRE(v.data() == t.data() + 8);
    REQUIRE(v(0, 0) == 8.0f);
    REQUIRE(v(2, 1) == 16.0f); // t(2, 4)
}
