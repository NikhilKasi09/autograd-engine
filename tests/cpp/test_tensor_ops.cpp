// Unit tests for the elementwise op layer. Written before the implementation,
// so they all fail on a fresh scaffold.
//
// The tests that matter most are the ones feeding an op a TRANSPOSED or an
// EXPAND'd input. A kernel that ignores strides and walks the buffer flat
// passes every contiguous case in this file and fails those, which is the
// whole reason they are here.

#include "tensor_ops.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

// Fills a contiguous tensor with 0, 1, 2, ... in memory order.
void fill_iota(Tensor &t) {
    for (std::size_t i = 0; i < t.numel(); i++) {
        t.data()[i] = static_cast<float>(i);
    }
}

// Fills a contiguous tensor with one repeated value.
void fill_with(Tensor &t, float v) {
    for (std::size_t i = 0; i < t.numel(); i++) {
        t.data()[i] = v;
    }
}

} // namespace

/* ------------------------------------------------------------------------ */
/* Values                                                                    */
/* ------------------------------------------------------------------------ */

TEST_CASE("add sums elementwise", "[ops]") {
    Tensor a({2, 3});
    Tensor b({2, 3});
    Tensor out({2, 3});

    fill_iota(a);        // 0 1 2 3 4 5
    fill_with(b, 10.0f); // 10 everywhere

    add(a, b, out);

    for (std::size_t i = 0; i < 6; i++) {
        REQUIRE(out.data()[i] == static_cast<float>(i) + 10.0f);
    }
}

TEST_CASE("mul multiplies elementwise, not as a matrix product", "[ops]") {
    Tensor a({2, 2});
    Tensor b({2, 2});
    Tensor out({2, 2});

    a.data()[0] = 1.0f; a.data()[1] = 2.0f;
    a.data()[2] = 3.0f; a.data()[3] = 4.0f;
    fill_with(b, 2.0f);

    mul(a, b, out);

    // Elementwise: {2, 4, 6, 8}. The matrix product of these two would be
    // {6, 6, 14, 14}, so this case tells the two apart.
    REQUIRE(out.data()[0] == 2.0f);
    REQUIRE(out.data()[1] == 4.0f);
    REQUIRE(out.data()[2] == 6.0f);
    REQUIRE(out.data()[3] == 8.0f);
}

TEST_CASE("scale multiplies by a scalar", "[ops]") {
    Tensor a({4});
    Tensor out({4});
    fill_iota(a); // 0 1 2 3

    scale(a, 2.5f, out);

    REQUIRE(out.data()[0] == 0.0f);
    REQUIRE(out.data()[1] == 2.5f);
    REQUIRE(out.data()[2] == 5.0f);
    REQUIRE(out.data()[3] == 7.5f);
}

TEST_CASE("relu clamps at zero", "[ops]") {
    Tensor a({5});
    Tensor out({5});

    a.data()[0] = -2.0f;
    a.data()[1] = -0.5f;
    a.data()[2] = 0.0f;
    a.data()[3] = 0.5f;
    a.data()[4] = 3.0f;

    relu(a, out);

    REQUIRE(out.data()[0] == 0.0f);
    REQUIRE(out.data()[1] == 0.0f);
    REQUIRE(out.data()[2] == 0.0f);
    REQUIRE(out.data()[3] == 0.5f);
    REQUIRE(out.data()[4] == 3.0f);
}

TEST_CASE("relu_backward masks the gradient on the reference's sign", "[ops]") {
    Tensor grad_out({4});
    Tensor ref({4});
    Tensor out({4});

    // ref carries the sign; grad_out carries the value. Deliberately unrelated,
    // so swapping the two lambda parameters gives a different answer rather
    // than a coincidence.
    ref.data()[0] = -2.0f; ref.data()[1] = 0.0f;
    ref.data()[2] = 3.0f;  ref.data()[3] = 1.0f;

    grad_out.data()[0] = 5.0f;  grad_out.data()[1] = 7.0f;
    grad_out.data()[2] = -4.0f; grad_out.data()[3] = 0.0f;

    relu_backward(grad_out, ref, out);

    REQUIRE(out.data()[0] == 0.0f);  // ref negative, gradient blocked
    REQUIRE(out.data()[1] == 0.0f);  // ref EXACTLY zero - the > vs >= case, and
                                     // the only element that distinguishes them
    REQUIRE(out.data()[2] == -4.0f); // a NEGATIVE gradient passes through a
                                     // positive ref. relu clamps its output;
                                     // relu_backward must not clamp a gradient
    REQUIRE(out.data()[3] == 0.0f);  // zero gradient, positive ref

    // Swapping the lambda's parameters gives {-2, 0, 0, 0}; ignoring ref gives
    // {5, 7, -4, 0}; using >= gives 7 at index 1. All three differ above.
}

TEST_CASE("relu_backward reads strided operands through their strides", "[ops]") {
    SECTION("a transposed gradient") {
        Tensor t({2, 3});
        fill_iota(t); // rows {0 1 2} and {3 4 5}

        const Tensor tr = t.transpose(0, 1); // 3x2: {0 3} {1 4} {2 5}

        Tensor ref({3, 2});
        fill_with(ref, 1.0f); // all positive: the mask passes everything, so
                              // the ONLY thing under test here is the walk
        Tensor out({3, 2});

        relu_backward(tr, ref, out);

        // Correct: {0 3 1 4 2 5} in memory order. A kernel walking tr's buffer
        // flat produces {0 1 2 3 4 5} - the same multiset, so a sum or a total
        // would wave it through.
        const float expected[6] = {0, 3, 1, 4, 2, 5};
        for (std::size_t i = 0; i < 6; i++) {
            REQUIRE(out.data()[i] == expected[i]);
        }
    }

    SECTION("a transposed reference") {
        Tensor t({2, 2});
        t(0, 0) = -1.0f; t(0, 1) = 2.0f;
        t(1, 0) = 3.0f;  t(1, 1) = -4.0f;

        const Tensor tr = t.transpose(0, 1); // {-1 3} {2 -4}

        Tensor grad_out({2, 2});
        fill_with(grad_out, 9.0f);
        Tensor out({2, 2});

        relu_backward(grad_out, tr, out);

        // The mask follows tr's LAYOUT, not its buffer order. A flat walk masks
        // positions {1, 2} instead of {1, 2} reordered - here that is
        // {0, 9, 9, 0} the wrong way round.
        REQUIRE(out.data()[0] == 0.0f); // -1
        REQUIRE(out.data()[1] == 9.0f); //  3
        REQUIRE(out.data()[2] == 9.0f); //  2
        REQUIRE(out.data()[3] == 0.0f); // -4
    }
}

TEST_CASE("relu_backward rejects bad shapes and a non-contiguous output", "[ops]") {
    Tensor grad_out({2, 3});
    Tensor ref({2, 3});
    Tensor out({2, 3});

    Tensor wrong_extents({3, 2});
    REQUIRE_THROWS_AS(relu_backward(grad_out, wrong_extents, out), std::invalid_argument);

    Tensor wrong_rank({6});
    REQUIRE_THROWS_AS(relu_backward(grad_out, wrong_rank, out), std::invalid_argument);

    REQUIRE_THROWS_AS(relu_backward(grad_out, ref, wrong_extents), std::invalid_argument);

    Tensor base({3, 2});
    Tensor tr_out = base.transpose(0, 1); // 2x3, strides {1, 2}
    REQUIRE_FALSE(tr_out.is_contiguous());
    REQUIRE_THROWS_AS(relu_backward(grad_out, ref, tr_out), std::invalid_argument);
}

TEST_CASE("sum totals every element", "[ops]") {
    Tensor a({2, 3});
    fill_iota(a); // 0 + 1 + 2 + 3 + 4 + 5

    REQUIRE(sum(a) == 15.0f);
}

/* ------------------------------------------------------------------------ */
/* Accumulation                                                              */
/* ------------------------------------------------------------------------ */

TEST_CASE("add_into accumulates rather than overwriting", "[ops]") {
    Tensor src({2, 2});
    fill_with(src, 3.0f);

    // Run twice, into a zeroed destination and a prefilled one. Zeroing before
    // every call is what makes = and += indistinguishable, and phase 6
    // accumulates into gradient buffers that are already populated.
    SECTION("into a zeroed destination") {
        Tensor dst({2, 2}); // zero-initialised on construction
        add_into(dst, src);
        for (std::size_t i = 0; i < 4; i++) {
            REQUIRE(dst.data()[i] == 3.0f);
        }
    }

    SECTION("into a prefilled destination") {
        Tensor dst({2, 2});
        fill_with(dst, 10.0f);
        add_into(dst, src);
        for (std::size_t i = 0; i < 4; i++) {
            REQUIRE(dst.data()[i] == 13.0f);
        }
        // And again, to prove it keeps accumulating.
        add_into(dst, src);
        for (std::size_t i = 0; i < 4; i++) {
            REQUIRE(dst.data()[i] == 16.0f);
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Strided inputs - the cases a flat buffer walk fails                       */
/* ------------------------------------------------------------------------ */

TEST_CASE("add reads a transposed input through its strides", "[ops]") {
    Tensor t({2, 3});
    fill_iota(t); // rows {0 1 2} and {3 4 5}

    const Tensor tr = t.transpose(0, 1); // 3x2: {0 3} {1 4} {2 5}

    Tensor b({3, 2});
    fill_with(b, 10.0f);
    Tensor out({3, 2});

    add(tr, b, out);

    // Correct: {10 13 11 14 12 15} in memory order.
    // A kernel walking tr's buffer flat produces {10 11 12 13 14 15}, which is
    // the same multiset - so checking a sum or a total would NOT catch it.
    const float expected[6] = {10, 13, 11, 14, 12, 15};
    for (std::size_t i = 0; i < 6; i++) {
        REQUIRE(out.data()[i] == expected[i]);
    }
}

TEST_CASE("relu reads a transposed input through its strides", "[ops]") {
    Tensor t({2, 2});
    t(0, 0) = -1.0f; t(0, 1) = 2.0f;
    t(1, 0) = 3.0f;  t(1, 1) = -4.0f;

    const Tensor tr = t.transpose(0, 1); // {-1 3} {2 -4}
    Tensor out({2, 2});

    relu(tr, out);

    REQUIRE(out.data()[0] == 0.0f); // -1
    REQUIRE(out.data()[1] == 3.0f);
    REQUIRE(out.data()[2] == 2.0f);
    REQUIRE(out.data()[3] == 0.0f); // -4
}

TEST_CASE("add broadcasts a bias through an expanded view", "[ops]") {
    // This is phase 8's Linear layer bias add, and the reason expand exists.
    // Nothing here allocates a 4x3 copy of the bias.
    Tensor bias({1, 3});
    bias(0, 0) = 7.0f;
    bias(0, 1) = 8.0f;
    bias(0, 2) = 9.0f;

    Tensor x({4, 3});
    fill_iota(x);
    Tensor out({4, 3});

    add(x, bias.expand({4, 3}), out);

    const float expected[12] = {7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 18, 20};
    for (std::size_t i = 0; i < 12; i++) {
        REQUIRE(out.data()[i] == expected[i]);
    }
}

TEST_CASE("mul reads an expanded input with stride 0", "[ops]") {
    Tensor scale_per_col({1, 3});
    scale_per_col(0, 0) = 0.0f;
    scale_per_col(0, 1) = 1.0f;
    scale_per_col(0, 2) = 2.0f;

    Tensor x({2, 3});
    fill_with(x, 5.0f);
    Tensor out({2, 3});

    mul(x, scale_per_col.expand({2, 3}), out);

    const float expected[6] = {0, 5, 10, 0, 5, 10};
    for (std::size_t i = 0; i < 6; i++) {
        REQUIRE(out.data()[i] == expected[i]);
    }
}

TEST_CASE("sum counts each repeat of an expanded view", "[ops]") {
    Tensor bias({1, 3});
    bias(0, 0) = 7.0f;
    bias(0, 1) = 8.0f;
    bias(0, 2) = 9.0f;

    REQUIRE(sum(bias) == 24.0f);

    // Expanded to four rows, every element is visited four times. Phase 6's
    // broadcast backward depends on exactly this: the gradient flowing back to
    // a broadcast tensor is the SUM over the dimensions it was stretched along.
    REQUIRE(sum(bias.expand({4, 3})) == 96.0f);
}

TEST_CASE("add_into accumulates from a transposed source", "[ops]") {
    Tensor t({2, 3});
    fill_iota(t);
    const Tensor tr = t.transpose(0, 1); // {0 3} {1 4} {2 5}

    Tensor dst({3, 2});
    fill_with(dst, 100.0f);

    add_into(dst, tr);

    const float expected[6] = {100, 103, 101, 104, 102, 105};
    for (std::size_t i = 0; i < 6; i++) {
        REQUIRE(dst.data()[i] == expected[i]);
    }
}

/* ------------------------------------------------------------------------ */
/* Precision                                                                 */
/* ------------------------------------------------------------------------ */

TEST_CASE("sum accumulates in double", "[ops]") {
    // 100000 copies of 0.1f. A float32 accumulator drifts to 9998.5566; a
    // double one gives 10000.000149, which narrows to exactly 10000.0f.
    // Measured, not estimated - the two differ by 1.4e-4 relative, which is
    // 14x this project's tolerance and would surface in phase 7 as a gradient
    // failure that is really a reduction failure.
    Tensor a({100000});
    fill_with(a, 0.1f);

    REQUIRE(std::fabs(sum(a) - 10000.0f) < 0.01f);
}

/* ------------------------------------------------------------------------ */
/* Rejections                                                                */
/* ------------------------------------------------------------------------ */

TEST_CASE("ops reject mismatched shapes", "[ops]") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    Tensor out({2, 3});

    // Same element count, different extents. No implicit broadcasting and no
    // implicit flattening.
    REQUIRE_THROWS_AS(add(a, b, out), std::invalid_argument);
    REQUIRE_THROWS_AS(mul(a, b, out), std::invalid_argument);

    // Rank mismatch.
    Tensor flat({6});
    REQUIRE_THROWS_AS(add(a, flat, out), std::invalid_argument);

    // Output shape must match the inputs too.
    Tensor c({2, 3});
    Tensor wrong_out({6});
    REQUIRE_THROWS_AS(add(a, c, wrong_out), std::invalid_argument);

    Tensor dst({3, 2});
    REQUIRE_THROWS_AS(add_into(dst, a), std::invalid_argument);
}

TEST_CASE("ops reject a non-contiguous output", "[ops]") {
    Tensor a({2, 3});
    Tensor b({2, 3});

    // A transposed destination has several ways to be walked and no single
    // right answer, and an expanded one aliases. Neither is writable.
    Tensor base({3, 2});
    Tensor tr_out = base.transpose(0, 1); // 2x3, strides {1, 2}
    REQUIRE_FALSE(tr_out.is_contiguous());

    REQUIRE_THROWS_AS(add(a, b, tr_out), std::invalid_argument);
    REQUIRE_THROWS_AS(mul(a, b, tr_out), std::invalid_argument);
    REQUIRE_THROWS_AS(scale(a, 2.0f, tr_out), std::invalid_argument);
    REQUIRE_THROWS_AS(relu(a, tr_out), std::invalid_argument);
    REQUIRE_THROWS_AS(add_into(tr_out, a), std::invalid_argument);
}

/* ------------------------------------------------------------------------ */
/* Aliasing                                                                  */
/* ------------------------------------------------------------------------ */

TEST_CASE("an op may write into one of its own inputs", "[ops]") {
    // Tensor copy is shallow, so passing the same tensor as input and output
    // is easy to do by accident and will happen deliberately in phase 8's
    // optimizer step. Elementwise ops read and write the same index, so this
    // is safe - unlike gemm, where C aliasing A or B is undefined.
    Tensor a({2, 2});
    fill_with(a, 3.0f);
    Tensor b({2, 2});
    fill_with(b, 4.0f);

    add(a, b, a);

    for (std::size_t i = 0; i < 4; i++) {
        REQUIRE(a.data()[i] == 7.0f);
    }
}
