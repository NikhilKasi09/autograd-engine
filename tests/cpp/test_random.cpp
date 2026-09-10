// Unit tests for Generator. Written before the implementation, so they fail on
// a fresh scaffold.
//
// The order is deliberate: reproducibility first, because that is the only
// reason this type exists rather than Tensor::randomize(), and the
// stream-advances case last, because it is the one a plausible wrong
// implementation fails while passing everything above it.

#include "random.hpp"
#include "tensor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

namespace {

// Values in flat buffer order. Every test compares whole buffers rather than a
// sum: two fills can share a sum and hold different numbers, and telling those
// apart is the entire point of three of the cases below.
std::vector<float> drain(Generator &g, std::size_t n, float lo, float hi) {
    Tensor t({n});
    g.uniform(t, lo, hi);
    return std::vector<float>(t.data(), t.data() + t.numel());
}

}   // namespace

TEST_CASE("the same seed gives the same sequence", "[random]") {
    Generator a(0xC0FFEE);
    Generator b(0xC0FFEE);

    REQUIRE(drain(a, 32, 0.0f, 1.0f) == drain(b, 32, 0.0f, 1.0f));
}

TEST_CASE("different seeds give different sequences", "[random]") {
    Generator a(1);
    Generator b(2);

    REQUIRE(drain(a, 32, 0.0f, 1.0f) != drain(b, 32, 0.0f, 1.0f));
}

TEST_CASE("successive calls advance the stream", "[random]") {
    // The case that earns this type.
    //
    // An engine constructed inside uniform() rather than held as a member
    // restarts from the same seed on every call, so these two fills come back
    // identical. That version passes every other test in this file: one seed
    // still reproduces, two seeds still differ, the range still holds, and
    // every element is still touched. Phase 8 would then initialise two Linear
    // layers from one generator and get correlated weights, silently.
    Generator g(7);

    const std::vector<float> first  = drain(g, 32, 0.0f, 1.0f);
    const std::vector<float> second = drain(g, 32, 0.0f, 1.0f);

    REQUIRE(first != second);
}

TEST_CASE("values land inside the requested range", "[random]") {
    Generator g(11);
    Tensor t({16, 4});
    g.uniform(t, -2.0f, 3.0f);

    for (std::size_t i = 0; i < t.numel(); i++) {
        REQUIRE(t.data()[i] >= -2.0f);
        REQUIRE(t.data()[i] < 3.0f);
    }
}

TEST_CASE("every element is written", "[random]") {
    // A range that excludes zero, so a do-nothing fill fails on the values
    // rather than needing a sentinel pass first. Tensor arrives zeroed.
    Generator g(13);
    Tensor t({8, 8});
    g.uniform(t, 1.0f, 2.0f);

    for (std::size_t i = 0; i < t.numel(); i++) {
        REQUIRE(t.data()[i] >= 1.0f);
    }
}

TEST_CASE("a non-contiguous destination is refused", "[random]") {
    Generator g(17);
    Tensor t({4, 6});

    // Transposed: strides (1, 4), so a flat write would land in the wrong
    // logical positions rather than out of bounds - invisible to ASan and to
    // any range check.
    Tensor transposed = t.transpose(0, 1);
    REQUIRE_THROWS_AS(g.uniform(transposed, 0.0f, 1.0f), std::invalid_argument);

    // Expanded: stride 0, so one float would take every write in the row.
    Tensor row({1, 6});
    Tensor expanded = row.expand({4, 6});
    REQUIRE_THROWS_AS(g.uniform(expanded, 0.0f, 1.0f), std::invalid_argument);
}

TEST_CASE("an inverted or empty range is refused", "[random]") {
    Generator g(19);
    Tensor t({4});

    REQUIRE_THROWS_AS(g.uniform(t, 1.0f, 0.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(g.uniform(t, 1.0f, 1.0f), std::invalid_argument);
}
