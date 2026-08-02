// Placeholder Catch2 test. Replaced by the ported shape harness; it is here so
// the Catch2 fetch, the ctest registration and the TSan setarch wrapper are all
// exercised from the first commit rather than the step that relies on them.

#include <catch2/catch_test_macros.hpp>

TEST_CASE("catch2 is wired up", "[smoke]") {
    REQUIRE(1 + 1 == 2);
}
