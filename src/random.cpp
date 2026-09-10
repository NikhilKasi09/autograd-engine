#include "random.hpp"
#include <random>
#include <stdexcept>

Generator::Generator(std::uint64_t seed) : engine_(seed) {}

void Generator::uniform(Tensor &out, float lo, float hi) {
    if (!out.is_contiguous()){
        throw std::invalid_argument("Error: Out tensor should be contiguous.");
    }

    if (lo >= hi){
        throw std::invalid_argument("Error: hi must be greater than lo");
    }

    std::uniform_real_distribution<float> dist(lo, hi);

    std::size_t n = out.numel();
    float *ptr = out.data();

    for (std::size_t i = 0; i < n; i++) {
        ptr[i] = dist(engine_);
    }
}
