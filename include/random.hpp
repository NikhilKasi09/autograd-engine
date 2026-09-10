#ifndef RANDOM_H
#define RANDOM_H

#include "tensor.hpp"

#include <cstdint>
#include <random>

/*
 Generator - a seeded stream of pseudorandom floats, and the state that stream
 needs.

 Tensor::randomize() stays where it is; it fills benchmark operands and nothing
 depends on which numbers it produces. This is the version phase 8 initialises
 parameters from, and it differs in the two ways that matter there: the seed is
 the caller's, so a training run reproduces, and the state is per-object rather
 than the one global stream rand() draws from.

 The C++ split this rests on: an ENGINE owns state and emits bits
 (std::mt19937_64), a DISTRIBUTION is a stateless mapping from those bits onto a
 range (std::uniform_real_distribution). C's rand() is both at once, which is
 why it has nowhere to put a per-object seed.
*/
class Generator {
public:
    // Seeds the engine. Two Generators built from the same seed produce the
    // same sequence; that is the point of taking one.
    explicit Generator(std::uint64_t seed);

    ~Generator() = default;

    // Neither copied nor moved, for the same reason Storage isn't - except that
    // here the deletion is catching a numerical bug rather than a lifetime one.
    // A copied Generator is a second object emitting an IDENTICAL stream, so
    // two layers initialised from what looks like two generators would come out
    // correlated, silently, and pass every range check written against them.
    //
    // pybind11 constructs in place through py::init, so the binding needs
    // neither. If you want copying back, these two lines are the change.
    Generator(const Generator &)            = delete;
    Generator &operator=(const Generator &) = delete;
    Generator(Generator &&)                 = delete;
    Generator &operator=(Generator &&)      = delete;

    // Overwrites every element of out with a draw from [lo, hi).
    //
    // Advances the engine by out.numel() draws, so successive calls return
    // different numbers. That is the whole reason engine_ is a member: an engine
    // constructed inside this function would restart the stream on every call
    // and hand out the same numbers each time - which passes a range check, and
    // passes any test that only ever calls uniform once.
    //
    // Requires a contiguous out and throws std::invalid_argument otherwise,
    // matching Tensor::zero() and for the same reason: this walks the buffer
    // flat, so a stride-0 dimension would write one float many times and a
    // sliced view would write past its own extent.
    //
    // Throws std::invalid_argument if lo >= hi. The standard's
    // uniform_real_distribution is undefined for an empty or inverted range
    // rather than merely unhelpful.
    void uniform(Tensor &out, float lo, float hi);

private:
    std::mt19937_64 engine_;
};

#endif
