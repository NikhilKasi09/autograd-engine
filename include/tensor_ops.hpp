#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.hpp"

/*
 Elementwise forward kernels. No graph, no gradients - phase 5 builds the DAG
 over these, and each one here has a backward that phase 6 will write.

 One contract, applied to every op below:

   - Inputs may have ANY strides. A transposed view and an expand'd dimension
     with stride 0 are both read in place, never materialised. That is what
     stops phase 8's bias add from allocating an MxN temporary per forward pass.
   - The output must be contiguous, and is overwritten, not accumulated into.
     add_into is the single exception and says so in its name.
   - Shapes must match EXACTLY. There is no implicit broadcasting: a caller who
     wants one writes .expand(...) at the call site, where it is visible. Phase
     6 needs the reverse operation - summing back over broadcast dimensions -
     before implicit promotion is safe to offer.
   - Violations throw std::invalid_argument. This differs from
     gemm_check_shapes, which returns bool and reports on stderr; that contract
     is load-bearing for the existing GEMM harness and is not a style to copy.

 These kernels are SCALAR, deliberately. Elementwise work is
 memory-bandwidth-bound rather than compute-bound, so vectorising it buys a
 fraction of what it buys in GEMM, and a stride-general walk that handles
 stride 0 and a transposed layout does not vectorise cleanly without a
 separate contiguous fast path. That fast path is a phase 9 measurement, gated
 on these showing up in an MLP training profile - not an oversight.
*/

// out = a + b
void add(const Tensor &a, const Tensor &b, Tensor &out);

// out = a * b, elementwise. Not a matrix product - that is gemm.
void mul(const Tensor &a, const Tensor &b, Tensor &out);

// out = a * s
void scale(const Tensor &a, float s, Tensor &out);

// out = max(a, 0)
void relu(const Tensor &a, Tensor &out);

// out = ref > 0 ? grad_out : 0. The backward half of relu.
//
// ref is whichever tensor carries the sign, and the graph passes relu's OUTPUT
// rather than its input - relu(x) > 0 exactly when x > 0, so the mask is
// identical and the node keeps one buffer alive instead of two. The parameter
// is named ref rather than input because both are correct arguments.
//
// The comparison is strict, matching relu's own x > 0.0f above. At exactly zero
// relu has no derivative and every framework picks a subgradient; picking 0
// keeps the forward and backward kernels agreeing at the boundary.
void relu_backward(const Tensor &grad_out, const Tensor &ref, Tensor &out);

// out = e^a. Values are not checked: a large input overflows to inf.
void exp(const Tensor &a, Tensor &out);

// out = ln(a). Values are not checked: a negative input gives NaN, zero -inf.
void log(const Tensor &a, Tensor &out);

// dst += src. The only op that reads its destination, and the reason it exists
// is phase 6: a tensor used twice in the forward pass receives a gradient
// contribution from each use, and they have to accumulate rather than the
// second overwriting the first.
//
// dst must be contiguous; src may have any strides.
void add_into(Tensor &dst, const Tensor &src);

// dst += src, summing src over every dimension where dst's extent is 1 and
// src's is larger. The exact inverse of Tensor::expand: expand gives a READ
// dimension stride 0 so one float is read many times, this gives a WRITE
// dimension stride 0 so many floats accumulate into one.
//
// Ranks must match, and each dst extent must be either src's or 1. Rank change
// is a caller's .reshape(...) at the call site, so one function never has two
// behaviours. dst must be contiguous; src may have any strides.
//
// Accumulates, like add_into and like every gemm kernel - the caller zeroes.
// {2,3} -> {1,3} is the bias gradient, {2,3} -> {1,1} is a full reduction into
// a tensor, and both are the same call.
void sum_into(Tensor &dst, const Tensor &src);

// out = max of a over every dimension where out's extent is 1. Same shape
// rules as sum_into: ranks match, each out extent is a's or 1, out contiguous.
//
// OVERWRITES, unlike sum_into. A max cannot start from a zeroed buffer - an
// all-negative row would come back 0 - so this seeds out itself and the caller
// zeroes nothing. Hence no _into suffix, and the output goes last.
void reduce_max(const Tensor &a, Tensor &out);

// Sum of every element, reading through a's strides - so summing an expand'd
// view counts each repeat, which is correct and is what phase 6's broadcast
// backward will rely on.
//
// Accumulates in double and returns float. NumPy sums pairwise, and a naive
// left-to-right float32 accumulation diverges from it by far more than this
// project's 1e-5 tolerances: over 100000 elements the two answers differ by
// 1.4e-4 relative, measured. Phase 7 would then report a correct gradient as a
// failing one, and the evening goes on the reduction instead of the gradient.
float sum(const Tensor &a);

#endif
