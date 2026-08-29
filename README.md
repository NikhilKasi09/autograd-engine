# Micro-PyTorch: autograd engine and accelerated BLAS

A deep learning framework built from scratch, in two halves that meet in the
middle:

1. A GEMM library in C++ using AVX2 intrinsics, cache tiling and `std::jthread`.
2. An autograd engine that records tensor operations and walks the graph
   backwards to get gradients.

The plan is for C++ to own tensor storage and the forward kernels, and Python
to own the graph and the backward pass, joined with pybind11.

No BLAS library is used anywhere. Writing the kernel is the point.

## Where it is now

The GEMM library is finished, handles arbitrary shapes, and has been ported
from C to C++. The tensor type it runs on is finished too, and it is now
exposed to Python: the tensor, its views, the elementwise ops and all six GEMM
kernels are callable from there, sharing memory with NumPy rather than copying
through it. The autograd half has started: there is a graph, a topological
sort and a working `backward()`, differentiating addition and elementwise
multiplication. The remaining operations — matmul, relu, sum — are next, and
they are the ones that need new kernels rather than new graph machinery.

The port replaced `posix_memalign`/`free` with RAII storage that owns its
aligned buffer, pthreads with `std::jthread`, the Makefile with CMake, and the
hand-written test harness with Catch2. It changed no kernel logic: the tiling,
the intrinsics and the ragged-edge peeling came across untouched, which is
what makes the performance claim below checkable.

Six kernels, kept as a performance ladder. Slower ones are not deleted when a
faster one lands, because the progression is the interesting part.

## The ladder

GFLOP/s, float32, gcc 13.3 with `-O3 -mavx2 -mfma`, on a Ryzen AI 9 HX 370
(12 cores, 24 threads). Higher is better.

These were measured on the C build and are unchanged by the C++ port. That is
not an assumption: the two binaries were run interleaved, seven runs each,
and the ranges overlap on all 36 shape/kernel pairs. Run-to-run noise on this
machine reaches 20%, so a single before/after pair would have shown nothing
either way.

| Kernel | 256³ | 1024³ | 1023³ |
|---|---|---|---|
| Naive (ijk) | 2.56 | 0.60 | 3.24 |
| Loop reordered (ikj) | 70.99 | 46.56 | 44.87 |
| Cache tiled | 60.77 | 39.72 | 37.61 |
| AVX2, no tiling | 19.25 | 7.49 | 14.76 |
| Tiled + AVX2 | 126.81 | 66.44 | 67.85 |
| Multithreaded, 8 threads | 45.56 | 330.73 | 337.89 |

Three things in that table are worth explaining, because none of them are
what you would guess:

**Tiling on its own is slower than just reordering the loops** at 256, and
still slower at 1024. Reordering to ikj already gives sequential access to B
and C, and at these sizes the tiling overhead is not paid back. Tiling only
earns its place once vectorisation makes the memory traffic the bottleneck,
which is why the fused kernel is nearly 2x either of its parts.

**AVX2 without tiling is the worst of the fast kernels.** It streams all of B
from memory for every row of A, so it is bandwidth bound and the vector units
sit idle. Vectorising the arithmetic does not help if you cannot feed it.

**Naive at 1023 is 5x faster than at 1024.** A stride of 1024 floats is exactly
4KB, so walking down a column of B hits the same cache set every time and
thrashes. One column off and the problem disappears. Nothing about the code
changes, only the number.

Threading is only worth it above about 512. At 256 the 8 threads cost more to
start than the work they save.

The multithreaded row is at 8 threads. `./build/gemm_benchmark` now defaults
to every hardware thread, so pass `8` to reproduce that column exactly.

## Compared to OpenBLAS

numpy 2.5.1 against OpenBLAS 0.3.33, running its AVX2 kernels so the comparison
is like for like. Same timing harness on both sides, median of 5.

Single threaded:

| size | this | OpenBLAS | ratio |
|---|---|---|---|
| 256 | 126.5 | 131.2 | 96% |
| 512 | 108.5 | 131.5 | 83% |
| 1024 | 65.4 | 123.8 | 53% |
| 2048 | 58.4 | 127.7 | 46% |

All threads:

| size | this | OpenBLAS | ratio |
|---|---|---|---|
| 1024 | 484.7 | 518.7 | 93% |
| 2048 | 616.7 | 718.5 | 86% |

At 256 the micro-kernel is essentially at the hardware limit. OpenBLAS peaks
around 131 GFLOP/s single threaded, which works out at 4.1 GHz times 32 FLOP
per cycle, so it is issuing an FMA every cycle and so are we.

The gap at larger sizes is one specific missing technique: **packing**. OpenBLAS
copies blocks of A and B into contiguous scratch buffers laid out in the order
the micro-kernel reads them, so it holds ~130 GFLOP/s flat from 256 to 2048.
This kernel reads B straight out of the original matrix with a stride, so as
the matrices grow it starts missing TLB entries and pulling partial cache
lines, and throughput decays. That is the whole difference.

Multithreaded the gap narrows to 7-14%, because both implementations are
memory bandwidth bound by then and packing matters less.

## Ragged shapes

The vectorised kernel handles leftover rows and columns by peeling them off and
running them through the scalar tiled kernel. That is correct but not fast:

| shape | this | OpenBLAS |
|---|---|---|
| 256x256 | 130.0 | 131.7 |
| 255x255 | 82.9 | 124.4 |

At 255 the edges are under 4% of the arithmetic but cost a third of the
runtime, since they are scalar and too thin for the cache to help. OpenBLAS
loses about 5% on the same shape.

In practice, keeping N a multiple of 8 and M a multiple of 4 avoids it
entirely. `_mm256_maskload_ps` would remove the strips altogether and is the
obvious next optimisation.

## The tensor type

The kernels used to take a `Matrix`: rank two, one stride, sole owner of its
buffer. That type is gone. Everything takes a `Tensor` now, which is a shape,
a stride per dimension, an offset, and a `shared_ptr` to a refcounted buffer.

The kernels themselves did not change at all. A rank-2 tensor with unit inner
stride is already the raw pointer and leading dimension they take, so swapping
the storage type came to 107 lines of signatures and accessors across 16 files
and not one line inside a loop. Keeping the storage type out of the inner loops
is what made that cheap, and this was the first real test of it.

Three decisions in it are worth explaining.

**Copying a tensor is shallow, where copying a matrix was a compile error.**
The old type deleted its copy constructor, on the grounds that copying a
1024x1024 matrix by accident costs 4MB and is invisible in a benchmark loop. A
tensor cannot keep that rule. A transpose, a slice and a broadcast are all just
a second handle onto one buffer, so two tensors sharing storage is the normal
case rather than the bug. `clone()` is the deep copy, and the compiler has
stopped being the thing that catches an accidental one.

**`reshape` throws on a non-contiguous tensor instead of quietly copying.**
Torch's `reshape` falls back to a copy where its `view` throws. One function
with two performance profiles behind identical syntax is how a training loop
ends up mysteriously slow, so there is one function here and it refuses.
Callers who want the copy write `.contiguous().reshape(...)` and can see
themselves paying for it.

**There is no rank-0 tensor and no empty one.** A scalar is shape `{1}`. Both
are divergences from torch, and both are there so that no kernel, no reduction
and no printing path needs a special case for a tensor with nothing in it.

The views are `transpose`, `permute`, `slice`, `expand`, `reshape` and
`contiguous`. All of them share storage and allocate nothing, except
`contiguous` when it has no choice. `expand` is the broadcast: stretching a
length-1 dimension gives it a stride of zero, so every index along it lands on
the same element, and adding a bias row to a whole batch costs no copy.

Bad offset arithmetic is the failure mode a type like this invites, and ASan is
poor at catching it, since a column overrun lands in the next row of the same
allocation on every row but the last. So every view checks, in debug builds,
that the furthest element it can address is still inside the buffer. Views can
only be built through one private constructor, which is what makes that check
impossible to forget rather than merely conventional.

## Elementwise ops

`add`, `mul`, `scale`, `relu`, `add_into` and `sum`. Inputs may have any
strides, so a transposed view or a broadcast is read where it lies; the output
has to be contiguous. Shapes have to match exactly. There is no implicit
broadcasting: a caller who wants one writes `expand` at the call site, where it
is visible.

These are scalar, on purpose. Elementwise work is memory bandwidth bound rather
than compute bound, so vectorising it buys a fraction of what it buys in GEMM,
and a stride-general walk that has to handle stride 0 does not vectorise
cleanly without a separate contiguous fast path. That fast path is worth
writing when a training loop says so, and not before.

`sum` accumulates in a double and returns a float. NumPy sums pairwise, and a
plain left-to-right float32 accumulation over 100000 elements drifts 1.4e-4
away from it, which is fourteen times the tolerance used everywhere else here.
Gradient checking would have reported that as a broken gradient rather than a
broken reduction, which is an evening nobody gets back.

## The Python surface

`autograd._core` exposes the tensor type and the forward kernels. No graph
logic and no gradient logic — that half is Python, and it sits on top of this
one. `autograd.Tensor` is the graph type and is documented in the next section;
`_core.Tensor` is the buffer it holds.

```python
from autograd import _core
import numpy as np

A = _core.from_numpy(np.random.rand(256, 256).astype(np.float32))
B = _core.from_numpy(np.random.rand(256, 256).astype(np.float32))
C = _core.zeros([256, 256])

_core.gemm(A, B, C, "tiled_simd")   # C += A @ B
np.asarray(C)                       # a view, not a copy
```

Names mirror the C++ ones — `rank` not `ndim`, `numel` not `size` — so the two
layers stay greppable. Ops take their output as a parameter and return `None`,
matching the C++ signatures rather than reading like torch. That is deliberate:
`C += A*B` is the kernels' contract, `add_into` exists because a tensor used
twice in a graph collects two gradient contributions, and a wrapper that
allocated and zeroed a fresh output would throw both away. Phase 5's graph
allocates its own outputs anyway, so the convenience would have bought one line
and cost a rebuild every time the graph layer changed its mind.

**Tensors share memory with NumPy.** `np.asarray(t)` is a view, because a
`Tensor` and an `ndarray` describe memory the same way — pointer, shape,
strides — so the buffer protocol can hand one to the other directly. A
transposed or sliced tensor crosses the boundary still being a view. Strides go
across in bytes where `Tensor::stride` counts elements, which is the one
conversion in the whole layer that fails silently if you get it wrong: NumPy
reads the right buffer with the wrong step and returns the right values in the
wrong order. `to_numpy()` and `from_numpy()` are the explicit copies, named
separately so the call site shows which one is being paid for.

An expanded tensor is exported read-only. Stride 0 means several logical
elements alias one float, so a single store would land in four places at once;
C++ blocks that by requiring contiguity on every mutating path, and the
read-only flag is what carries the same rule to NumPy.

**Two things the C++ side does silently, the binding does not.** A bad operand
makes the GEMM wrappers print to stderr and return with `C` untouched — right
for the Catch2 harness, whose magnitude guard depends on it, and a wrong answer
with no signal from Python. The binding calls `gemm_check_shapes` itself and
raises, so the accept/reject decision still lives in one place. And `C` sharing
storage with `A` or `B` is refused outright: the inner loops use `restrict`, so
overlap is undefined rather than slow, which is a fine contract between a C++
author and themselves and a segfault by typo from Python.

### The transpose cost, and why there is no `transa`

`gemm` requires unit inner stride on all three operands. A row stride wider
than the row is fine — that is what a slice produces, and the leading-dimension
handling exists for it — but a transposed operand is not, so it has to be
materialised with `.contiguous()` first.

Phase 6 will feel this. Backward wants `Xᵀ @ dY` and `dY @ Wᵀ`, so every
backprop GEMM pays an allocation and a full copy, per layer, per step. BLAS
solves it with `transa`/`transb` flags threaded through the call, which lets
the kernel walk the operand in the other order instead of rewriting it. That is
not hard to add, but doing it after the graph exists means touching every call
site in it, so it is written down here rather than discovered in a profile.

## The autograd graph

`autograd.Tensor` *holds* a `_core.Tensor` rather than being one. The C++ type
owns the buffer, the shape and the strides; the Python type owns the gradient
slot and the edge back to whatever produced it.

```python
import numpy as np
import autograd
from autograd import _core

x = autograd.Tensor(_core.from_numpy(np.array([2.0], np.float32)), requires_grad=True)

a = autograd.mul(x, x)          # x^2
b = autograd.add(a, x)          # x^2 + x
c = autograd.mul(b, a)          # x^4 + x^3

c.backward()
np.asarray(x.grad)              # [44.] = 4x^3 + 3x^2 at x = 2
```

**A `Function` instance is the node.** PyTorch splits a stateless `Function`
from a per-call context object; here one object is both, holding the tensors it
consumed and whatever the backward pass will need. One thing to inspect when a
graph is wrong instead of two.

**Every edge points backward, and that is what stops the graph leaking.** A
tensor holds its `grad_fn`, a `grad_fn` holds its parents, and nothing holds a
result. Dropping the loss frees the entire graph by reference counting, with
the cycle collector never involved — which the test suite asserts directly, by
taking weak references into a graph and checking they die on `del` *without*
calling `gc.collect()` first. Calling it would make that test pass on a graph
made entirely of cycles.

The rule that keeps it true is that a node saves raw `_core.Tensor` buffers and
never graph tensors. The temptation arrives in the next phase: `relu`'s
backward wants its own output to build a mask from, and saving the output
*tensor* would close the loop.

**Reverse topological order is what makes a tensor used twice correct.** Its
gradient is the sum of what comes back down each path, so no node may be
processed until every node that could send it a gradient already has. A
post-order walk reversed gives that; a breadth-first walk does not — though it
happens to be right on graphs this shallow, which is exactly how the bug would
have survived to the MNIST run. The suite pins it with a diamond whose answer
is 5; a stack-based walk reads 2 on the same graph.

**Gradients land on leaves only.** Intermediates keep theirs in a dictionary
that dies with the pass, unless asked otherwise with `retain_grad()`. What does
persist is accumulated rather than assigned, so two `backward()` calls sum and
`zero_grad()` in a training loop means something.

**`.grad` holds a raw `_core.Tensor`, not a graph tensor.** torch makes it the
same type, which is what allows gradients of gradients; here a gradient is data
rather than a node, and keeping it raw is part of what guarantees no edge
points forward. Accumulation buffers take their shape from the tensor they
belong to rather than from the incoming gradient, so a wrong-shaped gradient is
rejected by `add_into` instead of propagating quietly into `.grad`.

Two operations so far, `add` and `mul`, chosen because between them they cover
every node shape the engine has to handle — one with nothing saved, one with
both operands saved, and both able to receive two gradients at once. Neither
needed a new C++ kernel, so the graph arrived without touching the C++ side at
all.

## Building

```bash
python3 -m venv .venv && .venv/bin/pip install pybind11 numpy pytest

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/gemm_benchmark        # the ladder, all hardware threads
./build/gemm_benchmark 8      # ...or a specific thread count
./build/gemm_tests            # correctness, 1528 assertions
./build/gemm_tests "[tensor]" # ...or one tag: tensor, view, ops, gemm
.venv/bin/pytest tests/python # bindings, autograd, NumPy cross-check
ctest --test-dir build        # both suites together
```

`ctest` does not build, so the full command is
`cmake --build build && ctest --test-dir build`.

Each build tree writes its own complete `autograd` package — `_core` plus a
copy of the Python package — so a tree can be tested without disturbing any
other.
Only a Release, unsanitised tree writes to `python/`, which is why a bare
`pytest tests/python` needs no `PYTHONPATH`. Any other tree is selected by
naming it:

```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DGEMM_SANITIZE=off
cmake --build build-debug -j
PYTHONPATH=build-debug/python .venv/bin/pytest tests/python
```

That tree is the routine check for anything touching offsets or strides. Its
asserts are live, so `Tensor::assert_within_storage` fires on every view the
bindings construct — a stronger detector here than ASan, which cannot see a
column overrun landing in the next row of the same allocation. Every run prints
the module it imported in the pytest header, so a wrong-tree import names
itself.

Sanitisers get their own build directories, since the flags have to reach the
link line as well as the compile line:

```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DGEMM_SANITIZE=address
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DGEMM_SANITIZE=thread
cmake --build build-asan -j && ctest --test-dir build-asan
cmake --build build-tsan -j && ctest --test-dir build-tsan
```

Those trees build `_core` too, but only into themselves, and ctest runs just
the C++ suite there: importing an instrumented module into a stock CPython
needs `LD_PRELOAD`, `ASAN_OPTIONS=detect_leaks=0` and `PYTHONMALLOC=malloc`,
which is a manual one-off rather than something ctest should attempt.

TSan needs ASLR off on Ubuntu 24.04, which defaults `vm.mmap_rnd_bits` to 32
where TSan supports 28. CMake wraps the test in `setarch -R` automatically;
run the binary directly and it dies with "unexpected memory mapping".

## Testing

`./build/gemm_tests` runs every kernel against an independent reference over 21 shapes,
chosen so that each dimension independently crosses the vector width, the
register block and the tile boundary. That includes the awkward ones: `1x1x1`,
`17x31x13`, `1x512x1`, `129x130x131`.

Two things it does that a simpler harness would not:

- **Every kernel is checked against a plain triple loop**, including the naive
  one, rather than using naive as the source of truth. A bug in the reference
  cannot quietly bless five kernels.
- **Every case runs twice**, once with C zeroed and once prefilled. The kernels
  compute `C += A*B`, and zeroing C before every call makes `=` and `+=` look
  identical. The backward pass will accumulate into C, so that distinction has
  to be tested now.

`./build/gemm_tests "[.fuzz]"` adds 300 random shapes, giving 6600 assertions.
It is hidden behind a dot tag so the default run stays fast.

Two further checks the harness cannot make on its own:

- **A mutation test.** Swapping an `ldc` for an `lda` in one kernel passes
  every square shape and is caught only at `64x96x1`. Confirming the harness
  fails on a deliberate bug is the cheapest evidence it is not vacuously
  passing.
- **NumPy.** The reference implementation and the kernels were written by the
  same person on the same assumptions, so agreement between them is weaker
  evidence than it looks. `pytest tests/python` checks `np.allclose` against
  `a @ b` over 12 shapes and all six kernels, each run twice — zeroed and
  prefilled — so the Python layer keeps the `+=` distinction the C++ harness
  makes.

The Python suite carries two tests aimed at the binding rather than the
kernels, since the Catch2 suite already covers those. One hands `add` a
transposed operand and compares elementwise: a binding that "helpfully" called
`.contiguous()` on its inputs would give correct values everywhere and quietly
delete the stride-general reads, and nothing else would notice. The other runs
GEMM on an 8x16 sliced to 8x8 — a row stride of 16 with unit inner stride,
which must be accepted — for the same reason.

The tensor and op tests make up the rest of that count. The ones that earn
their place are the ones handing an op a transposed or broadcast input, since
a kernel that ignores strides and walks the buffer flat passes every contiguous
case and fails those immediately. One is arranged so that the wrong answer
holds the right values in the wrong order, which a checksum would wave through.

Worth knowing: ASan cannot catch a column overrun on any row except the last,
because it lands in the next row of the same allocation. The numeric comparison
is what actually catches those.

The graph tests are structural rather than numerical — gradient checking
against finite differences is a later phase, and for `add` and `mul` the
analytic gradients are exact anyway, so they are compared against hand-derived
NumPy expressions. Three of them carry the weight. The diamond pins the walk
order, and it is the only test in the file that a breadth-first walk fails.
The lifetime test asserts that weak references into a dropped graph die without
`gc.collect()`, which is the difference between "no cycles" and "cycles the
collector happens to clean up". And one test asserts that a node with no
gradient-requiring input builds no node at all, rather than building one and
flagging it unwanted — a distinction invisible in every gradient value and
visible only as memory growth.

Both were checked by mutation rather than trusted. Breaking the walk to a
stack-based order moves the diamond from 5 to 2. Swapping the pair returned by
multiplication's backward, or having it return the incoming gradient unchanged,
fails five or six tests each — the operands are distinct and the shape is 2x3
precisely so that those mutations cannot pass.

## Roadmap

- [x] Generalise the GEMM library to arbitrary M, N and K
- [x] Port to C++, CMake, Catch2
- [x] Tensor type: shape, strides, views, elementwise ops
- [x] pybind11 bindings
- [x] Autograd graph and `backward()`
- [ ] Backward kernels
- [ ] Gradient checking against finite differences and PyTorch
- [ ] `nn` layers, loss functions, SGD
- [ ] Train an MLP on MNIST against a PyTorch baseline

Packing is not on the list but is worth more than anything on it for the
performance story, so it may jump the queue.

## Origins

The original square-only C kernels came out of a group project at Imperial.
The history of that work is preserved in this repo's commits.

Everything since then is mine: generalising all six kernels to arbitrary
shapes, the leading-dimension rework, the test harness, the benchmarking
above, the port to C++, the tensor type, the pybind11 bindings, and the
autograd graph.
