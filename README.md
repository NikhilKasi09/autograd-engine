# Micro-PyTorch: autograd engine and accelerated BLAS

A deep learning framework built from scratch, in two halves that meet in the
middle:

1. A GEMM library in C++ using AVX2 intrinsics, cache tiling and `std::jthread`.
2. An autograd engine that records tensor operations and walks the graph
   backwards to get gradients.

C++ owns tensor storage and the forward kernels, Python owns the graph and the
backward pass, and pybind11 joins them.

No BLAS library is used anywhere. Writing the kernel is the point.

## Where it is now

Every phase on the roadmap is done. From the bottom up:

- **A GEMM library** of six kernels over arbitrary shapes, the fastest within
  4% of OpenBLAS at 256 and running an FMA every cycle.
- **A tensor type** with strides and views, so a transpose, a slice and a
  broadcast are all a second handle onto one buffer.
- **pybind11 bindings** that share memory with NumPy rather than copying
  through it.
- **An autograd graph** with a topological sort and a `backward()` over seven
  operations, checked against finite differences and against PyTorch.
- **Layers, two losses and SGD**, enough to train a model.
- **MNIST.** A 784-128-10 MLP reaches 97.4 to 97.6% test accuracy in ten
  epochs, follows PyTorch's loss step for step, and after three fixes found by
  profiling trains within noise of single-threaded PyTorch on that model.

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

`add`, `mul`, `scale`, `relu`, `exp`, `log`, `relu_backward`, `add_into`,
`sum_into`, `reduce_max` and `sum`. Inputs may have any strides, so a transposed view or a broadcast is read
where it lies; the output has to be contiguous. Shapes have to match exactly.
There is no implicit broadcasting: a caller who wants one writes `expand` at
the call site, where it is visible.

`sum_into` is the exception, and it is `expand` run backwards: `expand` gives a
read dimension stride 0 so one float is read many times, `sum_into` gives a
write dimension stride 0 so many floats land on one. Collapsing a `{4,3}` onto
a `{1,3}` is a bias gradient. It looks like `add_into` and is not — `add_into`
walks its destination with a counter, which only works because its shapes
match, and keeping that counter here writes past the end of the buffer.

`reduce_max` takes `sum_into`'s shape rules and the opposite write contract.
`sum_into` accumulates, so its caller zeroes the destination first. A max
cannot start from zero — an all-negative row would come back as 0 — so
`reduce_max` seeds its own output with `-inf` and overwrites. Seeding it with
zero instead fails exactly one test in each suite, the all-negative row, and
passes everything else.

These were scalar until a training loop said otherwise. The reasoning was that
elementwise work is memory bandwidth bound, so vectorising it buys a fraction
of what it buys in GEMM, and a stride-general walk that has to handle stride 0
does not vectorise without a separate contiguous fast path. That fast path was
left until something measured needed it.

MNIST needed it, and for a reason worse than missing vectorisation. The walk
padded every shape out to rank 4 on the right, so a matrix had its columns in
the second of four loops and the innermost loop ran once per element. Each
element paid a four-term offset sum per input. The kernels now pad on the left,
work out where a row starts once, and take a plain contiguous loop whenever
every input steps by one along it. A transposed or broadcast input still takes
the strided loop. `add_into` on a 784x128 went from 100 microseconds to 6. See
"Training MNIST" for what that was worth.

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

Backward feels this, since it wants `Xᵀ @ dY` and `dY @ Wᵀ` — so every backprop
GEMM pays an allocation and a full copy, per layer, per step. BLAS solves it
with `transa`/`transb` flags threaded through the call, letting the kernel walk
the operand in the other order instead of rewriting it.

The worry here was that adding the flags later would mean touching every call
site in the graph. It would not: there is only one. Every GEMM the graph runs
goes through a single private helper that owns the transposing, so `transa`
becomes a change to that function and nothing else — a decision a profile can
drive rather than one that had to be guessed now.

The profile has since been taken, and it said no. On MNIST the copy of the
transposed input batch cost 226 microseconds a step, more than both large
matrix products together. But the cost was in the copy routine, not in copying:
`clone()` rebuilt a multi-index with a divide per dimension for every element.
Walking it a row at a time brought that copy down to 14 microseconds, 3% of a
step, and there was nothing left for `transa` to save.

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

A layer looks like this, and is what the backward operations were for:

```python
h = autograd.relu(autograd.add(autograd.matmul(x, w), autograd.expand(b, (4, 3))))
loss = autograd.sum(h)
loss.backward()                 # w.grad, b.grad
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
never graph tensors. `relu` is where that stops being hypothetical: its
backward masks on its own output, so the node holds the buffer it just produced
while the tensor wrapping that buffer holds the node. Saving the *tensor*
instead is a one-word change that produces identical numbers everywhere and
hands the whole graph to the cycle collector.

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

Seven operations: `add`, `mul`, `matmul`, `relu`, `sum`, `expand` and `scale`.
The first two arrived without touching C++ at all, which was why they were
chosen. The next four needed two new kernels between them and no more, because
most of the work was already sitting in the tensor type. `scale` came last and
free — its kernel had been bound and unused since the bindings landed.

**`sum` and `expand` are one operation read in opposite directions.** `sum`
allocates going forward and returns a view coming back; `expand` returns a view
going forward and allocates coming back. Neither materialises a broadcast,
which matters because a bias add is a broadcast on every forward pass of every
layer. It does make `expand` the first op whose output does not own its buffer
— safe, since nothing here writes through an input, but new.

**Configuration reaches a node as a keyword argument, never as an input.** A
target shape is not something you differentiate. Positional arguments are graph
tensors and receive gradients; keyword arguments go to the node's constructor
and the backward walk never sees them.

**A node computes only the gradients somebody asked for.** It records which of
its inputs required a gradient when it ran, and its backward returns nothing
for the rest. This was not the original rule, which was that backward returns
a gradient for every input and the walk decides who keeps one. That is simpler
and costs nothing while every input is small. It stopped being free on MNIST:
a layer's input is data, and the first layer was computing the gradient of the
loss with respect to the pixels — a matrix product as large as its own forward
pass — and throwing it away.

Summing a `{2,3}` gives a `{1,1}`, not a bare scalar. Keeping the rank is what
makes the backward a bare `expand` with no allocation — at rank 1 it would need
a reshape first, and `reshape` refuses a non-contiguous tensor.

## The nn layer

Everything above computes a gradient. This is what uses one.

```python
import autograd
from autograd import _core, nn, optim

class MLP(nn.Module):
    def __init__(self, generator):
        self.fc1 = nn.Linear(784, 128, generator=generator)
        self.fc2 = nn.Linear(128, 10, generator=generator)

    def forward(self, x):
        return self.fc2(autograd.relu(self.fc1(x)))

model = MLP(_core.Generator(seed=0))
opt = optim.SGD(model.parameters(), lr=0.1)

for x, target in batches:                 # target is one-hot, {batch, 10}
    opt.zero_grad()
    loss = nn.cross_entropy(model(x), target)
    loss.backward()
    opt.step()
```

**A parameter is a type, and a module finds its parameters by looking.**
`Parameter` is a `Tensor` that requires a gradient by default and exists mostly
to be recognised. `Module.parameters()` walks the module's own attributes,
yields each `Parameter` and recurses into each child `Module`. There is no
registry and no `__setattr__` hook, which is what PyTorch uses; assigning an
attribute is just assigning an attribute. The one subtlety is that the set of
already-seen parameters is passed down the whole recursion rather than created
per module, or a weight shared between two layers would be listed twice and
stepped twice.

**A module is only a holder.** `Linear.forward` is one line calling the free
function `nn.linear(x, w, b)`. That is not tidiness: the finite-difference
checker rebuilds its inputs for every perturbation, so arithmetic locked inside
an object with fixed parameters could not be checked at all.

**Weights are stored `{in, out}`**, the transpose of PyTorch's `{out, in}`.
PyTorch computes `x @ W.T`; here that transpose would be a copy on every
forward pass, for the reason given under "The transpose cost". Stored this way
round the forward is a plain `x @ w`. Weights start uniform on
`±1/sqrt(in)`, which keeps the size of a layer's output roughly independent of
its width, and the bias starts at zero.

**Is a loss an operation or a function? Both, and for a reason.** `mse_loss`
is a function: it is `add`, `mul`, `sum` and `scale` composed, with no node of
its own, and cost no C++. `cross_entropy` is one fused node. Built from
separate `exp`, `log` and divide operations it overflows as soon as a logit
passes about 88, since `exp` of that is infinite in float32. Fused, each row
is shifted by its own maximum first, so every `exp` is of something at most
zero, and the gradient collapses to `softmax - target` with nothing
transcendental left to differentiate. Logits of ±1000 give the exact loss.

Targets are one-hot rows rather than class indices, because there is no
integer tensor here to hold an index.

**Initial weights come from a seeded generator, not from NumPy.** Nothing in
the engine imports NumPy to compute anything. `_core.Generator` owns a
`std::mt19937_64`, so a run reproduces from its seed to the bit, and two layers
built from one generator draw from one stream. It cannot be copied: a copy
would be a second generator emitting the same numbers, and two layers
initialised from it would be identical.

**`SGD.step` writes into the parameter's own buffer.** `p -= lr * grad`, with
`-lr * grad` built in a scratch buffer first. Scaling the gradient in place
would save that allocation and corrupt the gradient.

**Inference needs no `no_grad()`.** A forward pass through live parameters
does record a graph, but recording is cheap next to the arithmetic: on a
784-128-10 MLP at batch 64, 127.7 µs with the graph against 124.1 µs with
detached parameters, medians of 7, ranges overlapping. The graph is freed when
the output is dropped.

## Training MNIST

The demonstration the rest was built for: a 784-128-10 MLP, plain SGD at a
learning rate of 0.1, batches of 64, ten epochs. Nothing was tuned; those
numbers were fixed before the first run.

```bash
.venv/bin/python examples/mnist_data.py     # once: download and verify
.venv/bin/python examples/mnist_train.py    # train, print accuracy per epoch
.venv/bin/python examples/mnist_torch.py    # the same run in PyTorch
```

NumPy loads the data, shuffles it and counts the correct answers. Every number
the model computes comes from the engine.

### Against PyTorch

The PyTorch run is built to match rather than assumed to: it starts from this
engine's own initial weights, transposed into PyTorch's layout, and is fed the
same batches in the same order. What is left to differ is the arithmetic.

| seed | this engine | PyTorch |
|---|---|---|
| 0 | 97.39% | 97.36% |
| 1 | 97.50% | 97.51% |
| 2 | 97.42% | 97.40% |
| 3 | 97.63% | 97.63% |
| 4 | 97.50% | 97.47% |

Test accuracy after ten epochs. After one epoch the two agree to all four
digits on every seed.

The stronger check is the loss, step by step. For the first 300 steps the two
agree to 5.6e-7 relative, which is float32 rounding. They part company after
about an epoch, as any two float32 implementations must once a rounding
difference flips a relu, and they end within 0.03 points of each other.

### Where the time went

The first run took 1.69 seconds an epoch against PyTorch's 0.36: 4.7 times
slower. The expectation, written down in advance, was that GEMM would dominate
and the fix would be packing. A profile of one training step said otherwise:

| | share of a step |
|---|---|
| copying transposed operands | 39% |
| `add_into` | 21% |
| GEMM, all six calls | 17% |
| `scale` | 6% |
| Python, between the kernel calls | 10% |

The matrix products were under a fifth of the step. The rest was glue, and
three fixes took it out, each timed against the build before it.

| fix | seconds per epoch | |
|---|---|---|
| as first run | 1.73 | |
| skip gradients nobody asked for | 1.13 | 1.53x |
| walk elementwise ops a row at a time | 0.63 | 1.81x |
| copy a strided tensor a row at a time | 0.39 | 1.68x |

**Skipping unwanted gradients** removed a whole matrix product. The first layer
was differentiating the loss with respect to the input pixels: a GEMM the size
of its own forward pass, plus a transposed copy of the weight to feed it.

**The elementwise walk** had its loops in the wrong order for a matrix, as
described under "Elementwise ops". This was the largest single win.

**The strided copy** is the one that changed a plan. The intention had been to
give GEMM `transa` and `transb` flags. Fixing the copy made that unnecessary.

Two more fixes were on the list, an `axpy` kernel for the optimiser and
borrowing a gradient instead of copying it. After the elementwise fix they
were worth 1% and 3% of a step, and were dropped.

### The result

All four run alternately in one session, seven runs each, seconds per epoch:

| | median | range |
|---|---|---|
| this engine, as first run | 1.686 | 1.632 to 1.729 |
| this engine, now | 0.370 | 0.362 to 0.391 |
| PyTorch, 1 thread | 0.361 | 0.350 to 0.380 |
| PyTorch, 12 threads | 0.379 | 0.352 to 0.439 |

4.6 times faster than it started, and inside PyTorch's range. Two caveats
belong next to that. It is one workload, and a small one: at batch 64 the
largest matrix is 784x128, where this kernel is at its best and where
PyTorch's threads have nothing to do. And PyTorch is carrying far more
generality per call than this is.

What is left of a step is about 60% GEMM, and the two large products run at
133 GFLOP/s, which is this machine's single-core limit. So packing, the one
known gap in the kernel, would buy nothing on this model: it helps from about
1024 upwards and nothing here is that big. It is still the right next piece of
work for the kernel. It is not what MNIST was waiting for.

Timing on this machine drifts by up to 20% between sessions, so every
comparison above comes from builds alternating in a single run.
`examples/mnist_bench.py --compare --other-root <tree>` does that for any two
builds, and `--profile` produces the table of where a step goes.

## Building

```bash
python3 -m venv .venv && .venv/bin/pip install pybind11 numpy pytest

# Optional: the gradient cross-check skips without it, and says so in the
# pytest header. Configure with -DGEMM_REQUIRE_TORCH=ON to refuse to build
# without it; the default AUTO only requires it if it is present at configure
# time, so it catches a torch that breaks later, not one never installed.
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/gemm_benchmark        # the ladder, all hardware threads
./build/gemm_benchmark 8      # ...or a specific thread count
./build/gemm_tests            # correctness, 1907 assertions
./build/gemm_tests "[tensor]" # ...or one tag: tensor, view, ops, random, gemm
.venv/bin/pytest tests/python # bindings, autograd, gradients, training: 597 tests
ctest --test-dir build        # both suites together
```

`ctest` does not build, so the full command is
`cmake --build build && ctest --test-dir build`.

Five of the Python tests train on the real MNIST files and skip until they are
downloaded with `.venv/bin/python examples/mnist_data.py`, which puts 11 MB
under `data/`. The pytest header says whether they ran.

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

The graph tests themselves are mostly structural, and the gradients in them are
compared against hand-derived NumPy expressions; finite differences and the
PyTorch cross-check are below. The last of them is a two-layer MLP — matmul,
bias broadcast, relu, matmul, loss — run forward and backward and checked
against a NumPy pass on every parameter. Three others carry weight out of
proportion to their size. The diamond pins the walk
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

The backward operations were checked the same way, and one result is worth
keeping: a node saving its own output *wrapper* instead of the raw buffer fails
nothing at all except the lifetime test, which is the whole argument for having
one.

### Gradient checking

Every test above compares a gradient against a hand-derived NumPy expression —
the same calculus that produced the implementation, written out twice. It
catches a typo and cannot catch a mistake in the derivation. So gradients are
now also checked against finite differences, which know nothing about calculus,
and against PyTorch.

The textbook recipe does not survive contact with a float32 engine. Central
differences at `h = 1e-5` report a **correct** gradient as 4e-2 wrong here,
because subtracting two nearly-equal float32s destroys the digits the answer
lives in. `h` is squeezed from both sides:

| `h` | `sum(x*x*x)` | `matmul`, seeded | `relu`, margin 1e-2 |
|---|---|---|---|
| 1e-1 | 1.0e-2 | 6.0e-7 | **4.5e-1** |
| 1e-2 | 9.9e-5 | 7.5e-6 | 9.5e-7 |
| **3e-3** | **1.9e-5** | **1.7e-5** | **7.0e-6** |
| 1e-3 | 6.8e-5 | 5.3e-5 | 1.3e-5 |
| 1e-5 | 3.4e-3 | 5.1e-3 | 1.4e-3 |

Below, cancellation: the floor goes as `eps*|f|/h`. Above, the relu kink — a
step that straddles zero measures a slope that exists nowhere, which is the 45%
in the top right. `h = 3e-3` sits between them and at the bottom of the bowl for
the one case with a real `h^2` truncation term.

**The tolerance is computed, not chosen.** The noise floor turned out not to be
a property of the op at all. Summing `N` floats, where the gradient is
identically 1.0 for every `N`, the floor still grows with `N`:

| N | 6 | 24 | 96 | 384 | 1536 | 6144 |
|---|---|---|---|---|---|---|
| \|f\| | 2.1 | 5.0 | 7.9 | 13.3 | 33.6 | 83.1 |
| floor | 1.3e-5 | 8.6e-5 | 2.3e-4 | 5.5e-4 | 7.2e-4 | 2.0e-3 |

It tracks `|f|`, the magnitude of the numbers being subtracted, and `eps*|f|/h`
predicts it within 2–3x across a thousandfold range. That explains every
anomaly in the per-op table — `matmul` with `K=9` is noisier than `K=3` for the
same reason a longer sum is. So the tolerance is derived per check as
`max(3*eps*|f|/h, 2e-5)`, with a hard ceiling that raises rather than passing:
a case too large to resolve in float32 has to shrink, and quietly widening the
bound instead is how a check stops catching anything.

This matters because a gradient wrong by a factor of `(1+e)` produces exactly
`e` relative error — **the tolerance is the detection threshold**, the blind
spot stated as a number. A single global tolerance would set it for every op at
once, at the width the worst one needs.

Then PyTorch, in float64, as the tighter oracle:

| | agreement |
|---|---|
| finite differences, best case | 2e-5 |
| torch float64 vs this engine | **1.2e-7** |

which is float32 machine epsilon — the engine is as accurate as its storage
permits — and 170x tighter than any finite difference can be. Where the engine
diverges from torch deliberately, the torch side is constructed to match rather
than assumed: `.expand()` instead of implicit broadcasting, `keepdim=True`
because `sum` keeps rank with 1s.

Mutation results. Swapping either transpose in matmul's backward fails 15
tests; dropping the `expand` from sum's backward fails 9; setting `h` to the
1e-5 the textbook asks for fails 32. Two came back the other way and are worth
more than the ones that passed. Holding relu inputs off the kink catches
nothing — a uniform draw lands inside the step on 1 seed in 200, so the margin
removes a rare flake rather than a bug, and the explicit on-the-kink test is
what proves the mechanism. And finite differences cannot see the relu
subgradient at exactly zero at all, since a symmetric difference never samples
it; only the torch cross-check pins that down.

Finding the gradient checker found one real bug, before any test was written.
`matmul` could not be differentiated when `M == 1` — batch size one, which the
`nn` layer would have hit on its first step. `a.T` of a `{1,K}` is `{K,1}`,
which *is* contiguous, since the stride of an extent-1 dimension is never
stepped. But `gemm` wants `stride(1) == 1` literally, so `contiguous()` no-opped
and `gemm` rejected the operand. Two definitions of contiguous that agree
everywhere except on a dimension of extent one.

### Training

The last layer of tests builds a model the way a user would and trains it.

| | result |
|---|---|
| MLP on three clusters, 60 steps | loss 1.14 → 0.009, 100% |
| the same, loss on every step | strictly decreasing |
| MLP on XOR, 400 steps | loss 0.005, 4 of 4 |
| `Linear` under MSE, 200 steps | weights recovered to 6 digits |
| five SGD steps against PyTorch | parameters agree to 1e-5 |

XOR is there because it is not linearly separable: it only trains if gradient
is reaching the first layer through the relu. The PyTorch comparison is the
only oracle for the optimiser — same weights, same batch, five steps each.

Cross-entropy was the first function here that is not piecewise linear or
bilinear, so the finite-difference step was swept again before its checks were
written rather than assumed to carry over. It does: over 120 cases the worst
error was 0.54 of the derived tolerance.

Writing the loss found a bug in the engine. An operation may return `None`
from its backward for an input that is not differentiable, and that path had
been documented since the graph was first built — but no operation had ever
used it. The target of a loss was the first, and the walk raised `KeyError` on
reaching it.

Mutation results. A cross-entropy backward that ignores the gradient arriving
from above is right whenever the loss is the last thing in the graph, and the
full-Jacobian check passes it, because that check seeds a scalar output with
exactly 1. It is caught by seeding with anything else, by scaling the loss, and
by PyTorch — 8 tests. Dropping the shift by the row maximum fails one test, the
±1000 logits. An optimiser that holds the generator `parameters()` returns
steps once and then does nothing forever; the test asserts on the second step.
A `Linear` that remembers its last output fails nothing except the lifetime
test, which runs one full training step and requires every activation to be
freed by reference counting alone.

### MNIST

The loader is tested without the dataset: its tests write their own small IDX
files, 2x3 images with every pixel different, so a swapped height and width or
a header read in the wrong byte order shows up as wrong values. Downloads are
tested against `file://` mirrors. A file only takes its real name once its
SHA-256 has matched.

The tests on the real data assert thresholds that were measured over five
seeds first: a first loss within 0.1 of ln 10, over 90% after one epoch where
the worst seed scored 91.97%, and a run that repeats to the bit from its seed.

Mutation results. Shuffling the images and the labels with two different
permutations still produces well-formed batches, and drops accuracy to 12%.
Forgetting `zero_grad` drops it to 10%. A learning rate 1% too large is
invisible to the accuracy and is caught by the step-for-step comparison with
PyTorch, on 299 steps of 300.

The fixes were mutated too, and one result is worth keeping. Putting the
elementwise padding back on the right passes every test in both suites,
because it gives the same answers and is only slower. Nothing but the timing
guards it.

## Roadmap

- [x] Generalise the GEMM library to arbitrary M, N and K
- [x] Port to C++, CMake, Catch2
- [x] Tensor type: shape, strides, views, elementwise ops
- [x] pybind11 bindings
- [x] Autograd graph and `backward()`
- [x] Backward kernels
- [x] Gradient checking against finite differences and PyTorch
- [x] `nn` layers, loss functions, SGD
- [x] Train an MLP on MNIST against a PyTorch baseline

Packing is the next piece of work. It closes the gap to OpenBLAS above 1024,
and the MNIST profile showed it was right to leave it until after: nothing in
that model is large enough to need it.

## Origins

The original square-only C kernels came out of a group project at Imperial.
The history of that work is preserved in this repo's commits.

Everything since then is mine: generalising all six kernels to arbitrary
shapes, the leading-dimension rework, the test harness, the benchmarking
above, the port to C++, the tensor type, the pybind11 bindings, the autograd
graph, the backward operations on top of it, the gradient validation that
checks them, the layers, losses and optimiser that train on them, and the MNIST
run with the profiling that followed it.
