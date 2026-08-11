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
from C to C++. The autograd half has not been started yet.

The port replaced `posix_memalign`/`free` with an RAII `Matrix` that owns its
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

## Building

```bash
python3 -m venv .venv && .venv/bin/pip install pybind11 numpy pytest

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/gemm_benchmark        # the ladder, all hardware threads
./build/gemm_benchmark 8      # ...or a specific thread count
./build/gemm_tests            # correctness, 1085 assertions
ctest --test-dir build        # same, through ctest
.venv/bin/pytest tests/python # bindings and the NumPy cross-check
```

Sanitisers get their own build directories, since the flags have to reach the
link line as well as the compile line:

```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DGEMM_SANITIZE=address
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DGEMM_SANITIZE=thread
cmake --build build-asan -j && ctest --test-dir build-asan
cmake --build build-tsan -j && ctest --test-dir build-tsan
```

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
  `a @ b` over 12 shapes and all six kernels.

Worth knowing: ASan cannot catch a column overrun on any row except the last,
because it lands in the next row of the same allocation. The numeric comparison
is what actually catches those.

## Roadmap

- [x] Generalise the GEMM library to arbitrary M, N and K
- [x] Port to C++, CMake, Catch2
- [ ] Tensor type: shape, strides, contiguous storage
- [ ] pybind11 bindings
- [ ] Autograd graph and `backward()`
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
above, and the port to C++.
