// Minimal pybind11 surface, deliberately throwaway.
//
// This exists so the kernels can be checked against NumPy with np.allclose,
// which the C++ harness cannot do: it compares against its own reference, and
// two implementations agreeing is weaker evidence than agreeing with a third
// party nobody involved wrote.
//
// Roadmap phase 4 replaces all of this with the real Tensor binding. Do not
// grow it - if something wants a richer API, that is phase 4 arriving early.

#include "gemm.hpp"
#include "tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

// c_style forces a row-major contiguous buffer and forcecast converts dtype,
// so a float64 or Fortran-ordered array arrives as something the kernels can
// read rather than being rejected. The copy that implies is fine here; this is
// a correctness path, not a performance one.
using NpArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Every kernel converts to this, including the captureless lambda below, so
// there is nothing for std::function's type erasure to buy.
using GemmFn = void (*)(const Tensor &, const Tensor &, Tensor &);

void require_2d(const NpArray &arr, const char *who) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(who) + " must be 2-D, got " +
                                    std::to_string(arr.ndim()) + "-D");
    }
}

Tensor tensor_from_numpy(const NpArray &arr) {
    const size_t rows = static_cast<size_t>(arr.shape(0));
    const size_t cols = static_cast<size_t>(arr.shape(1));

    Tensor m({rows, cols});

    // Source is contiguous, destination walks its own row stride. Equal for a
    // freshly constructed Tensor, and not equal the moment one is a view.
    for (size_t i = 0; i < rows; i++) {
        memcpy(&m.data()[i * m.stride(0)], &arr.data()[i * cols], cols * sizeof(float));
    }

    return m;
}

NpArray numpy_from_tensor(const Tensor &m) {
    const py::ssize_t rows = static_cast<py::ssize_t>(m.shape(0));
    const py::ssize_t cols = static_cast<py::ssize_t>(m.shape(1));

    NpArray result = py::array_t<float>({rows, cols});

    for (py::ssize_t i = 0; i < rows; i++) {
        memcpy(&result.mutable_data()[i * cols],
               &m.data()[static_cast<size_t>(i) * m.stride(0)], static_cast<size_t>(cols) * sizeof(float));
    }

    return result;
}

GemmFn select_kernel(const std::string &name) {
    if (name == "naive")      return gemm_naive;
    if (name == "ikj")        return gemm_ikj;
    if (name == "tiled")      return gemm_tiled;
    if (name == "avx2")       return gemm_avx2;
    if (name == "tiled_simd") return gemm_tiled_simd;
    if (name == "multithreaded") {
        // Captures nothing, so it converts to a plain function pointer
        return [](const Tensor &a, const Tensor &b, Tensor &c) { gemm_multithreaded(a, b, c, 8); };
    }
    throw std::invalid_argument("unknown kernel: " + name);
}

NpArray gemm(const NpArray &a, const NpArray &b, const std::string &kernel) {
    // Rank before shape: reading shape(1) of a 1-D array is out of bounds, so
    // the dimension check has to come first to be the thing that reports it.
    require_2d(a, "a");
    require_2d(b, "b");

    if (a.shape(1) != b.shape(0)) {
        throw std::invalid_argument("shape mismatch: a is " + std::to_string(a.shape(0)) + "x" +
                                    std::to_string(a.shape(1)) + ", b is " +
                                    std::to_string(b.shape(0)) + "x" +
                                    std::to_string(b.shape(1)));
    }

    const GemmFn selected = select_kernel(kernel);

    Tensor A = tensor_from_numpy(a);
    Tensor B = tensor_from_numpy(b);

    // The kernels accumulate, so C has to start zeroed.
    Tensor C({A.shape(0), B.shape(1)});

    {
        // Released only around the kernel. gemm_multithreaded spawns real
        // threads, and holding the GIL across the call would block every other
        // Python thread in the process for its whole duration.
        py::gil_scoped_release release;
        selected(A, B, C);
    }
    // GIL held again from here: allocating the result array is a Python
    // operation and must not happen without it.

    return numpy_from_tensor(C);
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "autograd engine C++ core (phase 2: GEMM only)";

    m.def("add", [](double a, double b) { return a + b; }, py::arg("a"), py::arg("b"),
          "Add two numbers. Toolchain smoke test, retired in phase 4.");

    m.def("gemm", &gemm, py::arg("a"), py::arg("b"), py::arg("kernel") = "tiled_simd",
          R"(Compute a @ b with one of the GEMM kernels.

Both operands are converted to contiguous row-major float32. Returns a new
array; neither operand is modified.

kernel: naive, ikj, tiled, avx2, tiled_simd, or multithreaded.)");
}
