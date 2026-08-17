// The pybind11 surface.

#include "gemm.hpp"
#include "tensor.hpp"
#include "tensor_ops.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // std::vector <-> Python sequence. There is NO
                          // std::span caster, which is why every shape-taking
                          // lambda below takes a vector and builds the span at
                          // the call site.

#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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


// Tensor binding helpers                                                   

// Packs rank() extents into a tuple. Tensor has no whole-shape accessor on
// purpose - one would allocate and would overload confusingly against
// shape(dim) - so this aggregation is binding-local.
py::tuple shape_tuple(const Tensor &t) {
    
    py::tuple result(t.rank());

    for(std::size_t i = 0; i < t.rank(); i++){
        result[i] = t.shape(i);
    }

    return result;
}

// Strides IN ELEMENTS, matching Tensor::stride. NumPy's .strides is in BYTES.

py::tuple strides_tuple(const Tensor &t) {
    
    py::tuple result(t.rank());

    for(std::size_t i = 0; i < t.rank(); i++){
        result[i] = t.stride(i);
    }

    return result;
}

// Resolves a Python index expression to an offset in floats from data().

std::size_t index_offset(const Tensor &t, const py::object &idx) {
    std::vector<py::ssize_t> indices;

    if (py::isinstance<py::tuple>(idx)) {
        py::tuple t_idx = idx.cast<py::tuple>();
        for (auto elem : t_idx) {
            indices.push_back(elem.cast<py::ssize_t>());
        }
    } else {
        indices.push_back(idx.cast<py::ssize_t>());
    }

    if (indices.size() != t.rank()) {
        throw std::out_of_range("Error: wrong number of indices, must equal rank of tensor.");
    }

    std::size_t offset = 0;
    for (std::size_t d = 0; d < indices.size(); d++) {
        py::ssize_t i = indices[d];
        py::ssize_t extent = static_cast<py::ssize_t>(t.shape(d));

        if (i < 0) {
            i += extent;
        }

        if (i < 0 || i >= extent) {
            throw std::out_of_range("index out of range");
        }

        offset += static_cast<std::size_t>(i) * t.stride(d);
    }

    return offset;
}

// "Tensor(shape=(2, 3), strides=(3, 1), contiguous=True)" or similar. Values
// are deliberately not printed: a repr that walks a 4 MB buffer is a trap in a
// debugger, and to_numpy() at step 3 is the explicit way to see them.
std::string tensor_repr(const Tensor &t) {
    std::string shape_str = py::repr(shape_tuple(t)).cast<std::string>();
    std::string strides_str = py::repr(strides_tuple(t)).cast<std::string>();
    std::string contig_str = t.is_contiguous() ? "True" : "False";

    return "Tensor(shape=" + shape_str + ", strides=" + strides_str +
           ", contiguous=" + contig_str + ")";
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = R"(autograd engine C++ core.

Tensor storage and forward kernels. No graph logic and no gradient logic - the
autograd DAG lives in Python, from phase 5.

Errors. C++ exceptions arrive as pybind11's built-in mappings; there is no
custom exception hierarchy, and if the graph layer ever wants a ShapeError it
can subclass in Python.

    std::invalid_argument  -> ValueError    every Tensor view, every op
    std::out_of_range      -> IndexError    index bounds checks
    std::bad_alloc         -> MemoryError   Storage allocation failure
    std::length_error      -> ValueError    overflow paths
    std::logic_error       -> RuntimeError  an unimplemented scaffold stub

Threading. Tensor copies share a buffer and nothing protects the floats. The
GIL used to make that moot; from phase 4 the gemm binding releases it, so two
Python threads running gemm over tensors that share storage is a genuine data
race. This is the same statement tensor.hpp already makes about the C++ side.
)";

    // Lets a test assert the rank limit rather than hardcoding 4 next to a
    // comment that will not be updated.
    m.attr("MAX_RANK") = MAX_RANK;


    // Tensor


    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())

        // Every shape argument is a std::vector, converted from any Python
        // sequence by pybind11/stl.h, and the span is built at the call:
        //     Tensor(std::span<const std::size_t>{shape})
        // There is no std::span type caster - stl.h casts vector, array and
        // string_view but not span, and one would carry the lifetime problem
        // every view-type caster has. The cost is one copy of a <=4 element
        // vector per view construction, which is nothing beside a GEMM.
        .def(py::init([](const std::vector<std::size_t> &shape) -> Tensor {
                 return Tensor(std::span<const std::size_t>(shape));
             }),
             py::arg("shape"),
             "Allocate a zeroed, contiguous row-major tensor.\n\n"
             "Rank must be in [1, MAX_RANK] and no extent may be zero: there is\n"
             "no rank-0 tensor (a scalar is shape (1,)) and no empty tensor.\n"
             "Raises ValueError otherwise.")

        .def_property_readonly("shape", &shape_tuple, "Extents, as a tuple.")

        .def_property_readonly("strides", &strides_tuple,
                               "Strides IN ELEMENTS, matching Tensor::stride in C++.\n"
                               "NumPy's .strides is in BYTES - the two differ by 4.")

        .def("rank", [](const Tensor &t) -> std::size_t {
                 return t.rank();
             }, "Number of dimensions. At least 1, at most MAX_RANK.")

        .def("numel", [](const Tensor &t) -> std::size_t {
                 return t.numel();
             }, "Product of the shape. Not the size of the buffer behind it:\n"
                "a slice has fewer elements than the storage it views.")

        .def("is_contiguous", [](const Tensor &t) -> bool {
                 return t.is_contiguous();
             }, "True when the elements are dense row-major.")

        .def("shares_storage_with", [](const Tensor &t, const Tensor &other) -> bool {
                 return t.shares_storage_with(other);
             }, py::arg("other"),
             "True when both handles refer to the same buffer.\n\n"
             "This is what a test should assert to prove a view is a view.\n"
             "Comparing addresses instead false-negatives on two views sitting\n"
             "at different offsets in one buffer.")

        .def("clone", [](const Tensor &t) -> Tensor {
                 return t.clone();
             }, "Deep copy: fresh storage, contiguous, same values. Reads through\n"
                "this tensor's strides, so cloning a transposed view materialises\n"
                "the transpose.")

        .def("zero_", [](Tensor &t) -> void {
                 return t.zero();
             }, "Fill with zeros in place. Requires a contiguous tensor and raises\n"
                "ValueError otherwise: writing through an expanded view would have\n"
                "several logical elements aliasing one float.")


        // Views. All six share storage and return a new Tensor.        


        .def("transpose", [](const Tensor &t, std::size_t d0, std::size_t d1) -> Tensor {
                 return t.transpose(d0,d1);
             }, py::arg("d0"), py::arg("d1"),
             "Swap two dimensions. Nothing moves in memory, so the result is\n"
             "almost never contiguous. Raises ValueError on a bad dimension.")

        .def("permute", [](const Tensor &t, const std::vector<std::size_t> &dims) -> Tensor {
                 return t.permute(std::span<const std::size_t>(dims));
             }, py::arg("dims"),
             "Reorder dimensions: result dimension i takes its extent and stride\n"
             "from dims[i]. Raises ValueError unless dims is exactly a\n"
             "permutation of range(rank()).")

        .def("slice", [](const Tensor &t, std::size_t dim, std::size_t start, std::size_t count) -> Tensor {
                 return t.slice(dim, start, count);
             }, py::arg("dim"), py::arg("start"), py::arg("count"),
             "Narrow one dimension to [start, start + count). Strides are\n"
             "untouched; only the offset moves. This is the operation that\n"
             "produces a row stride wider than the row, which is what the GEMM\n"
             "kernels' leading-dimension handling exists for.")

        .def("expand", [](const Tensor &t, const std::vector<std::size_t> &shape) -> Tensor {
                 return t.expand(std::span<const std::size_t>(shape));
             }, py::arg("shape"),
             "Broadcast: any extent-1 dimension may stretch to any extent and\n"
             "gets stride 0. Rank is unchanged and every other extent must match.\n\n"
             "The result must never be written through - several logical elements\n"
             "alias one float. From step 3 numpy sees it as read-only.")

        .def("reshape", [](const Tensor &t, const std::vector<std::size_t> &shape) -> Tensor {
                 return t.reshape(std::span<const std::size_t>(shape));
             }, py::arg("shape"),
             "Reinterpret the same elements under a new shape.\n\n"
             "Raises ValueError on a non-contiguous tensor rather than silently\n"
             "copying, so one function never has two performance profiles. Write\n"
             ".contiguous().reshape(...) and see yourself paying for it.")

        .def("contiguous", [](const Tensor &t) -> Tensor {
                return t.contiguous();
             }, "Same values in dense row-major order. Returns a handle onto the\n"
                "same buffer when already contiguous, and only copies otherwise.")
        
        
        
        // Numpy


        // Exports this tensor's memory under PEP 3118, so np.asarray(t) is a view

        .def_buffer([](Tensor &t) -> py::buffer_info {
            std::vector<py::ssize_t> shape(t.rank());
            std::vector<py::ssize_t> strides(t.rank());
            bool readonly = false;

            for (std::size_t i = 0; i < t.rank(); i++){
                shape[i] = t.shape(i);
                strides[i] = t.stride(i) * sizeof(float);
                if (strides[i] == 0 ){
                    readonly = true;
                }
            }

            return py::buffer_info(t.data(), sizeof(float), py::format_descriptor<float>::format(), t.rank(), shape, strides, readonly);
        })

        // Explicit copy out: fresh, owning, C-contiguous, base is None.

        .def("to_numpy", [](const Tensor &t) -> py::array_t<float> {

            Tensor c = t.contiguous();

            std::vector<py::ssize_t> shape(c.rank());
            for (std::size_t d = 0; d < c.rank(); d++) {
                shape[d] = c.shape(d);
            }

            py::array_t<float> result(shape);
            std::memcpy(result.mutable_data(), c.data(), c.numel() * sizeof(float));

            return result;

        },
             "Copy to a new C-contiguous numpy array.\n\n"
             "np.asarray(t) is the zero-copy view; this is the copy, and the\n"
             "distinction is visible at the call site on purpose.")

        // Python protocol                                                    

        .def("__repr__", &tensor_repr)

        .def("__getitem__", [](const Tensor &t, const py::object &index) -> float {
                 return t.data()[index_offset(t, index)];
             }, py::arg("index"),
             "Read one element. t[i, j] on a rank-2, t[i] on a rank-1.\n\n"
             "Every index is bounds-checked in every build, unlike C++\n"
             "operator(), which checks only arity and only in debug. Negative\n"
             "indices count from the end. Partial indexing raises - use slice().")

        .def("__setitem__", [](Tensor &t, const py::object &index, float val) -> void {
                 for (std::size_t d = 0; d < t.rank(); d++) {
                     if (t.stride(d) == 0) {
                         throw std::invalid_argument(
                             "cannot write through an expanded tensor: dimension " +
                             std::to_string(d) +
                             " has stride 0, so one store would alias several "
                             "elements. Call .contiguous() first.");
                     }
                 }

                 t.data()[index_offset(t, index)] = val;
             }, py::arg("index"), py::arg("value"),
             "Write one element. Same indexing rules as __getitem__.\n\n"
             "Raises ValueError on an expanded tensor - any dimension with\n"
             "stride 0 - because one store would alias several elements.")

        // Copy is SHALLOW in C++ and stays shallow here. That is the single
        // most surprising fact about this type, and Python is where someone
        // will get bitten by it, so both halves are bound with the asymmetry
        // stated rather than left in the header.
        .def("__copy__", [](const Tensor &t) -> Tensor {
                 return t;
             }, "Shallow: a second handle onto the same buffer. copy.deepcopy is\n"
                "the one that copies the floats.")

        // The memo dict is not optional. copy.deepcopy always passes it, and a
        // one-parameter binding raises TypeError that reads like a Tensor bug.
        .def("__deepcopy__", [](const Tensor &t, const py::dict &) -> Tensor {
                 return t.clone();
             }, py::arg("memo"), "Deep: fresh storage, same values. Equivalent to clone().");

    // Factories                                                             

    m.def("from_numpy", [](const NpArray &arr) -> Tensor {
              if (arr.ndim() < 1 || arr.ndim() > static_cast<py::ssize_t>(MAX_RANK)) {
                  throw std::invalid_argument("Error: Rank should be in [1, MAX_RANK].");
              }

              for (py::ssize_t d = 0; d < arr.ndim(); d++) {
                  if (arr.shape(d) == 0) {
                      throw std::invalid_argument("Error: every extent must be non-zero.");
                  }
              }

              std::vector<std::size_t> shape(static_cast<std::size_t>(arr.ndim()));
              for (py::ssize_t d = 0; d < arr.ndim(); d++) {
                  shape[static_cast<std::size_t>(d)] = static_cast<std::size_t>(arr.shape(d));
              }

              Tensor result{std::span<const std::size_t>(shape)};
              std::memcpy(result.data(), arr.data(), result.numel() * sizeof(float));

              return result;
          }, py::arg("array"),
          "Copy a numpy array into a new Tensor.\n\n"
          "Accepts any dtype and either memory order; the data is converted to\n"
          "contiguous float32 on the way in. Rank must be in [1, MAX_RANK] and\n"
          "no extent may be zero, or ValueError.");

    m.def("zeros", [](const std::vector<std::size_t> &shape) -> Tensor {
            return Tensor(std::span<const std::size_t>(shape));
          }, py::arg("shape"), "Zeroed contiguous tensor. Same as Tensor(shape).");

    m.def("zeros_like", [](const Tensor &t) -> Tensor {
            std::vector<std::size_t> shape(t.rank());
            for (std::size_t d = 0; d < t.rank(); d++) {
                shape[d] = t.shape(d);
            }
            return Tensor(std::span<const std::size_t>(shape));
          }, py::arg("t"),
          "Zeroed CONTIGUOUS tensor with t's shape - not t's strides.\n\n"
          "That is exactly what an out-parameter needs, and it is why passing a\n"
          "transposed tensor here still gives you a legal output buffer.");

    // Elementwise ops.
    //
    // Out-parameter, mirroring C++, not `c = add(a, b)`. Phase 5's graph
    // allocates its own outputs anyway, and a wrapper that allocated and
    // zeroed would throw away the accumulate that phase 6 is built on.
    //
    // GIL stays HELD for all of these: scalar, single-threaded,
    // bandwidth-bound. It is released only where C++ threads actually run,
    // which in phase 4 is gemm and nothing else. Revisit if an elementwise
    // path is ever threaded or vectorised.

    m.def("add", [](const Tensor &a, const Tensor &b, Tensor &out) -> void {
            ::add(a,b, out);
          }, py::arg("a"), py::arg("b"), py::arg("out"),
          "out = a + b, elementwise.\n\n"
          "a and b may have ANY strides - a transposed view or a stride-0\n"
          "expand is read in place, never materialised. out must be contiguous\n"
          "and is OVERWRITTEN, not accumulated into; add_into is the one that\n"
          "accumulates.\n\n"
          "Shapes must match exactly. There is no implicit broadcasting: write\n"
          ".expand(...) at the call site, where it is visible. Raises\n"
          "ValueError on a shape mismatch or a non-contiguous out.");

    m.def("mul", [](const Tensor &a, const Tensor &b, Tensor &out) -> void {
            ::mul(a,b, out);
          }, py::arg("a"), py::arg("b"), py::arg("out"),
          "out = a * b, ELEMENTWISE. Not a matrix product - that is gemm.\n\n"
          "Same contract as add.");

    m.def("scale", [](const Tensor &in, float scalar, Tensor &out) -> void {
              ::scale(in, scalar, out);
          }, py::arg("a"), py::arg("s"), py::arg("out"),
          "out = a * s, for a Python float s. Same contract as add.");

    m.def("relu", [](const Tensor &a, Tensor &out) -> void {
              ::relu(a, out);
          }, py::arg("a"), py::arg("out"),
          "out = max(a, 0), elementwise. Same contract as add.");

    m.def("add_into", [](Tensor &dst, const Tensor &src) -> void {
              ::add_into(dst, src);
          }, py::arg("dst"), py::arg("src"),
          "dst += src. The ONLY op that reads its destination.\n\n"
          "This exists for phase 6: a tensor used twice in a graph receives two\n"
          "gradient contributions, and they have to accumulate rather than the\n"
          "second overwriting the first.\n\n"
          "dst must be contiguous; src may have any strides.");

    m.def("sum", [](const Tensor &a) -> float {
              return ::sum(a);
          }, py::arg("a"),
          "Sum of every element, reading through a's strides - so summing an\n"
          "expanded view counts each repeat, which is the arithmetically\n"
          "correct answer for what that view represents.\n\n"
          "Accumulates in double. Naive float32 accumulation drifts 1.4e-4\n"
          "relative over 100000 elements, fourteen times the tolerance used\n"
          "everywhere else here.");

    /* --------------------------------------------------------------------- */
    /* Phase 2 throwaway. Retired at step 6, once the real gemm binding is     */
    /* green. Do not grow either of these.                                    */
    /* --------------------------------------------------------------------- */

    m.def("add", [](double a, double b) { return a + b; }, py::arg("a"), py::arg("b"),
          "Add two numbers. Toolchain smoke test, retired in phase 4.");

    m.def("gemm", &gemm, py::arg("a"), py::arg("b"), py::arg("kernel") = "tiled_simd",
          R"(Compute a @ b with one of the GEMM kernels.

Both operands are converted to contiguous row-major float32. Returns a new
array; neither operand is modified.

kernel: naive, ikj, tiled, avx2, tiled_simd, or multithreaded.)");
}
