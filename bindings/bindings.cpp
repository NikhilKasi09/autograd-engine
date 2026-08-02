// Toolchain smoke test. Nothing here is meant to survive: it exists to prove
// that CMake finds pybind11, that the module builds against the venv's Python,
// and that pytest can import the result. The real tensor bindings land in
// roadmap phase 4.

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "autograd engine C++ core (toolchain smoke test)";

    m.def("add", [](double a, double b) { return a + b; },
          py::arg("a"), py::arg("b"),
          "Add two numbers. Placeholder to prove the binding layer works.");
}
