#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "underHFS native core";
  m.attr("__version__") = UNDERHFS_VERSION;

  py::class_<underhfs::TensorCore>(m, "TensorCore")
      .def(py::init<std::vector<double>, std::vector<std::size_t>>())
      .def_property_readonly("storage", &underhfs::TensorCore::storage)
      .def_property_readonly("shape", &underhfs::TensorCore::shape)
      .def("numel", &underhfs::TensorCore::numel)
      .def("__repr__", &underhfs::TensorCore::repr);
}
