#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "underHFS native core";
  m.attr("__version__") = UNDERHFS_VERSION;
#ifdef UNDERHFS_WITH_CUDA
  m.attr("cuda_enabled") = true;
#else
  m.attr("cuda_enabled") = false;
#endif

  py::class_<underhfs::TensorCore>(m, "TensorCore")
      .def(py::init<std::vector<double>, std::vector<std::size_t>>())
      .def_property_readonly("storage", &underhfs::TensorCore::storage)
      .def_property_readonly("shape", &underhfs::TensorCore::shape)
      .def_property_readonly("strides", &underhfs::TensorCore::strides)
      .def("numel", &underhfs::TensorCore::numel)
      .def("add", &underhfs::TensorCore::add)
      .def("mul", &underhfs::TensorCore::mul)
      .def("matmul", &underhfs::TensorCore::matmul)
      .def("sum", &underhfs::TensorCore::sum)
      .def("__repr__", &underhfs::TensorCore::repr);

  m.def("shape_numel", &underhfs::shape_numel);
  m.def("contiguous_strides", &underhfs::contiguous_strides);
}
