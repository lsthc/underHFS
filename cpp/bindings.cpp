#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor_core.hpp"
#ifdef UNDERHFS_WITH_CUDA
#include "kernels.hpp"
#endif

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
#ifdef UNDERHFS_WITH_CUDA
  m.def("cuda_add_f32", &underhfs::cuda_add_f32_host);
  py::class_<underhfs::CudaTensorF32>(m, "CudaTensorF32")
      .def(py::init<const std::vector<float>&, std::vector<std::size_t>>())
      .def_property_readonly("shape", &underhfs::CudaTensorF32::shape)
      .def("numel", &underhfs::CudaTensorF32::numel)
      .def("to_host", &underhfs::CudaTensorF32::to_host)
      .def("add", &underhfs::CudaTensorF32::add);
#endif
}
