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
#ifdef UNDERHFS_WITH_CUDNN
  m.attr("cudnn_enabled") = true;
#else
  m.attr("cudnn_enabled") = false;
#endif
#ifdef UNDERHFS_WITH_NCCL
  m.attr("nccl_enabled") = true;
#else
  m.attr("nccl_enabled") = false;
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
  m.def("cuda_fused_adamw_f32", &underhfs::cuda_fused_adamw_f32_host);
  m.def("cuda_attention_f32", &underhfs::cuda_attention_f32_host);
#ifdef UNDERHFS_WITH_CUDNN
  m.def("cudnn_conv2d_forward_f32", &underhfs::cudnn_conv2d_forward_f32_host);
#endif
  m.def("cuda_allocator_stats", &underhfs::cuda_allocator_stats);
  m.def("cuda_empty_cache", &underhfs::cuda_empty_cache);
  m.def("cuda_stream_stats", &underhfs::cuda_stream_stats);
  m.def("cuda_synchronize", &underhfs::cuda_synchronize);
  py::class_<underhfs::CudaTensorF32>(m, "CudaTensorF32")
      .def(py::init<const std::vector<float>&, std::vector<std::size_t>>())
      .def_property_readonly("shape", &underhfs::CudaTensorF32::shape)
      .def("numel", &underhfs::CudaTensorF32::numel)
      .def("to_host", &underhfs::CudaTensorF32::to_host)
      .def("add", &underhfs::CudaTensorF32::add)
      .def("mul", &underhfs::CudaTensorF32::mul)
      .def("matmul", &underhfs::CudaTensorF32::matmul)
      .def("sum", &underhfs::CudaTensorF32::sum);
  py::class_<underhfs::CudaTensorF16>(m, "CudaTensorF16")
      .def(py::init<const std::vector<float>&, std::vector<std::size_t>>())
      .def_property_readonly("shape", &underhfs::CudaTensorF16::shape)
      .def("numel", &underhfs::CudaTensorF16::numel)
      .def("to_host", &underhfs::CudaTensorF16::to_host)
      .def("add", &underhfs::CudaTensorF16::add)
      .def("mul", &underhfs::CudaTensorF16::mul);
  py::class_<underhfs::CudaTensorBF16>(m, "CudaTensorBF16")
      .def(py::init<const std::vector<float>&, std::vector<std::size_t>>())
      .def_property_readonly("shape", &underhfs::CudaTensorBF16::shape)
      .def("numel", &underhfs::CudaTensorBF16::numel)
      .def("to_host", &underhfs::CudaTensorBF16::to_host)
      .def("add", &underhfs::CudaTensorBF16::add)
      .def("mul", &underhfs::CudaTensorBF16::mul);
#endif
}
