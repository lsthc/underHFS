# Building underHFS

## Required local tools

- Python 3.13
- Git
- Visual Studio 2022 Build Tools with the C++ workload
- CMake 3.28+
- Ninja, optional when using the Visual Studio CMake generator
- CUDA Toolkit 13.x for native CUDA builds

The current machine has Python 3.13.12, Git, CMake, Visual Studio Build Tools,
Ninja, and CUDA Toolkit 13.2. The native `_core` extension builds successfully
through `scikit-build-core` and pybind11 with CUDA enabled.

## Python fallback development

```powershell
$env:PYTHONPATH = "src"
python -m underhfs.cli test
python -m underhfs.cli bench
```

## Native extension

Install Python build dependencies:

```powershell
python -m pip install pybind11 scikit-build-core
```

Build the native CPU extension:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64
python -m pip install -e . --no-build-isolation --config-settings=cmake.define.UNDERHFS_WITH_CUDA=OFF
python -m underhfs.cli doctor
```

CUDA is opt-in at CMake level:

```powershell
nvcc --version
python -m pip install -e . --config-settings=cmake.define.UNDERHFS_WITH_CUDA=ON
```

On Windows, the repo includes a helper for the current Visual Studio Enterprise
and CUDA 13.2 layout:

```powershell
scripts\build_cuda_editable.bat
```
