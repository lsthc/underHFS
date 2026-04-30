# Building underHFS

## Required local tools

- Python 3.13
- Git
- Visual Studio 2022 Build Tools with the C++ workload
- CMake 3.28+
- Ninja
- CUDA Toolkit 13.x for native CUDA builds

The current machine has Python 3.13.12, Git, and CMake. CUDA Toolkit was not
found under the standard installation path, and `nvcc` and `ninja` were not
found on `PATH` during implementation.

## Python fallback development

```powershell
$env:PYTHONPATH = "src"
python -m pytest
python -m underhfs.cli bench
```

## Native extension

Open a Visual Studio developer shell, verify tools, then build:

```powershell
cl
cmake --version
ninja --version
nvcc --version
python -m pip install -e . --no-build-isolation
```

CUDA is opt-in at CMake level:

```powershell
python -m pip install -e . --config-settings=cmake.define.UNDERHFS_WITH_CUDA=ON
```
