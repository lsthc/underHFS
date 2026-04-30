# underHFS Release and Wheel Matrix

underHFS publishes a CPU source wheel first and reserves CUDA wheels for native
runtime builds that are tied to a CUDA Toolkit major/minor line.

## Local Build Commands

```powershell
python -m pip install build scikit-build-core pybind11 ninja
python -m build --wheel
```

CUDA editable build:

```powershell
scripts\build_cuda_editable.bat
```

## Wheel Matrix

| Wheel | Python | Platform | Native | CUDA |
| --- | --- | --- | --- | --- |
| `underhfs-*-py3-none-any.whl` | 3.13 | any | Python fallback | none |
| `underhfs-*-cp313-win_amd64.whl` | 3.13 | Windows x64 | C++20 | none |
| `underhfs-*-cp313-win_amd64.cuda132.whl` | 3.13 | Windows x64 | C++20/CUDA | CUDA 13.2 |

## Release Gates

- `python -m compileall -q src tests bench`
- `python -m underhfs.cli test`
- `python -m underhfs.cli doctor`
- `python -m underhfs.cli bench --iterations 20 --warmup 3`
- Native CUDA probe must report allocator and stream stats on CUDA wheels.

