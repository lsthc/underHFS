# underHFS Release and Wheel Matrix

underHFS publishes CPU and CUDA native wheels tied to a Python and CUDA Toolkit
major/minor line.

## Local Build Commands

```powershell
python -m pip install build scikit-build-core pybind11 ninja
python -m build --wheel
```

CUDA editable build:

```powershell
scripts\build_cuda_editable.bat
```

CPU and CUDA wheel builds can be run through the checked-in helper:

```powershell
scripts\build_wheels.bat
```

By default the helper builds CPU and CUDA 13.2 wheels. Vendor-specific wheels
can be requested explicitly once the local SDKs are visible:

```powershell
$env:UNDERHFS_BUILD_VENDOR_WHEELS = "1"
scripts\build_wheels.bat
```

Set `UNDERHFS_SKIP_DEP_INSTALL=1` when dependencies are already pinned in the
active environment.

The local release gate runs compileall, CUDA editable build, the underHFS test
suite, doctor, ONNX export smoke, bench, and wheel build:

```powershell
scripts\release_check.bat
```

## Wheel Matrix

| Wheel | Python | Platform | Native | CUDA |
| --- | --- | --- | --- | --- |
| `underhfs-*-py3-none-any.whl` | 3.13 | any | Portable Python runtime | none |
| `underhfs-*-cp313-win_amd64.whl` | 3.13 | Windows x64 | C++20 | none |
| `underhfs-*-cp313-win_amd64.cuda132.whl` | 3.13 | Windows x64 | C++20/CUDA | CUDA 13.2 |

## Release Gates

- `python -m compileall -q src tests bench`
- `python -m underhfs.cli test`
- `python -m underhfs.cli doctor`
- `python -m underhfs.cli export .underhfs-test\release-smoke.onnx --format onnx`
- `python -m underhfs.cli bench --iterations 20 --warmup 3`
- `scripts\build_cuda_editable.bat`
- `scripts\build_wheels.bat`
- Native CUDA probe must report allocator and stream stats on CUDA wheels.
- Native CUDA probe should report fused AdamW and attention kernel availability
  for CUDA wheels.
