@echo off
setlocal
cd /d "%~dp0\.."

if "%UNDERHFS_SKIP_DEP_INSTALL%"=="" (
python -m pip install build scikit-build-core pybind11 ninja
if errorlevel 1 exit /b %errorlevel%
)

if "%UNDERHFS_WHEEL_OUTDIR%"=="" set UNDERHFS_WHEEL_OUTDIR=dist
if not exist "%UNDERHFS_WHEEL_OUTDIR%" mkdir "%UNDERHFS_WHEEL_OUTDIR%"

echo [underHFS] Building CPU wheel
set CMAKE_ARGS=-DUNDERHFS_WITH_CUDA=OFF
python -m build --wheel
if errorlevel 1 exit /b %errorlevel%

echo [underHFS] Building CUDA 13.2 wheel
set CMAKE_ARGS=-DUNDERHFS_WITH_CUDA=ON
python -m build --wheel
if errorlevel 1 exit /b %errorlevel%

if "%UNDERHFS_BUILD_VENDOR_WHEELS%"=="1" (
  echo [underHFS] Building CUDA 13.2 + cuDNN wheel
  set CMAKE_ARGS=-DUNDERHFS_WITH_CUDA=ON -DUNDERHFS_WITH_CUDNN=ON
  python -m build --wheel
  if errorlevel 1 exit /b %errorlevel%

  echo [underHFS] Building CUDA 13.2 + NCCL wheel
  set CMAKE_ARGS=-DUNDERHFS_WITH_CUDA=ON -DUNDERHFS_WITH_NCCL=ON
  python -m build --wheel
  if errorlevel 1 exit /b %errorlevel%
)

endlocal
