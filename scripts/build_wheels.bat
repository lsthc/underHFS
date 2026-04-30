@echo off
setlocal
cd /d "%~dp0\.."

python -m pip install build scikit-build-core pybind11 ninja
if errorlevel 1 exit /b %errorlevel%

set UNDERHFS_WITH_CUDA=OFF
python -m build --wheel
if errorlevel 1 exit /b %errorlevel%

set UNDERHFS_WITH_CUDA=ON
set CMAKE_ARGS=-DUNDERHFS_WITH_CUDA=ON
python -m build --wheel
if errorlevel 1 exit /b %errorlevel%

endlocal

