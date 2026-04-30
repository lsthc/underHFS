@echo off
setlocal

set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH_V13_2=%CUDA_PATH%"

if not exist "%VSDEVCMD%" (
  echo Visual Studio developer command script not found: %VSDEVCMD%
  exit /b 1
)

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo nvcc not found under CUDA_PATH: %CUDA_PATH%
  exit /b 1
)

call "%VSDEVCMD%" -arch=x64
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

cl >nul
if errorlevel 1 exit /b 1

nvcc --version
ninja --version
python -m pip install -e . --no-build-isolation --config-settings=cmake.define.UNDERHFS_WITH_CUDA=ON --config-settings=cmake.args=-GNinja
