@echo off
setlocal
cd /d "%~dp0\.."

python -m compileall -q src tests bench
if errorlevel 1 exit /b %errorlevel%

call scripts\build_cuda_editable.bat
if errorlevel 1 exit /b %errorlevel%

python -m underhfs.cli test
if errorlevel 1 exit /b %errorlevel%

python -m underhfs.cli doctor
if errorlevel 1 exit /b %errorlevel%

python -m underhfs.cli export .underhfs-test\release-smoke.onnx --format onnx
if errorlevel 1 exit /b %errorlevel%

python -m underhfs.cli bench --iterations 20 --warmup 3
if errorlevel 1 exit /b %errorlevel%

call scripts\build_wheels.bat
if errorlevel 1 exit /b %errorlevel%

endlocal
