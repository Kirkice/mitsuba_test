@echo off
echo Activating VS2022 build environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set DISTUTILS_USE_SDK=1
echo.
echo Installing nvdiffrast with CUDA support...
.venv\Scripts\python.exe -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation complete!
) else (
    echo.
    echo Installation failed!
)
pause
