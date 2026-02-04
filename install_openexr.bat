@echo off
echo Installing OpenEXR library for HDR/EXR file support...
echo.

.venv\Scripts\python.exe -m pip install OpenEXR

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS: OpenEXR installed!
    echo Now you can load .exr environment maps for IBL lighting.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo Installation failed!
    echo.
    echo Alternative: Try installing imageio plugins:
    echo   .venv\Scripts\python.exe -m pip install imageio[pyav]
    echo or
    echo   .venv\Scripts\python.exe -m pip install imageio[opencv]
    echo ============================================================
)

echo.
pause
