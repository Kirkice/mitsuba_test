@echo off
echo Installing DDS format support for environment maps...
echo.

.venv\Scripts\python.exe -m pip install Pillow

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS: DDS support installed!
    echo Now you can load .dds environment maps for IBL lighting.
    echo.
    echo Note: DDS files with mipmaps will only load the top level.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo Installation failed!
    echo.
    echo Alternative: Try installing imageio plugins:
    echo   .venv\Scripts\python.exe -m pip install imageio[pyav]
    echo ============================================================
)

echo.
pause
