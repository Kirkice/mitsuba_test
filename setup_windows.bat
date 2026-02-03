@echo off
echo ========================================
echo Mitsuba Studio Setup Script (Windows)
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.9-3.11 from python.org
    pause
    exit /b 1
)

echo [1/7] Python version:
python --version
echo.

:: Create virtual environment
echo [2/7] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

:: Activate virtual environment
echo [3/7] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

:: Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip
echo.

:: Install Mitsuba
echo [5/7] Installing Mitsuba 3...
pip install mitsuba
if errorlevel 1 (
    echo [ERROR] Failed to install Mitsuba
    pause
    exit /b 1
)
echo.

:: Install PyTorch (CPU version for compatibility)
echo [6/7] Installing PyTorch (CPU version)...
echo NOTE: For GPU support, manually install PyTorch from pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.

:: Install other dependencies
echo [7/7] Installing other dependencies...
pip install numpy imageio pillow trimesh matplotlib
echo.

:: Test installation
echo ========================================
echo Testing installation...
echo ========================================
python -c "import mitsuba as mi; mi.set_variant('scalar_rgb'); print('[OK] Mitsuba works!')"
python -c "import torch; print('[OK] PyTorch installed, CUDA:', torch.cuda.is_available())"
echo.

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Build GUI: cargo build --release
echo 2. Run GUI:   cargo run --release
echo 3. Or run quickstart: python quickstart_render.py
echo.
echo For GPU acceleration (optional):
echo - Install CUDA Toolkit from nvidia.com
echo - Reinstall PyTorch GPU version (see SETUP_GUIDE.md)
echo - Install nvdiffrast: pip install git+https://github.com/NVlabs/nvdiffrast.git
echo.
pause
