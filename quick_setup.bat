@echo off
chcp 65001 >nul
echo ========================================
echo Mitsuba Studio 快速设置
echo ========================================
echo.

echo [1/6] 检查 Python...
python --version
if errorlevel 1 (
    echo [错误] Python 未找到！
    echo 请确保 Python 已安装并添加到 PATH
    echo 你可能需要：
    echo 1. 重启终端/命令提示符
    echo 2. 或者使用完整路径，例如：C:\Python310\python.exe
    pause
    exit /b 1
)
echo.

echo [2/6] 创建虚拟环境...
if exist .venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功
)
echo.

echo [3/6] 激活虚拟环境...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b 1
)
echo.

echo [4/6] 升级 pip...
python -m pip install --upgrade pip
echo.

echo [5/6] 安装 Mitsuba 3...
pip install mitsuba
if errorlevel 1 (
    echo [警告] Mitsuba 安装失败，请检查网络连接
)
echo.

echo [6/6] 安装其他依赖...
echo 安装 PyTorch (CPU 版本)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
echo 安装其他库...
pip install numpy imageio pillow trimesh matplotlib
echo.

echo ========================================
echo 测试安装...
echo ========================================
python -c "import sys; print('Python:', sys.version)"
python -c "import mitsuba; print('Mitsuba: OK')" 2>nul && echo Mitsuba: OK || echo Mitsuba: 失败
python -c "import torch; print('PyTorch: OK, CUDA:', torch.cuda.is_available())" 2>nul && echo. || echo PyTorch: 失败
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步：
echo 1. 保持此窗口打开（虚拟环境已激活）
echo 2. 测试渲染: python quickstart_render.py
echo 3. 或启动 GUI（新窗口）: cargo run --release
echo.
echo 提示：
echo - 虚拟环境位置: .venv
echo - Python 路径: .venv\Scripts\python.exe
echo.
pause
