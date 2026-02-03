# Mitsuba Studio 设置指南

## 🚀 快速开始

### 前置要求

- **Python 3.9-3.11** （推荐 3.10）
- **Rust** （已安装，用于 GUI）
- **CUDA** （可选，用于 GPU 加速的拟合）
- **Git**

### Windows 快速设置

```powershell
# 1. 创建 Python 虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
.venv\Scripts\activate

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装 Mitsuba 3（CPU 版本）
pip install mitsuba

# 5. 安装其他依赖
pip install numpy torch torchvision imageio pillow trimesh

# 6. （可选）安装 nvdiffrast（需要 CUDA）
# 如果有 NVIDIA GPU 且安装了 CUDA：
pip install git+https://github.com/NVlabs/nvdiffrast.git

# 7. 测试 Mitsuba 安装
python -c "import mitsuba; print('Mitsuba version:', mitsuba.__version__)"

# 8. 构建并运行 GUI
cargo run --release
```

### Linux / macOS 快速设置

```bash
# 1. 创建 Python 虚拟环境
python3 -m venv .venv

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装 Mitsuba 3
pip install mitsuba

# 5. 安装其他依赖
pip install numpy torch torchvision imageio pillow trimesh

# 6. （可选）安装 nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git

# 7. macOS 特殊配置：安装 LLVM（用于 llvm_ad_* 变体）
# 使用 Homebrew:
brew install llvm

# 8. 测试 Mitsuba 安装
python -c "import mitsuba; print('Mitsuba version:', mitsuba.__version__)"

# 9. 构建并运行 GUI
cargo run --release
```

## 📦 详细安装步骤

### 1. 检查 Python 版本

```bash
python --version  # 应该显示 3.9.x - 3.11.x
```

如果版本不对，请从 [python.org](https://www.python.org/downloads/) 下载安装。

### 2. 创建虚拟环境

**为什么需要虚拟环境？**
- 隔离项目依赖
- 避免版本冲突
- 便于管理

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

激活成功后，命令行前会出现 `(.venv)` 前缀。

### 3. 安装 Mitsuba 3

```bash
pip install mitsuba
```

**验证安装：**
```bash
python -c "import mitsuba as mi; mi.set_variant('scalar_rgb'); print('✓ Mitsuba works!')"
```

### 4. 安装 PyTorch

**CPU 版本（适合测试）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本（推荐，用于 Disney BRDF 拟合）：**

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应的 CUDA 版本：

```bash
# 示例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 示例：CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**验证 PyTorch：**
```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 5. 安装 nvdiffrast（光栅化库）

**需要 CUDA（NVIDIA GPU）：**
```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

**如果没有 GPU：**
nvdiffrast 也支持 CPU 模式（较慢），安装命令相同。

**验证：**
```python
python -c "import nvdiffrast.torch as drt; print('✓ nvdiffrast installed')"
```

### 6. 安装其他依赖

```bash
pip install numpy imageio pillow trimesh matplotlib
```

### 7. macOS 特殊配置（LLVM）

如果要使用 `llvm_ad_rgb` 变体（可微渲染），需要安装 LLVM：

```bash
brew install llvm
```

GUI 会自动检测 LLVM 路径并设置 `DRJIT_LIBLLVM_PATH`。

### 8. 构建 Rust GUI

```bash
cargo build --release
```

### 9. 运行 GUI

```bash
cargo run --release
```

或直接运行编译好的可执行文件：

**Windows:**
```powershell
.\target\release\mitsuba_studio.exe
```

**Linux/macOS:**
```bash
./target/release/mitsuba_studio
```

## 🔧 配置 GUI

首次运行时，需要在 GUI 中配置 Python 路径：

1. 打开 GUI
2. 切换到左侧 **Render** 标签
3. 修改 **Python** 路径：
   - **Windows:** `.venv\Scripts\python.exe` 或 `python`
   - **Linux/macOS:** `.venv/bin/python` 或 `python3`

配置会自动保存到 `.mitsuba_studio_state.json`。

## ⚠️ 常见问题

### Q1: "系统找不到指定的路径"（Windows）

**原因：** Python 路径错误或虚拟环境未创建

**解决：**
1. 确认虚拟环境已创建：`dir .venv\Scripts\python.exe`
2. 在 GUI 中修改 Python 路径为 `python` 或完整路径
3. 或使用系统 Python：`python` 或 `C:\Python310\python.exe`

### Q2: "ModuleNotFoundError: No module named 'mitsuba'"

**原因：** 虚拟环境未激活或 Mitsuba 未安装

**解决：**
```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# 安装 Mitsuba
pip install mitsuba
```

### Q3: "CUDA not available"

**原因：** PyTorch 安装的是 CPU 版本或 CUDA 未安装

**解决：**
1. 检查 CUDA 是否安装：`nvidia-smi`
2. 重新安装 PyTorch GPU 版本（参考步骤 4）
3. 或使用 `--device cpu` 参数（较慢）

### Q4: "nvdiffrast not available"

**原因：** nvdiffrast 未安装

**解决：**
```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

如果编译失败，检查是否安装了 Visual Studio（Windows）或 GCC（Linux）。

### Q5: macOS 上 "LLVM not found"

**原因：** LLVM 未安装或路径未设置

**解决：**
```bash
brew install llvm
```

然后在 GUI 的 **Render → Advanced** 中设置：
```
/opt/homebrew/opt/llvm/lib/libLLVM.dylib
```

### Q6: 训练速度很慢

**原因：** 使用 CPU 而非 GPU

**解决：**
1. 确认 CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`
2. 在命令行使用 `--device cuda`
3. 降低分辨率或 SPP

### Q7: GUI 启动后立即崩溃

**原因：** 图形驱动问题或依赖缺失

**解决：**
1. 更新显卡驱动
2. Windows: 安装 Visual C++ Redistributable
3. Linux: 安装 `libxcb` 相关库

## 📊 依赖版本建议

| 包 | 推荐版本 | 最低版本 |
|----|---------|---------|
| Python | 3.10.x | 3.9.x |
| mitsuba | 最新 | 3.4.0 |
| torch | 2.0+ | 1.13.0 |
| nvdiffrast | 最新 | 0.3.1 |
| numpy | 1.24+ | 1.21.0 |

## 🎯 测试安装

运行完整的测试脚本：

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 运行测试
python -c "
import sys
print('Python version:', sys.version)

try:
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    print('✓ Mitsuba works')
except Exception as e:
    print('✗ Mitsuba error:', e)

try:
    import torch
    print('✓ PyTorch works, CUDA:', torch.cuda.is_available())
except Exception as e:
    print('✗ PyTorch error:', e)

try:
    import nvdiffrast.torch as drt
    print('✓ nvdiffrast works')
except Exception as e:
    print('✗ nvdiffrast error:', e)

print('Setup complete!')
"
```

预期输出：
```
Python version: 3.10.x
✓ Mitsuba works
✓ PyTorch works, CUDA: True
✓ nvdiffrast works
Setup complete!
```

## 🚀 下一步

1. **快速渲染测试：**
   ```bash
   python quickstart_render.py
   ```
   应该生成 `cbox.png`

2. **启动 GUI：**
   ```bash
   cargo run --release
   ```

3. **尝试简单拟合：**
   - GUI 中点击 "Fit diffuse albedo"
   - 观察 Log 标签的实时进度

4. **尝试 Disney BRDF 拟合：**
   - 点击 "Fit Disney BRDF"
   - 查看 `renders/fit_disney/` 的结果

## 📚 更多资源

- [Mitsuba 3 文档](https://mitsuba.readthedocs.io/)
- [nvdiffrast GitHub](https://github.com/NVlabs/nvdiffrast)
- [PyTorch 官网](https://pytorch.org/)
- [Disney BRDF README](DISNEY_BRDF_README.md)

---

**祝你使用愉快！🎨✨**

如果遇到其他问题，请查看 GitHub Issues 或提出新问题。
