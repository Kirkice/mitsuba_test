# 🚀 快速开始指南

## ⚡ 最快安装方式

### 方法 1：一键安装（推荐）

1. **双击运行**：`quick_setup.bat`
   - 会自动创建虚拟环境
   - 安装所有必要的依赖
   - 测试安装是否成功

2. **等待安装完成**（大约 5-10 分钟）

3. **测试渲染**：
   ```bash
   python quickstart_render.py
   ```
   应该生成 `cbox.png`

4. **启动 GUI**：
   ```bash
   cargo run --release
   ```

### 方法 2：手动安装

如果自动脚本失败，可以手动执行：

```powershell
# 1. 打开 PowerShell 或命令提示符

# 2. 进入项目目录
cd h:\Project\mitsuba_test

# 3. 创建虚拟环境
python -m venv .venv

# 4. 激活虚拟环境
.venv\Scripts\activate

# 5. 安装依赖
pip install mitsuba torch numpy imageio pillow trimesh

# 6. 测试
python quickstart_render.py
```

## ✅ 验证安装

运行以下命令检查：

```bash
# 激活虚拟环境（如果还没激活）
.venv\Scripts\activate

# 测试 Python
python --version

# 测试 Mitsuba
python -c "import mitsuba; print('Mitsuba OK')"

# 测试 PyTorch
python -c "import torch; print('PyTorch OK, CUDA:', torch.cuda.is_available())"
```

## 🎮 使用 GUI

1. **启动 GUI**：
   ```bash
   cargo run --release
   ```

2. **配置 Python 路径**（首次使用）：
   - 打开 GUI 后，切换到左侧 **Render** 标签
   - 修改 **Python** 字段为：`.venv\Scripts\python.exe`
   - GUI 会自动保存配置

3. **渲染测试**：
   - 点击顶部 **Render** 按钮
   - 切换到 **Log** 标签查看输出
   - 切换到 **Preview** 标签查看结果

4. **尝试材质拟合**：
   - 确保场景已配置好
   - 点击 **Fit diffuse albedo**（简单测试）
   - 或点击 **Fit Disney BRDF**（完整 PBR）
   - 在 **Log** 标签实时观察训练进度

## 🐛 常见问题

### Q: "python 不是内部或外部命令"

**A:** Python 未添加到 PATH，尝试：
1. 重启终端/命令提示符
2. 使用完整路径，例如：
   ```
   C:\Users\你的用户名\AppData\Local\Programs\Python\Python310\python.exe
   ```
3. 或重新安装 Python，确保勾选 "Add to PATH"

### Q: GUI 中点击 Render 报错 "系统找不到指定的路径"

**A:** Python 路径配置错误，请：
1. 在 GUI 的 **Render** 标签中
2. 将 **Python** 字段改为虚拟环境的完整路径：
   ```
   h:\Project\mitsuba_test\.venv\Scripts\python.exe
   ```
   或者系统 Python：
   ```
   python
   ```

### Q: 安装 Mitsuba 失败

**A:** 可能是网络问题，尝试：
```bash
pip install mitsuba -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: GPU 支持

**A:** 如果你有 NVIDIA GPU：
1. 确认 CUDA 已安装：`nvidia-smi`
2. 安装 GPU 版本的 PyTorch：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   （根据你的 CUDA 版本选择 cu118/cu121 等）
3. 安装 nvdiffrast：
   ```bash
   pip install git+https://github.com/NVlabs/nvdiffrast.git
   ```

## 📁 项目结构

```
mitsuba_test/
├── .venv/                      # Python 虚拟环境
├── src/main.rs                 # Rust GUI 源码
├── tools/                      # Python 工具脚本
│   ├── mitsuba_render.py
│   ├── mitsuba_raster_fit_nvdiffrast.py
│   └── mitsuba_raster_fit_disney.py     # Disney BRDF 拟合
├── scenes/                     # 场景文件
│   └── cbox.xml
├── renders/                    # 渲染输出（自动创建）
├── quick_setup.bat             # 一键安装脚本
└── quickstart_render.py        # 快速测试脚本
```

## 📚 下一步

- 阅读完整文档：[SETUP_GUIDE.md](SETUP_GUIDE.md)
- Disney BRDF 使用：[DISNEY_BRDF_README.md](DISNEY_BRDF_README.md)
- 技术细节：[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 💬 获得帮助

如果遇到问题：
1. 检查 **Log** 标签的错误信息
2. 查看 [SETUP_GUIDE.md](SETUP_GUIDE.md) 的常见问题部分
3. 提交 GitHub Issue

---

**祝你使用愉快！🎨✨**
