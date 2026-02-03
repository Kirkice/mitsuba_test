# 跨平台支持说明

## 🌍 支持的平台

Mitsuba Studio 现在完全支持以下平台：

- ✅ **Windows** (64-bit)
- ✅ **macOS** (Intel & Apple Silicon)
- ✅ **Linux** (x86_64)

## 🔧 Python 路径自动检测

### 智能路径管理

GUI 会自动检测并修正 Python 路径，支持以下场景：

1. **平台差异自动适配**
   - Windows: `.venv\Scripts\python.exe` 或 `.venv/Scripts/python.exe`
   - macOS: `.venv/bin/python3` 或 `.venv/bin/python`
   - Linux: `.venv/bin/python` 或 `.venv/bin/python3`

2. **路径验证与修正**
   - 启动时自动验证配置的 Python 路径
   - 如果路径不存在，自动尝试其他常见位置
   - 找不到虚拟环境时，回退到平台默认值

3. **支持多种配置**
   ```
   # 系统 Python（跨平台）
   python
   python3
   py

   # 虚拟环境（自动检测）
   .venv/Scripts/python.exe  (Windows)
   .venv/bin/python3         (macOS)
   .venv/bin/python          (Linux)

   # 完整路径
   C:\Python310\python.exe   (Windows)
   /usr/local/bin/python3    (Unix)
   ```

## 📂 虚拟环境检测顺序

当配置的路径无效时，按以下顺序自动搜索：

### Windows
1. `.venv/Scripts/python.exe`
2. `.venv\Scripts\python.exe`
3. `venv/Scripts/python.exe`
4. `venv\Scripts\python.exe`

### macOS
1. `.venv/bin/python`
2. `.venv/bin/python3`
3. `venv/bin/python`
4. `venv/bin/python3`

### Linux
1. `.venv/bin/python`
2. `.venv/bin/python3`
3. `venv/bin/python`
4. `venv/bin/python3`

## 🔄 自动修正机制

### 场景 1：配置文件路径错误

**问题：** 旧配置文件包含错误的平台路径
```json
{
  "python_exe": ".venv/bin/python"  // 在 Windows 上无效
}
```

**解决：** 启动时自动修正
- GUI 启动时调用 `normalize_python_path()`
- 检测文件是否存在
- 自动切换到正确的平台路径
- 下次保存时更新配置文件

### 场景 2：跨平台配置迁移

**问题：** 从 macOS 复制项目到 Windows

**解决：** 无需手动修改
1. 打开 GUI
2. 自动检测到路径不存在
3. 搜索 Windows 虚拟环境路径
4. 自动切换到正确路径

### 场景 3：运行时路径验证

**问题：** 点击 Render 时路径失效

**解决：** 启动任务前验证
- `start_python_job()` 调用前再次验证
- 确保使用最新的有效路径
- 防止运行时错误

## 🎯 用户体验

### 零配置启动（推荐）

```bash
# 1. 创建虚拟环境
python -m venv .venv                    # Windows
python3 -m venv .venv                   # macOS/Linux

# 2. 安装依赖
.venv\Scripts\activate                  # Windows
source .venv/bin/activate               # macOS/Linux
pip install mitsuba torch numpy imageio pillow

# 3. 启动 GUI（自动检测 Python）
cargo run --release
```

### 手动配置（可选）

如果自动检测失败，可以在 GUI 中手动设置：

**Windows:**
```
.venv\Scripts\python.exe
或
python
或
C:\Python310\python.exe
```

**macOS:**
```
.venv/bin/python3
或
python3
或
/usr/local/bin/python3
```

**Linux:**
```
.venv/bin/python
或
python3
或
/usr/bin/python3
```

## 🐛 故障排除

### 问题 1：GUI 显示 "系统找不到指定的路径"

**原因：** Python 未安装或虚拟环境未创建

**解决方案：**
```bash
# 验证 Python 是否安装
python --version        # Windows
python3 --version       # macOS/Linux

# 创建虚拟环境
python -m venv .venv

# 验证虚拟环境
ls .venv/Scripts/python.exe      # Windows (PowerShell)
ls .venv/bin/python3              # macOS/Linux
```

### 问题 2：路径配置不生效

**原因：** 配置文件权限问题或路径验证失败

**解决方案：**
1. 删除配置文件强制重新初始化：
   ```bash
   rm .mitsuba_studio_state.json
   ```
2. 重启 GUI
3. 在 Render 标签手动设置正确路径

### 问题 3：macOS 找不到 Python

**原因：** Python 3 命名为 `python3` 而非 `python`

**解决方案：**
- 使用 `python3` 命令创建虚拟环境
- GUI 会自动检测 `python3` 路径
- 或在 GUI 中设置为 `python3`

### 问题 4：Linux 权限错误

**原因：** Python 可执行文件无执行权限

**解决方案：**
```bash
chmod +x .venv/bin/python
chmod +x .venv/bin/python3
```

## 📋 检查清单

安装完成后，验证跨平台兼容性：

```bash
# 1. 验证 Python 可访问
python --version        # 或 python3 --version

# 2. 验证虚拟环境存在
# Windows
dir .venv\Scripts\python.exe

# macOS/Linux
ls -l .venv/bin/python*

# 3. 启动 GUI
cargo run --release

# 4. 检查 GUI 中的 Python 路径
# Render 标签 → Python 字段
# 应该显示正确的平台路径

# 5. 测试渲染
# 点击 Render 按钮
# 查看 Log 标签确认无错误
```

## 🔍 调试信息

如果遇到问题，检查以下信息：

### 1. 配置文件内容
```bash
cat .mitsuba_studio_state.json | grep python_exe
```

应该显示正确的平台路径。

### 2. Python 可执行文件
```bash
# Windows
.venv\Scripts\python.exe --version

# macOS/Linux
.venv/bin/python --version
```

### 3. GUI 日志
在 Log 标签中查看错误信息：
- "Failed to spawn process" → Python 路径错误
- "ModuleNotFoundError" → 依赖未安装
- "Permission denied" → 权限问题

## 🎓 开发者说明

### 实现细节

路径规范化逻辑位于 `src/main.rs`：

```rust
fn normalize_python_path(path: &str) -> String {
    // 1. 检查系统命令（python, python3, py）
    // 2. 验证提供的路径是否存在
    // 3. 搜索常见虚拟环境位置
    // 4. 返回平台默认值
}

fn get_default_python_path() -> String {
    // Windows: .venv/Scripts/python.exe
    // macOS:   .venv/bin/python3
    // Linux:   .venv/bin/python
}
```

### 调用时机

1. **GUI 启动时** (`new()`)
   - 加载配置后立即规范化
   - 确保启动时使用正确路径

2. **任务启动前** (`start_python_job()`)
   - 运行 Python 脚本前再次验证
   - 防止配置在运行时失效

3. **保存配置时**
   - 规范化后的路径写入配置文件
   - 保持配置文件的平台一致性

## 🌟 最佳实践

1. **使用虚拟环境**（推荐）
   - 隔离项目依赖
   - 避免版本冲突
   - 便于跨平台迁移

2. **避免硬编码路径**
   - 使用 `python` 或 `python3` 而非完整路径
   - 让 GUI 自动检测虚拟环境

3. **版本控制**
   - 不要提交 `.mitsuba_studio_state.json`
   - 已在 `.gitignore` 中排除

4. **CI/CD 兼容**
   - 自动检测逻辑支持无人值守运行
   - 适合自动化测试和部署

## 📚 相关文档

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - 详细安装指南
- [README_QUICKSTART.md](README_QUICKSTART.md) - 快速开始
- [DISNEY_BRDF_README.md](DISNEY_BRDF_README.md) - 功能说明

---

**现在你可以在任何平台上无缝使用 Mitsuba Studio！** 🎉
