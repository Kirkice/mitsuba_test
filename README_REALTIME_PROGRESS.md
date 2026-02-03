# 实时训练进度显示功能

## 功能概述

现在 Mitsuba Studio 可以在 GUI 面板上实时显示 nvdiffrast 材质拟合的训练进度。

## 新增功能

### 1. 实时日志流式输出
- 不再等待 Python 进程结束，而是**实时捕获**标准输出
- 使用多线程读取 stdout/stderr，立即更新到 GUI

### 2. 训练进度解析
自动解析 nvdiffrast 训练日志，提取关键信息：
- `step` - 当前迭代步数
- `loss` - 当前损失值
- `albedo` - 当前拟合的反照率 RGB 值

### 3. GUI 可视化
在 **Log 标签页**新增训练进度面板：
- ✅ **进度条** - 显示训练进度百分比 (step/total_steps)
- 📊 **Loss 值** - 实时显示损失值（精度到小数点后 6 位）
- 🎨 **Albedo 颜色** - RGB 值 + 颜色预览块
- 🔴 **实时指示器** - 显示 "● Live" 绿色指示器

### 4. 自动滚动日志
- 日志窗口自动滚动到最新输出
- 支持复制完整日志内容

## 使用方法

### 步骤 1：启动 GUI
```bash
cargo run --release
```

### 步骤 2：配置场景
在左侧面板的 **Render** 标签中：
- 设置 Python 路径（如 `.venv/bin/python` 或 `.venv/Scripts/python.exe`）
- 配置场景路径（如 `scenes/cbox.xml`）
- 设置 Fit 参数（Steps、LR 等）

### 步骤 3：启动训练
点击顶部工具栏的 **Fit diffuse albedo** 按钮（在 Render 标签下方的 "Fit material (nvdiffrast)" 折叠面板中）

### 步骤 4：查看实时进度
切换到中央面板的 **Log** 标签：
- 顶部显示训练进度面板
- 底部显示实时日志输出

## 技术实现细节

### 修改的文件
- `src/main.rs` - 主要实现文件

### 关键修改点

#### 1. 新增数据结构
```rust
struct TrainingProgress {
    step: u32,
    total_steps: u32,
    loss: f32,
    albedo: [f32; 3],
}
```

#### 2. 流式输出捕获
```rust
enum RenderJobState {
    Running {
        started_at: Instant,
        rx: mpsc::Receiver<RenderJobResult>,
        live_log: Arc<Mutex<Vec<String>>>,  // 新增：实时日志缓冲
    },
}
```

#### 3. 多线程实时读取
```rust
// 在后台线程中逐行读取 stdout
let reader = BufReader::new(stdout);
for line in reader.lines() {
    if let Ok(line) = line {
        // 立即追加到共享日志缓冲
        if let Ok(mut log) = live_log_clone.lock() {
            log.push(line.clone());
        }
    }
}
```

#### 4. 日志解析器
```rust
fn parse_training_progress(line: &str, total_steps: u32) -> Option<TrainingProgress> {
    // 解析格式：step=0025 loss=0.123456 albedo=[0.1 0.2 0.3]
    // ...
}
```

## 示例输出

训练运行时，Log 面板会显示：

```
┌─────────────────────────────────────┐
│    Training Progress                │
├─────────────────────────────────────┤
│ [████████████░░░░░░░░] Step 200/400 │
│                                     │
│ Loss:    0.015234                   │
│ Albedo:  [0.245, 0.352, 0.798] ■    │
└─────────────────────────────────────┘

Running for 45.3s...  ● Live

step=0000 loss=2.345678 albedo=[0.85 0.2 0.2]
step=0025 loss=0.456789 albedo=[0.45 0.28 0.5]
step=0050 loss=0.234567 albedo=[0.32 0.31 0.68]
...
step=0200 loss=0.015234 albedo=[0.245 0.352 0.798]
```

## 与原有功能的兼容性

✅ 完全向后兼容
- 普通渲染任务（不显示训练进度，只显示日志）
- 可微渲染测试（显示错误/成功信息）
- 所有现有功能保持不变

## 性能优化

- 使用 `Arc<Mutex<Vec<String>>>` 实现线程安全的日志共享
- 仅在 Running 状态时请求 UI 重绘 (`ctx.request_repaint()`)
- 日志解析仅检查最近 10 行（避免大量历史日志的重复解析）

## 已知限制

1. **日志格式依赖** - 需要 Python 脚本输出特定格式的日志
   - 当前支持：`step=XXXX loss=X.XXXXXX albedo=[X.X X.X X.X]`
   - 如果格式不匹配，不会显示进度面板（但日志仍正常显示）

2. **仅支持 nvdiffrast 拟合** - 其他 Python 脚本不会解析训练进度
   - `mitsuba_render.py` - 不显示训练进度
   - `mitsuba_diff_smoketest.py` - 不显示训练进度

## 未来改进方向

- [ ] 支持更多训练任务的进度解析（如 `mitsuba_diff_optimize_reflectance.py`）
- [ ] 添加训练 Loss 曲线图表
- [ ] 保存训练历史到文件
- [ ] 支持暂停/恢复训练
- [ ] 添加最佳参数自动保存

## 测试建议

1. 使用低 steps 数测试（如 50 steps）快速验证功能
2. 观察进度条是否平滑更新
3. 检查 Loss 是否逐步下降
4. 验证 Albedo 颜色预览是否准确

## 示例命令（不使用 GUI）

如果想在命令行测试 Python 脚本：
```bash
python tools/mitsuba_raster_fit_nvdiffrast.py \
  --scene scenes/cbox.xml \
  --gt-variant scalar_rgb \
  --gt-spp 256 \
  --steps 100 \
  --lr 0.05 \
  --out-dir renders/fit
```

日志格式示例：
```
step=0000 loss=2.345678 albedo=[0.85000001 0.2        0.2       ]
step=0025 loss=0.456789 albedo=[0.4512345  0.2801234  0.5123456 ]
...
```
