# Mitsuba Studio

Mitsuba Studio 是一个面向 **Mitsuba 3** 的“场景编辑 + 渲染 + 可微/拟合入口”的轻量前端。

目标：
- 用 GUI 更快地搭建/修改 Mitsuba XML 场景
- 一键调用 Python 工具脚本渲染预览、运行可微渲染 smoke test
- 为后续「路径追踪 Ground Truth vs 可微光栅化（nvdiffrast）」拟合工作流提供入口

> 备注：本项目以工程落地为导向，先把“最小可用链路”跑通，再逐步扩展材质模型、灯光模型与拟合能力。

---

## 功能概览

### 1) GUI：场景参数编辑（Rust + egui/eframe）
- 相机：FOV、lookat（origin/target/up）
- Film：分辨率
- 采样器：sample_count
- 灯光：顶灯 area emitter（强度/位置/尺寸）
- 物体：sphere/cube/obj/ply（平移/尺度/半径/网格路径）
- 材质：可编辑 BSDF 节点（含常见插件与参数）

GUI 会生成/预览一份 Mitsuba XML（默认写到 `scenes/cbox.xml`）。

### 2) 一键渲染（Python）
GUI 的 Render 页签会启动 Python 子进程来执行工具脚本：
- `tools/mitsuba_render.py`：渲染预览输出（默认 `renders/preview.png`）
- `tools/mitsuba_diff_smoketest.py`：可微渲染 smoke test（默认示例用 `llvm_ad_rgb`）

### 3) 拟合入口（nvdiffrast，建议在 Windows + NVIDIA CUDA 运行）
- `tools/mitsuba_raster_fit_nvdiffrast.py`
  - 用 Mitsuba 路径追踪渲 Ground Truth
  - 用 nvdiffrast 光栅化一个近似模型并对材质参数做梯度下降拟合
  - 当前版本先从“diffuse albedo（RGB）”跑通闭环，输出 `fit_params.json` 以及对比图

---

## 目录结构

- `src/`：Rust GUI 应用（Mitsuba Studio）
- `scenes/`：场景 XML（默认 `scenes/cbox.xml`）
- `tools/`：Python 工具脚本
- `renders/`：渲染输出（默认 `renders/preview.png`）

---

## 快速开始（macOS / 本机 GUI）

### 1) 运行 GUI
```bash
cargo run
```

### 2) 安装 Mitsuba 3（推荐：Python 3.12 + venv）
本项目建议使用项目根目录的 `.venv`：
```bash
/opt/homebrew/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install mitsuba matplotlib
```

### 3) 跑通最小渲染脚本（可选）
```bash
./.venv/bin/python quickstart_render.py
```
会输出 `cbox.png`。

---

## GUI 使用说明（推荐流程）

Mitsuba Studio 的布局是：
- 顶部工具栏：快速渲染/差分渲染按钮 + `Variant` / `SPP` 快速调整 + 运行状态
- 左侧：`Scene`（场景参数）/ `Render`（渲染与拟合配置）页签
- 中央：`Preview` / `Log` / `XML` 页签

### A. Scene（场景参数）
- **Camera**：调整 `FOV / origin / target / up`。相机与物体、灯光强耦合，建议先把构图调对。
- **Film**：调整输出分辨率。预览阶段建议先用较小分辨率（例如 512×512）。
- **Sampler**：`sample_count` 主要影响路径追踪渲染的噪声与耗时。
- **Light**：开关顶灯、调 `radiance`、灯的尺寸与高度。
- **Object**：选择物体类型（sphere/cube/obj/ply），并调平移/尺度。网格路径支持相对路径（一般相对 `scenes/` 目录）。
- **Material (BSDF)**：编辑材质节点与参数。当前 UI 支持常见 BSDF 插件的快捷选择，也可直接输入自定义插件名。

修改 Scene 参数后，GUI 会自动更新 `XML` 页签中的场景 XML 预览。

### B. Render（渲染与拟合配置）
- **Python**：Python 解释器路径（默认 `.venv/bin/python`）。
- **Scene**：要写入并渲染的 XML 路径（默认 `scenes/cbox.xml`）。
- **Variant / SPP / Output**：对应 Mitsuba 渲染参数与输出路径（默认 `renders/preview.png`）。
- **Write XML to scene path**：将当前 GUI 的场景配置写入 `Scene` 指定的 XML 文件。

#### Fit material (nvdiffrast)
- 用于“路径追踪 Ground Truth vs 可微光栅化”拟合工作流。
- **注意**：nvdiffrast 依赖 CUDA，建议在 Windows + NVIDIA 机器运行。
- 当前版本提供 `Fit diffuse albedo`：会先写出 XML，再启动 `tools/mitsuba_raster_fit_nvdiffrast.py`，输出到 `Out dir`。

### C. Preview（预览图）
- `Path`：预览图片路径（默认 `renders/preview.png`）。
- `Use output`：一键把预览路径设为 Render 的 Output。
- `Reload`：手动重新加载预览图。
- `Auto` + `Every (s)`：自动轮询刷新预览图。
- `Zoom`：预览缩放。

### D. Log（日志）
- 展示 Python 子进程输出（stdout/stderr）。
- `Copy`：复制完整日志，便于你粘贴到 issue 或发给我定位问题。

### E. XML（场景 XML）
- 展示当前生成的 XML。
- `Copy`：复制 XML。
- 渲染前实际写入文件的是 Render 页签的 `Scene` 路径（通过 “Write XML…” 或渲染按钮触发）。

---

## nvdiffrast 拟合（Windows + CUDA 机器）

### 环境建议
- Windows 10/11
- NVIDIA GPU + CUDA
- Python（建议 3.10/3.11/3.12 其一）

### 依赖（示例）
```bash
pip install mitsuba drjit torch trimesh imageio
# nvdiffrast 安装方式可能需要按官方说明（有时需要从源码编译）
```

### 运行示例
在仓库根目录：
```bash
python tools/mitsuba_raster_fit_nvdiffrast.py \
  --scene scenes/cbox.xml \
  --gt-variant scalar_rgb \
  --gt-spp 256 \
  --steps 400 \
  --lr 0.05 \
  --out-dir renders/fit
```
输出目录会包含：
- `gt.png`：路径追踪 Ground Truth（展示用）
- `pred.png`：光栅化拟合结果（展示用）
- `diff.png`：差异可视化（展示用）
- `fit_params.json`：拟合出的数值参数（当前是 albedo_rgb）

---

## 运行提示与常见问题

- Mitsuba 可微渲染需要选择 `*_ad_*` 变体（例如 `llvm_ad_rgb`）。
- nvdiffrast 依赖 CUDA；macOS 本机通常不适合跑这一段拟合脚本。
- GUI 状态会写入项目根目录：`.mitsuba_studio_state.json`（用于记住上次的参数与路径）。

---

## Roadmap（方向）
- 更完整的 PBR 拟合参数：roughness/metallic/IBL 等
- 更强的对齐与 loss：多尺度、mask、感知损失、色彩映射参数
- 更完善的跨平台任务分发（本机编辑，Win CUDA 机器训练/拟合）
