# Disney BRDF 材质拟合系统

## 🎨 概述

这是一个完整的 **Disney principled BRDF** 材质拟合系统，用于将光栅化渲染拟合到 Mitsuba 3 路径追踪的 Ground Truth。

## ✨ 主要特性

### 1. **Disney BRDF 实现**
基于 Burley 2012 "Physically-Based Shading at Disney" 论文，实现了完整的 PBR 材质模型：

- **Base Color** - 基础颜色（albedo）
- **Roughness** - 表面粗糙度（0 = 镜面，1 = 完全粗糙）
- **Metallic** - 金属度（0 = 电介质，1 = 金属）
- **Specular** - 镜面反射强度

### 2. **BRDF 组件**
- **Disney Diffuse** - 带粗糙度的漫反射（非简单 Lambert）
- **GGX 法线分布函数** (Trowbridge-Reitz)
- **Smith GGX 几何遮蔽函数**
- **Fresnel-Schlick 菲涅尔项**

### 3. **多光源支持**
- 自动解析 XML 场景中的所有面积光
- 每个光源独立计算贡献
- 支持环境光（简化实现）

### 4. **实时训练监控**
- GUI 实时显示所有 4 个材质参数
- 进度条可视化 roughness、metallic、specular
- 颜色预览显示 base color

## 📂 文件结构

```
tools/
├── mitsuba_raster_fit_nvdiffrast.py  # 原始简单漫反射拟合
└── mitsuba_raster_fit_disney.py      # 新增 Disney BRDF 拟合 ⭐

src/
└── main.rs                            # 已扩展支持多参数显示
```

## 🚀 使用方法

### 方法 1：通过 GUI（推荐）

1. **启动 GUI**
   ```bash
   cargo run --release
   ```

2. **配置场景**
   - 左侧 **Render** 标签
   - 展开 "Fit material (nvdiffrast)"
   - 设置 Steps（推荐 400）、LR（推荐 0.01）

3. **选择拟合模式**
   - **"Fit diffuse albedo"** - 简单漫反射（快速测试）
   - **"Fit Disney BRDF"** - 完整 PBR 材质 ⭐

4. **查看实时进度**
   - 切换到 **Log** 标签
   - 观察训练进度面板：
     - Base Color + 颜色预览
     - Roughness 进度条
     - Metallic 进度条
     - Specular 进度条
     - Loss 值变化

### 方法 2：命令行

```bash
python tools/mitsuba_raster_fit_disney.py \
  --scene scenes/cbox.xml \
  --gt-variant scalar_rgb \
  --gt-spp 256 \
  --steps 400 \
  --lr 0.01 \
  --out-dir renders/fit_disney \
  --init-base-color "0.8,0.8,0.8" \
  --init-roughness 0.5 \
  --init-metallic 0.0 \
  --init-specular 0.5
```

## 📊 输出结果

训练完成后，在 `renders/fit_disney/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `gt.png` | Ground Truth（Mitsuba 路径追踪） |
| `pred.png` | 拟合结果（nvdiffrast 光栅化） |
| `diff.png` | 差异图（放大 4 倍便于查看） |
| `fit_params.json` | 拟合的材质参数 |

### 示例 `fit_params.json`
```json
{
  "base_color": [0.245, 0.352, 0.798],
  "roughness": 0.423,
  "metallic": 0.012,
  "specular": 0.567,
  "steps": 400,
  "lr": 0.01,
  "final_loss": 0.015234
}
```

## 🎓 技术细节

### Disney BRDF 公式

```python
# 漫反射（Disney diffuse with retro-reflection）
fd90 = 0.5 + 2.0 * (l·h)² * roughness
Fd = base_color * lerp(1, fd90, (1-n·l)⁵) * lerp(1, fd90, (1-n·v)⁵) / π

# 镜面反射（Cook-Torrance）
D = GGX(n·h, roughness)           # 法线分布
G = Smith-GGX(n·l, n·v, roughness) # 几何遮蔽
F = Fresnel-Schlick(l·h, F0)       # 菲涅尔

Specular = D * G * F / (4 * n·l * n·v)

# 最终 BRDF
kd = (1 - F) * (1 - metallic)
BRDF = kd * Fd + Specular
```

### 参数化策略

所有参数使用 **logit 参数化** 确保值在 [0, 1] 范围内：

```python
# 训练时
param_logit = torch.tensor([...], requires_grad=True)
param = torch.sigmoid(param_logit)  # 映射到 [0, 1]

# 初始化
init_value = 0.5
param_logit = log(init_value / (1 - init_value))
```

### 照明模型

1. **直接光照**
   - 从 XML 解析所有面积光
   - 简化为点光源（位置 = 面积光中心）
   - 距离平方衰减

2. **环境光（简化）**
   ```python
   ambient = base_color * [0.05, 0.05, 0.05] * (1 - metallic)
   ```

3. **未来扩展**
   - [ ] 基于图像的照明（IBL）
   - [ ] 球谐函数环境光
   - [ ] 阴影映射

## 📈 训练建议

### 学习率调整

| 场景类型 | 推荐 LR | 说明 |
|---------|---------|------|
| 简单几何（球体、立方体） | 0.01 - 0.02 | 较快收敛 |
| 复杂网格 | 0.005 - 0.01 | 需要更稳定的梯度 |
| 高对比度材质 | 0.005 | 避免震荡 |

### 迭代步数

- **快速预览**：100 steps
- **标准拟合**：400 steps
- **高精度**：1000 steps

### 初始值设置

```bash
# 金属材质
--init-metallic 0.8 --init-roughness 0.3

# 粗糙塑料
--init-metallic 0.0 --init-roughness 0.7

# 光滑玻璃
--init-metallic 0.0 --init-roughness 0.05 --init-specular 0.9
```

## 🔍 与原始实现对比

| 特性 | Simple Diffuse | Disney BRDF |
|------|----------------|-------------|
| 可优化参数 | 1 个（albedo RGB） | 4 个（base color, roughness, metallic, specular） |
| BRDF 模型 | Lambert 漫反射 + 环境光 | Disney principled BRDF |
| 镜面反射 | ❌ 无 | ✅ 完整实现 |
| 金属材质 | ❌ 不支持 | ✅ 支持 |
| 粗糙度 | ❌ 固定 | ✅ 可优化 |
| 多光源 | ⚠️ 单光源 | ✅ 多光源 |
| 训练速度 | 快（~2-3s/step） | 中等（~4-5s/step） |
| 拟合精度 | 仅适合纯漫反射 | 适合真实 PBR 材质 |

## 🧪 测试场景

### 1. 简单材质测试
```bash
# Cornell box 蓝色球体
cargo run --release
# 点击 "Fit Disney BRDF"
# 预期：base_color ≈ [0.25, 0.35, 0.8]
```

### 2. 金属材质测试
手动编辑 `scenes/cbox.xml`，将球体 BSDF 改为：
```xml
<bsdf type="conductor">
    <rgb name="eta" value="0.2, 0.9, 1.0"/>
    <float name="k" value="3.0"/>
</bsdf>
```
预期拟合结果：metallic ≈ 0.9+

### 3. 粗糙塑料测试
```xml
<bsdf type="roughplastic">
    <rgb name="diffuse_reflectance" value="0.8, 0.1, 0.1"/>
    <float name="alpha" value="0.2"/>
</bsdf>
```
预期：roughness ≈ 0.45 (alpha = √roughness)

## ⚠️ 已知限制

1. **间接照明**
   - 当前不支持全局光照（GI）
   - 适用于简单直接光照场景
   - 解决方案：使用高 SPP 渲染 GT + 环境光近似

2. **阴影**
   - 光栅化不计算阴影
   - 导致被遮挡区域的拟合不准确
   - 解决方案：添加 shadow mapping

3. **次表面散射（SSS）**
   - 不支持半透明材质
   - 解决方案：扩展 BRDF 为 BSSRDF

4. **各向异性**
   - 当前实现为各向同性
   - 解决方案：添加 anisotropic GGX

## 🔧 调试技巧

### Loss 不下降
1. 降低学习率（0.005）
2. 检查 GT 图像是否过于复杂
3. 增加迭代步数

### 颜色不匹配
1. 检查光源强度是否正确解析
2. 确认 sRGB/Linear 颜色空间一致
3. 调整环境光强度（代码中 ambient_color）

### 金属度异常
1. 确保 GT 使用了金属材质
2. 检查 Fresnel 计算是否正确
3. 尝试不同的初始值

## 📚 参考文献

1. **Disney BRDF**
   - Burley, B. (2012). "Physically-Based Shading at Disney"
   - [Slides](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)

2. **GGX/Trowbridge-Reitz**
   - Walter et al. (2007). "Microfacet Models for Refraction"

3. **Smith GGX**
   - Heitz (2014). "Understanding the Masking-Shadowing Function"

## 🎯 未来改进方向

- [ ] 添加 **IBL（基于图像的照明）**
- [ ] 实现 **阴影映射**
- [ ] 支持 **各向异性 BRDF**
- [ ] 添加 **clearcoat** 涂层
- [ ] 实现 **sheen** 和 **subsurface** 参数
- [ ] 多尺度 loss（Laplacian pyramid）
- [ ] 感知损失（LPIPS）
- [ ] 训练曲线可视化（Loss curve）

## 💡 示例工作流

```bash
# 1. 启动 GUI
cargo run --release

# 2. 配置场景
# - 左侧 Scene 面板设置物体和材质
# - Render 面板设置 SPP=256, Steps=400

# 3. 开始训练
# 点击 "Fit Disney BRDF"

# 4. 观察进度
# Log 标签实时显示：
# step=0000 loss=2.345 baseColor=[0.8 0.8 0.8] roughness=0.5 ...
# step=0025 loss=0.987 baseColor=[0.45 0.52 0.79] roughness=0.43 ...
# ...
# step=0400 loss=0.012 baseColor=[0.245 0.352 0.798] roughness=0.423 ...

# 5. 查看结果
# 在 renders/fit_disney/ 目录对比 gt.png 和 pred.png
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个系统！

特别欢迎的改进：
- 更高级的 BRDF 模型
- 更好的照明近似
- 性能优化
- 新的测试场景

---

**Happy Material Fitting! 🎨✨**
