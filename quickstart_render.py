"""Mitsuba 3 快速上手

这个脚本的目标是把“最小可运行链路”跑通：
1) 选择 Mitsuba variant（决定后端/是否可微）
2) 从 XML 加载场景
3) 渲染得到线性空间的 HDR 图像
4) 做一个简单的显示用 tonemap（γ≈2.2）并保存到 PNG

注意：
- 这里使用的 variant 是 `scalar_rgb`：纯 CPU 标量路径追踪（不可微）。
- 若要学习可微渲染（对场景参数求梯度并优化），通常要选 `*_ad_*` 变体，例如
  `scalar_ad_rgb`（最容易跑通，但慢）、`llvm_ad_rgb` / `cuda_ad_rgb`（更快）。
  我在同目录额外提供了一个可微渲染示例脚本，专门演示“反向传播 + 梯度下降”。
"""

import os

# 可选：降低 Dr.Jit/Mitsuba 的日志噪声（不影响结果）
os.environ.setdefault("DRJIT_LOG_LEVEL", "warn")

import mitsuba as mi


def main() -> None:
    # 1) 选择 variant
    # `scalar_rgb` 对应文章中的 `mi.set_variant("scalar_rgb")`
    mi.set_variant("scalar_rgb")

    # 2) 加载 XML 场景
    # 路径以脚本运行目录为基准（当前脚本在项目根目录，所以用 scenes/xxx.xml）
    scene = mi.load_file("scenes/cbox.xml")

    # 3) 渲染：返回的是线性空间的 HDR 图像（float RGB）
    # spp = samples per pixel，越大噪声越少、越慢。
    image = mi.render(scene, spp=64)

    # 4) 保存：为了“看起来像普通图片”，做一个近似 sRGB 的 gamma 校正。
    # 严格的 sRGB 不是简单幂函数，但对快速预览足够。
    ldr_preview = image ** (1.0 / 2.2)
    mi.util.write_bitmap("cbox.png", ldr_preview)
    print("Wrote cbox.png")

    # 可选：用 matplotlib 弹窗预览（没有也不影响保存）
    try:
        import matplotlib.pyplot as plt

        plt.axis("off")
        plt.imshow(ldr_preview)
        plt.show()
    except Exception as exc:
        print(f"Matplotlib preview skipped: {exc}")


if __name__ == "__main__":
    main()
