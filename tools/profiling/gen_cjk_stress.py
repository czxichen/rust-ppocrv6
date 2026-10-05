"""生成中文高宽比压力测试图，用于验证 rec_max_w 压缩导致的行尾丢字。

默认 max_w=960 时，高宽比 > 20 的中文行会被横向压缩，这里生成两张：
  bench_cjk_ar33.png  宽高比 ~33，自然 rec 宽度 ~1600 -> 被 960 压缩 1.67x
  bench_cjk_ar20.png  宽高比 ~20，自然 rec 宽度 ~960  -> 基本不压缩（对照组）

用法: python examples/gen_cjk_stress.py
"""

import os

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data"
)

TRUTH = "文本检测与识别端到端耗时统计包含预处理前向推理与后处理阶段以及内存带宽限制"

CASES = [
    ("bench_cjk_wide.png", 56, 44),
    ("bench_cjk_ref.png", 56, 26),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    for name, height, size in CASES:
        font = ImageFont.truetype(font_path, size)
        probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        bbox = probe.textbbox((0, 0), TRUTH, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        width = text_w + 40
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, (height - text_h) // 2 - bbox[1]), TRUTH, font=font, fill="black")
        path = os.path.join(OUT_DIR, name)
        img.save(path)
        natural = 48 * width / height
        print(
            f"{name:24s} {width}x{height}  aspect={width / height:5.1f}  "
            f"natural_rec_w={natural:6.0f}  max_w=960 压缩 {natural / 960:.2f}x"
        )
    print(f"\n真值 ({len(TRUTH)} 字): {TRUTH}")


if __name__ == "__main__":
    main()
