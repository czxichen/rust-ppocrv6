"""Generate representative benchmark images for rust_ppocrv6 speed testing.

Produces (under tests/data/):
  bench_single_line.png   400x100   单行短文本（对齐仓库自带测试图）
  bench_long_line.png     1440x72   单行超长文本（考验变长识别）
  bench_doc.png           1000x1400 多行文档（~30 行，考验检测 + 多线程识别）
  bench_hd.png            1920x1080 高分辨率混排（考验检测缩放链路）

Usage:
    python examples/gen_bench_images.py
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data")
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]

CN = "深度学习是机器学习的一个重要分支，它通过多层神经网络自动学习数据的层次化特征表示。"
EN = "PP-OCRv6 is a pure Rust inference engine with zero C++ runtime dependencies."
MIX = "端到端 OCR 管线：检测 DBNet + 识别 SVTR 精度 98.30%"


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_single_line() -> Image.Image:
    # NOTE: font size 40 with tight margins (400x100) is missed by the DBNet stage
    # (det returns 0 boxes); 36pt with a comfortable margin detects reliably.
    img = Image.new("RGB", (480, 128), "white")
    d = ImageDraw.Draw(img)
    d.text((24, 36), "深度学习 PP-OCR", font=pick_font(36), fill="black")
    return img


def make_long_line() -> Image.Image:
    img = Image.new("RGB", (1440, 72), "white")
    d = ImageDraw.Draw(img)
    text = (CN + EN)
    d.text((12, 12), text, font=pick_font(38), fill="black")
    return img


def make_doc() -> Image.Image:
    w, h = 1000, 1400
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = pick_font(24)
    title_font = pick_font(30)
    d.text((60, 30), "纯 Rust PP-OCRv6 推理引擎性能测试报告", font=title_font, fill="black")

    pool = [CN, EN, MIX, "文本检测与识别端到端耗时统计，包含预处理、前向推理与后处理阶段。",
            "AVX2 与 FMA 微内核在 Windows 平台可达到硬件理论峰值的 80% 以上。"]
    y = 90
    for i in range(28):
        line = "[%02d] %s" % (i + 1, pool[i % len(pool)])
        d.text((60, y), line, font=font, fill="black")
        y += 45
    return img


def make_hd() -> Image.Image:
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    random.seed(20260921)
    pool = [CN, EN, MIX, "检测模型 DBNet 输出概率图后进行连通域分析。"]
    for _ in range(18):
        f = pick_font(random.choice([26, 30, 34, 38]))
        d.text((random.randint(40, 1200), random.randint(40, 1000)), random.choice(pool), font=f, fill="black")
    return img


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = [
        ("bench_single_line.png", make_single_line),
        ("bench_long_line.png", make_long_line),
        ("bench_doc.png", make_doc),
        ("bench_hd.png", make_hd),
    ]
    for name, fn in cases:
        path = os.path.join(OUT_DIR, name)
        img = fn()
        img.save(path)
        print("wrote %s  (%dx%d)" % (path, img.width, img.height))


if __name__ == "__main__":
    main()
