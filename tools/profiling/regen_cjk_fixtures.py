"""Regenerate the four CJK squeeze fixtures with intact PNGs.

`tests/data/bench_cjk_ar20.png` and `bench_cjk_ar33.png` are committed but truncated --
both fail to decode ("broken data stream"), and their byte sizes match the intact
`bench_cjk_ref.png` / `bench_cjk_wide.png` exactly, so they are partial copies rather
than images at a different aspect ratio. `gen_cjk_stress.py` only emits `wide` and `ref`,
so the `ar20`/`ar33` pair predates it and has no live generator.

They exist to stress `rec_max_w`: the 37-char ground truth is drawn at a font size that
lands the natural rec width at roughly 20 and 33 characters' worth, so rec has to squeeze
and 33 is the worst case. The target here is the natural rec width, not the pixel width:

    natural_rec_w = 48 * width / height

so for ~20 chars (~960 px) and ~33 chars (~1600 px) the width is width = target * height/48.

Usage: python tools/profiling/regen_cjk_fixtures.py
"""

import io
import os
import re

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "tests", "data")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_cjk_stress.py")

# name, target natural rec width (or None to use the size the original generator used)
#
# The canvas is always `text_w + 40`, never derived from the target. Sizing the canvas from
# the target instead leaves the text short of the right edge: a first attempt did that and
# produced a 1866px canvas holding 1221px of text, i.e. a third of the image was blank
# margin. rec then decoded *nothing at all* for that fixture, which looks exactly like an
# engine failure at high squeeze but is really a badly formed image. `ar20` and `ar33` are
# the squeezed cases, so their font size is solved for the target and the canvas follows
# from the text.
CASES = [
    ("bench_cjk_ar20.png", 960),
    ("bench_cjk_ar33.png", 1600),
    ("bench_cjk_ref.png", None),  # 26px, as gen_cjk_stress.py
    ("bench_cjk_wide.png", None),  # 44px, as gen_cjk_stress.py
]
ORIGINAL_SIZE_PX = {"bench_cjk_ref.png": 26, "bench_cjk_wide.png": 44}


def font_size_for(rec_width: int, truth: str, font_path: str) -> int:
    """Smallest integer font size whose text is at least `rec_width * 56/48 - 40` px wide.

    Binary search on the rendered text width, so the canvas that follows (`text_w + 40`)
    lands the natural rec width on the target instead of leaving the text short of it.
    """
    want = rec_width * 56 / 48 - 40
    lo, hi = 10, 90
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    while lo < hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        b = probe.textbbox((0, 0), truth, font=f)
        if b[2] - b[0] < want:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main() -> None:
    d = open(SRC, "rb").read().decode("utf-8")
    truth = re.search(r'TRUTH\s*=\s*"(.*?)"', d, re.S).group(1)
    assert len(truth) == 37, f"unexpected ground truth length {len(truth)}"
    os.makedirs(OUT_DIR, exist_ok=True)

    font_path = "C:/Windows/Fonts/msyh.ttc"
    for name, target_w in CASES:
        size = (
            ORIGINAL_SIZE_PX[name] if target_w is None else font_size_for(target_w, truth, font_path)
        )
        font = ImageFont.truetype(font_path, size)
        probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        bbox = probe.textbbox((0, 0), truth, font=font)
        text_h = bbox[3] - bbox[1]
        text_w = bbox[2] - bbox[0]
        height = 56
        # Always text-driven: the 40px of margin is the same fraction for every case.
        width = text_w + 40
        natural = 48 * width / height
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, (height - text_h) // 2 - bbox[1]), truth, font=font, fill="black")
        path = os.path.join(OUT_DIR, name)
        # Encode to memory first, then write the bytes and re-verify what landed. Two silent
        # failures motivated this: `Image.save(path)` over an existing file left the old
        # bytes in place on this host, and an `OUT_DIR` with one `..` too few wrote the
        # fixtures to a fresh `tools/tests/data/` while the script still reported success.
        # Reading the size back from the file we just opened -- and asserting the directory
        # exists before writing -- is what makes either mistake visible.
        assert os.path.isdir(OUT_DIR), f"fixture directory does not exist: {OUT_DIR}"
        buf = io.BytesIO()
        img.save(buf, "PNG")
        data = buf.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        on_disk = os.path.getsize(path)
        if on_disk != len(data):
            raise RuntimeError(
                f"{name}: wrote {len(data)} bytes but the file is {on_disk} on disk"
            )
        # Verify the round trip: re-open and confirm it decodes at the intended size.
        with Image.open(path) as check:
            check.load()
            if check.size != (width, height):
                raise RuntimeError(
                    f"{name}: wrote {check.size}, expected {(width, height)}"
                )
        print(
            f"{name:24s} {width}x{height}  {size}px  text_w={text_w:5d}  "
            f"natural_rec_w={natural:6.0f}  squeeze={natural / 960:.2f}x  "
            f"margin={40 * 100 // width}%  ({on_disk} B, re-read OK)"
        )

    print(f"\nground truth ({len(truth)} chars): {truth}")


if __name__ == "__main__":
    main()
