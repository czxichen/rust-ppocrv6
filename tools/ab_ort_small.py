#!/usr/bin/env python3
"""Head-to-head: this Rust engine vs ONNX Runtime, small models only.

Both sides run the *same graph* on the *same input tensor* at the *same fixed
input shape*, with the *same thread count*, alternating between the two so a
drifting machine cannot bias one side. Reported figures are min and p50 over
repeats; min is the more reproducible of the two on a shared box.

  python tools/ab_ort_small.py --repeats 5 --iters 30
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build_onnx")
DUMP = os.path.join(ROOT, "target", "release", "examples", "dump_forward.exe" if sys.platform == "win32" else "dump_forward")

# small models only, as requested
MODELS = [
    ("det_small", "det_small.onnx", "PP-OCRv6_small_det.safetensors", (1, 3, 960, 704)),
    ("rec_small", "rec_small.onnx", "PP-OCRv6_small_rec.safetensors", (1, 3, 48, 960)),
    ("cls", "cls.onnx", "PP-OCRv6_cls.safetensors", (1, 3, 48, 192)),
]


def synth_input(numel):
    """Byte-identical to examples/dump_forward.rs::synth_input."""
    idx = np.arange(numel, dtype=np.uint64)
    h = (idx * np.uint64(2654435761)) >> np.uint64(8)
    return ((h % np.uint64(10007)).astype(np.float32) / np.float32(10007.0))


def make_session(onnx_path, threads):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])


def bench_ort(session, x, iters, warmup=3):
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


RUST_RE = re.compile(
    r"RUST ms: min=([\d.]+) p50=([\d.]+) mean=([\d.]+) max=([\d.]+)"
)


def bench_rust(st_path, shape, iters, threads):
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(threads)
    out_bin = os.path.join(BUILD, "rust_ab_tmp.bin")
    r = subprocess.run(
        [
            DUMP,
            st_path,
            out_bin,
            *[str(d) for d in shape],
            "--bench",
            str(iters),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        # The Rust binary prints UTF-8 Chinese; the Windows locale codec is GBK and would
        # raise UnicodeDecodeError on it. The timing line is pure ASCII regardless.
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"dump_forward failed: {r.stderr[-500:]}")
    m = RUST_RE.search(r.stdout)
    if not m:
        raise RuntimeError(f"could not parse Rust timing:\n{r.stdout}")
    # The Rust side reports summary stats; take min/p50 from it directly and
    # re-run per repeat so both sides get the same number of samples.
    return float(m.group(1)), float(m.group(2))


def p50(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--models", default="")
    args = ap.parse_args()

    wanted = set(args.models.split(",")) if args.models else None
    print(f"onnxruntime {ort.__version__}  |  threads={args.threads}  |  "
          f"iters={args.iters}  repeats={args.repeats}")
    print("=" * 92)

    out_rows = []
    for label, onnx_file, st_file, shape in MODELS:
        if wanted and label not in wanted:
            continue
        onnx_path = os.path.join(BUILD, onnx_file)
        st_path = os.path.join(ROOT, "models", st_file)
        if not os.path.exists(onnx_path) or not os.path.exists(st_path):
            print(f"{label:10s} skipped (missing asset)")
            continue

        x = synth_input(int(np.prod(shape))).reshape(shape)
        sess = make_session(onnx_path, args.threads)

        ort_mins, ort_p50s, rust_mins, rust_p50s = [], [], [], []
        for _ in range(args.repeats):
            ts = bench_ort(sess, x, args.iters)
            ort_mins.append(min(ts))
            ort_p50s.append(p50(ts))
            rm, rp = bench_rust(st_path, shape, args.iters, args.threads)
            rust_mins.append(rm)
            rust_p50s.append(rp)

        o_min, o_p50 = min(ort_mins), min(ort_p50s)
        r_min, r_p50 = min(rust_mins), min(rust_p50s)
        # Report the spread of the per-repeat minima as well as the best figure. On a
        # shared workstation the ratio moves by tens of percent between runs, so a single
        # ratio without its spread is not reproducible and should not be quoted as one.
        def spread(v):
            return f"[{min(v):.1f}..{max(v):.1f}]"

        print(
            f"{label:10s} {'x'.join(map(str, shape)):14s}\n"
            f"           ORT  min {o_min:7.2f} {spread(ort_mins):16s} p50 {o_p50:7.2f} ms\n"
            f"           Rust min {r_min:7.2f} {spread(rust_mins):16s} p50 {r_p50:7.2f} ms\n"
            f"           gap (best case) {r_min / o_min:5.2f}x   "
            f"(p50 {r_p50 / o_p50:5.2f}x, unreliable under load)"
        )
        out_rows.append(
            {
                "label": label,
                "shape": list(shape),
                "ort_min": o_min,
                "ort_p50": o_p50,
                "ort_min_spread": [min(ort_mins), max(ort_mins)],
                "rust_min": r_min,
                "rust_p50": r_p50,
                "rust_min_spread": [min(rust_mins), max(rust_mins)],
                "gap_min": r_min / o_min,
                "gap_p50": r_p50 / o_p50,
            }
        )

    print("=" * 92)
    dst = os.path.join(BUILD, "ab_ort_small.json")
    with open(dst, "w") as fh:
        json.dump(out_rows, fh, indent=2)
    print(f"saved: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
