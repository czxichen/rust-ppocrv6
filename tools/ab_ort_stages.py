#!/usr/bin/env python3
"""Stage-by-stage comparison against ONNX Runtime for the *small* models.

The per-op forward numbers in ab_ort_small.py answer "how fast is one forward".
This answers "where does the end-to-end gap come from", using the tensors the
engine actually feeds each network, dumped by `onnx_ab dump <image> <dir>`:

  det  :  ORT forward on det_in.bin   vs  Rust det forward on the same tensor
  rec  :  ORT forward on every rec_in_*.bin, one at a time (the engine's own
          strategy is per-box batch=1, parallel across boxes) vs the Rust engine
          recognising the same 29 crops

Only forward passes are compared. The engine's pre- and post-processing has no
ORT counterpart here, so it is reported separately as Rust-only cost rather than
folded into the ratio -- a full end-to-end number would need a Python pipeline
reimplementation, which this repo does not have.

  python tools/ab_ort_stages.py --dump build_onnx/dump_small --repeats 8
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
DUMP_EXE = os.path.join(ROOT, "target", "release", "examples", "dump_forward.exe")

SHAPES = {
    "det_small.onnx": "models/PP-OCRv6_small_det.safetensors",
    "rec_small.onnx": "models/PP-OCRv6_small_rec.safetensors",
}


def read_manifest(path):
    shapes, boxes, text = {}, None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("TENSOR "):
            parts = line.split()
            shapes[parts[1]] = tuple(int(d) for d in parts[2].split("=")[1].split(","))
        elif line.startswith("BOX "):
            boxes = boxes or 0
            boxes += 1
        elif line.startswith("TEXT "):
            text = line[5:]
    return shapes, boxes, text


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
    return min(ts)


RUST_RE = re.compile(r"RUST ms: min=([\d.]+)")


def bench_rust(st_path, shape, iters, threads, data=None):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    args = [DUMP_EXE, st_path, os.path.join(BUILD, "stage_tmp.bin")]
    args += [str(d) for d in shape] + ["--bench", str(iters)]
    r = subprocess.run(
        args, cwd=ROOT, env=env, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"dump_forward failed: {r.stderr[-400:]}")
    m = RUST_RE.search(r.stdout)
    if not m:
        raise RuntimeError(f"unparsed Rust timing: {r.stdout}")
    return float(m.group(1))


TIME_RE = re.compile(r"REC_PAR min=\s*([\d.]+)")


def bench_rust_page(threads, repeats):
    """Best-of-N `REC_PAR` from `onnx_ab time`: the engine recognising one real page.

    This is the honest way to get the Rust side of the rec stage -- the 29 crops have
    different widths, so scaling a single 1x3x48x960 forward is not equivalent.
    """
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    exe = os.path.join(ROOT, "target", "release", "examples", "onnx_ab.exe")
    best = None
    for _ in range(repeats):
        r = subprocess.run(
            [exe, "time", "tests/data/bench_doc.png", "5", str(threads)],
            cwd=ROOT, env=env, capture_output=True,
            encoding="utf-8", errors="replace",
        )
        m = TIME_RE.search(r.stdout)
        if m:
            v = float(m.group(1))
            best = v if best is None else min(best, v)
    if best is None:
        raise RuntimeError(
            f"could not parse REC_PAR from onnx_ab time:\n{r.stdout[-400:]}"
        )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="build_onnx/dump_small")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()

    d = args.dump if os.path.isabs(args.dump) else os.path.join(ROOT, args.dump)
    shapes, n_boxes, text = read_manifest(os.path.join(d, "manifest.txt"))
    print(f"onnxruntime {ort.__version__} | threads={args.threads} | "
          f"dump={args.dump} | boxes={n_boxes}")
    print("=" * 92)

    results = {}

    # --- detection -----------------------------------------------------------
    det_shape = shapes["det_in"]
    det_x = np.fromfile(os.path.join(d, "det_in.bin"), dtype="<f4").reshape(det_shape)
    det_sess = make_session(os.path.join(BUILD, "det_small.onnx"), args.threads)
    om, rm = [], []
    for _ in range(args.repeats):
        om.append(bench_ort(det_sess, det_x, args.iters))
        rm.append(bench_rust(
            os.path.join(ROOT, SHAPES["det_small.onnx"]), det_shape,
            args.iters, args.threads,
        ))
    o, r = min(om), min(rm)
    print(f"det  {'x'.join(map(str, det_shape))}")
    print(f"      ORT  min {o:8.2f} ms   [{min(om):.1f}..{max(om):.1f}]")
    print(f"      Rust min {r:8.2f} ms   [{min(rm):.1f}..{max(rm):.1f}]   gap {r / o:5.2f}x")
    results["det"] = {"shape": list(det_shape), "ort": o, "rust": r, "gap": r / o}

    # --- recognition, per crop, matching the engine's batch=1 strategy ---------
    # Every crop has its own width, so each one gets its own shape from the manifest.
    rec_entries = []
    for key, shp in sorted(shapes.items()):
        if key.startswith("rec_in_"):
            rec_entries.append((key, shp))
    rec_entries.sort()
    sess = make_session(os.path.join(BUILD, "rec_small.onnx"), args.threads)
    name = sess.get_inputs()[0].name
    inputs = [
        np.fromfile(os.path.join(d, key + ".bin"), dtype="<f4").reshape(shp)
        for key, shp in rec_entries
    ]
    widths = sorted({shp[3] for _, shp in rec_entries})
    print(f"      crop widths: {widths}")
    sess.run(None, {name: inputs[0]})  # warmup

    ort_seq = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for x in inputs:
            sess.run(None, {name: x})
        ort_seq.append((time.perf_counter() - t0) * 1000.0)

    # A thread-pool fan-out matching the engine's `recognize`: one ORT run per worker,
    # each ORT session pinned to a single thread so the two strategies are comparable.
    from concurrent.futures import ThreadPoolExecutor

    single = make_session(os.path.join(BUILD, "rec_small.onnx"), 1)
    sname = single.get_inputs()[0].name
    for x in inputs[:3]:
        single.run(None, {sname: x})

    ort_par = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            list(ex.map(lambda x: single.run(None, {sname: x}), inputs))
        ort_par.append((time.perf_counter() - t0) * 1000.0)

    # The Rust engine is timed on a real page through `onnx_ab time`, not scaled from a
    # single crop: recognition crops differ in width, so a linear extrapolation is wrong.
    rust_page = bench_rust_page(args.threads, args.repeats)

    os_, op = min(ort_seq), min(ort_par)
    print(f"rec  {len(inputs)} crops, widths {min(widths)}..{max(widths)}")
    print(f"      ORT  all crops, 16-thread pool, sequential        {os_:8.1f} ms")
    print(f"      ORT  all crops, 1 thread each, across pool         {op:8.1f} ms")
    print(f"      Rust engine `recognize` on the same 29 boxes       {rust_page:8.1f} ms")
    print(f"      gap vs ORT sequential {rust_page / os_:5.2f}x   "
          f"vs ORT across-pool {rust_page / op:5.2f}x")
    results["rec"] = {
        "crops": len(inputs),
        "widths": widths,
        "ort_sequential": os_,
        "ort_across_pool": op,
        "rust_recognize": rust_page,
        "gap_vs_sequential": rust_page / os_,
        "gap_vs_across_pool": rust_page / op,
    }

    print("=" * 92)
    dst = os.path.join(BUILD, "ab_ort_stages.json")
    with open(dst, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"saved: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
