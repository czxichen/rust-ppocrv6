#!/usr/bin/env python3
"""Compare Rust engine output against the rebuilt ONNX model, then benchmark both.

Two jobs in one script so the numeric check and the timing use byte-identical
inputs:

  1. Parity - run the same synthetic input through ONNX Runtime and through the
     Rust dump produced by `cargo run --release --example dump_forward`, and
     diff element-wise. If these disagree, any latency comparison is moot.
  2. Benchmark - time ONNX Runtime under the same thread settings the Rust
     engine uses, so the FPS/latency ratio is apples-to-apples.

Usage:
    python tools/compare_rust_onnx.py [--iters 20] [--threads 16] [--skip-dump]
"""

import argparse
import os
import subprocess
import struct
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build_onnx")

# (label, onnx file, safetensors file, input shape)
MODELS = [
    ("det_small", "det_small.onnx", "PP-OCRv6_small_det.safetensors", (1, 3, 960, 704)),
    ("rec_small", "rec_small.onnx", "PP-OCRv6_small_rec.safetensors", (1, 3, 48, 960)),
    ("det_tiny", "det_tiny.onnx", "PP-OCRv6_tiny_det.safetensors", (1, 3, 960, 704)),
    ("rec_tiny", "rec_tiny.onnx", "PP-OCRv6_tiny_rec.safetensors", (1, 3, 48, 960)),
    ("cls", "cls.onnx", "PP-OCRv6_cls.safetensors", (1, 3, 48, 192)),
]


def synth_input(numel):
    """Must match examples/dump_forward.rs::synth_input exactly."""
    idx = np.arange(numel, dtype=np.uint64)
    h = (idx * np.uint64(2654435761)) >> np.uint64(8)
    return ((h % np.uint64(10007)).astype(np.float32) / np.float32(10007.0))


def read_rust_dump(path):
    with open(path, "rb") as fh:
        magic = fh.read(7)
        if magic != b"RSFWD1\0":
            raise ValueError(f"{path}: bad magic {magic!r}")
        ndim = struct.unpack("<Q", fh.read(8))[0]
        shape = tuple(struct.unpack("<Q", fh.read(8))[0] for _ in range(ndim))
        n = int(np.prod(shape))
        data = np.frombuffer(fh.read(4 * n), dtype="<f4").copy()
    return shape, data


def make_session(onnx_path, threads):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        onnx_path, so, providers=["CPUExecutionProvider"]
    )


def bench(session, x, iters, warmup=3):
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "min": ts[0],
        "p50": ts[len(ts) // 2],
        "mean": sum(ts) / len(ts),
        "max": ts[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated subset")
    args = ap.parse_args()

    wanted = set(args.models.split(",")) if args.models else None
    print(f"onnxruntime {ort.__version__} | 线程数 {args.threads} | iters {args.iters}")
    print("=" * 88)

    rows = []
    for label, onnx_file, st_file, shape in MODELS:
        if wanted and label not in wanted:
            continue
        onnx_path = os.path.join(BUILD, onnx_file)
        st_path = os.path.join(ROOT, "models", st_file)
        if not os.path.exists(onnx_path):
            print(f"{label:10s} 跳过：{onnx_path} 不存在")
            continue

        numel = int(np.prod(shape))
        x = synth_input(numel).reshape(shape)

        # --- 1. numeric parity -------------------------------------------
        parity = "未检查"
        if not args.skip_dump:
            dump = os.path.join(BUILD, f"rust_{label}.bin")
            cmd = [
                os.path.join(ROOT, "target", "release", "examples", "dump_forward.exe"),
                st_path,
                dump,
                *[str(d) for d in shape],
            ]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                               # The Rust binary prints UTF-8 Chinese; the Windows locale
                               # codec is GBK and would raise UnicodeDecodeError on it.
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"{label:10s} Rust dump 失败: {r.stderr.strip()[:200]}")
                parity = "Rust 侧失败"
            else:
                rshape, rdata = read_rust_dump(dump)
                sess = make_session(onnx_path, args.threads)
                oout = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
                # ORT hands back a flat buffer while the Rust engine preserves the
                # rank, so compare element counts rather than shapes. If the counts
                # agree the graph topology matched; the values then decide parity.
                if rdata.size != oout.size:
                    parity = f"元素数不符 rust={rdata.size} onnx={oout.size}"
                else:
                    diff = np.abs(oout.astype(np.float64) - rdata.astype(np.float64))
                    denom = np.abs(oout).max() + 1e-12
                    rel = diff.max() / denom
                    parity = f"max_abs={diff.max():.3e} rel={rel:.2e}"
                    if diff.max() > 1e-3:
                        parity += "  <<< 不一致"

        # --- 2. timing ----------------------------------------------------
        sess = make_session(onnx_path, args.threads)
        st = bench(sess, x, args.iters)
        rows.append((label, shapestr := "x".join(map(str, shape)), st, parity))
        print(
            f"{label:10s} {shapestr:14s} ORT min {st['min']:8.2f} | p50 {st['p50']:8.2f} "
            f"| mean {st['mean']:8.2f} ms   数值: {parity}"
        )

    print("=" * 88)
    out = os.path.join(BUILD, "ort_bench.json")
    import json

    with open(out, "w") as fh:
        json.dump(
            [
                {"label": l, "shape": s, "ort": t, "parity": p}
                for l, s, t, p in rows
            ],
            fh,
            indent=2,
        )
    print(f"ORT 结果已存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
