#!/usr/bin/env python3
"""Drive ONNX Runtime with the *same* tensors `examples/onnx_ab.rs` dumps.

Two jobs:

1. **Numerical equivalence gate.** ORT and the Rust engine are supposed to be
   evaluating the same graph on the same weights. If their outputs disagree,
   any latency comparison is meaningless, so we check first and refuse to
   report speed numbers otherwise.
2. **Latency.** Same input tensors, matched thread counts, min/p50/mean plus
   CPU seconds so a noisy shared machine cannot fake a result.

    # 1. produce the tensors with the Rust side
    cargo run --release --example onnx_ab -- dump tests/data/bench_doc.png build_onnx/dump
    # 2. run this
    python tools/profiling/ort_ab.py --dump build_onnx/dump --iters 15 --threads 12

Requires `numpy` and `onnxruntime`; run from the repository root so that the
`models_v6/` and `models/` relative paths above resolve.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

DET_ONNX = "models_v6/ppocrv6_det.onnx"
REC_ONNX = "models_v6/ppocrv6_rec.onnx"
CLS_ONNX = "models_v6/ppocr_mobile_v2.0_cls_infer.onnx"
DICT = "models/ppocrv6_keys.txt"


def read_manifest(dump_dir):
    """name -> (shape, path) for every dumped tensor, plus image/boxes/text."""
    tensors, meta = {}, {}
    with open(os.path.join(dump_dir, "manifest.txt"), encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("TENSOR "):
                _, name, shapekv = line.split(" ", 2)
                shape = tuple(int(d) for d in shapekv.split("=", 1)[1].split(","))
                arr = np.fromfile(os.path.join(dump_dir, name + ".bin"), dtype="<f4")
                if arr.size != int(np.prod(shape)):
                    raise SystemExit(
                        f"{name}: {arr.size} values on disk but manifest says {shape}"
                    )
                tensors[name] = arr.reshape(shape).copy()
            elif line.split(" ", 1)[0] in ("IMAGE", "BOXES", "TEXT"):
                # `TEXT` carries tab-separated lines that themselves contain spaces,
                # so keep the whole remainder rather than one whitespace token.
                key, value = line.split(" ", 1)
                meta[key] = value
    return tensors, meta


def load_vocab(path):
    vocab = ["blank"]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            vocab.append(line.rstrip("\r\n"))
    vocab.append(" ")
    return vocab


def ctc_greedy(logits, vocab):
    """Mirror of `CtcDecoder::decode`: argmax, collapse repeats, drop blank."""
    idx = logits.argmax(axis=-1)  # [T]
    text, scores, last = [], [], 0
    for t, i in enumerate(idx):
        if i != 0 and i != last:
            text.append(vocab[int(i)])
            scores.append(float(logits[t, i]))
        last = int(i)
    return "".join(text), (sum(scores) / len(scores) if scores else 0.0)


def make_session(path, threads, profile=False):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 0 would mean "let ORT choose" (= all cores); we pin it so the comparison is
    # against a known thread count, and so both sides can be run single-threaded.
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    if profile:
        opts.enable_profiling = True
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return sess, (time.perf_counter() - t0) * 1000.0


def dump_ort_profile(model_path, threads, feeds, label, top, out_prefix, runs=5):
    """ORT's chrome trace carries real per-node durations.

    Two traps, both hit on the way here:

    * A brand-new session's first inference is cold, so the *last* run in the
      trace is the steady state — take it, not the sum over all runs.
    * Each node emits one event per run, so events must be split by timestamp
      into equal chunks. Grouping by name and taking "the latest occurrence"
      silently mixes runs and understates everything.

    Profiling also slows the run down (roughly 1.8x here), so use these numbers
    for *shares*, not for absolute latency — that is what `--iters` measures.
    """
    import collections
    import json as _json

    sess, _ = make_session(model_path, threads, profile=True)
    for _ in range(runs):
        sess.run(None, feeds)
    trace_path = sess.end_profiling()
    with open(trace_path, encoding="utf-8") as fh:
        trace = _json.load(fh)

    nodes = sorted(
        (e for e in trace if e.get("cat") == "Node" and "dur" in e),
        key=lambda e: e["ts"],
    )
    chunk = len(nodes) // runs
    per_run = [
        sum(e["dur"] for e in nodes[i * chunk : (i + 1) * chunk]) / 1000.0
        for i in range(runs)
    ]
    last = nodes[(runs - 1) * chunk :]

    agg = collections.Counter()
    for e in last:
        agg[e["args"].get("op_name", e.get("name", "?"))] += e["dur"]
    total_us = sum(agg.values())
    items = sorted(agg.items(), key=lambda kv: -kv[1])
    print(
        f"PROFILE {label} warm_run={total_us / 1000.0:.2f} ms  nodes={len(items)}"
        f"  (profiled; unprofiled runs were {', '.join(f'{v:.0f}' for v in per_run)} ms)"
    )
    for name, us in items[:top]:
        print(f"  {name:30} {us / 1000.0:8.2} ms  {100.0 * us / total_us:5.1f}%")
    with open(f"{out_prefix}_profile.json", "w", encoding="utf-8") as fh:
        _json.dump(items, fh, indent=2)


def time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "min": samples[0],
        "p50": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "n": len(samples),
    }


def compare(name, got, ref):
    """Max/mean absolute error and a scale-aware relative error."""
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = np.abs(ref).max()
    return {
        "tensor": name,
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "ref_max": float(scale),
        "rel": float(d.max() / scale) if scale > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument(
        "--light",
        action="store_true",
        help="only det/single-crop forward; skip the page-level loops "
        "(single-threaded page loops take minutes)",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--profile", action="store_true",
                    help="emit ORT's per-node breakdown for det and one rec crop")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    tensors, meta = read_manifest(args.dump)
    n_crops = sum(1 for k in tensors if k.startswith("rec_in_"))
    print(f"dump: {meta.get('IMAGE')}  boxes={meta.get('BOXES')}  crops={n_crops}")
    print(f"ORT {ort.__version__}  providers={ort.get_available_providers()}")

    if args.profile:
        os.makedirs("/tmp/onnx_bench", exist_ok=True)
        d = ort.InferenceSession(DET_ONNX, providers=["CPUExecutionProvider"])
        dump_ort_profile(
            DET_ONNX, args.threads,
            {d.get_inputs()[0].name: tensors["det_in"]},
            "det", args.top, "/tmp/onnx_bench/ort_det",
        )
        r = ort.InferenceSession(REC_ONNX, providers=["CPUExecutionProvider"])
        dump_ort_profile(
            REC_ONNX, args.threads,
            {r.get_inputs()[0].name: tensors["rec_in_000"]},
            "rec", args.top, "/tmp/onnx_bench/ort_rec",
        )
        return

    det_sess, det_load = make_session(DET_ONNX, args.threads)
    rec_sess, rec_load = make_session(REC_ONNX, args.threads)
    det_in_name = det_sess.get_inputs()[0].name
    rec_in_name = rec_sess.get_inputs()[0].name
    print(f"session load: det={det_load:.0f} ms  rec={rec_load:.0f} ms "
          f"(intra_op={args.threads or 'ORT default'})")

    # ---- numerical equivalence gate -------------------------------------
    checks = []
    det_got = det_sess.run(None, {det_in_name: tensors["det_in"]})[0]
    checks.append(compare("det_out", det_got, tensors["det_out"]))

    vocab = load_vocab(DICT)
    rec_got, texts = [], []
    for i in range(n_crops):
        out = rec_sess.run(None, {rec_in_name: tensors[f"rec_in_{i:03}"]})[0]
        rec_got.append(out)
        checks.append(compare(f"rec_out_{i:03}", out, tensors[f"rec_out_{i:03}"]))
        texts.append(ctc_greedy(out[0], vocab)[0])

    worst_det = max(c["rel"] for c in checks if c["tensor"] == "det_out")
    worst_rec = max(c["rel"] for c in checks if c["tensor"].startswith("rec_out"))
    max_abs_rec = max(c["max_abs"] for c in checks if c["tensor"].startswith("rec_out"))
    print("\n--- numerical equivalence (ORT vs engine, identical inputs) ---")
    print(f"det_out : max_abs={checks[0]['max_abs']:.3e}  ref_max={checks[0]['ref_max']:.4f}"
          f"  rel={checks[0]['rel']:.3e}")
    print(f"rec_out : worst rel={worst_rec:.3e}  max_abs={max_abs_rec:.3e}  over {n_crops} crops")

    engine_text = [t for t in meta.get("TEXT", "").split("\t") if t]
    ort_text = [f"{t}|{s:.4f}" for t, s in (ctc_greedy(o[0], vocab) for o in rec_got)]
    same_text = engine_text == ort_text
    n_same = sum(1 for a, b in zip(engine_text, ort_text) if a == b)
    print(f"decoded text: {n_same}/{len(engine_text)} lines identical"
          f"  ({'ALL IDENTICAL' if same_text else 'MISMATCH'})")
    if not same_text:
        for a, b in zip(engine_text, ort_text):
            if a != b:
                print(f"   engine={a!r}\n   ort   ={b!r}")
                break

    result = {
        "threads": args.threads,
        "iters": args.iters,
        "boxes": int(meta.get("BOXES", 0)),
        "session_load_ms": {"det": det_load, "rec": rec_load},
        "numerics": {
            "det_max_abs": checks[0]["max_abs"],
            "det_rel": worst_det,
            "rec_worst_rel": worst_rec,
            "rec_max_abs": max_abs_rec,
            "text_lines_identical": n_same,
            "text_lines_total": len(engine_text),
        },
    }

    if args.check_only:
        print("\n" + json.dumps(result))
        return

    # ---- latency ---------------------------------------------------------
    # Detection and single-crop recognition are cheap, so sample them more heavily;
    # the page-level loops dominate the wall clock and stay at `iters`.
    warmup = max(3, args.iters // 4)
    light_iters = args.iters * 3
    cpu0 = time.process_time()
    wall0 = time.perf_counter()

    det_in = tensors["det_in"]
    r = time_call(lambda: det_sess.run(None, {det_in_name: det_in}), light_iters, warmup)
    result["det_fwd"] = r

    rec_inputs = [tensors[f"rec_in_{i:03}"] for i in range(n_crops)]
    r1 = time_call(
        lambda: rec_sess.run(None, {rec_in_name: rec_inputs[0]}), light_iters, warmup
    )
    result["rec1_fwd"] = r1

    def run_all():
        for t in rec_inputs:
            rec_sess.run(None, {rec_in_name: t})

    if not args.light:
        result["recall_fwd"] = time_call(run_all, args.iters, warmup)

        # Batched pass: pad the odd-width crop up to the common width. Only meaningful
        # because every crop here is already ~48x960; a ragged batch would need far
        # more padding.
        max_w = max(t.shape[-1] for t in rec_inputs)
        batch = np.zeros((n_crops, 3, 48, max_w), dtype=np.float32)
        for i, t in enumerate(rec_inputs):
            batch[i, :, :, : t.shape[-1]] = t
        result["recbatch_fwd"] = time_call(
            lambda: rec_sess.run(None, {rec_in_name: batch}), args.iters, warmup
        )
        result["recbatch_shape"] = [int(d) for d in batch.shape]

    cpu1 = time.process_time()
    wall1 = time.perf_counter()
    result["cpu_seconds"] = cpu1 - cpu0
    result["wall_seconds"] = wall1 - wall0

    print("\n--- ONNX Runtime latency (ms) ---")
    labels = [
        ("det_fwd", "det  forward  1x3x960x704"),
        ("rec1_fwd", "rec  forward  1x3x48x960 (1 crop, batch=1)"),
    ]
    if not args.light:
        labels.append(("recall_fwd", "rec  forward  29 crops sequentially"))
        b = result["recbatch_shape"]
        labels.append(("recbatch_fwd", f"rec  forward  batch {b[0]}x3x48x{b[3]}"))
    for k, label in labels:
        s = result[k]
        print(f"{label:46s} min={s['min']:8.2f}  p50={s['p50']:8.2f}  mean={s['mean']:8.2f}")
    print(f"cpu(user+sys)={result['cpu_seconds']:.1f}s  wall={result['wall_seconds']:.1f}s "
          f"over all timing loops")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    summary = {k: result[k]["min"] for k, _ in labels}
    summary["cpu_seconds"] = result["cpu_seconds"]
    print("\n" + json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
