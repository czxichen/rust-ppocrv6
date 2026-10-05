#!/usr/bin/env python3
"""Rust tiny vs ORT tiny recognition rate on the SAME rec_accuracy cases.

Usage:
    cargo build --release --example rec_dump_tiny
    ./target/release/examples/rec_dump_tiny /tmp/rec_dump_tiny
    python3 tools/profiling/eval_tiny_vs_ort.py /tmp/rec_dump_tiny

Reads the dumped Rust logits (rec_out_*.bin) and the Rust inputs (rec_in_*.bin),
runs ORT tiny on the same inputs, decodes both with the same CTC greedy rule and
tiny keys, and reports char accuracy per side plus text identity per case.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

VOCAB_PATH = "models/ppocrv6_tiny_keys.txt"
REC_ONNX = "build_onnx/rec_tiny.onnx"

meta = {}
manifest_path = os.path.join(sys.argv[1], "manifest.txt") if len(sys.argv) > 1 else ""
try:
    with open(manifest_path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
except (FileNotFoundError, IndexError) as err:
    raise SystemExit("usage: eval_tiny_vs_ort.py <dump_dir>") from err

for line in lines:
    if line.startswith("TENSOR"):
        _, name, kv = line.split(" ", 2)
        shape = tuple(int(x) for x in kv.split("=", 1)[1].split(","))
        arr = np.fromfile(os.path.join(sys.argv[1], name + ".bin"), dtype="<f4")
        meta[name] = (shape, arr.reshape(shape))
    elif line.startswith("CASE"):
        _, i, note, img, truth = line.split("\t", 4)
        meta[f"case_{i}"] = (note, img, truth)

vocab = ["blank"]
try:
    with open(VOCAB_PATH, encoding="utf-8") as fh:
        for raw in fh:
            vocab.append(raw.rstrip("\r\n"))
except FileNotFoundError as err:
    raise SystemExit(f"missing vocab: {VOCAB_PATH}") from err
vocab.append(" ")

def ctc_greedy(logits):
    idx = logits.argmax(axis=-1)
    text, scores, last = [], [], 0
    for t, i in enumerate(idx):
        if i != 0 and i != last:
            text.append(vocab[int(i)])
            scores.append(float(logits[t, i]))
        last = int(i)
    return "".join(text), (sum(scores) / len(scores) if scores else 0.0)

def lev(a, b):
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
            prev = cur
    return dp[m]

sess = ort.InferenceSession(REC_ONNX, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
cases = sorted((k for k in meta if k.startswith("case_")), key=lambda k: int(k.split("_")[1]))
tot_len = tot_er = tot_eo = same = 0
print(f"{'case':<28}{'truth_len':>9}{'rust':>6}{'ort':>6}{'same?':>6}")
for k in cases:
    i = int(k.split("_")[1])
    note, img, truth = meta[k]
    r = meta[f"rec_out_{i:03d}"][1][0]
    o = np.asarray(sess.run(None, {iname: meta[f"rec_in_{i:03d}"][1]})[0])[0]
    rt, _ = ctc_greedy(r)
    ot, _ = ctc_greedy(o)
    ln = len(truth)
    er = lev(rt, truth)
    eo = lev(ot, truth)
    tot_len += ln
    tot_er += er
    tot_eo += eo
    same += (rt == ot)
    print(f"{note:<28}{ln:>9}{er:>6}{eo:>6}{('Y' if rt==ot else 'N'):>6}")
print("-" * 60)
print(f"rust tiny: acc={(tot_len-tot_er)/tot_len:.4f}  (edits={tot_er}/{tot_len})")
print(f"ort  tiny: acc={(tot_len-tot_eo)/tot_len:.4f}  (edits={tot_eo}/{tot_len})")
print(f"text identical on {same}/{len(cases)} cases")
