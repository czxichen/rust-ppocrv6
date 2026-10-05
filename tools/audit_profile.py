"""Audit the rust_ppocrv6 operator profile dumped by `examples/profile_all.rs`.

Reconstructs MAC counts from the operator labels, then reports per-operator
effective GFLOPS and memory traffic rate, so we can tell whether a kernel is
compute-bound, bandwidth-bound, or just inefficient.

    python tools/audit_profile.py profile_dump.txt
"""

import re
import sys
from collections import defaultdict

BYTES_PER_F32 = 4
BANDWIDTH_GBS = 10.0   # measured platform ceiling (see report)
PEAK1_GFLOPS = 146.0   # measured single-core AVX2/FMA peak
PHYS_CORES = 8

RE_1X1 = re.compile(r"^Conv1x1: .*\((\d+)x(\d+)\) in=\[(\d+), (\d+), (\d+), (\d+)\]")
RE_DW = re.compile(
    r"^ConvDW: .*\[(\d+), (\d+), (\d+), (\d+)\] s=\[(\d+), (\d+)\] in=\[(\d+), (\d+), (\d+), (\d+)\]"
)
RE_CONV = re.compile(
    r"^Conv: .*\[(\d+), (\d+), (\d+), (\d+)\] s=\[(\d+), (\d+)\] p=\[([\d,\s]*)\] in=\[(\d+), (\d+), (\d+), (\d+)\]"
)
RE_MM = re.compile(r"^MatMul: .*\[(\d+), (\d+), (\d+)\] x \[\]")


def out_dim(size, k, s, p):
    v = (size + 2 * p - k) // s + 1
    return max(v, 0)


def parse_case(path):
    cases = []
    cur = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("CASE\t"):
                parts = line.split("\t")
                if parts[2] == "ERROR":
                    cur = None
                    continue
                cur = {"name": parts[1], "total": float(parts[2]), "ops": []}
                cases.append(cur)
            elif line == "ENDCASE":
                cur = None
            elif cur is not None and "\t" in line:
                label, ms = line.rsplit("\t", 1)
                cur["ops"].append((label, float(ms)))
    return cases


VOCAB = 18710  # PP-OCRv6 Small recognition head vocabulary


def analyze(label):
    """-> dict(macs, traffic_bytes, kind, detail) or None if not accounted."""
    m = RE_1X1.match(label)
    if m:
        co, ci, n, c, h, w = (int(x) for x in m.groups())
        macs = co * h * w * ci
        elems = n * c * h * w + n * co * h * w + co * ci
        return dict(macs=macs, traffic=elems * BYTES_PER_F32, kind="1x1 GEMM",
                    out=(co, h, w))

    m = RE_DW.match(label)
    if m:
        co, ci, kh, kw, sh, sw, n, c, h, w = (int(x) for x in m.groups())
        oh = out_dim(h, kh, sh, kh // 2)
        ow = out_dim(w, kw, sw, kw // 2)
        macs = c * oh * ow * kh * kw
        elems = n * c * h * w + n * c * oh * ow + c * kh * kw
        return dict(macs=macs, traffic=elems * BYTES_PER_F32, kind="depthwise",
                    out=(c, oh, ow))

    m = RE_CONV.match(label)
    if m:
        co, ci, kh, kw, sh, sw, pstr, n, c, h, w = m.groups()
        co, ci, kh, kw, sh, sw = (int(x) for x in (co, ci, kh, kw, sh, sw))
        n, c, h, w = (int(x) for x in (n, c, h, w))
        ps = [int(x) for x in pstr.split(",") if x.strip()]
        pt, pl = (ps[0], ps[1]) if len(ps) == 4 else (0, 0)
        pb, pr = (ps[2], ps[3]) if len(ps) == 4 else (0, 0)
        oh = out_dim(h, kh, sh, pt) if pt == pb else out_dim(h, kh, sh, (pt + pb) // 2)
        ow = out_dim(w, kw, sw, pl) if pl == pr else out_dim(w, kw, sw, (pl + pr) // 2)
        macs = co * oh * ow * c * kh * kw
        elems = n * c * h * w + n * co * oh * ow + co * c * kh * kw
        return dict(macs=macs, traffic=elems * BYTES_PER_F32, kind="conv",
                    out=(co, oh, ow))

    m = RE_MM.match(label)
    if m:
        n, t, k = (int(x) for x in m.groups())
        macs = n * t * k * VOCAB
        elems = n * t * k + k * VOCAB + n * t * VOCAB
        return dict(macs=macs, traffic=elems * BYTES_PER_F32, kind="matmul",
                    out=(n, t, VOCAB))
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile_dump.txt"
    cases = parse_case(path)

    for case in cases:
        total = sum(ms for _, ms in case["ops"])
        acc = defaultdict(lambda: [0.0, 0, 0])   # kind -> [ms, macs, traffic]
        unknown_ms = 0.0
        unknown_ops = defaultdict(float)
        slow = []

        n_ops = 0
        for label, ms in case["ops"]:
            n_ops += 1
            info = analyze(label)
            if info is None:
                unknown_ms += ms
                unknown_ops[label.split(":")[0].split("[")[0].strip()] += ms
                continue
            a = acc[info["kind"]]
            a[0] += ms
            a[1] += info["macs"]
            a[2] += info["traffic"]
            if ms > 0.05:
                slow.append((ms / (info["macs"] / 1e9) if info["macs"] else 0, label, ms,
                             info["macs"], info["traffic"]))

        print("=" * 100)
        print(f"CASE {case['name']}   nodes={n_ops}   sum(op_ms)={total:.2f} ms   forward={case['total']:.2f} ms")
        print("-" * 100)
        grand_macs = 0
        grand_ms = 0.0
        for kind, (ms, macs, traffic) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
            gflops = (macs * 2) / ms / 1e6 if ms > 0 else 0
            gbs = traffic / ms / 1e6 if ms > 0 else 0
            share = ms / total * 100 if total else 0
            grand_macs += macs
            grand_ms += ms
            print(f"  {kind:<11} ms={ms:8.2f} ({share:5.1f}%)  GMAC={macs/1e9:7.2f}  "
                  f"eff={gflops:7.2f} GFLOPS ({gflops/(PEAK1_GFLOPS*PHYS_CORES)*100:5.2f}% of {PHYS_CORES}-core peak)  "
                  f"traffic={gbs:6.2f} GB/s ({gbs/BANDWIDTH_GBS*100:5.1f}% of ceiling)")
        if unknown_ms > 0:
            print(f"  {'other':<11} ms={unknown_ms:8.2f} ({unknown_ms/total*100:5.1f}%)  "
                  f"top: " + ", ".join(f"{k}={v:.2f}ms" for k, v in
                                       sorted(unknown_ops.items(), key=lambda kv: -kv[1])[:6]))
        tot_gflops = (grand_macs * 2) / grand_ms / 1e6 if grand_ms else 0
        print(f"  -> compute ops total: {grand_ms:.2f} ms, {grand_macs/1e9:.2f} GMAC, "
              f"aggregate {tot_gflops:.2f} GFLOPS")
        print(f"  -> slowest per-MAC ops (ms per GMAC):")
        slow.sort(key=lambda x: -x[0])
        for ms_per_gmac, label, ms, macs, traffic in slow[:4]:
            print(f"       {ms_per_gmac:9.1f} ms/GMAC  {ms:7.3f} ms  {label[:72]}")
        print()


if __name__ == "__main__":
    main()
