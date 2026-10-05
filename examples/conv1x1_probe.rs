// Micro-benchmark for the 1x1 convolutions that dominate the rec forward.
//
// A 1x1 conv is a plain GEMM: out[oc, s] = sum_ic w[oc, ic] * in[ic, s], with the
// input in NCHW so it is already [k, n] row-major and the weight is [m, k].
//
// The shapes below are the real ones from `onnx_ab profile` on the small rec model,
// e.g. `Conv.33 (384x192) in=[1, 192, 6, 24]` -> m=384, k=192, n=144.
//
// The engine splits by output channels (Strategy B: `m.div_ceil(threads)` rows per
// thread), so each `sgemm` call sees a small m and the full n. This probe measures
// that split against the alternatives, and against a transposed-B layout where the
// reduction axis is contiguous, which is the layout `matrixmultiply` prefers.
//
//   cargo run --release --example conv1x1_probe
use std::hint::black_box;
use std::time::Instant;

/// Minimum of many timed runs. This is a shared workstation where a single run can
/// vary 2x, so a mean cannot resolve a 10-20% change.
fn bench_ms<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

fn main() {
    let iters = 40;
    // (m, k, n) as reported by the profiler: out_channels, in_channels, h*w.
    let shapes = [
        (384usize, 192usize, 144usize), // Conv.33 / 38 / 41, 6x24
        (192, 384, 144),                // Conv.34 / 39 / 42
        (384, 192, 9216),               // Conv.52, 3x3072 (the wide one)
        (96, 48, 2304),                 // Conv.11 / 8, 24x96
        (192, 96, 2304),                // Conv.31
    ];

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
        // B as the engine sees it: [k, n] row-major (NCHW input, channel-major).
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1e-4).cos() * 0.01).collect();
        // B^T as [n, k] row-major: the reduction axis contiguous.
        let mut bt = vec![0.0f32; n * k];
        for j in 0..n {
            for p in 0..k {
                bt[j * k + p] = b[p * n + j];
            }
        }
        let mut out = vec![0.0f32; m * n];
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        // A: engine layout, sgemm(m, k, n) with B strided by n.
        let ms_nchw = bench_ms(
            || unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    a.as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    out.as_mut_ptr(),
                    n as isize,
                    1,
                );
            },
            iters,
        );

        // B: transposed weight, sgemm(n, k, m) writing C^T, then we would need a
        // transpose to get [m, n]. Measured *without* the transpose so the number is
        // the GEMM only; the transpose cost is called out in the printed note.
        let mut out_t = vec![0.0f32; n * m];
        let ms_transposed = bench_ms(
            || unsafe {
                matrixmultiply::sgemm(
                    n,
                    k,
                    m,
                    1.0,
                    bt.as_ptr(),
                    k as isize,
                    1,
                    a.as_ptr(),
                    k as isize,
                    1,
                    0.0,
                    out_t.as_mut_ptr(),
                    m as isize,
                    1,
                );
            },
            iters,
        );

        let g = |ms: f64| flops / (ms * 1e-3) / 1e9;
        println!(
            "m={m:<4} k={k:<4} n={n:<6} NCHW-in {ms_nchw:7.3} ms ({:6.1} GFLOP/s)   \
             Bt-contig {ms_transposed:7.3} ms ({:6.1} GFLOP/s)   {:.2}x",
            g(ms_nchw),
            g(ms_transposed),
            ms_nchw / ms_transposed
        );
    }
}
