// Single-thread sgemm on the exact calls the 1x1 conv path makes, for matrixmultiply
// `constconf` tuning.
//
// `constconf` is a compile-time cache-blocking knob, so its effect is a per-call,
// single-thread question and this probe measures it that way -- one `sgemm` call, no rayon,
// so the thread pool cannot hide a change behind scheduling.
//
// The shapes are the per-chunk calls Strategy A issues: `sgemm(m, k, chunk_n)` where
// `chunk_n = n.div_ceil(threads).max(64)`.
//
//   cargo run --release --example constconf_probe
use std::hint::black_box;
use std::time::Instant;

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
    // (m, k, chunk_n, what it is)
    let calls: &[(usize, usize, usize, &str)] = &[
        (384, 192, 64, "Conv.30/33/38/41  n=144 -> 3 chunks of 64"),
        (192, 384, 64, "Conv.31/34/39/42  n=144 -> 3 chunks of 64"),
        (96, 48, 144, "Conv.11/8         n=2304 -> 16 chunks of 144"),
        (192, 96, 144, "Conv.31          n=2304"),
        (768, 384, 45, "Conv.49/52       n=708 -> 16 chunks of 45"),
        (384, 768, 45, "Conv.53          n=708"),
        (384, 192, 45, "Conv.50          n=708"),
        (768, 384, 708, "whole row, n=708 unchunked"),
    ];

    println!(
        "{:<38} {:>6} {:>10} {:>9}  {:>10} {:>7}",
        "call", "MFLOP", "ms", "GFLOP/s", "us/call", "label"
    );
    let mut total_flops = 0.0f64;
    let mut total_ms = 0.0f64;
    for &(m, k, n, label) in calls {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1e-4).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let ms = bench_ms(
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
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                );
            },
            40,
        );
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        total_flops += flops;
        total_ms += ms;
        println!(
            "sgemm({m:>4},{k:>4},{n:>4}){:<12} {:6.1} {:10.3} {:9.1}  {:10.1}  {}",
            "",
            flops / 1e6,
            ms,
            flops / (ms * 1e-3) / 1e9,
            ms * 1000.0,
            label
        );
    }
    println!(
        "\ntotal {total_flops:.1} MFLOP in {total_ms:.3} ms = {:.1} GFLOP/s aggregate",
        total_flops / (total_ms * 1e-3) / 1e9
    );
}
