// Calibrates what this machine actually achieves, so "X% of peak" means something.
//
// Every optimisation attempt should be decided by measurement, and the
// first four were refuted. This example exists so a claim like "the kernels only reach
// Y% of peak" is anchored to a number measured on this host rather than to the datasheet.
//
//   cargo run --release --example peak_probe
use std::hint::black_box;
use std::time::Instant;

fn gflops(ms: f64, flops: f64) -> f64 {
    flops / (ms * 1e-3) / 1e9
}

/// Theoretical single-thread FP32 peak with AVX2 FMA. The i7-10700 base clock is 2.9 GHz
/// but turbo is 4.6 GHz, and a single thread gets the turbo clock, so the ceiling is
/// 8 lanes * 2 (FMA) * 2 flops * 4.6 GHz = 147 GFLOP/s. Using the base clock instead
/// produces the nonsense "109% of one core" that the first draft of this file printed.
const ONE_CORE_PEAK: f64 = 8.0 * 2.0 * 2.0 * 4.6;

fn main() {
    let threads = rayon::current_num_threads();
    println!("rayon threads = {threads} (about {} physical cores)", threads / 2);
    println!("theoretical one-core peak (turbo 4.6 GHz) = {ONE_CORE_PEAK:.1} GFLOP/s");

    // 1. Large square sgemm: what a well-tuned blocked FP32 GEMM reaches here. This is the
    //    number every other GEMM in this crate should be compared against -- not the
    //    datasheet peak, which nothing in the crate gets near.
    for n in [512usize, 1024, 2048] {
        let a: Vec<f32> = (0..n * n).map(|i| (i % 97) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i % 89) as f32 * 0.01).collect();
        let mut c = vec![0.0f32; n * n];
        let iters = if n <= 1024 { 5 } else { 2 };
        let mut run = || unsafe {
            matrixmultiply::sgemm(
                n,
                n,
                n,
                1.0,
                a.as_ptr(),
                n as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            )
        };
        for _ in 0..2 {
            run();
        }
        let t = Instant::now();
        for _ in 0..iters {
            run();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        black_box(&c);
        let gf = gflops(ms, 2.0 * n as f64 * n as f64 * n as f64);
        println!(
            "sgemm {n:5}^3 single-thread : {ms:8.3} ms  {gf:6.1} GFLOP/s  ({:3.0}% of one core)",
            gf / ONE_CORE_PEAK * 100.0
        );
    }

    // 2. Streaming read, single thread, 64 MB so it cannot sit in cache.
    //
    //    NOTE: this is a scalar `iter().sum()`, so it under-reports badly. Treat it as an
    //    order-of-magnitude check that the GEMM numbers above are not bandwidth-bound, not
    //    as a bandwidth figure.
    let n = 64 * 1024 * 1024 / 4;
    let a: Vec<f32> = (0..n).map(|i| (i % 1009) as f32).collect();
    let mut sink = 0.0f32;
    let t = Instant::now();
    for _ in 0..3 {
        sink += a.iter().sum::<f32>();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 3.0;
    black_box(sink);
    println!(
        "stream read 64 MB, 1 thread : {ms:8.3} ms  {:6.1} GB/s (scalar sum, lower bound)",
        (n * 4) as f64 / (ms * 1e-3) / 1e9
    );
}
