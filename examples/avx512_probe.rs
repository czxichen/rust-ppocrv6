// Is the AVX-512 sgemm kernel in matrixmultiply actually being reached on this host?
//
// The CPU is a Comet Lake (family 6 model 165), which does have AVX-512. matrixmultiply
// 0.3.11 has an AVX-512 sgemm microkernel, gated on Rust 1.89+ (this toolchain is 1.98),
// and detects it at runtime. So in principle it should already be in use.
//
// The evidence says it is not. `peak_probe` measured single-thread sgemm at 43-63 GFLOP/s,
// which is the AVX2 ceiling (16 FLOP/cycle -> 46.4 GFLOP/s at 2.9 GHz) rather than the
// AVX-512 one (32 FLOP/cycle -> 92.8 GFLOP/s). This probe separates the two by measuring
// the same GEMM at sizes large enough to be compute-bound.
//
//   cargo run --release --example avx512_probe
use matrixmultiply::sgemm;
use std::hint::black_box;
use std::time::Instant;

fn bench_ms<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..2 {
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
    println!("avx2  : {}", is_x86_feature_detected!("avx2"));
    println!("fma   : {}", is_x86_feature_detected!("fma"));
    println!("avx512f       : {}", is_x86_feature_detected!("avx512f"));
    println!("avx512dq     : {}", is_x86_feature_detected!("avx512dq"));
    println!("avx512bw     : {}", is_x86_feature_detected!("avx512bw"));
    println!("avx512vl     : {}", is_x86_feature_detected!("avx512vl"));
    println!("avx512_vnni  : {}", is_x86_feature_detected!("avx512vnni"));
    println!("avx512_vnni  : {}", is_x86_feature_detected!("avx512vnni"));
    println!("avx512_bf16  : {}", is_x86_feature_detected!("avx512bf16"));

    // 2.9 GHz base. Two FMA ports is the usual Comet Lake figure.
    let base_ghz = 2.90f64;
    println!("\ntheoretical at {base_ghz} GHz base:");
    println!("  AVX2   16 FLOP/cycle -> {:6.1} GFLOP/s", 16.0 * base_ghz);
    println!("  AVX-512 32 FLOP/cycle -> {:6.1} GFLOP/s", 32.0 * base_ghz);

    println!("\nsingle-thread sgemm, f32 (min of 8):");
    for n in [256usize, 512, 1024] {
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];
        let mut c = vec![0.0f32; n * n];
        let ms = bench_ms(
            || unsafe {
                sgemm(
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
                );
            },
            8,
        );
        let gf = 2.0 * n as f64 * n as f64 * n as f64 / (ms * 1e-3) / 1e9;
        let pct = gf / (16.0 * base_ghz) * 100.0;
        println!("  {n:5}^3  {ms:8.3} ms  {gf:7.1} GFLOP/s  = {pct:5.1}% of the AVX2 ceiling");
    }
}
