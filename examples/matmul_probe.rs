//! Isolate the vocabulary-projection MatMul that dominates recognition latency.
//!
//! ORT fuses MatMul+Transpose and still beats the Rust engine by ~8x on rec_tiny;
//! the last projection (M=time steps, K=hidden, N=vocab) is the single biggest
//! node in that graph, so this measures the Rust kernel on exactly that shape and
//! reports achieved GFLOPS and effective weight-read bandwidth.
//!
//! Reading the numbers: K=80 means the weight tile is tiny relative to N=6906, so
//! the op is memory-bound on the weight matrix. Achieved GFLOPS far below the
//! machine's FMA peak is expected; the useful signal is whether we are near the
//! platform's memory bandwidth, which the project has measured at ~11 GB/s.
//!
//! Usage: cargo run --release --example matmul_probe [-- --iters 30]

use std::time::Instant;

use ppocr::Tensor;
use ppocr::ops;

fn synth(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = i.wrapping_mul(2654435761) >> 8;
            (h % 10007) as f32 / 10007.0
        })
        .collect()
}

fn bench(label: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a = Tensor::from_vec(&[m, k], synth(m * k)).unwrap();
    // MatMul weight is [K, N] in the graph; matmul_transposed_b takes [N, K].
    let w = Tensor::from_vec(&[n, k], synth(n * k)).unwrap();

    // Warm up: the first call may trigger per-shape preparation.
    let mut out = ops::matmul_transposed_b(&a, &w, None).unwrap();
    std::hint::black_box(&out);

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        out = ops::matmul_transposed_b(&a, &w, None).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&out);
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p50 = times[times.len() / 2];

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    // Every output tile needs the full weight matrix, but the weight stays hot in
    // cache across the M loop, so the DRAM-relevant traffic is min over: full
    // weight read per call, vs. total bytes touched.
    let w_bytes = (k * n * 4) as f64;
    let gflops = flops / (p50 / 1000.0) / 1e9;
    let wbw = w_bytes / (p50 / 1000.0) / 1e9;

    println!(
        "{label:12} M={m:4} K={k:4} N={n:5}  p50={p50:8.3} ms  min={:8.3} ms  \
         {gflops:7.1} GFLOPS  权重 {:.2} MB → {wbw:6.2} GB/s",
        times[0],
        w_bytes / 1e6,
    );
}

fn main() {
    let iters = std::env::args()
        .position(|a| a == "--iters")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);

    println!("线程池大小: {}", rayon::current_num_threads());
    println!("{}", "=".repeat(104));
    println!("【rec_tiny 词表投影】linear_1.w_0 = [80, 6906]");
    for m in [120usize, 240, 480] {
        bench("vocab_tiny", m, 80, 6906, iters);
    }
    println!();
    println!("【rec_small 词表投影】实际词表 18710");
    for m in [120usize, 240, 480] {
        bench("vocab_small", m, 120, 18710, iters);
    }
    println!();
    println!("【对照：大 K 的方阵 GEMM】权重驻留性更好，应显著更快");
    bench("square", 512, 512, 512, iters);
    bench("square", 120, 512, 512, iters);
}
