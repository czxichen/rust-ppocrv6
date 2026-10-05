// Measures how well the 1x1 conv path actually parallelises.
use rayon::prelude::*;
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
    let iters = 30;
    let shapes = [
        (768usize, 384usize, 720usize), // Conv.49, 52
        (384, 768, 720),                // Conv.50, 53
        (384, 192, 1440),               // Conv.22, 25, 30, 33, 38, 41
        (192, 384, 1440),               // Conv.23, 26, 31, 34, 39, 42
        (192, 96, 2880),                // Conv.8, 11, 14
        (96, 192, 2880),                // Conv.9, 12, 15
    ];

    println!("threads = {}", rayon::current_num_threads());

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1e-4).cos() * 0.01).collect();
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let g = |ms: f64| flops / (ms * 1e-3) / 1e9;

        let nt = rayon::current_num_threads();

        // Strategy B: split m
        let chunk_m = m.div_ceil(nt).max(4).min(m);
        let nchunks_b = m.div_ceil(chunk_m);
        let mut out_b = vec![0.0f32; m * n];
        let p_b = out_b.as_mut_ptr() as usize;
        let ms_b = bench_ms(
            || {
                (0..nchunks_b).into_par_iter().for_each(|ci| {
                    let oc = ci * chunk_m;
                    let am = (m - oc).min(chunk_m);
                    unsafe {
                        ppocr::ops::gemm_row_major(
                            am,
                            k,
                            n,
                            a.as_ptr().add(oc * k),
                            k,
                            b.as_ptr(),
                            n,
                            (p_b as *mut f32).add(oc * n),
                            n,
                        );
                    }
                });
            },
            iters,
        );

        // Strategy A: split n
        let chunk_n = n.div_ceil(nt).max(64).min(n);
        let nchunks_a = n.div_ceil(chunk_n);
        let mut out_a = vec![0.0f32; m * n];
        let p_a = out_a.as_mut_ptr() as usize;
        let ms_a = bench_ms(
            || {
                (0..nchunks_a).into_par_iter().for_each(|ci| {
                    let cs = ci * chunk_n;
                    let an = (n - cs).min(chunk_n);
                    unsafe {
                        ppocr::ops::gemm_row_major(
                            m,
                            k,
                            an,
                            a.as_ptr(),
                            k,
                            b.as_ptr().add(cs),
                            n,
                            (p_a as *mut f32).add(cs),
                            n,
                        );
                    }
                });
            },
            iters,
        );

        println!(
            "m={:4} k={:4} n={:4} | stratB-m: {:6.3} ms ({:5.1} GF, {} chunks) | stratA-n: {:6.3} ms ({:5.1} GF, {} chunks) | B vs A: {:.2}x",
            m, k, n, ms_b, g(ms_b), nchunks_b, ms_a, g(ms_a), nchunks_a, ms_a / ms_b
        );
    }
}
