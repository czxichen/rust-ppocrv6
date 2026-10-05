// Compares the two layouts for the recognition head's classification GEMM.
//
// The safetensors weight is [k, n] = [120, 18710] with n contiguous, which is exactly
// what a standard blocked GEMM wants. `Model::load` transposes it to [n, k] to feed the
// hand-rolled dot-product kernel in matmul.rs, which reaches only ~10% of peak.
//
//   cargo run --release --example gemm_probe
use ppocr::ops::matmul_transposed_b;
use ppocr::tensor::Tensor;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let (k, n) = (120usize, 18710usize);
    for m in [37usize, 118] {
        // A: [m, k] row-major.
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-3).cos()).collect();
        let a_t = Tensor::from_vec(&[1, m, k], a.clone()).expect("a");

        // [k, n] row-major -- the layout actually stored in the model file.
        let w_kn: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1e-4).sin() * 0.01).collect();
        // [n, k] row-major -- what the engine transposes to.
        let mut w_nk = vec![0.0f32; n * k];
        for j in 0..n {
            for p in 0..k {
                w_nk[j * k + p] = w_kn[p * n + j];
            }
        }
        let w_nk_t = Tensor::from_vec(&[n, k], w_nk).expect("w_nk");

        let iters = 30;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        // Hand-rolled kernel, [n, k] layout.
        for _ in 0..3 {
            black_box(matmul_transposed_b(&a_t, &w_nk_t, None).expect("tr"));
        }
        // Take the *minimum* of many timed runs rather than the mean. This is a shared
        // workstation: a neighbour's load stretches a single run to 2x, but it cannot make
        // a run faster than the hardware allows, so the minimum is the closest estimate of
        // the true cost. Means here vary by up to 2x run to run and cannot resolve a
        // 10-20% kernel change.
        let mut best_tr = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            black_box(matmul_transposed_b(&a_t, &w_nk_t, None).expect("tr"));
            best_tr = best_tr.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        let ms_tr = best_tr;

        // matrixmultiply, [k, n] layout, split across columns with plain `sgemm`.
        // (`parallel_sgemm` lives behind the crate's unenabled `threading` feature; the
        // engine already drives `sgemm` itself for the 1x1 conv path, same pattern.)
        let mut out = vec![0.0f32; m * n];
        let out_ptr = out.as_mut_ptr();
        let threads = rayon::current_num_threads();
        let chunk = n.div_ceil(threads).clamp(1, 2048);
        let nchunks = n.div_ceil(chunk);

        // Column spans are disjoint, so the shared write pointer is sound; a `*mut f32`
        // round-trip keeps the probe short (the engine uses an audited `Send` newtype).
        let out_addr = out_ptr as usize;

        let mm_once = move || {
            (0..nchunks).into_par_iter().for_each(|c| {
                let col_start = c * chunk;
                let cn = (n - col_start).min(chunk);
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        cn,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        w_kn.as_ptr().add(col_start),
                        n as isize,
                        1,
                        0.0,
                        (out_addr as *mut f32).add(col_start),
                        n as isize,
                        1,
                    )
                };
            });
            nchunks
        };

        for _ in 0..3 {
            black_box(mm_once());
        }
        let mut best_mm = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            black_box(mm_once());
            best_mm = best_mm.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        let ms_mm = best_mm;

        println!(
            "m={m:<4} hand_rolled[n,k] {ms_tr:7.3} ms ({:5.1} GFLOP/s)   \
             matrixmultiply[k,n] {ms_mm:7.3} ms ({:5.1} GFLOP/s)   speedup {:.2}x",
            flops / (ms_tr * 1e-3) / 1e9,
            flops / (ms_mm * 1e-3) / 1e9,
            ms_tr / ms_mm
        );
    }
}
