use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, MatMut, MatRef, Par};
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    println!("Threads: {}", rayon::current_num_threads());

    let shapes = [
        (768, 384, 720, "Rec Conv.49/52 (768x384, n=720)"),
        (384, 192, 1440, "Rec Conv.22/30 (384x192, n=1440)"),
        (192, 96, 2880, "Rec Conv.8/11/14 (192x96, n=2880)"),
        (96, 48, 42240, "Det Conv.11 (96x48, n=42240)"),
        (48, 96, 42240, "Det Conv.9/12 (48x96, n=42240)"),
        (24, 96, 42240, "Det Conv.79 (24x96, n=42240)"),
        (48, 24, 42240, "Det Conv.4 (48x24, n=42240)"),
        (192, 96, 10560, "Det Conv.19/22 (192x96, n=10560)"),
        (96, 192, 10560, "Det Conv.20/23 (96x192, n=10560)"),
        (384, 192, 2640, "Det Conv.38/41 (384x192, n=2640)"),
    ];

    for &(m, k, n, label) in &shapes {
        let a = vec![0.01f32; m * k];
        let b = vec![0.02f32; k * n];

        // Strategy A: split n
        let mut c_a = vec![0.0f32; m * n];
        let chunk_n = n.div_ceil(threads).max(64).min(n);
        let num_col_chunks = n.div_ceil(chunk_n);

        let c_a_ptr = c_a.as_mut_ptr() as usize;

        // Warmup
        for _ in 0..3 {
            (0..num_col_chunks).into_par_iter().for_each(|chunk_idx| {
                let col_start = chunk_idx * chunk_n;
                let actual_n = (n - col_start).min(chunk_n);
                unsafe {
                    let a_t = MatRef::from_raw_parts(a.as_ptr(), k, m, 1isize, k as isize);
                    let b_t = MatRef::from_raw_parts(b.as_ptr().add(col_start), actual_n, k, 1isize, n as isize);
                    let mut c_t = MatMut::from_raw_parts_mut((c_a_ptr as *mut f32).add(col_start), actual_n, m, 1isize, n as isize);
                    faer_matmul(&mut c_t, Accum::Replace, &b_t, &a_t, 1.0, Par::Seq);
                }
            });
        }

        let mut t_a_best = f64::INFINITY;
        for _ in 0..20 {
            let t0 = Instant::now();
            (0..num_col_chunks).into_par_iter().for_each(|chunk_idx| {
                let col_start = chunk_idx * chunk_n;
                let actual_n = (n - col_start).min(chunk_n);
                unsafe {
                    let a_t = MatRef::from_raw_parts(a.as_ptr(), k, m, 1isize, k as isize);
                    let b_t = MatRef::from_raw_parts(b.as_ptr().add(col_start), actual_n, k, 1isize, n as isize);
                    let mut c_t = MatMut::from_raw_parts_mut((c_a_ptr as *mut f32).add(col_start), actual_n, m, 1isize, n as isize);
                    faer_matmul(&mut c_t, Accum::Replace, &b_t, &a_t, 1.0, Par::Seq);
                }
            });
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            if elapsed < t_a_best { t_a_best = elapsed; }
        }

        // Strategy B: split m
        let mut c_b = vec![0.0f32; m * n];
        let chunk_m = m.div_ceil(threads).max(4).min(m);
        let num_row_chunks = m.div_ceil(chunk_m);
        let c_b_ptr = c_b.as_mut_ptr() as usize;

        for _ in 0..3 {
            (0..num_row_chunks).into_par_iter().for_each(|chunk_idx| {
                let oc_start = chunk_idx * chunk_m;
                let actual_m = (m - oc_start).min(chunk_m);
                let a_offset = oc_start * k;
                unsafe {
                    let a_t = MatRef::from_raw_parts(a.as_ptr().add(a_offset), k, actual_m, 1isize, k as isize);
                    let b_t = MatRef::from_raw_parts(b.as_ptr(), n, k, 1isize, n as isize);
                    let mut c_t = MatMut::from_raw_parts_mut((c_b_ptr as *mut f32).add(oc_start * n), n, actual_m, 1isize, n as isize);
                    faer_matmul(&mut c_t, Accum::Replace, &b_t, &a_t, 1.0, Par::Seq);
                }
            });
        }

        let mut t_b_best = f64::INFINITY;
        for _ in 0..20 {
            let t0 = Instant::now();
            (0..num_row_chunks).into_par_iter().for_each(|chunk_idx| {
                let oc_start = chunk_idx * chunk_m;
                let actual_m = (m - oc_start).min(chunk_m);
                let a_offset = oc_start * k;
                unsafe {
                    let a_t = MatRef::from_raw_parts(a.as_ptr().add(a_offset), k, actual_m, 1isize, k as isize);
                    let b_t = MatRef::from_raw_parts(b.as_ptr(), n, k, 1isize, n as isize);
                    let mut c_t = MatMut::from_raw_parts_mut((c_b_ptr as *mut f32).add(oc_start * n), n, actual_m, 1isize, n as isize);
                    faer_matmul(&mut c_t, Accum::Replace, &b_t, &a_t, 1.0, Par::Seq);
                }
            });
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            if elapsed < t_b_best { t_b_best = elapsed; }
        }

        let diff: f32 = c_a.iter().zip(c_b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        println!(
            "{:<35}: Strat A (split N): {:6.3} ms | Strat B (split M): {:6.3} ms | Speedup B/A: {:5.2}x (diff: {:.1e})",
            label, t_a_best, t_b_best, t_a_best / t_b_best, diff
        );
    }

    // Now test actual ppocr::ops::conv2d on Det Conv.11
    {
        let input = ppocr::Tensor::zeros(&[1, 48, 240, 176]);
        let weight = ppocr::Tensor::zeros(&[96, 48, 1, 1]);
        let bias = ppocr::Tensor::zeros(&[96]);
        let params = ppocr::ops::conv2d::Conv2dParams {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
            group: 1,
            activation: ppocr::ops::conv2d::ConvActivation::HardSwish,
        };

        for _ in 0..3 {
            let _ = ppocr::ops::conv2d::conv2d(&input, &weight, Some(&bias), &params);
        }

        let mut t_conv_best = f64::INFINITY;
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = ppocr::ops::conv2d::conv2d(&input, &weight, Some(&bias), &params);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            if elapsed < t_conv_best { t_conv_best = elapsed; }
        }
        println!("ppocr::ops::conv2d on Det Conv.11: {:6.3} ms", t_conv_best);
    }
}
