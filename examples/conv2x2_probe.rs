// Isolated timing for the rec model's two 2x2 convolutions.
//
// The per-node profiler said the direct 2x2 kernel is 0.90x of the im2col path, i.e.
// slower. 23 GFLOP/s is far below what the loop nest should reach, so before concluding
// the kernel is the problem this measures the op directly, with the same alternation the
// rest of the probes use.
//
//   cargo run --release --example conv2x2_probe
use image::DynamicImage;
use ppocr::ops::conv2d;
use ppocr::tensor::Tensor;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    // (out_c, in_c, h, w) as the rec model uses them, from `onnx_ab profile`.
    let cases: &[(usize, usize, usize, usize, &str)] = &[
        (24, 48, 24, 471, "Conv.1 rec 24x471"),
        (48, 24, 24, 471, "Conv.2 rec 24x471"),
        (24, 12, 480, 336, "Conv.1 det 480x336"),
    ];

    for &(cout, cin, h, w, label) in cases {
        let input: Vec<f32> = (0..cin * h * w).map(|i| (i as f32 * 1e-4).sin()).collect();
        let weight: Vec<f32> = (0..cout * cin * 4)
            .map(|i| (i as f32 * 1e-4).cos())
            .collect();
        let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(&[1, cin, h, w], input.clone()).expect("x");
        let k = Tensor::from_vec(&[cout, cin, 2, 2], weight.clone()).expect("k");
        let bb = Tensor::from_vec(&[cout], bias.clone()).expect("b");
        let params = ppocr::ops::Conv2dParams::default();

        let out_h = h - 1;
        let out_w = w - 1;
        let flops = 2.0 * cout as f64 * (cin * 4) as f64 * (out_h * out_w) as f64;
        // The im2col buffer the generic path would build.
        let _im2col_bytes = (4 * cin) * out_h * out_w * 4;

        for _ in 0..3 {
            black_box(conv2d(&x, &k, Some(&bb), &params).expect("conv"));
        }
        let mut best = f64::INFINITY;
        for _ in 0..30 {
            let t = Instant::now();
            black_box(conv2d(&x, &k, Some(&bb), &params).expect("conv"));
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        // Benchmark direct kernel
        let mut out_direct = Tensor::empty(&[1, cout, out_h, out_w]);
        let mut best_direct = f64::INFINITY;

        let mut run_direct = || {
            use rayon::prelude::*;
            let in_p_usize = input.as_ptr() as usize;
            let w_p_usize = weight.as_ptr() as usize;
            let b_p_usize = bias.as_ptr() as usize;
            let out_p_usize = out_direct.as_mut_slice().as_mut_ptr() as usize;
            let in_spatial = h * w;
            let out_spatial = out_h * out_w;

            (0..out_h).into_par_iter().for_each(|oh| {
                let in_p = in_p_usize as *const f32;
                let w_p = w_p_usize as *const f32;
                let b_p = b_p_usize as *const f32;
                let out_p = out_p_usize as *mut f32;
                for oc in 0..cout {
                    let b_val = unsafe { *b_p.add(oc) };
                    let out_row = unsafe { out_p.add(oc * out_spatial + oh * out_w) };

                    // Initialize row with bias
                    let mut ow = 0;
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        use std::arch::x86_64::*;
                        let v_bias = _mm256_set1_ps(b_val);
                        while ow + 8 <= out_w {
                            _mm256_storeu_ps(out_row.add(ow), v_bias);
                            ow += 8;
                        }
                    }
                    while ow < out_w {
                        unsafe { *out_row.add(ow) = b_val; }
                        ow += 1;
                    }

                    // Accumulate each cin
                    for ic in 0..cin {
                        let w_base = (oc * cin + ic) * 4;
                        let r0 = unsafe { in_p.add(ic * in_spatial + oh * w) };
                        let r1 = unsafe { in_p.add(ic * in_spatial + (oh + 1) * w) };

                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            use std::arch::x86_64::*;
                            let w00 = _mm256_set1_ps(*w_p.add(w_base));
                            let w01 = _mm256_set1_ps(*w_p.add(w_base + 1));
                            let w10 = _mm256_set1_ps(*w_p.add(w_base + 2));
                            let w11 = _mm256_set1_ps(*w_p.add(w_base + 3));

                            let mut ow = 0;
                            while ow + 16 <= out_w {
                                let mut acc0 = _mm256_loadu_ps(out_row.add(ow));
                                let mut acc1 = _mm256_loadu_ps(out_row.add(ow + 8));

                                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow)), w00, acc0);
                                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow + 8)), w00, acc1);

                                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow + 1)), w01, acc0);
                                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow + 9)), w01, acc1);

                                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow)), w10, acc0);
                                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow + 8)), w10, acc1);

                                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow + 1)), w11, acc0);
                                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow + 9)), w11, acc1);

                                _mm256_storeu_ps(out_row.add(ow), acc0);
                                _mm256_storeu_ps(out_row.add(ow + 8), acc1);
                                ow += 16;
                            }
                            while ow + 8 <= out_w {
                                let mut acc = _mm256_loadu_ps(out_row.add(ow));
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow)), w00, acc);
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow + 1)), w01, acc);
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow)), w10, acc);
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow + 1)), w11, acc);
                                _mm256_storeu_ps(out_row.add(ow), acc);
                                ow += 8;
                            }
                            while ow < out_w {
                                let v00 = *r0.add(ow);
                                let v01 = *r0.add(ow + 1);
                                let v10 = *r1.add(ow);
                                let v11 = *r1.add(ow + 1);
                                *out_row.add(ow) += v00 * *w_p.add(w_base)
                                    + v01 * *w_p.add(w_base + 1)
                                    + v10 * *w_p.add(w_base + 2)
                                    + v11 * *w_p.add(w_base + 3);
                                ow += 1;
                            }
                        }
                    }
                }
            });
        };

        for _ in 0..3 {
            black_box(run_direct());
        }
        for _ in 0..30 {
            let t = Instant::now();
            black_box(run_direct());
            best_direct = best_direct.min(t.elapsed().as_secs_f64() * 1000.0);
        }

        // Verify numeric equality
        let ref_out = conv2d(&x, &k, Some(&bb), &params).unwrap();
        let max_err = out_direct
            .as_slice()
            .iter()
            .zip(ref_out.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "{:24} cout={:<4} cin={:<4} {}x{} -> {}x{}   im2col: {:8.3} ms ({:5.1} GF/s)  direct: {:8.3} ms ({:5.1} GF/s)  speedup: {:5.2}x  max_err: {:e}",
            label,
            cout,
            cin,
            h,
            w,
            out_h,
            out_w,
            best,
            flops / (best * 1e-3) / 1e9,
            best_direct,
            flops / (best_direct * 1e-3) / 1e9,
            best / best_direct,
            max_err,
        );
    }
    let _ = DynamicImage::new_rgb8(1, 1);
}
