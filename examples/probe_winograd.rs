use ppocr::Tensor;
use ppocr::ops::conv2d::ConvActivation;
use ppocr::ops::winograd::{pretransform_weights_f23, winograd_f23_conv2d};
use std::time::Instant;

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    let cout = 24;
    let cin = 96;
    let in_h = 240;
    let in_w = 176;

    // Det Conv.82: [24, 96, 3, 3], in=[1, 96, 240, 176]
    let input = Tensor::zeros(&[1, cin, in_h, in_w]);
    let weight = Tensor::zeros(&[cout, cin, 3, 3]);
    let bias = Tensor::zeros(&[cout]);
    let u_weights = pretransform_weights_f23(&weight).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = winograd_f23_conv2d(&input, &u_weights, Some(&bias), ConvActivation::HardSwish).unwrap();
    }

    let mut best = f64::INFINITY;
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = winograd_f23_conv2d(&input, &u_weights, Some(&bias), ConvActivation::HardSwish).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best { best = ms; }
    }
    println!("Conv.82 Current Winograd total: {:.3} ms", best);

    // Test spatially chunked Winograd
    let tiles_h = in_h / 2; // 120
    let tiles_w = in_w / 2; // 88
    let chunk_th = 10; // 12 chunks of 10 tile rows
    let num_chunks = tiles_h.div_ceil(chunk_th);

    let mut out_test = ppocr::Tensor::empty(&[1, cout, in_h, in_w]);
    let out_slice_ptr = out_test.as_mut_slice().as_mut_ptr() as usize;
    let in_slice_ptr = input.as_slice().as_ptr() as usize;

    let run_chunked_winograd = || {
        use rayon::prelude::*;
        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let th_start = chunk_idx * chunk_th;
            let th_end = (th_start + chunk_th).min(tiles_h);
            let actual_th = th_end - th_start;
            let chunk_tiles = actual_th * tiles_w;

            let mut v_chunk = vec![0.0f32; 16 * cin * chunk_tiles];
            let mut m_chunk = vec![0.0f32; 16 * cout * chunk_tiles];

            let in_p = in_slice_ptr as *const f32;
            let out_p = out_slice_ptr as *mut f32;

            // 1. Input transform for this chunk's tiles
            for ic in 0..cin {
                let in_ic_p = unsafe { in_p.add(ic * in_h * in_w) };
                for th_rel in 0..actual_th {
                    let th = th_start + th_rel;
                    let ih_base = (th * 2) as isize - 1;
                    for tw in 0..tiles_w {
                        let iw_base = (tw * 2) as isize - 1;
                        let tile_idx = th_rel * tiles_w + tw;
                        unsafe {
                            use std::arch::x86_64::*;
                            if ih_base >= 0 && (ih_base + 3) < in_h as isize && iw_base >= 0 && (iw_base + 3) < in_w as isize {
                                let p0 = in_ic_p.add(ih_base as usize * in_w + iw_base as usize);
                                let r0 = _mm_loadu_ps(p0);
                                let r1 = _mm_loadu_ps(p0.add(in_w));
                                let r2 = _mm_loadu_ps(p0.add(in_w * 2));
                                let r3 = _mm_loadu_ps(p0.add(in_w * 3));
                                let mut bt0 = _mm_sub_ps(r0, r2);
                                let mut bt1 = _mm_add_ps(r1, r2);
                                let mut bt2 = _mm_sub_ps(r2, r1);
                                let mut bt3 = _mm_sub_ps(r1, r3);
                                _MM_TRANSPOSE4_PS(&mut bt0, &mut bt1, &mut bt2, &mut bt3);
                                let mut v0 = _mm_sub_ps(bt0, bt2);
                                let mut v1 = _mm_add_ps(bt1, bt2);
                                let mut v2 = _mm_sub_ps(bt2, bt1);
                                let mut v3 = _mm_sub_ps(bt1, bt3);
                                _MM_TRANSPOSE4_PS(&mut v0, &mut v1, &mut v2, &mut v3);
                                let mut res = [[0.0f32; 4]; 4];
                                _mm_storeu_ps(res[0].as_mut_ptr(), v0);
                                _mm_storeu_ps(res[1].as_mut_ptr(), v1);
                                _mm_storeu_ps(res[2].as_mut_ptr(), v2);
                                _mm_storeu_ps(res[3].as_mut_ptr(), v3);
                                for r in 0..4 {
                                    for c in 0..4 {
                                        let xi = r * 4 + c;
                                        *v_chunk.as_mut_ptr().add(xi * cin * chunk_tiles + ic * chunk_tiles + tile_idx) = res[r][c];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 2. 16 GEMMs for this chunk
            for xi in 0..16 {
                let u_xi = u_weights[xi].as_slice();
                let v_xi_off = xi * cin * chunk_tiles;
                let m_xi_off = xi * cout * chunk_tiles;
                unsafe {
                    ppocr::ops::gemm_row_major(
                        cout,
                        cin,
                        chunk_tiles,
                        u_xi.as_ptr(),
                        cin,
                        v_chunk.as_ptr().add(v_xi_off),
                        chunk_tiles,
                        m_chunk.as_mut_ptr().add(m_xi_off),
                        chunk_tiles,
                    );
                }
            }

            // 3. Output transform for this chunk
            for oc in 0..cout {
                for th_rel in 0..actual_th {
                    let th = th_start + th_rel;
                    let oh0 = th * 2;
                    let oh1 = oh0 + 1;
                    if oh1 >= in_h { continue; }
                    let mut tw = 0;
                    use std::arch::x86_64::*;
                    while tw + 4 <= tiles_w && (tw + 4) * 2 <= in_w {
                        let tile_idx = th_rel * tiles_w + tw;
                        unsafe {
                            let mut m = [[_mm_setzero_ps(); 4]; 4];
                            for r in 0..4 {
                                for c in 0..4 {
                                    let xi = r * 4 + c;
                                    m[r][c] = _mm_loadu_ps(m_chunk.as_ptr().add(xi * cout * chunk_tiles + oc * chunk_tiles + tile_idx));
                                }
                            }
                            let at_m00 = _mm_add_ps(_mm_add_ps(m[0][0], m[1][0]), m[2][0]);
                            let at_m10 = _mm_sub_ps(_mm_sub_ps(m[1][0], m[2][0]), m[3][0]);
                            let at_m01 = _mm_add_ps(_mm_add_ps(m[0][1], m[1][1]), m[2][1]);
                            let at_m11 = _mm_sub_ps(_mm_sub_ps(m[1][1], m[2][1]), m[3][1]);
                            let at_m02 = _mm_add_ps(_mm_add_ps(m[0][2], m[1][2]), m[2][2]);
                            let at_m12 = _mm_sub_ps(_mm_sub_ps(m[1][2], m[2][2]), m[3][2]);
                            let at_m03 = _mm_add_ps(_mm_add_ps(m[0][3], m[1][3]), m[2][3]);
                            let at_m13 = _mm_sub_ps(_mm_sub_ps(m[1][3], m[2][3]), m[3][3]);

                            let y00 = _mm_add_ps(_mm_add_ps(at_m00, at_m01), at_m02);
                            let y01 = _mm_sub_ps(_mm_sub_ps(at_m01, at_m02), at_m03);
                            let y10 = _mm_add_ps(_mm_add_ps(at_m10, at_m11), at_m12);
                            let y11 = _mm_sub_ps(_mm_sub_ps(at_m11, at_m12), at_m13);

                            let lo0 = _mm_unpacklo_ps(y00, y01);
                            let hi0 = _mm_unpackhi_ps(y00, y01);
                            let r0 = _mm256_set_m128(hi0, lo0);
                            let lo1 = _mm_unpacklo_ps(y10, y11);
                            let hi1 = _mm_unpackhi_ps(y10, y11);
                            let r1 = _mm256_set_m128(hi1, lo1);

                            // store
                            let ow = tw * 2;
                            _mm256_storeu_ps(out_p.add(oc * in_h * in_w + oh0 * in_w + ow), r0);
                            _mm256_storeu_ps(out_p.add(oc * in_h * in_w + oh1 * in_w + ow), r1);
                        }
                        tw += 4;
                    }
                }
            }
        });
    };

    for _ in 0..3 { run_chunked_winograd(); }
    let mut best_chunked = f64::INFINITY;
    for _ in 0..10 {
        let t0 = Instant::now();
        run_chunked_winograd();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best_chunked { best_chunked = ms; }
    }
    println!("Conv.82 Chunked Winograd total: {:.3} ms | Speedup vs Global: {:.2}x",
        best_chunked, best / best_chunked);
}
