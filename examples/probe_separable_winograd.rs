use std::time::Instant;

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    let cin = 96;
    let in_h = 240;
    let in_w = 176;
    let tiles_h = in_h / 2;
    let tiles_w = in_w / 2;
    let num_tiles = tiles_h * tiles_w;

    let in_data = vec![0.02f32; cin * in_h * in_w];
    let mut v_xi_old = vec![0.0f32; 16 * cin * num_tiles];
    let mut v_xi_new = vec![0.0f32; 16 * cin * num_tiles];

    // Benchmark Old Stage 1
    let t0 = Instant::now();
    for _ in 0..10 {
        use rayon::prelude::*;
        let in_slice = in_data.as_slice();
        let v_ptr_usize = v_xi_old.as_mut_ptr() as usize;
        (0..cin).into_par_iter().for_each(|ic| {
            let in_ic_p = unsafe { in_slice.as_ptr().add(ic * in_h * in_w) };
            let v_p = v_ptr_usize as *mut f32;
            let mut xi_ptrs = [std::ptr::null_mut(); 16];
            for xi in 0..16 {
                xi_ptrs[xi] = unsafe { v_p.add(xi * cin * num_tiles + ic * num_tiles) };
            }
            for th in 0..tiles_h {
                let ih_base = (th * 2) as isize - 1;
                for tw in 0..tiles_w {
                    let iw_base = (tw * 2) as isize - 1;
                    let tile_idx = th * tiles_w + tw;
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
                                    *xi_ptrs[r * 4 + c].add(tile_idx) = res[r][c];
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    let t_old = t0.elapsed().as_secs_f64() * 100.0;

    // Benchmark New Separable AVX2 Stage 1
    let t0 = Instant::now();
    for _ in 0..10 {
        use rayon::prelude::*;
        let in_slice = in_data.as_slice();
        let v_ptr_usize = v_xi_new.as_mut_ptr() as usize;
        (0..cin).into_par_iter().for_each(|ic| {
            let in_ic_p = unsafe { in_slice.as_ptr().add(ic * in_h * in_w) };
            let v_p = v_ptr_usize as *mut f32;
            let mut xi_ptrs = [std::ptr::null_mut(); 16];
            for xi in 0..16 {
                xi_ptrs[xi] = unsafe { v_p.add(xi * cin * num_tiles + ic * num_tiles) };
            }

            // Stack buffer of width in_w + 4 (with 1 pad left, 2 pad right)
            let padded_w = in_w + 4;
            let mut t_rows = [0.0f32; 4 * 512]; // fits in L1 cache

            for th in 0..tiles_h {
                let ih_base = (th * 2) as isize - 1;
                // Gather 4 rows with vertical padding
                let get_row_ptr = |r: isize| -> Option<*const f32> {
                    let ih = ih_base + r;
                    if ih >= 0 && (ih as usize) < in_h {
                        Some(unsafe { in_ic_p.add(ih as usize * in_w) })
                    } else {
                        None
                    }
                };
                let r0_p = get_row_ptr(0);
                let r1_p = get_row_ptr(1);
                let r2_p = get_row_ptr(2);
                let r3_p = get_row_ptr(3);

                // Zero out boundary padding
                for r in 0..4 {
                    let base = r * padded_w;
                    t_rows[base] = 0.0;
                    t_rows[base + in_w + 1] = 0.0;
                    t_rows[base + in_w + 2] = 0.0;
                    t_rows[base + in_w + 3] = 0.0;
                }

                // AVX2 vertical transform over the width
                unsafe {
                    use std::arch::x86_64::*;
                    let zero = _mm256_setzero_ps();
                    let t0_p = t_rows.as_mut_ptr().add(1);
                    let t1_p = t_rows.as_mut_ptr().add(padded_w + 1);
                    let t2_p = t_rows.as_mut_ptr().add(2 * padded_w + 1);
                    let t3_p = t_rows.as_mut_ptr().add(3 * padded_w + 1);

                    let mut w = 0;
                    while w + 8 <= in_w {
                        let v0 = r0_p.map(|p| _mm256_loadu_ps(p.add(w))).unwrap_or(zero);
                        let v1 = r1_p.map(|p| _mm256_loadu_ps(p.add(w))).unwrap_or(zero);
                        let v2 = r2_p.map(|p| _mm256_loadu_ps(p.add(w))).unwrap_or(zero);
                        let v3 = r3_p.map(|p| _mm256_loadu_ps(p.add(w))).unwrap_or(zero);

                        let bt0 = _mm256_sub_ps(v0, v2);
                        let bt1 = _mm256_add_ps(v1, v2);
                        let bt2 = _mm256_sub_ps(v2, v1);
                        let bt3 = _mm256_sub_ps(v1, v3);

                        _mm256_storeu_ps(t0_p.add(w), bt0);
                        _mm256_storeu_ps(t1_p.add(w), bt1);
                        _mm256_storeu_ps(t2_p.add(w), bt2);
                        _mm256_storeu_ps(t3_p.add(w), bt3);
                        w += 8;
                    }
                    while w < in_w {
                        let v0 = r0_p.map(|p| *p.add(w)).unwrap_or(0.0);
                        let v1 = r1_p.map(|p| *p.add(w)).unwrap_or(0.0);
                        let v2 = r2_p.map(|p| *p.add(w)).unwrap_or(0.0);
                        let v3 = r3_p.map(|p| *p.add(w)).unwrap_or(0.0);

                        *t0_p.add(w) = v0 - v2;
                        *t1_p.add(w) = v1 + v2;
                        *t2_p.add(w) = v2 - v1;
                        *t3_p.add(w) = v1 - v3;
                        w += 1;
                    }
                }

                // Step 2: horizontal transform per tile row into xi_ptrs
                let row_offset = th * tiles_w;
                unsafe {
                    for r in 0..4 {
                        let t_row = t_rows.as_ptr().add(r * padded_w);
                        let xi_base = r * 4;
                        let p_xi0 = xi_ptrs[xi_base].add(row_offset);
                        let p_xi1 = xi_ptrs[xi_base + 1].add(row_offset);
                        let p_xi2 = xi_ptrs[xi_base + 2].add(row_offset);
                        let p_xi3 = xi_ptrs[xi_base + 3].add(row_offset);

                        for tw in 0..tiles_w {
                            let idx = tw * 2;
                            let c0 = *t_row.add(idx);
                            let c1 = *t_row.add(idx + 1);
                            let c2 = *t_row.add(idx + 2);
                            let c3 = *t_row.add(idx + 3);

                            *p_xi0.add(tw) = c0 - c2;
                            *p_xi1.add(tw) = c1 + c2;
                            *p_xi2.add(tw) = c2 - c1;
                            *p_xi3.add(tw) = c1 - c3;
                        }
                    }
                }
            }
        });
    }
    let t_new = t0.elapsed().as_secs_f64() * 100.0;

    let diff: f32 = v_xi_old.iter().zip(v_xi_new.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("Stage 1: Old: {:.3} ms | New Separable AVX2: {:.3} ms | Speedup: {:.2}x | Diff: {:.1e}",
        t_old, t_new, t_old / t_new, diff);
}
