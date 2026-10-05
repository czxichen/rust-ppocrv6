use crate::error::{OcrError, Result};
use crate::ops::conv2d::{ConvActivation, apply_conv_activation};
use crate::tensor::Tensor;
use rayon::prelude::*;

/// Pre-transforms 3x3 convolution weights into Winograd F(2x2, 3x3) domain.
/// Input weight shape: [cout, cin, 3, 3]
/// Returns 16 tensors, each of shape [cout, cin] for frequency point xi in 0..16.
#[allow(clippy::needless_range_loop)]
pub fn pretransform_weights_f23(weights: &Tensor) -> Result<Vec<Tensor>> {
    if weights.ndim() != 4 || weights.shape()[2] != 3 || weights.shape()[3] != 3 {
        return Err(OcrError::OperatorError {
            op: "WinogradPretransform".into(),
            msg: format!(
                "Expected weight of shape [cout, cin, 3, 3], got {:?}",
                weights.shape()
            ),
        });
    }

    let w_shape = weights.shape();
    let cout = w_shape[0];
    let cin = w_shape[1];
    let w_cont_holder;
    let w_slice = if weights.is_contiguous() {
        weights.as_slice()
    } else {
        w_cont_holder = weights.to_contiguous();
        w_cont_holder.as_slice()
    };

    let mut transformed = vec![vec![0.0f32; cout * cin]; 16];

    for oc in 0..cout {
        let oc_off = oc * cin * 9;
        for ic in 0..cin {
            let filter_off = oc_off + ic * 9;
            let mut g = [[0.0f32; 3]; 3];
            for kh in 0..3 {
                for kw in 0..3 {
                    g[kh][kw] = w_slice[filter_off + kh * 3 + kw];
                }
            }

            let u = transform_filter_3x3(&g);
            let idx = oc * cin + ic;
            for r in 0..4 {
                for c in 0..4 {
                    let xi = r * 4 + c;
                    transformed[xi][idx] = u[r][c];
                }
            }
        }
    }

    let mut result = Vec::with_capacity(16);
    for t in transformed {
        result.push(Tensor::from_vec(&[cout, cin], t)?);
    }

    Ok(result)
}

#[inline(always)]
fn transform_filter_3x3(g: &[[f32; 3]; 3]) -> [[f32; 4]; 4] {
    let mut gg = [[0.0f32; 3]; 4];
    for j in 0..3 {
        gg[0][j] = g[0][j];
        gg[1][j] = 0.5 * (g[0][j] + g[1][j] + g[2][j]);
        gg[2][j] = 0.5 * (g[0][j] - g[1][j] + g[2][j]);
        gg[3][j] = g[2][j];
    }
    let mut u = [[0.0f32; 4]; 4];
    for i in 0..4 {
        u[i][0] = gg[i][0];
        u[i][1] = 0.5 * (gg[i][0] + gg[i][1] + gg[i][2]);
        u[i][2] = 0.5 * (gg[i][0] - gg[i][1] + gg[i][2]);
        u[i][3] = gg[i][2];
    }
    u
}

#[inline(always)]
fn transform_input_4x4(d: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut bt_d = [[0.0f32; 4]; 4];
    for j in 0..4 {
        bt_d[0][j] = d[0][j] - d[2][j];
        bt_d[1][j] = d[1][j] + d[2][j];
        bt_d[2][j] = d[2][j] - d[1][j];
        bt_d[3][j] = d[1][j] - d[3][j];
    }
    let mut v = [[0.0f32; 4]; 4];
    for i in 0..4 {
        v[i][0] = bt_d[i][0] - bt_d[i][2];
        v[i][1] = bt_d[i][1] + bt_d[i][2];
        v[i][2] = bt_d[i][2] - bt_d[i][1];
        v[i][3] = bt_d[i][1] - bt_d[i][3];
    }
    v
}

#[inline(always)]
fn transform_output_4x4(m: &[[f32; 4]; 4]) -> [[f32; 2]; 2] {
    let mut at_m = [[0.0f32; 4]; 2];
    for j in 0..4 {
        at_m[0][j] = m[0][j] + m[1][j] + m[2][j];
        at_m[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    let mut y = [[0.0f32; 2]; 2];
    for i in 0..2 {
        y[i][0] = at_m[i][0] + at_m[i][1] + at_m[i][2];
        y[i][1] = at_m[i][1] - at_m[i][2] - at_m[i][3];
    }
    y
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_act_avx2(
    val: std::arch::x86_64::__m256,
    act: ConvActivation,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        match act {
            ConvActivation::None => val,
            ConvActivation::Relu => _mm256_max_ps(val, _mm256_setzero_ps()),
            ConvActivation::HardSigmoid { alpha, beta } => {
                let zero = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);
                let a = _mm256_set1_ps(alpha);
                let b = _mm256_set1_ps(beta);
                let scaled = _mm256_add_ps(_mm256_mul_ps(val, a), b);
                _mm256_min_ps(_mm256_max_ps(scaled, zero), one)
            }
            ConvActivation::HardSwish => {
                let zero = _mm256_setzero_ps();
                let six = _mm256_set1_ps(6.0);
                let three = _mm256_set1_ps(3.0);
                let one_sixth = _mm256_set1_ps(1.0 / 6.0);
                let clamped = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(val, three), zero), six);
                _mm256_mul_ps(_mm256_mul_ps(val, clamped), one_sixth)
            }
            ConvActivation::Gelu => {
                let mut arr = [0.0f32; 8];
                _mm256_storeu_ps(arr.as_mut_ptr(), val);
                for x in &mut arr {
                    *x = apply_conv_activation(*x, act);
                }
                _mm256_loadu_ps(arr.as_ptr())
            }
        }
    }
}

/// Executes Winograd F(2x2, 3x3) convolution for stride 1, padding 1.
#[allow(clippy::needless_range_loop)]
pub fn winograd_f23_conv2d(
    input: &Tensor,
    u_weights: &[Tensor], // 16 pre-transformed weight matrices [cout, cin]
    bias: Option<&Tensor>,
    activation: ConvActivation,
) -> Result<Tensor> {
    let in_shape = input.shape();
    let batch_size = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let cout = u_weights[0].shape()[0];
    let out_h = in_h;
    let out_w = in_w;

    let tiles_h = out_h.div_ceil(2);
    let tiles_w = out_w.div_ceil(2);
    let num_tiles = tiles_h * tiles_w;

    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };

    let bias_cont_holder;
    let bias_slice: Option<&[f32]> = match bias {
        Some(b) => {
            if b.is_contiguous() {
                Some(b.as_slice())
            } else {
                bias_cont_holder = b.to_contiguous();
                Some(bias_cont_holder.as_slice())
            }
        }
        None => None,
    };

    let mut output = Tensor::empty(&[batch_size, cout, out_h, out_w]);
    let out_slice = output.as_mut_slice();

    for b in 0..batch_size {
        let in_b_off = b * in_channels * in_h * in_w;
        let out_b_off = b * cout * out_h * out_w;

        // 1. Directly transform input tiles into V_xi[16, cin, num_tiles]
        let v_xi_len = 16 * in_channels * num_tiles;
        let mut v_xi = crate::tensor::acquire_buffer(v_xi_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            v_xi.set_len(v_xi_len);
        }
        let v_xi_ptr_usize = v_xi.as_mut_ptr() as usize;
        let in_hw = in_h * in_w;

        // Parallelize input transformation over channels
        let run_ic = |ic: usize| {
            let in_ic_off = in_b_off + ic * in_hw;
            let in_channel_data = &in_slice[in_ic_off..in_ic_off + in_hw];
            let v_xi_p = v_xi_ptr_usize as *mut f32;

            let mut xi_ptrs = [std::ptr::null_mut(); 16];
            for xi in 0..16 {
                xi_ptrs[xi] = unsafe { v_xi_p.add(xi * in_channels * num_tiles + ic * num_tiles) };
            }

            for th in 0..tiles_h {
                let ih_base = (th * 2) as isize - 1;
                for tw in 0..tiles_w {
                    let iw_base = (tw * 2) as isize - 1;
                    let tile_idx = th * tiles_w + tw;

                    let is_interior = ih_base >= 0
                        && (ih_base + 3) < in_h as isize
                        && iw_base >= 0
                        && (iw_base + 3) < in_w as isize;

                    let v: [[f32; 4]; 4] = if is_interior {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            use std::arch::x86_64::*;
                            let p0 = in_channel_data
                                .as_ptr()
                                .add(ih_base as usize * in_w + iw_base as usize);
                            let p1 = p0.add(in_w);
                            let p2 = p1.add(in_w);
                            let p3 = p2.add(in_w);

                            let r0 = _mm_loadu_ps(p0);
                            let r1 = _mm_loadu_ps(p1);
                            let r2 = _mm_loadu_ps(p2);
                            let r3 = _mm_loadu_ps(p3);

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
                            res
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let mut d = [[0.0f32; 4]; 4];
                            for r in 0..4 {
                                let row_off = (ih_base as usize + r) * in_w + iw_base as usize;
                                d[r].copy_from_slice(&in_channel_data[row_off..row_off + 4]);
                            }
                            transform_input_4x4(&d)
                        }
                    } else {
                        let mut d = [[0.0f32; 4]; 4];
                        for r in 0..4 {
                            let ih = ih_base + r as isize;
                            if ih >= 0 && (ih as usize) < in_h {
                                let row_off = ih as usize * in_w;
                                for c in 0..4 {
                                    let iw = iw_base + c as isize;
                                    if iw >= 0 && (iw as usize) < in_w {
                                        d[r][c] = in_channel_data[row_off + iw as usize];
                                    }
                                }
                            }
                        }
                        transform_input_4x4(&d)
                    };

                    for r in 0..4 {
                        for c in 0..4 {
                            let xi = r * 4 + c;
                            unsafe {
                                *xi_ptrs[xi].add(tile_idx) = v[r][c];
                            }
                        }
                    }
                }
            }
        };

        if crate::tensor::should_parallelize(in_channels, 8) {
            (0..in_channels).into_par_iter().for_each(run_ic);
        } else {
            (0..in_channels).for_each(run_ic);
        }

        // 2. Parallel GEMM for all 16 frequency points: M[16, cout, num_tiles] = U[16, cout, cin] * V[16, cin, num_tiles]
        let m_buf_len = 16 * cout * num_tiles;
        let mut m_buf = crate::tensor::acquire_buffer(m_buf_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            m_buf.set_len(m_buf_len);
        }

        // Run 16 GEMMs concurrently with Rayon
        let v_xi_slice = v_xi.as_slice();
        let run_gemm = |(xi, m_xi_chunk): (usize, &mut [f32])| {
            let u_xi = u_weights[xi].as_slice();
            let v_xi_off = xi * in_channels * num_tiles;

            unsafe {
                crate::ops::gemm_row_major(
                    cout,
                    in_channels,
                    num_tiles,
                    u_xi.as_ptr(),
                    in_channels,
                    v_xi_slice.as_ptr().add(v_xi_off),
                    num_tiles,
                    m_xi_chunk.as_mut_ptr(),
                    num_tiles,
                );
            }
        };

        let gemm_flops = 16 * cout * in_channels * num_tiles;
        if crate::tensor::should_parallelize(gemm_flops, 32_768) {
            m_buf
                .par_chunks_mut(cout * num_tiles)
                .enumerate()
                .for_each(run_gemm);
        } else {
            m_buf
                .chunks_mut(cout * num_tiles)
                .enumerate()
                .for_each(run_gemm);
        }

        // 3. Transform output tiles and apply bias + activation
        let m_slice = m_buf.as_slice();
        let out_hw = out_h * out_w;
        let out_batch_slice = &mut out_slice[out_b_off..out_b_off + cout * out_hw];

        let run_out_tile = |(oc, out_channel_slice): (usize, &mut [f32])| {
            let b_val = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);

            let mut m_xi_ptrs = [std::ptr::null::<f32>(); 16];
            for xi in 0..16 {
                m_xi_ptrs[xi] =
                    unsafe { m_slice.as_ptr().add(xi * cout * num_tiles + oc * num_tiles) };
            }

            for th in 0..tiles_h {
                let oh0 = th * 2;
                let oh1 = oh0 + 1;
                let mut tw = 0;

                #[cfg(target_arch = "x86_64")]
                if crate::ops::avx2_fma_enabled() && oh1 < out_h {
                    use std::arch::x86_64::*;
                    let bias_v256 = unsafe { _mm256_set1_ps(b_val) };

                    while tw + 4 <= tiles_w && (tw + 4) * 2 <= out_w {
                        let tile_idx = th * tiles_w + tw;
                        unsafe {
                            let m00 = _mm_loadu_ps(m_xi_ptrs[0].add(tile_idx));
                            let m01 = _mm_loadu_ps(m_xi_ptrs[1].add(tile_idx));
                            let m02 = _mm_loadu_ps(m_xi_ptrs[2].add(tile_idx));
                            let m03 = _mm_loadu_ps(m_xi_ptrs[3].add(tile_idx));
                            let m10 = _mm_loadu_ps(m_xi_ptrs[4].add(tile_idx));
                            let m11 = _mm_loadu_ps(m_xi_ptrs[5].add(tile_idx));
                            let m12 = _mm_loadu_ps(m_xi_ptrs[6].add(tile_idx));
                            let m13 = _mm_loadu_ps(m_xi_ptrs[7].add(tile_idx));
                            let m20 = _mm_loadu_ps(m_xi_ptrs[8].add(tile_idx));
                            let m21 = _mm_loadu_ps(m_xi_ptrs[9].add(tile_idx));
                            let m22 = _mm_loadu_ps(m_xi_ptrs[10].add(tile_idx));
                            let m23 = _mm_loadu_ps(m_xi_ptrs[11].add(tile_idx));
                            let m30 = _mm_loadu_ps(m_xi_ptrs[12].add(tile_idx));
                            let m31 = _mm_loadu_ps(m_xi_ptrs[13].add(tile_idx));
                            let m32 = _mm_loadu_ps(m_xi_ptrs[14].add(tile_idx));
                            let m33 = _mm_loadu_ps(m_xi_ptrs[15].add(tile_idx));

                            let at_m00 = _mm_add_ps(_mm_add_ps(m00, m10), m20);
                            let at_m10 = _mm_sub_ps(_mm_sub_ps(m10, m20), m30);

                            let at_m01 = _mm_add_ps(_mm_add_ps(m01, m11), m21);
                            let at_m11 = _mm_sub_ps(_mm_sub_ps(m11, m21), m31);

                            let at_m02 = _mm_add_ps(_mm_add_ps(m02, m12), m22);
                            let at_m12 = _mm_sub_ps(_mm_sub_ps(m12, m22), m32);

                            let at_m03 = _mm_add_ps(_mm_add_ps(m03, m13), m23);
                            let at_m13 = _mm_sub_ps(_mm_sub_ps(m13, m23), m33);

                            let y00 = _mm_add_ps(_mm_add_ps(at_m00, at_m01), at_m02);
                            let y01 = _mm_sub_ps(_mm_sub_ps(at_m01, at_m02), at_m03);

                            let y10 = _mm_add_ps(_mm_add_ps(at_m10, at_m11), at_m12);
                            let y11 = _mm_sub_ps(_mm_sub_ps(at_m11, at_m12), at_m13);

                            let lo0 = _mm_unpacklo_ps(y00, y01);
                            let hi0 = _mm_unpackhi_ps(y00, y01);
                            let mut r0 = _mm256_set_m128(hi0, lo0);

                            let lo1 = _mm_unpacklo_ps(y10, y11);
                            let hi1 = _mm_unpackhi_ps(y10, y11);
                            let mut r1 = _mm256_set_m128(hi1, lo1);

                            r0 = _mm256_add_ps(r0, bias_v256);
                            r1 = _mm256_add_ps(r1, bias_v256);

                            r0 = apply_act_avx2(r0, activation);
                            r1 = apply_act_avx2(r1, activation);

                            let ow = tw * 2;
                            _mm256_storeu_ps(
                                out_channel_slice.as_mut_ptr().add(oh0 * out_w + ow),
                                r0,
                            );
                            _mm256_storeu_ps(
                                out_channel_slice.as_mut_ptr().add(oh1 * out_w + ow),
                                r1,
                            );
                        }
                        tw += 4;
                    }
                }

                while tw < tiles_w {
                    let tile_idx = th * tiles_w + tw;
                    let mut m = [[0.0f32; 4]; 4];

                    for r in 0..4 {
                        for c in 0..4 {
                            let xi = r * 4 + c;
                            m[r][c] = unsafe { *m_xi_ptrs[xi].add(tile_idx) };
                        }
                    }

                    let y = transform_output_4x4(&m);

                    for r in 0..2 {
                        let oh = th * 2 + r;
                        if oh < out_h {
                            let row_off = oh * out_w;
                            for c in 0..2 {
                                let ow = tw * 2 + c;
                                if ow < out_w {
                                    let val = y[r][c] + b_val;
                                    out_channel_slice[row_off + ow] =
                                        apply_conv_activation(val, activation);
                                }
                            }
                        }
                    }
                    tw += 1;
                }
            }
        };

        if crate::tensor::should_parallelize(cout, 8) {
            out_batch_slice
                .par_chunks_mut(out_hw)
                .enumerate()
                .for_each(run_out_tile);
        } else {
            out_batch_slice
                .chunks_mut(out_hw)
                .enumerate()
                .for_each(run_out_tile);
        }

        crate::tensor::release_buffer(v_xi);
        crate::tensor::release_buffer(m_buf);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::conv2d::{Conv2dParams, conv2d};

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
    }

    /// Winograd F(2x2, 3x3) is a different (and numerically sloppier) factorisation of the
    /// same convolution, so it is validated against the direct kernel rather than against
    /// hand-computed values.
    fn check_against_direct(in_h: usize, in_w: usize, activation: ConvActivation, ctx: &str) {
        let (cin, cout) = (16usize, 8usize);
        let input: Vec<f32> = (0..cin * in_h * in_w)
            .map(|i| ((i as f32) * 0.061).sin() * 2.5)
            .collect();
        let weight: Vec<f32> = (0..cout * cin * 9)
            .map(|i| ((i as f32) * 0.023).cos() * 0.4)
            .collect();
        let bias: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.07 - 0.2).collect();

        let x = tensor(&[1, cin, in_h, in_w], &input);
        let w = tensor(&[cout, cin, 3, 3], &weight);
        let b = tensor(&[cout], &bias);

        let direct = conv2d(
            &x,
            &w,
            Some(&b),
            &Conv2dParams {
                pads: [1, 1, 1, 1],
                activation,
                ..Default::default()
            },
        )
        .expect("direct conv2d");

        let u = pretransform_weights_f23(&w).expect("pretransform");
        assert_eq!(u.len(), 16, "{ctx}: expected 16 frequency matrices");
        assert_eq!(u[0].shape(), &[cout, cin]);
        let wx = x.clone();
        let wino = winograd_f23_conv2d(&wx, &u, Some(&b), activation).expect("winograd");

        assert_eq!(wino.shape(), direct.shape(), "{ctx}: shape");
        let mut worst = 0.0f64;
        for (&a, &d) in wino.as_slice().iter().zip(direct.as_slice()) {
            let denom = d.abs().max(1.0) as f64;
            worst = worst.max((a - d).abs() as f64 / denom);
        }
        assert!(
            worst < 5e-3,
            "{ctx}: winograd diverges from the direct kernel (worst rel err {worst})"
        );
    }

    #[test]
    fn winograd_matches_direct_conv_on_even_spatial_dims() {
        check_against_direct(8, 8, ConvActivation::None, "8x8 no activation");
    }

    #[test]
    fn winograd_matches_direct_conv_on_odd_spatial_dims() {
        // Odd sizes exercise the clipped last tile row/column.
        check_against_direct(7, 9, ConvActivation::None, "7x9 no activation");
    }

    #[test]
    fn winograd_matches_direct_conv_with_bias_and_activation() {
        check_against_direct(8, 6, ConvActivation::Relu, "8x6 relu");
        check_against_direct(6, 6, ConvActivation::HardSwish, "6x6 hardswish");
    }

    #[test]
    fn winograd_handles_a_single_tile() {
        // 2x2 input: exactly one F(2x2,3x3) tile, so no tiling arithmetic is exercised.
        check_against_direct(2, 2, ConvActivation::None, "2x2 single tile");
    }

    #[test]
    fn pretransform_rejects_non_3x3_weights() {
        let w = tensor(&[4, 4, 1, 1], &[0.0; 16]);
        assert!(pretransform_weights_f23(&w).is_err());
        let w3 = tensor(&[4, 4, 3, 2], &vec![0.0; 96]);
        assert!(pretransform_weights_f23(&w3).is_err());
    }
}
