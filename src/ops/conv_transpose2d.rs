use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

pub struct ConvTranspose2dParams {
    pub strides: [usize; 2],
    pub pads: [usize; 4],
    pub dilations: [usize; 2],
    pub group: usize,
    pub activation: crate::ops::ConvActivation,
}

impl Default for ConvTranspose2dParams {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
            group: 1,
            activation: crate::ops::ConvActivation::None,
        }
    }
}

pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvTranspose2dParams,
) -> Result<Tensor> {
    if input.ndim() != 4 || weight.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: "Input and weight must be 4D tensors".into(),
        });
    }

    let input_shape = input.shape();
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let weight_shape = weight.shape();
    // In ONNX ConvTranspose: weight is [C_in, C_out / group, K_h, K_w]
    let weight_in_channels = weight_shape[0];
    let out_channels_per_group = weight_shape[1];
    let k_h = weight_shape[2];
    let k_w = weight_shape[3];

    let group = params.group;
    if group == 0 || in_channels % group != 0 {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: format!("Invalid group {}: in_channels={}", group, in_channels),
        });
    }
    let out_channels = out_channels_per_group * group;

    if weight_in_channels != in_channels {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: format!(
                "Weight in_channels ({}) must match input in_channels ({})",
                weight_in_channels, in_channels
            ),
        });
    }

    // The bias is indexed as `bs[oc]` all over the kernels below with no length check, so a
    // short bias panicked mid-kernel. `conv2d` now validates the same thing; this brings
    // ConvTranspose in line.
    if let Some(b) = bias
        && b.numel() != out_channels
    {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: format!(
                "bias has {} elements but out_channels is {out_channels}",
                b.numel()
            ),
        });
    }

    if in_h == 0 || in_w == 0 {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: format!(
                "Input spatial dimensions must be > 0, got [{}, {}]",
                in_h, in_w
            ),
        });
    }

    let [stride_h, stride_w] = params.strides;
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;
    let [dilation_h, dilation_w] = params.dilations;

    let eff_k_h = dilation_h * (k_h.saturating_sub(1)) + 1;
    let eff_k_w = dilation_w * (k_w.saturating_sub(1)) + 1;

    let unpadded_h = (in_h - 1) * stride_h + eff_k_h;
    let pad_h = pad_top + pad_bottom;
    if unpadded_h < pad_h {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: format!(
                "Output height underflow: unpadded {} < pad {}",
                unpadded_h, pad_h
            ),
        });
    }
    let out_h = unpadded_h - pad_h;

    let unpadded_w = (in_w - 1) * stride_w + eff_k_w;
    let pad_w = pad_left + pad_right;
    if unpadded_w < pad_w {
        return Err(OcrError::OperatorError {
            op: "ConvTranspose2d".into(),
            msg: format!(
                "Output width underflow: unpadded {} < pad {}",
                unpadded_w, pad_w
            ),
        });
    }
    let out_w = unpadded_w - pad_w;

    let mut output = Tensor::empty(&[batch_size, out_channels, out_h, out_w]);

    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let weight_cont_holder;
    let weight_slice = if weight.is_contiguous() {
        weight.as_slice()
    } else {
        weight_cont_holder = weight.to_contiguous();
        weight_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

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

    let in_channels_per_group = in_channels / group;

    let is_2x2_s2 = group == 1
        && k_h == 2
        && k_w == 2
        && stride_h == 2
        && stride_w == 2
        && pad_top == 0
        && pad_left == 0
        && pad_bottom == 0
        && pad_right == 0
        && dilation_h == 1
        && dilation_w == 1;

    if is_2x2_s2 {
        let in_hw = in_h * in_w;
        let act = params.activation;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = crate::ops::avx2_fma_enabled();

        #[cfg(target_arch = "x86_64")]
        let precomputed: Vec<TransposeWeight2x2> = if has_avx2 {
            unsafe { build_transpose_weights_fma(weight_slice, in_channels * out_channels) }
        } else {
            Vec::new()
        };

        let out_ptr_usize = out_slice.as_mut_ptr() as usize;
        let in_ptr_usize = in_slice.as_ptr() as usize;
        let weight_ptr_usize = weight_slice.as_ptr() as usize;

        let run_plane = |b: usize, oc: usize, ih_start: usize, ih_end: usize| {
            let in_b_off = b * in_channels * in_hw;
            let out_b_off = b * out_channels * out_h * out_w;
            let c_bias = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);
            let out_ptr =
                unsafe { (out_ptr_usize as *mut f32).add(out_b_off + oc * out_h * out_w) };
            let in_ptr = in_ptr_usize as *const f32;
            let weight_ptr = weight_ptr_usize as *const f32;

            for ih in ih_start..ih_end {
                let out_row0 = (2 * ih) * out_w;
                let out_row1 = (2 * ih + 1) * out_w;
                let in_row = ih * in_w;

                let mut iw = 0;
                #[cfg(target_arch = "x86_64")]
                if has_avx2 {
                    iw = unsafe {
                        ct2x2s2_row_fma(
                            in_ptr,
                            out_ptr,
                            &precomputed,
                            in_channels,
                            in_hw,
                            out_channels,
                            oc,
                            in_b_off,
                            in_row,
                            iw,
                            in_w,
                            out_row0,
                            out_row1,
                            c_bias,
                            act,
                        )
                    };
                }

                while iw < in_w {
                    let mut s00 = c_bias;
                    let mut s01 = c_bias;
                    let mut s10 = c_bias;
                    let mut s11 = c_bias;

                    for ic in 0..in_channels {
                        let in_val = unsafe { *in_ptr.add(in_b_off + ic * in_hw + in_row + iw) };
                        if in_val != 0.0 {
                            let w_off = (ic * out_channels + oc) * 4;
                            unsafe {
                                let wp = weight_ptr.add(w_off);
                                s00 += in_val * *wp;
                                s01 += in_val * *wp.add(1);
                                s10 += in_val * *wp.add(2);
                                s11 += in_val * *wp.add(3);
                            }
                        }
                    }

                    unsafe {
                        let ow2 = 2 * iw;
                        *out_ptr.add(out_row0 + ow2) = crate::ops::apply_conv_activation(s00, act);
                        *out_ptr.add(out_row0 + ow2 + 1) =
                            crate::ops::apply_conv_activation(s01, act);
                        *out_ptr.add(out_row1 + ow2) = crate::ops::apply_conv_activation(s10, act);
                        *out_ptr.add(out_row1 + ow2 + 1) =
                            crate::ops::apply_conv_activation(s11, act);
                    }
                    iw += 1;
                }
            }
        };

        let total_planes = batch_size * out_channels;
        let total_work = total_planes * in_h * in_w;
        let num_threads = rayon::current_num_threads();
        let parallel = crate::tensor::should_parallelize(total_work, 16_384);

        if parallel && total_planes >= num_threads {
            (0..total_planes).into_par_iter().for_each(|plane_idx| {
                let b = plane_idx / out_channels;
                let oc = plane_idx % out_channels;
                run_plane(b, oc, 0, in_h);
            });
        } else if parallel && in_h >= 4 {
            for b in 0..batch_size {
                for oc in 0..out_channels {
                    let chunk_size = (in_h / num_threads).clamp(4, 32);
                    let num_chunks = in_h.div_ceil(chunk_size);
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let ih_start = chunk_idx * chunk_size;
                        let ih_end = (ih_start + chunk_size).min(in_h);
                        run_plane(b, oc, ih_start, ih_end);
                    });
                }
            }
        } else {
            for b in 0..batch_size {
                for oc in 0..out_channels {
                    run_plane(b, oc, 0, in_h);
                }
            }
        }
        return Ok(output);
    }

    let total_planes = batch_size * group * out_channels_per_group;
    let out_ptr_usize = out_slice.as_mut_ptr() as usize;
    let in_ptr_usize = in_slice.as_ptr() as usize;
    let weight_ptr_usize = weight_slice.as_ptr() as usize;

    let run_plane = |plane_idx: usize| {
        let oc = plane_idx % out_channels_per_group;
        let b_g = plane_idx / out_channels_per_group;
        let g = b_g % group;
        let b = b_g / group;

        let in_batch_offset = b * in_channels * in_h * in_w;
        let out_batch_offset = b * out_channels * out_h * out_w;
        let in_g_offset = in_batch_offset + g * in_channels_per_group * in_h * in_w;
        let out_g_offset = out_batch_offset + g * out_channels_per_group * out_h * out_w;

        let out_c_offset = out_g_offset + oc * out_h * out_w;
        let c_bias = bias_slice
            .map(|bs| bs[g * out_channels_per_group + oc])
            .unwrap_or(0.0);

        let out_ptr = unsafe { (out_ptr_usize as *mut f32).add(out_c_offset) };
        let in_ptr = in_ptr_usize as *const f32;
        let weight_ptr = weight_ptr_usize as *const f32;

        // Initialize with bias
        for i in 0..out_h * out_w {
            unsafe {
                *out_ptr.add(i) = c_bias;
            }
        }

        for ic in 0..in_channels_per_group {
            let in_c_offset = in_g_offset + ic * in_h * in_w;
            let actual_ic = g * in_channels_per_group + ic;
            let w_c_offset = (actual_ic * out_channels_per_group + oc) * k_h * k_w;

            for ih in 0..in_h {
                let base_oh = (ih * stride_h) as isize - pad_top as isize;
                for iw in 0..in_w {
                    let in_val = unsafe { *in_ptr.add(in_c_offset + ih * in_w + iw) };
                    if in_val == 0.0 {
                        continue;
                    }
                    let base_ow = (iw * stride_w) as isize - pad_left as isize;

                    for kh in 0..k_h {
                        let oh = base_oh + (kh * dilation_h) as isize;
                        if oh >= 0 && oh < out_h as isize {
                            let out_row_offset = oh as usize * out_w;
                            let w_row_offset = kh * k_w;
                            for kw in 0..k_w {
                                let ow = base_ow + (kw * dilation_w) as isize;
                                if ow >= 0 && ow < out_w as isize {
                                    unsafe {
                                        let w = *weight_ptr.add(w_c_offset + w_row_offset + kw);
                                        *out_ptr.add(out_row_offset + ow as usize) += in_val * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if params.activation != crate::ops::ConvActivation::None {
            for i in 0..out_h * out_w {
                unsafe {
                    let p = out_ptr.add(i);
                    *p = crate::ops::apply_conv_activation(*p, params.activation);
                }
            }
        }
    };

    if crate::tensor::should_parallelize(total_planes, 2) {
        (0..total_planes).into_par_iter().for_each(run_plane);
    } else {
        (0..total_planes).for_each(run_plane);
    }

    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct TransposeWeight2x2 {
    w00: std::arch::x86_64::__m256,
    w01: std::arch::x86_64::__m256,
    w10: std::arch::x86_64::__m256,
    w11: std::arch::x86_64::__m256,
}

/// Broadcast the four 2x2 transposed-conv taps for every (ic, oc) pair.
///
/// # Safety
/// Requires AVX2+FMA (see [`crate::ops::avx2_fma_enabled`]); `weight_slice` must
/// hold at least `count * 4` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn build_transpose_weights_fma(
    weight_slice: &[f32],
    count: usize,
) -> Vec<TransposeWeight2x2> {
    unsafe {
        use std::arch::x86_64::*;
        (0..count)
            .map(|idx| {
                let wp = weight_slice.as_ptr().add(idx * 4);
                TransposeWeight2x2 {
                    w00: _mm256_set1_ps(*wp),
                    w01: _mm256_set1_ps(*wp.add(1)),
                    w10: _mm256_set1_ps(*wp.add(2)),
                    w11: _mm256_set1_ps(*wp.add(3)),
                }
            })
            .collect()
    }
}

/// Vectorised activation for a 256-bit accumulator.
///
/// # Safety
/// Requires AVX2+FMA (see [`crate::ops::avx2_fma_enabled`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_act_fma(
    v: std::arch::x86_64::__m256,
    act: crate::ops::ConvActivation,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        match act {
            crate::ops::ConvActivation::None => v,
            crate::ops::ConvActivation::Relu => _mm256_max_ps(v, _mm256_setzero_ps()),
            crate::ops::ConvActivation::HardSigmoid { alpha, beta } => {
                let scaled = _mm256_fmadd_ps(v, _mm256_set1_ps(alpha), _mm256_set1_ps(beta));
                _mm256_min_ps(
                    _mm256_max_ps(scaled, _mm256_setzero_ps()),
                    _mm256_set1_ps(1.0),
                )
            }
            crate::ops::ConvActivation::HardSwish => crate::ops::conv2d::avx2_hardswish(v),
            crate::ops::ConvActivation::Gelu => crate::ops::conv2d::avx2_gelu(v),
        }
    }
}

/// Write the vectorised part of one ConvTranspose 2x2 / stride 2 output row.
/// Returns the first input column that still needs the scalar tail.
///
/// # Safety
/// Requires AVX2+FMA (see [`crate::ops::avx2_fma_enabled`]). All raw pointers must
/// be valid for the accesses the loop performs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ct2x2s2_row_fma(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    precomputed: &[TransposeWeight2x2],
    in_channels: usize,
    in_hw: usize,
    out_channels: usize,
    oc: usize,
    in_b_off: usize,
    in_row: usize,
    iw_start: usize,
    in_w: usize,
    out_row0: usize,
    out_row1: usize,
    c_bias: f32,
    act: crate::ops::ConvActivation,
) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let mut iw = iw_start;
        let v_bias = _mm256_set1_ps(c_bias);

        while iw + 16 <= in_w {
            let mut s00_0 = v_bias;
            let mut s01_0 = v_bias;
            let mut s10_0 = v_bias;
            let mut s11_0 = v_bias;

            let mut s00_1 = v_bias;
            let mut s01_1 = v_bias;
            let mut s10_1 = v_bias;
            let mut s11_1 = v_bias;

            for ic in 0..in_channels {
                let pw = &precomputed[ic * out_channels + oc];
                let in_p = in_ptr.add(in_b_off + ic * in_hw + in_row + iw);
                let vx0 = _mm256_loadu_ps(in_p);
                let vx1 = _mm256_loadu_ps(in_p.add(8));

                s00_0 = _mm256_fmadd_ps(vx0, pw.w00, s00_0);
                s01_0 = _mm256_fmadd_ps(vx0, pw.w01, s01_0);
                s10_0 = _mm256_fmadd_ps(vx0, pw.w10, s10_0);
                s11_0 = _mm256_fmadd_ps(vx0, pw.w11, s11_0);

                s00_1 = _mm256_fmadd_ps(vx1, pw.w00, s00_1);
                s01_1 = _mm256_fmadd_ps(vx1, pw.w01, s01_1);
                s10_1 = _mm256_fmadd_ps(vx1, pw.w10, s10_1);
                s11_1 = _mm256_fmadd_ps(vx1, pw.w11, s11_1);
            }

            s00_0 = vec_act_fma(s00_0, act);
            s01_0 = vec_act_fma(s01_0, act);
            s10_0 = vec_act_fma(s10_0, act);
            s11_0 = vec_act_fma(s11_0, act);

            s00_1 = vec_act_fma(s00_1, act);
            s01_1 = vec_act_fma(s01_1, act);
            s10_1 = vec_act_fma(s10_1, act);
            s11_1 = vec_act_fma(s11_1, act);

            let unpck0_lo0 = _mm256_unpacklo_ps(s00_0, s01_0);
            let unpck0_hi0 = _mm256_unpackhi_ps(s00_0, s01_0);
            let r0_lo0 = _mm256_permute2f128_ps(unpck0_lo0, unpck0_hi0, 0x20);
            let r0_hi0 = _mm256_permute2f128_ps(unpck0_lo0, unpck0_hi0, 0x31);
            _mm256_storeu_ps(out_ptr.add(out_row0 + 2 * iw), r0_lo0);
            _mm256_storeu_ps(out_ptr.add(out_row0 + 2 * iw + 8), r0_hi0);

            let unpck1_lo0 = _mm256_unpacklo_ps(s10_0, s11_0);
            let unpck1_hi0 = _mm256_unpackhi_ps(s10_0, s11_0);
            let r1_lo0 = _mm256_permute2f128_ps(unpck1_lo0, unpck1_hi0, 0x20);
            let r1_hi0 = _mm256_permute2f128_ps(unpck1_lo0, unpck1_hi0, 0x31);
            _mm256_storeu_ps(out_ptr.add(out_row1 + 2 * iw), r1_lo0);
            _mm256_storeu_ps(out_ptr.add(out_row1 + 2 * iw + 8), r1_hi0);

            let unpck0_lo1 = _mm256_unpacklo_ps(s00_1, s01_1);
            let unpck0_hi1 = _mm256_unpackhi_ps(s00_1, s01_1);
            let r0_lo1 = _mm256_permute2f128_ps(unpck0_lo1, unpck0_hi1, 0x20);
            let r0_hi1 = _mm256_permute2f128_ps(unpck0_lo1, unpck0_hi1, 0x31);
            _mm256_storeu_ps(out_ptr.add(out_row0 + 2 * iw + 16), r0_lo1);
            _mm256_storeu_ps(out_ptr.add(out_row0 + 2 * iw + 24), r0_hi1);

            let unpck1_lo1 = _mm256_unpacklo_ps(s10_1, s11_1);
            let unpck1_hi1 = _mm256_unpackhi_ps(s10_1, s11_1);
            let r1_lo1 = _mm256_permute2f128_ps(unpck1_lo1, unpck1_hi1, 0x20);
            let r1_hi1 = _mm256_permute2f128_ps(unpck1_lo1, unpck1_hi1, 0x31);
            _mm256_storeu_ps(out_ptr.add(out_row1 + 2 * iw + 16), r1_lo1);
            _mm256_storeu_ps(out_ptr.add(out_row1 + 2 * iw + 24), r1_hi1);

            iw += 16;
        }

        while iw + 8 <= in_w {
            let mut s00 = v_bias;
            let mut s01 = v_bias;
            let mut s10 = v_bias;
            let mut s11 = v_bias;

            for ic in 0..in_channels {
                let pw = &precomputed[ic * out_channels + oc];
                let in_p = in_ptr.add(in_b_off + ic * in_hw + in_row + iw);
                let vx = _mm256_loadu_ps(in_p);

                s00 = _mm256_fmadd_ps(vx, pw.w00, s00);
                s01 = _mm256_fmadd_ps(vx, pw.w01, s01);
                s10 = _mm256_fmadd_ps(vx, pw.w10, s10);
                s11 = _mm256_fmadd_ps(vx, pw.w11, s11);
            }

            s00 = vec_act_fma(s00, act);
            s01 = vec_act_fma(s01, act);
            s10 = vec_act_fma(s10, act);
            s11 = vec_act_fma(s11, act);

            let unpck0_lo = _mm256_unpacklo_ps(s00, s01);
            let unpck0_hi = _mm256_unpackhi_ps(s00, s01);
            let r0_lo = _mm256_permute2f128_ps(unpck0_lo, unpck0_hi, 0x20);
            let r0_hi = _mm256_permute2f128_ps(unpck0_lo, unpck0_hi, 0x31);
            _mm256_storeu_ps(out_ptr.add(out_row0 + 2 * iw), r0_lo);
            _mm256_storeu_ps(out_ptr.add(out_row0 + 2 * iw + 8), r0_hi);

            let unpck1_lo = _mm256_unpacklo_ps(s10, s11);
            let unpck1_hi = _mm256_unpackhi_ps(s10, s11);
            let r1_lo = _mm256_permute2f128_ps(unpck1_lo, unpck1_hi, 0x20);
            let r1_hi = _mm256_permute2f128_ps(unpck1_lo, unpck1_hi, 0x31);
            _mm256_storeu_ps(out_ptr.add(out_row1 + 2 * iw), r1_lo);
            _mm256_storeu_ps(out_ptr.add(out_row1 + 2 * iw + 8), r1_hi);

            iw += 8;
        }
        iw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_transpose2d_rejects_zero_and_underflow() {
        let input = Tensor::from_vec(&[1, 1, 0, 10], vec![]).expect("tensor");
        let weight = Tensor::from_vec(&[1, 1, 2, 2], vec![1.0; 4]).expect("tensor");
        let mut params = ConvTranspose2dParams::default();
        // Zero in_h
        assert!(conv_transpose2d(&input, &weight, None, &params).is_err());

        // Zero group
        let valid_in = Tensor::from_vec(&[1, 1, 4, 4], vec![1.0; 16]).expect("tensor");
        params.group = 0;
        assert!(conv_transpose2d(&valid_in, &weight, None, &params).is_err());

        // Excessive padding causing underflow
        params.group = 1;
        params.pads = [100, 100, 100, 100];
        assert!(conv_transpose2d(&valid_in, &weight, None, &params).is_err());
    }
    /// A bias whose length disagrees with `out_channels` is now rejected rather than
    /// indexed unchecked inside the kernels.
    #[test]
    fn a_wrong_length_bias_is_rejected() {
        let x = Tensor::from_vec(&[1, 2, 4, 4], vec![1.0; 32]).expect("tensor");
        // weight is [in=2, out=3, 2, 2] with group 1, so out_channels is 3.
        let w = Tensor::from_vec(&[2, 3, 2, 2], vec![1.0; 24]).expect("tensor");
        for n in [0usize, 1, 2, 4] {
            let b = Tensor::from_vec(&[n], vec![0.5; n]).expect("tensor");
            assert!(
                conv_transpose2d(&x, &w, Some(&b), &ConvTranspose2dParams::default()).is_err(),
                "bias of {n} must be rejected"
            );
        }
        let b = Tensor::from_vec(&[3], vec![0.5; 3]).expect("tensor");
        conv_transpose2d(&x, &w, Some(&b), &ConvTranspose2dParams::default())
            .expect("a valid bias must be accepted");
    }

    /// Pins the 2x2 / stride-2 fast path -- the FPN upsample in the det model -- against a
    /// naive transposed-convolution reference, and across both SIMD paths.
    ///
    /// The per-call 128-byte-per-(ic, oc) pre-broadcast table is kept deliberately. Replacing
    /// it with an in-loop `_mm256_broadcast_ss` (16 B per (ic, oc) instead of 128 B, and no
    /// 8 MB temporary for a 256 -> 256 layer) measured *slower* on this machine, reproducibly
    /// across three shapes:
    ///
    /// ```text
    ///                      table      broadcast_ss
    /// cin=256 hw=20        8.42 ms        9.03 ms
    /// cin=256 hw=40       13.46 ms       14.99 ms
    /// cin=96  hw=75        6.16 ms        6.70 ms
    /// ```
    ///
    /// The table's writes are sequential and prefetch-friendly, and it turns the inner loop
    /// into pure L1 loads with no broadcast latency. The temporary allocation is not the cost
    /// it looks like. Left as a comment so the next person does not repeat the experiment.
    #[test]
    fn fast_path_2x2_s2_matches_reference_and_both_simd_paths() {
        let (cin, cout, ih, iw) = (5usize, 4usize, 9usize, 11usize);
        let x_data: Vec<f32> = (0..cin * ih * iw)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let w_data: Vec<f32> = (0..cin * cout * 4)
            .map(|i| (i as f32 * 0.13).cos() * 0.5)
            .collect();
        let b_data: Vec<f32> = (0..cout).map(|i| (i as f32 * 0.29).sin()).collect();

        let x = Tensor::from_vec(&[1, cin, ih, iw], x_data.clone()).expect("tensor");
        let w = Tensor::from_vec(&[cin, cout, 2, 2], w_data.clone()).expect("tensor");
        let b = Tensor::from_vec(&[cout], b_data.clone()).expect("tensor");
        let params = ConvTranspose2dParams {
            strides: [2, 2],
            ..Default::default()
        };

        let got = conv_transpose2d(&x, &w, Some(&b), &params).expect("conv_transpose2d");
        let (oh, ow) = (ih * 2, iw * 2);
        assert_eq!(got.shape(), &[1, cout, oh, ow]);

        // Naive reference: out[oc, 2i + dy, 2j + dx] = bias[oc] + sum_ic x[ic,i,j] * w[ic,oc,dy,dx]
        for oc in 0..cout {
            for i in 0..ih {
                for j in 0..iw {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let mut acc = 0.0f64;
                            for ic in 0..cin {
                                acc += x_data[ic * ih * iw + i * iw + j] as f64
                                    * w_data[(ic * cout + oc) * 4 + dy * 2 + dx] as f64;
                            }
                            let idx = oc * oh * ow + (2 * i + dy) * ow + (2 * j + dx);
                            let want = acc + b_data[oc] as f64;
                            let g = got.as_slice()[idx];
                            assert!(
                                (g as f64 - want).abs() < 1e-4,
                                "oc={oc} i={i} j={j} dy={dy} dx={dx}: got {g} want {want}"
                            );
                        }
                    }
                }
            }
        }

        let scalar = crate::ops::simd::with_avx2_override(false, || {
            conv_transpose2d(&x, &w, Some(&b), &params).expect("scalar")
        });
        let wide = crate::ops::simd::with_avx2_override(true, || {
            conv_transpose2d(&x, &w, Some(&b), &params).expect("wide")
        });
        assert_eq!(wide.shape(), scalar.shape());
        for (i, (&a, &s)) in wide.as_slice().iter().zip(scalar.as_slice()).enumerate() {
            assert!(
                (a - s).abs() <= 1e-4 + 1e-4 * s.abs(),
                "index {i}: wide={a} scalar={s}"
            );
        }
    }
}
