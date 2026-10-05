use crate::error::{OcrError, Result};
use crate::tensor::{Tensor, should_parallelize};
use rayon::prelude::*;

/// Output extent of a pooled dimension, with an explicit error instead of a
/// `usize` underflow when the padded input is smaller than the kernel.
fn pooled_out_size(
    op: &str,
    in_size: usize,
    pad_lo: usize,
    pad_hi: usize,
    k: usize,
    stride: usize,
) -> Result<usize> {
    let padded = in_size + pad_lo + pad_hi;
    if padded < k {
        return Err(OcrError::OperatorError {
            op: op.into(),
            msg: format!(
                "padded input size {} ({} + {} + {}) is smaller than kernel size {}",
                padded, in_size, pad_lo, pad_hi, k
            ),
        });
    }
    if stride == 0 {
        return Err(OcrError::OperatorError {
            op: op.into(),
            msg: "stride must be non-zero".into(),
        });
    }
    Ok((padded - k) / stride + 1)
}

pub fn global_avg_pool2d(input: &Tensor) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "GlobalAveragePool".into(),
            msg: format!("Expected 4D tensor, got ndim={}", input.ndim()),
        });
    }

    let [batch_size, channels, h, w] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];

    // Every element is written below, so skip the zero-fill.
    let mut output = Tensor::empty(&[batch_size, channels, 1, 1]);
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let spatial_size = (h * w) as f32;
    let inv_spatial = 1.0 / spatial_size;

    let run_elem = |(c_idx, out_val): (usize, &mut f32)| {
        let in_offset = c_idx * (h * w);
        let slice = &in_slice[in_offset..in_offset + (h * w)];
        let sum = avx2_sum_slice(slice);
        *out_val = sum * inv_spatial;
    };

    if should_parallelize(out_slice.len(), 512) {
        out_slice.par_iter_mut().enumerate().for_each(run_elem);
    } else {
        out_slice.iter_mut().enumerate().for_each(run_elem);
    }

    Ok(output)
}

/// Sum a slice, using AVX2+FMA when the running CPU implements it.
#[inline(always)]
fn avx2_sum_slice(slice: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        return unsafe { avx2_sum_slice_fma(slice) };
    }
    slice.iter().sum()
}

/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_sum_slice_fma(slice: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let n = slice.len();
        let ptr = slice.as_ptr();
        let mut i = 0;
        let mut v_sum0 = _mm256_setzero_ps();
        let mut v_sum1 = _mm256_setzero_ps();

        while i + 16 <= n {
            v_sum0 = _mm256_add_ps(v_sum0, _mm256_loadu_ps(ptr.add(i)));
            v_sum1 = _mm256_add_ps(v_sum1, _mm256_loadu_ps(ptr.add(i + 8)));
            i += 16;
        }
        while i + 8 <= n {
            v_sum0 = _mm256_add_ps(v_sum0, _mm256_loadu_ps(ptr.add(i)));
            i += 8;
        }

        v_sum0 = _mm256_add_ps(v_sum0, v_sum1);
        let v_low = _mm256_castps256_ps128(v_sum0);
        let v_high = _mm256_extractf128_ps(v_sum0, 1);
        let v128 = _mm_add_ps(v_low, v_high);
        let shuf = _mm_movehdup_ps(v128);
        let sums = _mm_add_ps(v128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let total = _mm_add_ss(sums, shuf2);
        let mut scalar_total = _mm_cvtss_f32(total);

        while i < n {
            scalar_total += *ptr.add(i);
            i += 1;
        }
        scalar_total
    }
}

/// Vectorised 2x2/stride-1 max-pool over one output row.
/// Returns the first column index that still needs the scalar fallback.
///
/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`] and must
/// guarantee that `r0`/`r1` are readable up to `out_w` elements and that
/// `out_row` is writable for `out_w` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_pool2x2_row_fma(
    r0: *const f32,
    r1: *const f32,
    out_row: *mut f32,
    out_w: usize,
) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let mut ow = 0;
        while ow + 32 <= out_w {
            let a0_0 = _mm256_loadu_ps(r0.add(ow));
            let a1_0 = _mm256_loadu_ps(r0.add(ow + 1));
            let b0_0 = _mm256_loadu_ps(r1.add(ow));
            let b1_0 = _mm256_loadu_ps(r1.add(ow + 1));

            let a0_1 = _mm256_loadu_ps(r0.add(ow + 8));
            let a1_1 = _mm256_loadu_ps(r0.add(ow + 9));
            let b0_1 = _mm256_loadu_ps(r1.add(ow + 8));
            let b1_1 = _mm256_loadu_ps(r1.add(ow + 9));

            let a0_2 = _mm256_loadu_ps(r0.add(ow + 16));
            let a1_2 = _mm256_loadu_ps(r0.add(ow + 17));
            let b0_2 = _mm256_loadu_ps(r1.add(ow + 16));
            let b1_2 = _mm256_loadu_ps(r1.add(ow + 17));

            let a0_3 = _mm256_loadu_ps(r0.add(ow + 24));
            let a1_3 = _mm256_loadu_ps(r0.add(ow + 25));
            let b0_3 = _mm256_loadu_ps(r1.add(ow + 24));
            let b1_3 = _mm256_loadu_ps(r1.add(ow + 25));

            let res0 = _mm256_max_ps(_mm256_max_ps(a0_0, a1_0), _mm256_max_ps(b0_0, b1_0));
            let res1 = _mm256_max_ps(_mm256_max_ps(a0_1, a1_1), _mm256_max_ps(b0_1, b1_1));
            let res2 = _mm256_max_ps(_mm256_max_ps(a0_2, a1_2), _mm256_max_ps(b0_2, b1_2));
            let res3 = _mm256_max_ps(_mm256_max_ps(a0_3, a1_3), _mm256_max_ps(b0_3, b1_3));

            _mm256_storeu_ps(out_row.add(ow), res0);
            _mm256_storeu_ps(out_row.add(ow + 8), res1);
            _mm256_storeu_ps(out_row.add(ow + 16), res2);
            _mm256_storeu_ps(out_row.add(ow + 24), res3);
            ow += 32;
        }
        while ow + 8 <= out_w {
            let a0 = _mm256_loadu_ps(r0.add(ow));
            let a1 = _mm256_loadu_ps(r0.add(ow + 1));
            let b0 = _mm256_loadu_ps(r1.add(ow));
            let b1 = _mm256_loadu_ps(r1.add(ow + 1));
            let res = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(b0, b1));
            _mm256_storeu_ps(out_row.add(ow), res);
            ow += 8;
        }
        ow
    }
}

pub struct Pool2dParams {
    pub kernel_shape: [usize; 2],
    pub strides: [usize; 2],
    pub pads: [usize; 4],
}

impl Default for Pool2dParams {
    fn default() -> Self {
        Self {
            kernel_shape: [2, 2],
            strides: [1, 1],
            pads: [0, 0, 0, 0],
        }
    }
}

pub fn max_pool2d(input: &Tensor, params: &Pool2dParams) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "MaxPool".into(),
            msg: format!("Expected 4D tensor, got ndim={}", input.ndim()),
        });
    }

    let [batch_size, channels, in_h, in_w] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];

    let [k_h, k_w] = params.kernel_shape;
    let [stride_h, stride_w] = params.strides;
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;

    let out_h = pooled_out_size("MaxPool", in_h, pad_top, pad_bottom, k_h, stride_h)?;
    let out_w = pooled_out_size("MaxPool", in_w, pad_left, pad_right, k_w, stride_w)?;

    let mut output = Tensor::empty(&[batch_size, channels, out_h, out_w]);
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();
    let out_spatial = out_h * out_w;

    // Fast Path: 2x2 MaxPool with stride 1, pad_top=0, pad_left=0, pad_bottom<=1, pad_right<=1
    // Covers both pad=0 (valid) and pad=SAME_UPPER (where pad_bottom=1, pad_right=1)
    if k_h == 2
        && k_w == 2
        && stride_h == 1
        && stride_w == 1
        && pad_top == 0
        && pad_left == 0
        && pad_bottom <= 1
        && pad_right <= 1
    {
        let in_ptr_usize = in_slice.as_ptr() as usize;
        let out_ptr_usize = out_slice.as_mut_ptr() as usize;
        #[cfg(target_arch = "x86_64")]
        let use_simd = crate::ops::avx2_fma_enabled();

        let valid_h = (in_h.saturating_sub(1)).min(out_h);
        let valid_w = (in_w.saturating_sub(1)).min(out_w);

        let run_channel = |c_idx: usize| {
            let in_c_offset = c_idx * in_h * in_w;
            let out_c_offset = c_idx * out_spatial;
            let in_ptr = in_ptr_usize as *const f32;
            let out_ptr = out_ptr_usize as *mut f32;

            for oh in 0..valid_h {
                let r0 = unsafe { in_ptr.add(in_c_offset + oh * in_w) };
                let r1 = unsafe { in_ptr.add(in_c_offset + (oh + 1) * in_w) };
                let out_row = unsafe { out_ptr.add(out_c_offset + oh * out_w) };

                let mut ow = 0;

                #[cfg(target_arch = "x86_64")]
                if use_simd {
                    ow = unsafe { max_pool2x2_row_fma(r0, r1, out_row, valid_w) };
                }

                while ow < valid_w {
                    unsafe {
                        let v00 = *r0.add(ow);
                        let v01 = *r0.add(ow + 1);
                        let v10 = *r1.add(ow);
                        let v11 = *r1.add(ow + 1);
                        *out_row.add(ow) = v00.max(v01).max(v10).max(v11);
                    }
                    ow += 1;
                }

                if valid_w < out_w {
                    unsafe {
                        let v0 = *r0.add(valid_w);
                        let v1 = *r1.add(valid_w);
                        *out_row.add(valid_w) = v0.max(v1);
                    }
                }
            }

            for oh in valid_h..out_h {
                let r0 = unsafe { in_ptr.add(in_c_offset + oh * in_w) };
                let out_row = unsafe { out_ptr.add(out_c_offset + oh * out_w) };

                for ow in 0..valid_w {
                    unsafe {
                        let v00 = *r0.add(ow);
                        let v01 = *r0.add(ow + 1);
                        *out_row.add(ow) = v00.max(v01);
                    }
                }
                if valid_w < out_w {
                    unsafe {
                        *out_row.add(valid_w) = *r0.add(valid_w);
                    }
                }
            }
        };

        let total_work = batch_size * channels * out_spatial;
        if should_parallelize(total_work, 65536) {
            (0..batch_size * channels)
                .into_par_iter()
                .for_each(run_channel);
        } else {
            (0..batch_size * channels).for_each(run_channel);
        }

        return Ok(output);
    }

    if stride_h == 0 || stride_w == 0 {
        return Err(OcrError::OperatorError {
            op: "MaxPool".into(),
            msg: "stride must be non-zero".into(),
        });
    }

    let run_chunk = |(c_idx, out_chunk): (usize, &mut [f32])| {
        let in_c_offset = c_idx * in_h * in_w;

        for oh in 0..out_h {
            let base_ih = (oh * stride_h) as isize - pad_top as isize;
            for ow in 0..out_w {
                let base_iw = (ow * stride_w) as isize - pad_left as isize;
                let mut max_val = f32::NEG_INFINITY;

                for kh in 0..k_h {
                    let ih = base_ih + kh as isize;
                    if ih >= 0 && ih < in_h as isize {
                        let row_offset = in_c_offset + (ih as usize) * in_w;
                        for kw in 0..k_w {
                            let iw = base_iw + kw as isize;
                            if iw >= 0 && iw < in_w as isize {
                                let val = in_slice[row_offset + iw as usize];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                }

                out_chunk[oh * out_w + ow] = max_val;
            }
        }
    };

    let total_work = batch_size * channels * out_spatial;
    if should_parallelize(total_work, 65536) {
        out_slice
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_chunk);
    } else {
        out_slice
            .chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_chunk);
    }

    Ok(output)
}

pub fn avg_pool2d(
    input: &Tensor,
    params: &Pool2dParams,
    count_include_pad: bool,
) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "AveragePool".into(),
            msg: format!("Expected 4D tensor, got ndim={}", input.ndim()),
        });
    }

    let [batch_size, channels, in_h, in_w] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];

    let [k_h, k_w] = params.kernel_shape;
    let [stride_h, stride_w] = params.strides;
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;

    let out_h = pooled_out_size("AveragePool", in_h, pad_top, pad_bottom, k_h, stride_h)?;
    let out_w = pooled_out_size("AveragePool", in_w, pad_left, pad_right, k_w, stride_w)?;

    // Every element is written below, so skip the zero-fill.
    let mut output = Tensor::empty(&[batch_size, channels, out_h, out_w]);
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let out_spatial = out_h * out_w;

    let run_chunk = |(c_idx, out_chunk): (usize, &mut [f32])| {
        let in_c_offset = c_idx * in_h * in_w;

        for oh in 0..out_h {
            let base_ih = (oh * stride_h) as isize - pad_top as isize;
            for ow in 0..out_w {
                let base_iw = (ow * stride_w) as isize - pad_left as isize;
                let mut sum = 0.0f32;
                let mut count = 0usize;

                for kh in 0..k_h {
                    let ih = base_ih + kh as isize;
                    for kw in 0..k_w {
                        let iw = base_iw + kw as isize;
                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let idx = in_c_offset + (ih as usize) * in_w + (iw as usize);
                            sum += in_slice[idx];
                            count += 1;
                        } else if count_include_pad {
                            count += 1;
                        }
                    }
                }

                let denom = if count > 0 { count as f32 } else { 1.0 };
                out_chunk[oh * out_w + ow] = sum / denom;
            }
        }
    };

    if should_parallelize(batch_size * channels, 2) {
        out_slice
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_chunk);
    } else {
        out_slice
            .chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_chunk);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_out_size_reports_error_instead_of_underflowing() {
        let err = pooled_out_size("MaxPool", 2, 0, 0, 4, 1);
        assert!(
            err.is_err(),
            "kernel larger than input must be a clean error"
        );
    }

    #[test]
    fn max_pool2d_matches_reference_2x2_stride1() {
        // 1 x 1 x 4 x 4 with a strictly increasing ramp
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let input = Tensor::from_vec(&[1, 1, 4, 4], data).unwrap();
        let params = Pool2dParams {
            kernel_shape: [2, 2],
            strides: [1, 1],
            pads: [0, 0, 0, 0],
        };
        let out = max_pool2d(&input, &params).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        // Reference: max of each 2x2 window on the ramp
        let expect = [
            5.0f32, 6.0, 7.0, //
            9.0, 10.0, 11.0, //
            13.0, 14.0, 15.0,
        ];
        assert_eq!(out.as_slice(), &expect);
    }

    #[test]
    fn max_pool2d_matches_reference_2x2_stride1_with_pad() {
        // 1 x 1 x 4 x 4 with a strictly increasing ramp
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let input = Tensor::from_vec(&[1, 1, 4, 4], data).unwrap();
        let params = Pool2dParams {
            kernel_shape: [2, 2],
            strides: [1, 1],
            pads: [0, 0, 1, 1],
        };
        let out = max_pool2d(&input, &params).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
        let expect = [
            5.0f32, 6.0, 7.0, 7.0, //
            9.0, 10.0, 11.0, 11.0, //
            13.0, 14.0, 15.0, 15.0, //
            13.0, 14.0, 15.0, 15.0,
        ];
        assert_eq!(out.as_slice(), &expect);
    }

    #[test]
    fn global_avg_pool2d_matches_scalar_reference() {
        let n = 3 * 5 * 8 * 8;
        let data: Vec<f32> = (0..n).map(|v| ((v % 17) as f32) - 8.0).collect();
        let input = Tensor::from_vec(&[3, 5, 8, 8], data.clone()).unwrap();
        let out = global_avg_pool2d(&input).unwrap();
        assert_eq!(out.shape(), &[3, 5, 1, 1]);
        for c in 0..15 {
            let plane = &data[c * 64..(c + 1) * 64];
            let expect = plane.iter().sum::<f32>() / 64.0;
            assert!((out.as_slice()[c] - expect).abs() < 1e-4);
        }
    }
}
