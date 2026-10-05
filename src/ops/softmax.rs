use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
#[inline]
pub(crate) unsafe fn avx2_exp_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.336544f32));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.722839f32));

    let log2e = _mm256_set1_ps(1.4426950408889634f32);
    let fx = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let c1 = _mm256_set1_ps(0.693145751953125f32);
    let c2 = _mm256_set1_ps(1.42860682030941723212e-6f32);
    let g = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(fx, c1)),
        _mm256_mul_ps(fx, c2),
    );

    let p6 = _mm256_set1_ps(1.3888889e-3f32);
    let p5 = _mm256_set1_ps(8.3333333e-3f32);
    let p4 = _mm256_set1_ps(4.1666668e-2f32);
    let p3 = _mm256_set1_ps(1.6666667e-1f32);
    let p2 = _mm256_set1_ps(0.5f32);
    let p1 = _mm256_set1_ps(1.0f32);

    let mut y = _mm256_fmadd_ps(p6, g, p5);
    y = _mm256_fmadd_ps(y, g, p4);
    y = _mm256_fmadd_ps(y, g, p3);
    y = _mm256_fmadd_ps(y, g, p2);
    y = _mm256_fmadd_ps(y, g, p1);
    y = _mm256_fmadd_ps(y, g, p1);

    let emm0 = _mm256_cvtps_epi32(fx);
    let emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    let emm0 = _mm256_slli_epi32(emm0, 23);
    let pow2n = _mm256_castsi256_ps(emm0);

    _mm256_mul_ps(y, pow2n)
}

pub fn softmax(input: &Tensor, axis: isize) -> Result<Tensor> {
    let ndim = input.ndim();
    let norm_axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if norm_axis >= ndim {
        return Err(OcrError::OperatorError {
            op: "Softmax".into(),
            msg: format!("Axis {} out of bounds for ndim {}", axis, ndim),
        });
    }

    let mut output = Tensor::empty(input.shape());
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    // An empty input means an empty output, and `Tensor::empty`'s "every element written
    // before read" contract is satisfied vacuously. Returning here also avoids the division
    // below: `axis_size * inner_size` is 0 for a legal `[3, 0]` or `[2, 5, 0]` input.
    if out_slice.is_empty() {
        return Ok(output);
    }

    let axis_size = input.shape()[norm_axis];
    let inner_size: usize = input.shape()[norm_axis + 1..].iter().product();
    let slice_stride = axis_size * inner_size;
    debug_assert!(
        slice_stride > 0,
        "a non-empty output must have a non-zero reduce slice"
    );
    let total_slices = out_slice.len() / slice_stride;

    if inner_size == 1 {
        // Fast path for 1D rows (e.g. Attention weights [..., seq_len, seq_len] or Vocab Softmax)
        let rows_per_chunk = if axis_size >= 4096 {
            1
        } else {
            (4096 / axis_size).clamp(1, 32).min(total_slices)
        };
        let chunk_size = rows_per_chunk * axis_size;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = crate::ops::avx2_fma_enabled();

        let compute_row = |in_row: &[f32], out_row: &mut [f32]| {
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                unsafe {
                    compute_row_avx2(in_row, out_row);
                }
                return;
            }
            compute_row_scalar(in_row, out_row);
        };

        if total_slices <= rows_per_chunk
            || !crate::tensor::should_parallelize(total_slices * axis_size, 32_768)
        {
            for r in 0..total_slices {
                let offset = r * axis_size;
                compute_row(
                    &in_slice[offset..offset + axis_size],
                    &mut out_slice[offset..offset + axis_size],
                );
            }
        } else {
            out_slice
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let chunk_start = chunk_idx * chunk_size;
                    let num_rows = out_chunk.len() / axis_size;
                    for r in 0..num_rows {
                        let offset = r * axis_size;
                        let in_offset = chunk_start + offset;
                        compute_row(
                            &in_slice[in_offset..in_offset + axis_size],
                            &mut out_chunk[offset..offset + axis_size],
                        );
                    }
                });
        }

        return Ok(output);
    }

    // General multi-dimensional softmax
    let slices_per_chunk = (16_384 / slice_stride).max(1).min(total_slices);
    let chunk_size = slices_per_chunk * slice_stride;

    let run_slice = |(outer, out_chunk): (usize, &mut [f32])| {
        let in_outer_offset = outer * slice_stride;

        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let v = in_slice[in_outer_offset + a * inner_size + inner];
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp = 0.0f32;
            for a in 0..axis_size {
                let exp_val = (in_slice[in_outer_offset + a * inner_size + inner] - max_val).exp();
                out_chunk[a * inner_size + inner] = exp_val;
                sum_exp += exp_val;
            }

            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
            for a in 0..axis_size {
                out_chunk[a * inner_size + inner] *= inv_sum;
            }
        }
    };

    if total_slices <= slices_per_chunk
        || !crate::tensor::should_parallelize(total_slices * slice_stride, 32_768)
    {
        out_slice
            .chunks_mut(slice_stride)
            .enumerate()
            .for_each(run_slice);
    } else {
        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_slice_idx = chunk_idx * slices_per_chunk;
                chunk
                    .chunks_mut(slice_stride)
                    .enumerate()
                    .for_each(|(i, sub_chunk)| {
                        run_slice((base_slice_idx + i, sub_chunk));
                    });
            });
    }

    Ok(output)
}

pub fn softmax_inplace(input: &mut Tensor, axis: isize) -> Result<()> {
    let ndim = input.ndim();
    let norm_axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if norm_axis >= ndim {
        return Err(OcrError::OperatorError {
            op: "Softmax".into(),
            msg: format!("Axis {} out of bounds for ndim {}", axis, ndim),
        });
    }

    if !input.is_contiguous() {
        let out = softmax(input, axis)?;
        *input = out;
        return Ok(());
    }

    let axis_size = input.shape()[norm_axis];
    let inner_size: usize = input.shape()[norm_axis + 1..].iter().product();
    let slice_stride = axis_size * inner_size;
    let slice = input.as_mut_slice();
    if slice.is_empty() {
        return Ok(());
    }

    let total_slices = slice.len() / slice_stride;

    if inner_size == 1 {
        let rows_per_chunk = if axis_size >= 4096 {
            1
        } else {
            (4096 / axis_size).clamp(1, 32).min(total_slices)
        };
        let chunk_size = rows_per_chunk * axis_size;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = crate::ops::avx2_fma_enabled();

        let compute_row = |row: &mut [f32]| {
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                unsafe {
                    let ptr = row.as_mut_ptr();
                    let n = row.len();
                    let in_slice = std::slice::from_raw_parts(ptr, n);
                    compute_row_avx2(in_slice, row);
                }
                return;
            }
            let n = row.len();
            let ptr = row.as_mut_ptr();
            unsafe {
                let in_slice = std::slice::from_raw_parts(ptr, n);
                compute_row_scalar(in_slice, row);
            }
        };

        if total_slices <= rows_per_chunk
            || !crate::tensor::should_parallelize(total_slices * axis_size, 32_768)
        {
            for r in 0..total_slices {
                let offset = r * axis_size;
                compute_row(&mut slice[offset..offset + axis_size]);
            }
        } else {
            slice
                .par_chunks_mut(chunk_size)
                .for_each(|out_chunk| {
                    let num_rows = out_chunk.len() / axis_size;
                    for r in 0..num_rows {
                        let offset = r * axis_size;
                        compute_row(&mut out_chunk[offset..offset + axis_size]);
                    }
                });
        }
        return Ok(());
    }

    // Non-inner-1 fallback
    let out = softmax(input, axis)?;
    *input = out;
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, hi64);
    let hi32 = _mm_shuffle_ps(max64, max64, 1);
    _mm_cvtss_f32(_mm_max_ss(max64, hi32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_row_avx2(in_row: &[f32], out_row: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        let n = in_row.len();
        let in_ptr = in_row.as_ptr();
        let out_ptr = out_row.as_mut_ptr();

        // 1. Find max - unroll 4x (32 floats)
        let mut v_max0 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut v_max1 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut v_max2 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut v_max3 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 32 <= n {
            v_max0 = _mm256_max_ps(v_max0, _mm256_loadu_ps(in_ptr.add(i)));
            v_max1 = _mm256_max_ps(v_max1, _mm256_loadu_ps(in_ptr.add(i + 8)));
            v_max2 = _mm256_max_ps(v_max2, _mm256_loadu_ps(in_ptr.add(i + 16)));
            v_max3 = _mm256_max_ps(v_max3, _mm256_loadu_ps(in_ptr.add(i + 24)));
            i += 32;
        }
        let mut v_max = _mm256_max_ps(_mm256_max_ps(v_max0, v_max1), _mm256_max_ps(v_max2, v_max3));
        while i + 8 <= n {
            v_max = _mm256_max_ps(v_max, _mm256_loadu_ps(in_ptr.add(i)));
            i += 8;
        }
        let mut max_val = if i > 0 { hmax256(v_max) } else { f32::NEG_INFINITY };
        while i < n {
            if *in_ptr.add(i) > max_val {
                max_val = *in_ptr.add(i);
            }
            i += 1;
        }

        // 2. Compute exp and sum
        let v_max = _mm256_set1_ps(max_val);
        let mut v_sum0 = _mm256_setzero_ps();
        let mut v_sum1 = _mm256_setzero_ps();
        let mut v_sum2 = _mm256_setzero_ps();
        let mut v_sum3 = _mm256_setzero_ps();
        i = 0;
        while i + 32 <= n {
            let diff0 = _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), v_max);
            let diff1 = _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i + 8)), v_max);
            let diff2 = _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i + 16)), v_max);
            let diff3 = _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i + 24)), v_max);
            let e0 = avx2_exp_ps(diff0);
            let e1 = avx2_exp_ps(diff1);
            let e2 = avx2_exp_ps(diff2);
            let e3 = avx2_exp_ps(diff3);
            _mm256_storeu_ps(out_ptr.add(i), e0);
            _mm256_storeu_ps(out_ptr.add(i + 8), e1);
            _mm256_storeu_ps(out_ptr.add(i + 16), e2);
            _mm256_storeu_ps(out_ptr.add(i + 24), e3);
            v_sum0 = _mm256_add_ps(v_sum0, e0);
            v_sum1 = _mm256_add_ps(v_sum1, e1);
            v_sum2 = _mm256_add_ps(v_sum2, e2);
            v_sum3 = _mm256_add_ps(v_sum3, e3);
            i += 32;
        }
        let mut v_sum = _mm256_add_ps(_mm256_add_ps(v_sum0, v_sum1), _mm256_add_ps(v_sum2, v_sum3));
        while i + 8 <= n {
            let v = _mm256_loadu_ps(in_ptr.add(i));
            let diff = _mm256_sub_ps(v, v_max);
            let e = avx2_exp_ps(diff);
            _mm256_storeu_ps(out_ptr.add(i), e);
            v_sum = _mm256_add_ps(v_sum, e);
            i += 8;
        }
        let mut sum_exp = if i > 0 { hsum256(v_sum) } else { 0.0 };
        while i < n {
            let e = (*in_ptr.add(i) - max_val).exp();
            *out_ptr.add(i) = e;
            sum_exp += e;
            i += 1;
        }

        // 3. Normalize
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
        let v_inv = _mm256_set1_ps(inv_sum);
        i = 0;
        while i + 32 <= n {
            let v0 = _mm256_loadu_ps(out_ptr.add(i));
            let v1 = _mm256_loadu_ps(out_ptr.add(i + 8));
            let v2 = _mm256_loadu_ps(out_ptr.add(i + 16));
            let v3 = _mm256_loadu_ps(out_ptr.add(i + 24));
            _mm256_storeu_ps(out_ptr.add(i), _mm256_mul_ps(v0, v_inv));
            _mm256_storeu_ps(out_ptr.add(i + 8), _mm256_mul_ps(v1, v_inv));
            _mm256_storeu_ps(out_ptr.add(i + 16), _mm256_mul_ps(v2, v_inv));
            _mm256_storeu_ps(out_ptr.add(i + 24), _mm256_mul_ps(v3, v_inv));
            i += 32;
        }
        while i + 8 <= n {
            let v = _mm256_loadu_ps(out_ptr.add(i));
            _mm256_storeu_ps(out_ptr.add(i), _mm256_mul_ps(v, v_inv));
            i += 8;
        }
        while i < n {
            *out_ptr.add(i) *= inv_sum;
            i += 1;
        }
    }
}

#[inline(always)]
fn compute_row_scalar(in_row: &[f32], out_row: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in in_row {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum_exp = 0.0f32;
    for (i, &v) in in_row.iter().enumerate() {
        let e = (v - max_val).exp();
        out_row[i] = e;
        sum_exp += e;
    }
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
    for v in out_row.iter_mut() {
        *v *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape, data).expect("valid tensor")
    }

    /// Straightforward f64 implementation used as the reference.
    fn softmax_f64(input: &[f32], shape: &[usize], axis: usize) -> Vec<f64> {
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let stride = axis_size * inner;
        let slices = input.len() / stride;
        let mut out = vec![0.0f64; input.len()];
        for s in 0..slices {
            let base = s * stride;
            for i in 0..inner {
                let max = (0..axis_size)
                    .map(|a| input[base + a * inner + i] as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for a in 0..axis_size {
                    let e = ((input[base + a * inner + i] as f64) - max).exp();
                    out[base + a * inner + i] = e;
                    sum += e;
                }
                for a in 0..axis_size {
                    out[base + a * inner + i] /= sum;
                }
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f64], tol: f64, ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                ((a as f64) - e).abs() <= tol,
                "{ctx}: index {i}: got {a}, want {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn softmax_matches_f64_reference_on_last_axis() {
        let (shape, data) = (
            [3usize, 5usize],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, //
                -1.0, 0.0, 0.0, 0.5, 100.0, //
                0.25, -0.25, 0.75, -0.75, 0.0,
            ],
        );
        let out = softmax(&tensor(&shape, data.clone()), -1).expect("softmax");
        assert_eq!(out.shape(), &shape);
        assert_close(
            out.as_slice(),
            &softmax_f64(&data, &shape, 1),
            1e-6,
            "axis=-1",
        );

        for row in out.as_slice().chunks(5) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row must sum to 1, got {sum}");
        }
    }

    #[test]
    fn softmax_matches_f64_reference_on_middle_axis() {
        // inner_size != 1 exercises the general multi-dimensional path.
        let shape = [2usize, 4usize, 3usize];
        let data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let out = softmax(&tensor(&shape, data.clone()), 1).expect("softmax");
        assert_eq!(out.shape(), &shape);
        assert_close(
            out.as_slice(),
            &softmax_f64(&data, &shape, 1),
            1e-6,
            "axis=1",
        );
    }

    #[test]
    fn softmax_is_stable_on_large_magnitudes() {
        // Without max-subtraction this would overflow to inf/inf = NaN.
        let shape = [2usize, 4usize];
        let data = vec![
            1000.0, 1001.0, 999.0, 1000.5, -1000.0, -999.0, -1001.0, -1000.5,
        ];
        let out = softmax(&tensor(&shape, data), -1).expect("softmax");
        assert!(
            out.as_slice().iter().all(|v| v.is_finite()),
            "must not produce inf/NaN: {:?}",
            out.as_slice()
        );
        for row in out.as_slice().chunks(4) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row must sum to 1, got {sum}");
        }
    }

    #[test]
    fn softmax_rejects_out_of_range_axis() {
        let t = tensor(&[2, 3], vec![0.0; 6]);
        assert!(softmax(&t, 2).is_err());
        assert!(softmax(&t, -3).is_err());
    }

    /// A zero-length reduce axis is legal ONNX. `total_slices = out.len() / (axis_size *
    /// inner_size)` divided by zero before this guard.
    #[test]
    fn empty_input_returns_an_empty_tensor_instead_of_dividing_by_zero() {
        for shape in [vec![0usize], vec![3, 0], vec![2, 5, 0], vec![1, 1, 0, 4]] {
            let n: usize = shape.iter().product();
            let t = tensor(&shape, vec![0.5; n]);
            let r = std::panic::catch_unwind(|| softmax(&t, -1));
            assert!(r.is_ok(), "shape {shape:?} must not panic");
            let out = r
                .expect("no panic")
                .expect("an empty input is not an error");
            assert_eq!(out.shape(), &shape[..], "shape {shape:?}");
            assert_eq!(out.numel(), 0, "shape {shape:?}");
        }
    }

    /// Sanity control for the guard above: a normal softmax still works.
    #[test]
    fn single_row_softmax_still_works() {
        let t = tensor(&[3], vec![1.0, 2.0, 3.0]);
        let out = softmax(&t, -1).expect("softmax");
        let sum: f32 = out.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
        assert!(out.as_slice()[2] > out.as_slice()[1]);
    }

    /// The 1-D fast path (inner_size == 1) has a dedicated AVX2 kernel; it must agree with
    /// the scalar fallback.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn softmax_avx2_row_matches_scalar() {
        let shape = [1usize, 1usize, 40usize, 40usize];
        let data: Vec<f32> = (0..1600)
            .map(|i| ((i as f32) * 0.113).sin() * 6.0 + (i % 7) as f32 * 0.5)
            .collect();
        let t = tensor(&shape, data);

        let scalar =
            crate::ops::simd::with_avx2_override(false, || softmax(&t, -1).expect("softmax"));
        let avx2 = crate::ops::simd::with_avx2_override(true, || softmax(&t, -1).expect("softmax"));

        assert_eq!(avx2.shape(), scalar.shape());
        for (i, (&a, &b)) in avx2.as_slice().iter().zip(scalar.as_slice()).enumerate() {
            // The AVX2 path uses a polynomial exp approximation, so allow a little slack.
            assert!(
                (a - b).abs() <= 1e-6 + 1e-5 * b.abs(),
                "row-kernel mismatch at {i}: avx2={a} scalar={b}"
            );
        }
    }
}
