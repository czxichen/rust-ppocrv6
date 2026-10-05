use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let ndim = input.ndim();
    let norm_ndim = normalized_shape.len();
    if norm_ndim > ndim {
        return Err(OcrError::OperatorError {
            op: "LayerNorm".into(),
            msg: "Normalized shape has more dimensions than input".into(),
        });
    }

    let start_dim = ndim - norm_ndim;
    if &input.shape()[start_dim..] != normalized_shape {
        return Err(OcrError::OperatorError {
            op: "LayerNorm".into(),
            msg: "Input trailing dimensions do not match normalized shape".into(),
        });
    }

    let norm_size: usize = normalized_shape.iter().product();

    // `par_chunks_mut(0)` / `chunks_mut(0)` panic on a zero-length chunk size. A zero
    // `normalized_shape` is expressible through the public `layer_norm` signature.
    if norm_size == 0 {
        return Ok(Tensor::empty(input.shape()));
    }

    // The AVX2 kernel below does unchecked `_mm256_loadu_ps` over `w` and `b` for
    // `norm_size` elements, and `as_slice()` returns raw storage without a contiguity
    // check. Validate both here, on a `pub fn` that a caller can invoke directly — the
    // in-graph path happens to always supply matching constants, which is exactly why this
    // was never noticed.
    for (label, t) in [("weight", weight), ("bias", bias)] {
        if let Some(t) = t {
            if t.numel() != norm_size {
                return Err(OcrError::OperatorError {
                    op: "LayerNorm".into(),
                    msg: format!(
                        "{label} has {} elements but normalized_shape has {norm_size}",
                        t.numel()
                    ),
                });
            }
        }
    }
    let w_holder = weight.and_then(|w| {
        if w.is_contiguous() {
            None
        } else {
            Some(w.to_contiguous())
        }
    });
    let b_holder = bias.and_then(|b| {
        if b.is_contiguous() {
            None
        } else {
            Some(b.to_contiguous())
        }
    });

    let mut output = Tensor::empty(input.shape());
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let w_slice = weight.map(|w| match &w_holder {
        Some(h) => h.as_slice(),
        None => w.as_slice(),
    });
    let b_slice = bias.map(|b| match &b_holder {
        Some(h) => h.as_slice(),
        None => b.as_slice(),
    });

    let run_chunk = |(outer, out_chunk): (usize, &mut [f32])| {
        let offset = outer * norm_size;
        let slice = &in_slice[offset..offset + norm_size];
        compute_layer_norm_row(slice, out_chunk, w_slice, b_slice, eps);
    };

    // Route through `should_parallelize` like every other parallel site. This was the one
    // place that hand-rolled the three conditions with its own magic threshold, which
    // `tensor.rs` explicitly documents as not allowed because a future threshold change
    // there would not reach it.
    if crate::tensor::should_parallelize(out_slice.len(), 32_000) {
        out_slice
            .par_chunks_mut(norm_size)
            .enumerate()
            .for_each(run_chunk);
    } else {
        out_slice
            .chunks_mut(norm_size)
            .enumerate()
            .for_each(run_chunk);
    }

    Ok(output)
}

#[inline]
fn compute_layer_norm_row(
    slice: &[f32],
    out_chunk: &mut [f32],
    w_slice: Option<&[f32]>,
    b_slice: Option<&[f32]>,
    eps: f32,
) {
    let n = slice.len();
    if n == 0 || out_chunk.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        unsafe { compute_layer_norm_row_fma(slice, out_chunk, w_slice, b_slice, eps) };
        return;
    }

    let mean = slice.iter().sum::<f32>() / (n as f32);
    let var = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / (n as f32);
    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..n {
        let mut val = (slice[i] - mean) * inv_std;
        if let Some(w) = w_slice {
            val *= w[i];
        }
        if let Some(b) = b_slice {
            val += b[i];
        }
        out_chunk[i] = val;
    }
}

/// AVX2+FMA implementation of one LayerNorm row.
///
/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`], and
/// `w_slice`/`b_slice` (when present) must cover `slice`/`out_chunk`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_layer_norm_row_fma(
    slice: &[f32],
    out_chunk: &mut [f32],
    w_slice: Option<&[f32]>,
    b_slice: Option<&[f32]>,
    eps: f32,
) {
    let n = slice.len();
    let mean: f32;
    let var: f32;

    unsafe {
        use std::arch::x86_64::*;
        let mut v_sum = _mm256_setzero_ps();
        let mut i = 0;
        let ptr = slice.as_ptr();
        while i + 8 <= n {
            v_sum = _mm256_add_ps(v_sum, _mm256_loadu_ps(ptr.add(i)));
            i += 8;
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), v_sum);
        let mut sum = buf.iter().sum::<f32>();
        while i < n {
            sum += *ptr.add(i);
            i += 1;
        }
        mean = sum / (n as f32);

        let v_mean = _mm256_set1_ps(mean);
        let mut v_var = _mm256_setzero_ps();
        i = 0;
        while i + 8 <= n {
            let diff = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(i)), v_mean);
            v_var = _mm256_fmadd_ps(diff, diff, v_var);
            i += 8;
        }
        _mm256_storeu_ps(buf.as_mut_ptr(), v_var);
        let mut var_sum = buf.iter().sum::<f32>();
        while i < n {
            let d = *ptr.add(i) - mean;
            var_sum += d * d;
            i += 1;
        }
        var = var_sum / (n as f32);
    }

    let inv_std = 1.0 / (var + eps).sqrt();

    unsafe {
        use std::arch::x86_64::*;
        let v_mean = _mm256_set1_ps(mean);
        let v_inv_std = _mm256_set1_ps(inv_std);
        let in_ptr = slice.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();
        let w_ptr = w_slice.map(|w| w.as_ptr());
        let b_ptr = b_slice.map(|b| b.as_ptr());

        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(in_ptr.add(i));
            let mut v_norm = _mm256_mul_ps(_mm256_sub_ps(v, v_mean), v_inv_std);
            if let Some(wp) = w_ptr {
                let vw = _mm256_loadu_ps(wp.add(i));
                if let Some(bp) = b_ptr {
                    let vb = _mm256_loadu_ps(bp.add(i));
                    v_norm = _mm256_fmadd_ps(v_norm, vw, vb);
                } else {
                    v_norm = _mm256_mul_ps(v_norm, vw);
                }
            } else if let Some(bp) = b_ptr {
                let vb = _mm256_loadu_ps(bp.add(i));
                v_norm = _mm256_add_ps(v_norm, vb);
            }
            _mm256_storeu_ps(out_ptr.add(i), v_norm);
            i += 8;
        }

        while i < n {
            let mut val = (*in_ptr.add(i) - mean) * inv_std;
            if let Some(w) = w_slice {
                val *= w[i];
            }
            if let Some(b) = b_slice {
                val += b[i];
            }
            *out_ptr.add(i) = val;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape, data).expect("valid tensor")
    }

    /// f64 reference: normalise over the trailing `norm_size` elements, then apply the
    /// optional affine transform elementwise.
    fn reference(
        input: &[f32],
        norm_size: usize,
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; input.len()];
        for (row, chunk) in input.chunks(norm_size).enumerate() {
            let n = chunk.len() as f64;
            let mean = chunk.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = chunk
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            for (i, &v) in chunk.iter().enumerate() {
                let mut val = (v as f64 - mean) * inv_std;
                if let Some(w) = w {
                    val *= w[i] as f64;
                }
                if let Some(b) = b {
                    val += b[i] as f64;
                }
                out[row * norm_size + i] = val;
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f64], tol: f64, ctx: &str) {
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                ((a as f64) - e).abs() <= tol,
                "{ctx}: index {i}: got {a}, want {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn layer_norm_matches_f64_reference_with_and_without_affine() {
        let shape = [4usize, 8usize];
        let data: Vec<f32> = (0..32)
            .map(|i| ((i as f32) * 0.31).sin() * 2.0 + 0.5)
            .collect();
        // Deliberately not a multiple of 8 so the vectorised tail is covered too.
        let shape_odd = [3usize, 13usize];
        let data_odd: Vec<f32> = (0..39).map(|i| ((i as f32) * 0.17).cos() * 1.5).collect();
        let eps = 1e-5f32;

        for (sh, d) in [(&shape[..], &data), (&shape_odd[..], &data_odd)] {
            let norm = *sh.last().unwrap();
            // Per-shape affine parameters: LayerNorm's weight/bias are indexed by position
            // within the normalised window, so they must have exactly `norm` entries.
            let w_data: Vec<f32> = (0..norm).map(|i| 1.0 - i as f32 * 0.12).collect();
            let b_data: Vec<f32> = (0..norm).map(|i| i as f32 * 0.3 - 0.5).collect();
            let w = tensor(&[norm], w_data.clone());
            let b = tensor(&[norm], b_data.clone());

            let t = tensor(sh, d.clone());

            let plain = layer_norm(&t, &[norm], None, None, eps).expect("layernorm");
            assert_eq!(plain.shape(), sh);
            assert_close(
                plain.as_slice(),
                &reference(d, norm, None, None, eps),
                1e-5,
                "no affine",
            );

            let affined = layer_norm(&t, &[norm], Some(&w), Some(&b), eps).expect("layernorm");
            assert_close(
                affined.as_slice(),
                &reference(d, norm, Some(&w_data), Some(&b_data), eps),
                1e-5,
                "with affine",
            );
        }
    }

    #[test]
    fn layer_norm_produces_zero_mean_unit_variance() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 1.7 - 9.0).collect();
        let t = tensor(&[2, 8], data);
        let out = layer_norm(&t, &[8], None, None, 0.0).expect("layernorm");
        for row in out.as_slice().chunks(8) {
            let mean = row.iter().sum::<f32>() / 8.0;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
            assert!(mean.abs() < 1e-5, "mean should be 0, got {mean}");
            assert!((var - 1.0).abs() < 1e-4, "var should be 1, got {var}");
        }
    }

    #[test]
    fn layer_norm_handles_constant_rows_without_nan() {
        let t = tensor(&[2, 8], vec![3.5f32; 16]);
        let out = layer_norm(&t, &[8], None, None, 1e-5).expect("layernorm");
        assert!(
            out.as_slice().iter().all(|v| v.is_finite()),
            "constant rows must not yield NaN/inf: {:?}",
            out.as_slice()
        );
        assert!(out.as_slice().iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn layer_norm_rejects_shape_mismatch() {
        let t = tensor(&[2, 6], vec![0.0; 12]);
        assert!(layer_norm(&t, &[8], None, None, 1e-5).is_err());
        assert!(
            layer_norm(&t, &[2, 6, 1], None, None, 1e-5).is_err(),
            "normalized shape cannot have more dims than the input"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn layer_norm_fma_matches_scalar() {
        let shape = [5usize, 37usize];
        let data: Vec<f32> = (0..185)
            .map(|i| ((i as f32) * 0.211).sin() * 4.0 + (i % 5) as f32)
            .collect();
        let t = tensor(&shape, data);
        let w = tensor(&[37], (0..37).map(|i| 0.5 + i as f32 * 0.05).collect());
        let b = tensor(&[37], (0..37).map(|i| (i as f32 * 0.3).cos()).collect());

        let scalar = crate::ops::simd::with_avx2_override(false, || {
            layer_norm(&t, &[37], Some(&w), Some(&b), 1e-5).expect("layernorm")
        });
        let fma = crate::ops::simd::with_avx2_override(true, || {
            layer_norm(&t, &[37], Some(&w), Some(&b), 1e-5).expect("layernorm")
        });

        for (i, (&a, &b)) in fma.as_slice().iter().zip(scalar.as_slice()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
                "fma-vs-scalar mismatch at {i}: fma={a} scalar={b}"
            );
        }
    }
    /// `par_chunks_mut(0)` / `chunks_mut(0)` panic on a zero-length chunk size, and
    /// `normalized_shape` reaches this `pub fn` from a caller directly.
    #[test]
    fn zero_normalized_shape_does_not_panic() {
        let t = tensor(&[2, 0], vec![]);
        let r = std::panic::catch_unwind(|| layer_norm(&t, &[0], None, None, 1e-5));
        assert!(r.is_ok(), "a zero normalized_shape must not panic");
        let out = r.expect("no panic").expect("not an error");
        assert_eq!(out.shape(), &[2, 0]);
        assert_eq!(out.numel(), 0);
    }

    /// The AVX2 kernel loads `weight` and `bias` with unchecked `_mm256_loadu_ps` over
    /// `norm_size` elements, and the length was never compared.
    #[test]
    fn a_wrong_length_weight_or_bias_is_an_error() {
        let t = tensor(&[1, 8], (0..8).map(|i| i as f32).collect::<Vec<f32>>());
        for n in [0usize, 4, 7, 9] {
            let w = tensor(&[n], vec![1.0; n]);
            let err = layer_norm(&t, &[8], Some(&w), None, 1e-5)
                .expect_err(&format!("weight of {n} must be rejected"));
            assert!(
                matches!(err, crate::error::OcrError::OperatorError { .. }),
                "weight of {n}: got {err:?}"
            );

            let b = tensor(&[n], vec![0.0; n]);
            let err = layer_norm(&t, &[8], None, Some(&b), 1e-5)
                .expect_err(&format!("bias of {n} must be rejected"));
            assert!(
                matches!(err, crate::error::OcrError::OperatorError { .. }),
                "bias of {n}: got {err:?}"
            );
        }
        // The matching length still works.
        let w = tensor(&[8], vec![1.0; 8]);
        let b = tensor(&[8], vec![0.5; 8]);
        layer_norm(&t, &[8], Some(&w), Some(&b), 1e-5).expect("a valid weight/bias must work");
    }
}
