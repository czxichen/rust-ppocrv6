use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvActivation {
    None,
    Relu,
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    /// `x * clamp(x/6 + 0.5, 0, 1)`, i.e. `x * HardSigmoid_{1/6, 0.5}(x)`.
    ///
    /// Produced by the graph optimizer when it folds `Mul(x, HardSigmoid(x))` (or a
    /// `Conv -> HardSigmoid -> Mul` chain) into a single fused node.
    HardSwish,
    Gelu,
}

#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn apply_conv_activation(val: f32, act: ConvActivation) -> f32 {
    match act {
        ConvActivation::None => val,
        ConvActivation::Relu => val.max(0.0),
        ConvActivation::HardSigmoid { alpha, beta } => (val * alpha + beta).clamp(0.0, 1.0),
        ConvActivation::HardSwish => val * ((val + 3.0).clamp(0.0, 6.0) / 6.0),
        ConvActivation::Gelu => {
            let v = val;
            let scaled = v * 0.7071067811865475;
            let sign = if scaled >= 0.0 { 1.0 } else { -1.0 };
            let ax = scaled.abs();
            let p = 0.3275911;
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let t = 1.0 / (1.0 + p * ax);
            let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
            let erf_val = sign * (1.0 - poly * (-ax * ax).exp());
            0.5 * v * (1.0 + erf_val)
        }
    }
}

/// GELU on a 256-bit vector.
///
/// # Safety
/// The caller must have verified that the running CPU implements AVX2 **and** FMA,
/// e.g. through [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::excessive_precision, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn avx2_gelu(v: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    unsafe {
        use std::arch::x86_64::*;
        let v_sqrt2_inv = _mm256_set1_ps(0.7071067811865475);
        let v_p = _mm256_set1_ps(0.3275911);
        let v_a1 = _mm256_set1_ps(0.254829592);
        let v_a2 = _mm256_set1_ps(-0.284496736);
        let v_a3 = _mm256_set1_ps(1.421413741);
        let v_a4 = _mm256_set1_ps(-1.453152027);
        let v_a5 = _mm256_set1_ps(1.061405429);
        let v_one = _mm256_set1_ps(1.0);
        let v_half = _mm256_set1_ps(0.5);
        let sign_mask = _mm256_set1_ps(-0.0);

        let scaled = _mm256_mul_ps(v, v_sqrt2_inv);
        let ax = _mm256_andnot_ps(sign_mask, scaled);
        let t = _mm256_div_ps(v_one, _mm256_fmadd_ps(v_p, ax, v_one));
        let mut poly = _mm256_fmadd_ps(v_a5, t, v_a4);
        poly = _mm256_fmadd_ps(poly, t, v_a3);
        poly = _mm256_fmadd_ps(poly, t, v_a2);
        poly = _mm256_fmadd_ps(poly, t, v_a1);
        poly = _mm256_mul_ps(poly, t);

        let neg_ax2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), ax), ax);
        let v_exp = crate::ops::softmax::avx2_exp_ps(neg_ax2);

        let erf_abs = _mm256_fnmadd_ps(poly, v_exp, v_one);
        let sign = _mm256_blendv_ps(v_one, _mm256_set1_ps(-1.0), scaled);
        let erf_val = _mm256_mul_ps(sign, erf_abs);
        _mm256_mul_ps(_mm256_mul_ps(v_half, v), _mm256_add_ps(v_one, erf_val))
    }
}

/// HardSwish on a 256-bit vector: `v * clamp(v*(1/6) + 0.5, 0, 1)`.
///
/// # Safety
/// The caller must have verified that the running CPU implements AVX2 **and** FMA,
/// e.g. through [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
#[inline]
pub unsafe fn avx2_hardswish(v: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let v_alpha = _mm256_set1_ps(1.0 / 6.0);
    let v_beta = _mm256_set1_ps(0.5);
    let v_zero = _mm256_setzero_ps();
    let v_one = _mm256_set1_ps(1.0);
    let gate = _mm256_min_ps(
        _mm256_max_ps(_mm256_fmadd_ps(v, v_alpha, v_beta), v_zero),
        v_one,
    );
    _mm256_mul_ps(v, gate)
}

/// Add `bias` to a row and apply `act`, using AVX2+FMA when available.
///
/// This stays a safe function: it probes the CPU itself and falls back to a
/// scalar loop. (Previously it was safe but emitted AVX2 unconditionally.)
#[inline]
pub fn apply_bias_act_simd(row: &mut [f32], bias: f32, act: ConvActivation) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        unsafe { apply_bias_act_fma(row, bias, act) };
        return;
    }
    apply_bias_act_scalar(row, bias, act);
}

/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_bias_act_fma(row: &mut [f32], bias: f32, act: ConvActivation) {
    unsafe {
        apply_bias_act_fma_rows(row.as_mut_ptr(), 1, 0, row.len(), Some(&[bias]), 0, act);
    }
}

/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
unsafe fn apply_bias_act_fma_ptr(ptr: *mut f32, n: usize, bias: f32, act: ConvActivation) {
    unsafe {
        apply_bias_act_fma_rows(ptr, 1, 0, n, Some(&[bias]), 0, act);
    }
}

/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_bias_act_fma_rows(
    out_ptr: *mut f32,
    num_rows: usize,
    row_stride: usize,
    n: usize,
    bias_slice: Option<&[f32]>,
    bias_offset: usize,
    act: ConvActivation,
) {
    unsafe {
        use std::arch::x86_64::*;
        let bias_ptr = bias_slice.map(|b| b.as_ptr().add(bias_offset));
        match act {
            ConvActivation::None => {
                for r in 0..num_rows {
                    let bias = match bias_ptr {
                        Some(bp) => *bp.add(r),
                        None => 0.0,
                    };
                    let ptr = out_ptr.add(r * row_stride);
                    let v_bias = _mm256_set1_ps(bias);
                    let mut i = 0;
                    while i + 64 <= n {
                        let v0 = _mm256_loadu_ps(ptr.add(i));
                        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                        let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                        let v4 = _mm256_loadu_ps(ptr.add(i + 32));
                        let v5 = _mm256_loadu_ps(ptr.add(i + 40));
                        let v6 = _mm256_loadu_ps(ptr.add(i + 48));
                        let v7 = _mm256_loadu_ps(ptr.add(i + 56));
                        _mm256_storeu_ps(ptr.add(i), _mm256_add_ps(v0, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 8), _mm256_add_ps(v1, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 16), _mm256_add_ps(v2, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 24), _mm256_add_ps(v3, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 32), _mm256_add_ps(v4, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 40), _mm256_add_ps(v5, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 48), _mm256_add_ps(v6, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 56), _mm256_add_ps(v7, v_bias));
                        i += 64;
                    }
                    while i + 32 <= n {
                        let v0 = _mm256_loadu_ps(ptr.add(i));
                        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                        let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                        _mm256_storeu_ps(ptr.add(i), _mm256_add_ps(v0, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 8), _mm256_add_ps(v1, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 16), _mm256_add_ps(v2, v_bias));
                        _mm256_storeu_ps(ptr.add(i + 24), _mm256_add_ps(v3, v_bias));
                        i += 32;
                    }
                    while i + 8 <= n {
                        let v = _mm256_loadu_ps(ptr.add(i));
                        _mm256_storeu_ps(ptr.add(i), _mm256_add_ps(v, v_bias));
                        i += 8;
                    }
                    while i < n {
                        *ptr.add(i) += bias;
                        i += 1;
                    }
                }
            }
            ConvActivation::Relu => {
                let zero = _mm256_setzero_ps();
                for r in 0..num_rows {
                    let bias = match bias_ptr {
                        Some(bp) => *bp.add(r),
                        None => 0.0,
                    };
                    let ptr = out_ptr.add(r * row_stride);
                    let v_bias = _mm256_set1_ps(bias);
                    let mut i = 0;
                    while i + 64 <= n {
                        let v0 = _mm256_loadu_ps(ptr.add(i));
                        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                        let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                        let v4 = _mm256_loadu_ps(ptr.add(i + 32));
                        let v5 = _mm256_loadu_ps(ptr.add(i + 40));
                        let v6 = _mm256_loadu_ps(ptr.add(i + 48));
                        let v7 = _mm256_loadu_ps(ptr.add(i + 56));
                        _mm256_storeu_ps(ptr.add(i), _mm256_max_ps(_mm256_add_ps(v0, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 8), _mm256_max_ps(_mm256_add_ps(v1, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 16), _mm256_max_ps(_mm256_add_ps(v2, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 24), _mm256_max_ps(_mm256_add_ps(v3, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 32), _mm256_max_ps(_mm256_add_ps(v4, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 40), _mm256_max_ps(_mm256_add_ps(v5, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 48), _mm256_max_ps(_mm256_add_ps(v6, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 56), _mm256_max_ps(_mm256_add_ps(v7, v_bias), zero));
                        i += 64;
                    }
                    while i + 32 <= n {
                        let v0 = _mm256_loadu_ps(ptr.add(i));
                        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                        let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                        _mm256_storeu_ps(ptr.add(i), _mm256_max_ps(_mm256_add_ps(v0, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 8), _mm256_max_ps(_mm256_add_ps(v1, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 16), _mm256_max_ps(_mm256_add_ps(v2, v_bias), zero));
                        _mm256_storeu_ps(ptr.add(i + 24), _mm256_max_ps(_mm256_add_ps(v3, v_bias), zero));
                        i += 32;
                    }
                    while i + 8 <= n {
                        let v = _mm256_loadu_ps(ptr.add(i));
                        let res = _mm256_max_ps(_mm256_add_ps(v, v_bias), zero);
                        _mm256_storeu_ps(ptr.add(i), res);
                        i += 8;
                    }
                    while i < n {
                        *ptr.add(i) = (*ptr.add(i) + bias).max(0.0);
                        i += 1;
                    }
                }
            }
            ConvActivation::HardSigmoid { alpha, beta } => {
                let v_alpha = _mm256_set1_ps(alpha);
                let v_beta = _mm256_set1_ps(beta);
                let zero = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);
                for r in 0..num_rows {
                    let bias = match bias_ptr {
                        Some(bp) => *bp.add(r),
                        None => 0.0,
                    };
                    let ptr = out_ptr.add(r * row_stride);
                    let v_bias = _mm256_set1_ps(bias);
                    let mut i = 0;
                    while i + 32 <= n {
                        let v0 = _mm256_loadu_ps(ptr.add(i));
                        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                        let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                        let s0 = _mm256_fmadd_ps(_mm256_add_ps(v0, v_bias), v_alpha, v_beta);
                        let s1 = _mm256_fmadd_ps(_mm256_add_ps(v1, v_bias), v_alpha, v_beta);
                        let s2 = _mm256_fmadd_ps(_mm256_add_ps(v2, v_bias), v_alpha, v_beta);
                        let s3 = _mm256_fmadd_ps(_mm256_add_ps(v3, v_bias), v_alpha, v_beta);
                        _mm256_storeu_ps(ptr.add(i), _mm256_min_ps(_mm256_max_ps(s0, zero), one));
                        _mm256_storeu_ps(ptr.add(i + 8), _mm256_min_ps(_mm256_max_ps(s1, zero), one));
                        _mm256_storeu_ps(ptr.add(i + 16), _mm256_min_ps(_mm256_max_ps(s2, zero), one));
                        _mm256_storeu_ps(ptr.add(i + 24), _mm256_min_ps(_mm256_max_ps(s3, zero), one));
                        i += 32;
                    }
                    while i + 8 <= n {
                        let v = _mm256_loadu_ps(ptr.add(i));
                        let added = _mm256_add_ps(v, v_bias);
                        let scaled = _mm256_fmadd_ps(added, v_alpha, v_beta);
                        let res = _mm256_min_ps(_mm256_max_ps(scaled, zero), one);
                        _mm256_storeu_ps(ptr.add(i), res);
                        i += 8;
                    }
                    while i < n {
                        let v = (*ptr.add(i) + bias) * alpha + beta;
                        *ptr.add(i) = v.clamp(0.0, 1.0);
                        i += 1;
                    }
                }
            }
            ConvActivation::HardSwish => {
                let v_alpha = _mm256_set1_ps(1.0 / 6.0);
                let v_beta = _mm256_set1_ps(0.5);
                let zero = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);
                for r in 0..num_rows {
                    let bias = match bias_ptr {
                        Some(bp) => *bp.add(r),
                        None => 0.0,
                    };
                    let ptr = out_ptr.add(r * row_stride);
                    let v_bias = _mm256_set1_ps(bias);
                    let mut i = 0;
                    while i + 32 <= n {
                        let v0 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i)), v_bias);
                        let v1 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_bias);
                        let v2 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_bias);
                        let v3 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_bias);
                        let g0 = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(v0, v_alpha, v_beta), zero), one);
                        let g1 = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(v1, v_alpha, v_beta), zero), one);
                        let g2 = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(v2, v_alpha, v_beta), zero), one);
                        let g3 = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(v3, v_alpha, v_beta), zero), one);
                        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v0, g0));
                        _mm256_storeu_ps(ptr.add(i + 8), _mm256_mul_ps(v1, g1));
                        _mm256_storeu_ps(ptr.add(i + 16), _mm256_mul_ps(v2, g2));
                        _mm256_storeu_ps(ptr.add(i + 24), _mm256_mul_ps(v3, g3));
                        i += 32;
                    }
                    while i + 8 <= n {
                        let v = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i)), v_bias);
                        let gate = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(v, v_alpha, v_beta), zero), one);
                        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, gate));
                        i += 8;
                    }
                    while i < n {
                        let v = *ptr.add(i) + bias;
                        *ptr.add(i) = v * ((v + 3.0).clamp(0.0, 6.0)) / 6.0;
                        i += 1;
                    }
                }
            }
            ConvActivation::Gelu => {
                let v_sqrt2_inv = _mm256_set1_ps(0.707_106_77);
                let v_p = _mm256_set1_ps(0.3275911);
                let v_a1 = _mm256_set1_ps(0.254_829_6);
                let v_a2 = _mm256_set1_ps(-0.284_496_72);
                let v_a3 = _mm256_set1_ps(1.421_413_8);
                let v_a4 = _mm256_set1_ps(-1.453_152_1);
                let v_a5 = _mm256_set1_ps(1.061_405_4);
                let v_one = _mm256_set1_ps(1.0);
                let v_half = _mm256_set1_ps(0.5);
                let sign_mask = _mm256_set1_ps(-0.0);

                let eval_gelu_step = |v: __m256| -> __m256 {
                    let scaled = _mm256_mul_ps(v, v_sqrt2_inv);
                    let ax = _mm256_andnot_ps(sign_mask, scaled);
                    let t = _mm256_div_ps(v_one, _mm256_fmadd_ps(v_p, ax, v_one));
                    let mut poly = _mm256_fmadd_ps(v_a5, t, v_a4);
                    poly = _mm256_fmadd_ps(poly, t, v_a3);
                    poly = _mm256_fmadd_ps(poly, t, v_a2);
                    poly = _mm256_fmadd_ps(poly, t, v_a1);
                    poly = _mm256_mul_ps(poly, t);

                    let neg_ax2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), ax), ax);
                    let v_exp = crate::ops::softmax::avx2_exp_ps(neg_ax2);

                    let erf_abs = _mm256_fnmadd_ps(poly, v_exp, v_one);
                    let sign = _mm256_blendv_ps(v_one, _mm256_set1_ps(-1.0), scaled);
                    let erf_val = _mm256_mul_ps(sign, erf_abs);
                    _mm256_mul_ps(_mm256_mul_ps(v_half, v), _mm256_add_ps(v_one, erf_val))
                };

                for r in 0..num_rows {
                    let bias = match bias_ptr {
                        Some(bp) => *bp.add(r),
                        None => 0.0,
                    };
                    let ptr = out_ptr.add(r * row_stride);
                    let v_bias = _mm256_set1_ps(bias);
                    let mut i = 0;
                    while i + 32 <= n {
                        let v0 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i)), v_bias);
                        let v1 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_bias);
                        let v2 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_bias);
                        let v3 = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_bias);
                        _mm256_storeu_ps(ptr.add(i), eval_gelu_step(v0));
                        _mm256_storeu_ps(ptr.add(i + 8), eval_gelu_step(v1));
                        _mm256_storeu_ps(ptr.add(i + 16), eval_gelu_step(v2));
                        _mm256_storeu_ps(ptr.add(i + 24), eval_gelu_step(v3));
                        i += 32;
                    }
                    while i + 8 <= n {
                        let v = _mm256_add_ps(_mm256_loadu_ps(ptr.add(i)), v_bias);
                        _mm256_storeu_ps(ptr.add(i), eval_gelu_step(v));
                        i += 8;
                    }
                    while i < n {
                        *ptr.add(i) = apply_conv_activation(*ptr.add(i) + bias, act);
                        i += 1;
                    }
                }
            }
        }
    }
}

fn apply_bias_act_scalar(row: &mut [f32], bias: f32, act: ConvActivation) {
    for v in row.iter_mut() {
        *v = apply_conv_activation(*v + bias, act);
    }
}

pub struct Conv2dParams {
    pub strides: [usize; 2],
    pub pads: [usize; 4], // [top, left, bottom, right]
    pub dilations: [usize; 2],
    pub group: usize,
    pub activation: ConvActivation,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
            group: 1,
            activation: ConvActivation::None,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_act_avx2(
    v: std::arch::x86_64::__m256,
    act: ConvActivation,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        match act {
            ConvActivation::None => v,
            ConvActivation::Relu => _mm256_max_ps(v, _mm256_setzero_ps()),
            ConvActivation::HardSigmoid { alpha, beta } => {
                let scaled = _mm256_fmadd_ps(v, _mm256_set1_ps(alpha), _mm256_set1_ps(beta));
                _mm256_min_ps(_mm256_max_ps(scaled, _mm256_setzero_ps()), _mm256_set1_ps(1.0))
            }
            ConvActivation::HardSwish => avx2_hardswish(v),
            ConvActivation::Gelu => avx2_gelu(v),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw_stride1_d1_row_fma(
    out_row_ptr: *mut f32,
    row_ptrs: &[*const f32],
    w_ptr: *const f32,
    k_h: usize,
    k_w: usize,
    pad_left: usize,
    c_bias: f32,
    act: ConvActivation,
    mut ow: usize,
    interior_end: usize,
) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let b_vec = _mm256_set1_ps(c_bias);

        while ow + 16 <= interior_end {
            let biw = ow - pad_left;
            let mut acc0 = b_vec;
            let mut acc1 = b_vec;

            for kh in 0..k_h {
                let in_row = row_ptrs[kh];
                if in_row.is_null() {
                    continue;
                }
                let w_row = w_ptr.add(kh * k_w);
                for kw in 0..k_w {
                    let vw = _mm256_set1_ps(*w_row.add(kw));
                    let vx0 = _mm256_loadu_ps(in_row.add(biw + kw));
                    let vx1 = _mm256_loadu_ps(in_row.add(biw + kw + 8));
                    acc0 = _mm256_fmadd_ps(vw, vx0, acc0);
                    acc1 = _mm256_fmadd_ps(vw, vx1, acc1);
                }
            }

            let res0 = apply_act_avx2(acc0, act);
            let res1 = apply_act_avx2(acc1, act);
            _mm256_storeu_ps(out_row_ptr.add(ow), res0);
            _mm256_storeu_ps(out_row_ptr.add(ow + 8), res1);
            ow += 16;
        }

        while ow + 8 <= interior_end {
            let biw = ow - pad_left;
            let mut acc = b_vec;

            for kh in 0..k_h {
                let in_row = row_ptrs[kh];
                if in_row.is_null() {
                    continue;
                }
                let w_row = w_ptr.add(kh * k_w);
                for kw in 0..k_w {
                    let vw = _mm256_set1_ps(*w_row.add(kw));
                    let vx = _mm256_loadu_ps(in_row.add(biw + kw));
                    acc = _mm256_fmadd_ps(vw, vx, acc);
                }
            }

            let res = apply_act_avx2(acc, act);
            _mm256_storeu_ps(out_row_ptr.add(ow), res);
            ow += 8;
        }

        ow
    }
}

/// Depthwise 3x3 / stride-1 / dilation-1 inner loop for one output row.
/// Returns the first output column still needing the scalar tail.
///
/// # Safety
/// Requires AVX2+FMA (see [`crate::ops::avx2_fma_enabled`]). `ow_start` must be
/// `>= pad_left`, `out_row_ptr` writable for `out_w` elements, and every `r*`
/// offset must be in range for the `in_ptr` allocation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3x3_accum_row_fma(
    out_row_ptr: *mut f32,
    in_ptr: *const f32,
    ow_start: usize,
    pad_left: usize,
    in_w: usize,
    out_w: usize,
    w: [f32; 9],
    c_bias: f32,
    act: ConvActivation,
    r0: Option<usize>,
    r1: Option<usize>,
    r2: Option<usize>,
) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let vw0 = _mm256_set1_ps(w[0]);
        let vw1 = _mm256_set1_ps(w[1]);
        let vw2 = _mm256_set1_ps(w[2]);
        let vw3 = _mm256_set1_ps(w[3]);
        let vw4 = _mm256_set1_ps(w[4]);
        let vw5 = _mm256_set1_ps(w[5]);
        let vw6 = _mm256_set1_ps(w[6]);
        let vw7 = _mm256_set1_ps(w[7]);
        let vw8 = _mm256_set1_ps(w[8]);
        let v_bias = _mm256_set1_ps(c_bias);

        let max_safe_ow = if in_w >= 2 + pad_left {
            (in_w - 2 + pad_left).min(out_w)
        } else {
            0
        };

        let mut ow = ow_start;
        while ow + 8 <= max_safe_ow {
            let biw = ow.saturating_sub(pad_left);
            let mut v_sum = v_bias;

            if let Some(off) = r0 {
                let p = in_ptr.add(off + biw);
                let d0 = _mm256_loadu_ps(p);
                let d1 = _mm256_loadu_ps(p.add(1));
                let d2 = _mm256_loadu_ps(p.add(2));
                v_sum = _mm256_fmadd_ps(d0, vw0, v_sum);
                v_sum = _mm256_fmadd_ps(d1, vw1, v_sum);
                v_sum = _mm256_fmadd_ps(d2, vw2, v_sum);
            }
            if let Some(off) = r1 {
                let p = in_ptr.add(off + biw);
                let d0 = _mm256_loadu_ps(p);
                let d1 = _mm256_loadu_ps(p.add(1));
                let d2 = _mm256_loadu_ps(p.add(2));
                v_sum = _mm256_fmadd_ps(d0, vw3, v_sum);
                v_sum = _mm256_fmadd_ps(d1, vw4, v_sum);
                v_sum = _mm256_fmadd_ps(d2, vw5, v_sum);
            }
            if let Some(off) = r2 {
                let p = in_ptr.add(off + biw);
                let d0 = _mm256_loadu_ps(p);
                let d1 = _mm256_loadu_ps(p.add(1));
                let d2 = _mm256_loadu_ps(p.add(2));
                v_sum = _mm256_fmadd_ps(d0, vw6, v_sum);
                v_sum = _mm256_fmadd_ps(d1, vw7, v_sum);
                v_sum = _mm256_fmadd_ps(d2, vw8, v_sum);
            }

            let v_out = match act {
                ConvActivation::None => v_sum,
                ConvActivation::Relu => _mm256_max_ps(v_sum, _mm256_setzero_ps()),
                ConvActivation::HardSigmoid { alpha, beta } => {
                    let scaled =
                        _mm256_fmadd_ps(v_sum, _mm256_set1_ps(alpha), _mm256_set1_ps(beta));
                    _mm256_min_ps(
                        _mm256_max_ps(scaled, _mm256_setzero_ps()),
                        _mm256_set1_ps(1.0),
                    )
                }
                ConvActivation::HardSwish => avx2_hardswish(v_sum),
                ConvActivation::Gelu => avx2_gelu(v_sum),
            };

            _mm256_storeu_ps(out_row_ptr.add(ow), v_out);
            ow += 8;
        }
        ow
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw5x5_accum_row_fma(
    out_row_ptr: *mut f32,
    in_ptr: *const f32,
    ow_start: usize,
    pad_left: usize,
    in_w: usize,
    out_w: usize,
    w: [f32; 25],
    c_bias: f32,
    act: ConvActivation,
    r: [Option<usize>; 5],
) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let mut vw = [_mm256_setzero_ps(); 25];
        for i in 0..25 {
            vw[i] = _mm256_set1_ps(w[i]);
        }
        let v_bias = _mm256_set1_ps(c_bias);

        let max_safe_ow = if in_w >= 4 + pad_left {
            (in_w - 4 + pad_left).min(out_w)
        } else {
            0
        };

        let mut ow = ow_start;
        while ow + 8 <= max_safe_ow {
            let biw = ow.saturating_sub(pad_left);
            let mut v_sum = v_bias;

            for kh in 0..5 {
                if let Some(off) = r[kh] {
                    let p = in_ptr.add(off + biw);
                    let d0 = _mm256_loadu_ps(p);
                    let d1 = _mm256_loadu_ps(p.add(1));
                    let d2 = _mm256_loadu_ps(p.add(2));
                    let d3 = _mm256_loadu_ps(p.add(3));
                    let d4 = _mm256_loadu_ps(p.add(4));
                    let w_base = kh * 5;
                    v_sum = _mm256_fmadd_ps(d0, vw[w_base], v_sum);
                    v_sum = _mm256_fmadd_ps(d1, vw[w_base + 1], v_sum);
                    v_sum = _mm256_fmadd_ps(d2, vw[w_base + 2], v_sum);
                    v_sum = _mm256_fmadd_ps(d3, vw[w_base + 3], v_sum);
                    v_sum = _mm256_fmadd_ps(d4, vw[w_base + 4], v_sum);
                }
            }

            let v_out = match act {
                ConvActivation::None => v_sum,
                ConvActivation::Relu => _mm256_max_ps(v_sum, _mm256_setzero_ps()),
                ConvActivation::HardSigmoid { alpha, beta } => {
                    let scaled =
                        _mm256_fmadd_ps(v_sum, _mm256_set1_ps(alpha), _mm256_set1_ps(beta));
                    _mm256_min_ps(
                        _mm256_max_ps(scaled, _mm256_setzero_ps()),
                        _mm256_set1_ps(1.0),
                    )
                }
                ConvActivation::HardSwish => avx2_hardswish(v_sum),
                ConvActivation::Gelu => avx2_gelu(v_sum),
            };

            _mm256_storeu_ps(out_row_ptr.add(ow), v_out);
            ow += 8;
        }
        ow
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: *const f32, b: *const f32, n: usize) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut v_acc0 = _mm256_setzero_ps();
        let mut v_acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let va0 = _mm256_loadu_ps(a.add(i));
            let vb0 = _mm256_loadu_ps(b.add(i));
            v_acc0 = _mm256_fmadd_ps(va0, vb0, v_acc0);
            let va1 = _mm256_loadu_ps(a.add(i + 8));
            let vb1 = _mm256_loadu_ps(b.add(i + 8));
            v_acc1 = _mm256_fmadd_ps(va1, vb1, v_acc1);
            i += 16;
        }
        while i + 8 <= n {
            let va0 = _mm256_loadu_ps(a.add(i));
            let vb0 = _mm256_loadu_ps(b.add(i));
            v_acc0 = _mm256_fmadd_ps(va0, vb0, v_acc0);
            i += 8;
        }
        let v_sum = _mm256_add_ps(v_acc0, v_acc1);
        let hi128 = _mm256_extractf128_ps(v_sum, 1);
        let lo128 = _mm256_castps256_ps128(v_sum);
        let sum128 = _mm_add_ps(lo128, hi128);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sum64, hi32));
        while i < n {
            sum += *a.add(i) * *b.add(i);
            i += 1;
        }
        sum
    }
}

/// # Safety
/// `a` and `b` must point to at least `n` valid floats.
#[inline]
pub(crate) unsafe fn dot_product_fma(a: *const f32, b: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        return unsafe { dot_product_avx2_fma(a, b, n) };
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += unsafe { *a.add(i) * *b.add(i) };
    }
    sum
}

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2dParams,
) -> Result<Tensor> {
    if input.ndim() != 4 || weight.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: format!(
                "Input and weight must be 4D tensors, got input ndim={}, weight ndim={}",
                input.ndim(),
                weight.ndim()
            ),
        });
    }

    let input_shape = input.shape();
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let weight_shape = weight.shape();
    let out_channels = weight_shape[0];
    let kernel_in_channels = weight_shape[1];
    let k_h = weight_shape[2];
    let k_w = weight_shape[3];

    let group = params.group;
    // `group` reaches here straight from the graph attribute. `engine::model::group_attr`
    // rejects a negative value, but `conv2d` is a `pub fn` and must not divide by zero on
    // its own account either — `conv_transpose2d` already guarded this.
    if group == 0 {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: "group must be at least 1".to_string(),
        });
    }
    if in_channels % group != 0 || out_channels % group != 0 {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: format!(
                "in_channels ({}) and out_channels ({}) must be divisible by group ({})",
                in_channels, out_channels, group
            ),
        });
    }

    if kernel_in_channels != in_channels / group {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: format!(
                "Weight in_channels ({}) must match in_channels / group ({})",
                kernel_in_channels,
                in_channels / group
            ),
        });
    }

    // A zero-sized kernel would underflow `k - 1`: a debug panic, and a silent wrap in
    // release that then feeds a nonsense `eff_k` into the output-size computation and a
    // wrongly-sized `Tensor::empty`. `conv_transpose2d` uses `saturating_sub(1)`; the two
    // files disagreed, so both now reject it outright.
    if k_h == 0 || k_w == 0 {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: format!("kernel must be at least 1x1, got {k_h}x{k_w}"),
        });
    }

    // A bias whose length disagrees with `out_channels` used to be indexed with no check,
    // panicking inside the kernel. `batchnorm` validates its equivalent; this brings the
    // convolution in line.
    if let Some(b) = bias
        && b.numel() != out_channels
    {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: format!(
                "bias has {} elements but out_channels is {out_channels}",
                b.numel()
            ),
        });
    }

    let [stride_h, stride_w] = params.strides;
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;
    let [dilation_h, dilation_w] = params.dilations;

    if stride_h == 0 || stride_w == 0 {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: format!("strides must be at least 1x1, got {stride_h}x{stride_w}"),
        });
    }

    let eff_k_h = dilation_h * (k_h - 1) + 1;
    let eff_k_w = dilation_w * (k_w - 1) + 1;

    if in_h + pad_top + pad_bottom < eff_k_h || in_w + pad_left + pad_right < eff_k_w {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: "Input spatial dimensions + padding smaller than dilated kernel".into(),
        });
    }

    let out_h = (in_h + pad_top + pad_bottom - eff_k_h) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right - eff_k_w) / stride_w + 1;

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

    // Special Fast Path: Depthwise Convolution (group == in_channels == out_channels)
    if group == in_channels && in_channels == out_channels && kernel_in_channels == 1 {
        let is_3x3_d1 = k_h == 3 && k_w == 3 && dilation_h == 1 && dilation_w == 1;
        let is_5x5_d1_sw1 =
            k_h == 5 && k_w == 5 && dilation_h == 1 && dilation_w == 1 && stride_w == 1;

        for b in 0..batch_size {
            let in_batch_offset = b * in_channels * in_h * in_w;
            let out_batch_offset = b * out_channels * out_h * out_w;
            let out_hw = out_h * out_w;
            let out_batch_slice =
                &mut out_slice[out_batch_offset..out_batch_offset + out_channels * out_hw];

            let dw_work = out_channels * out_h * out_w;
            let run_channel = |(c, out_channel_chunk): (usize, &mut [f32])| {
                let w_offset = c * k_h * k_w;
                let in_c_offset = in_batch_offset + c * in_h * in_w;
                let c_bias = bias_slice.map(|bs| bs[c]).unwrap_or(0.0);

                let out_ptr = out_channel_chunk.as_mut_ptr();
                let in_ptr = unsafe { in_slice.as_ptr().add(in_c_offset) };
                let w_ptr = unsafe { weight_slice.as_ptr().add(w_offset) };

                let is_stride1_d1 =
                    stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
                if !is_3x3_d1 && is_stride1_d1 {
                    let act = params.activation;
                    let pad_left_i = pad_left as isize;
                    let pad_top_i = pad_top as isize;

                    let interior_end = if in_w >= k_w - 1 + pad_left {
                        (in_w - (k_w - 1) + pad_left).min(out_w)
                    } else {
                        0
                    };

                    for oh in 0..out_h {
                        let out_row_ptr = unsafe { out_ptr.add(oh * out_w) };
                        let base_ih = oh as isize - pad_top_i;

                        let mut row_ptrs = [std::ptr::null::<f32>(); 16];
                        for kh in 0..k_h {
                            let ih = base_ih + kh as isize;
                            if ih >= 0 && (ih as usize) < in_h {
                                row_ptrs[kh] = unsafe { in_ptr.add(ih as usize * in_w) };
                            }
                        }

                        // Left boundary scalar
                        let left_boundary = pad_left.min(out_w);
                        for ow in 0..left_boundary {
                            let base_iw = ow as isize - pad_left_i;
                            let mut sum = c_bias;
                            for kh in 0..k_h {
                                let in_row = row_ptrs[kh];
                                if in_row.is_null() {
                                    continue;
                                }
                                let w_row = unsafe { w_ptr.add(kh * k_w) };
                                for kw in 0..k_w {
                                    let iw = base_iw + kw as isize;
                                    if iw >= 0 && (iw as usize) < in_w {
                                        unsafe {
                                            sum += *in_row.add(iw as usize) * *w_row.add(kw);
                                        }
                                    }
                                }
                            }
                            unsafe {
                                *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                            }
                        }

                        // Interior vectorised path
                        let mut ow = left_boundary;
                        #[cfg(target_arch = "x86_64")]
                        if crate::ops::avx2_fma_enabled() {
                            ow = unsafe {
                                dw_stride1_d1_row_fma(
                                    out_row_ptr,
                                    &row_ptrs[..k_h],
                                    w_ptr,
                                    k_h,
                                    k_w,
                                    pad_left,
                                    c_bias,
                                    act,
                                    ow,
                                    interior_end,
                                )
                            };
                        }

                        // Remainder / Right boundary scalar
                        while ow < out_w {
                            let base_iw = ow as isize - pad_left_i;
                            let mut sum = c_bias;
                            for kh in 0..k_h {
                                let in_row = row_ptrs[kh];
                                if in_row.is_null() {
                                    continue;
                                }
                                let w_row = unsafe { w_ptr.add(kh * k_w) };
                                for kw in 0..k_w {
                                    let iw = base_iw + kw as isize;
                                    if iw >= 0 && (iw as usize) < in_w {
                                        unsafe {
                                            sum += *in_row.add(iw as usize) * *w_row.add(kw);
                                        }
                                    }
                                }
                            }
                            unsafe {
                                *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                            }
                            ow += 1;
                        }
                    }
                    return;
                }

                if is_3x3_d1 {
                    let w0 = unsafe { *w_ptr };
                    let w1 = unsafe { *w_ptr.add(1) };
                    let w2 = unsafe { *w_ptr.add(2) };
                    let w3 = unsafe { *w_ptr.add(3) };
                    let w4 = unsafe { *w_ptr.add(4) };
                    let w5 = unsafe { *w_ptr.add(5) };
                    let w6 = unsafe { *w_ptr.add(6) };
                    let w7 = unsafe { *w_ptr.add(7) };
                    let w8 = unsafe { *w_ptr.add(8) };

                    let act = params.activation;

                    for oh in 0..out_h {
                        let base_ih = (oh * stride_h) as isize - pad_top as isize;
                        let r0 = if base_ih >= 0 && (base_ih as usize) < in_h {
                            Some(base_ih as usize * in_w)
                        } else {
                            None
                        };
                        let r1 = if base_ih + 1 >= 0 && ((base_ih + 1) as usize) < in_h {
                            Some((base_ih + 1) as usize * in_w)
                        } else {
                            None
                        };
                        let r2 = if base_ih + 2 >= 0 && ((base_ih + 2) as usize) < in_h {
                            Some((base_ih + 2) as usize * in_w)
                        } else {
                            None
                        };

                        let out_row_ptr = unsafe { out_ptr.add(oh * out_w) };

                        if stride_w == 1 {
                            let mut ow = 0;
                            // Left boundary scalar
                            while ow < out_w && (ow as isize - pad_left as isize) < 0 {
                                let base_iw = ow as isize - pad_left as isize;
                                let mut sum = c_bias;
                                for kh in 0..3 {
                                    let ih = base_ih + kh as isize;
                                    if ih >= 0 && (ih as usize) < in_h {
                                        let in_row = ih as usize * in_w;
                                        let w_row = kh * 3;
                                        for kw in 0..3 {
                                            let iw = base_iw + kw as isize;
                                            if iw >= 0 && (iw as usize) < in_w {
                                                unsafe {
                                                    sum += *in_ptr.add(in_row + iw as usize)
                                                        * *w_ptr.add(w_row + kw);
                                                }
                                            }
                                        }
                                    }
                                }
                                unsafe {
                                    *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                                }
                                ow += 1;
                            }

                            // Interior vectorised path
                            #[cfg(target_arch = "x86_64")]
                            if crate::ops::avx2_fma_enabled() {
                                ow = unsafe {
                                    dw3x3_accum_row_fma(
                                        out_row_ptr,
                                        in_ptr,
                                        ow,
                                        pad_left,
                                        in_w,
                                        out_w,
                                        [w0, w1, w2, w3, w4, w5, w6, w7, w8],
                                        c_bias,
                                        act,
                                        r0,
                                        r1,
                                        r2,
                                    )
                                };
                            }

                            // Remainder / Right boundary scalar
                            while ow < out_w {
                                let base_iw = ow as isize - pad_left as isize;
                                let mut sum = c_bias;
                                if base_iw >= 0 && ((base_iw + 2) as usize) < in_w {
                                    let biw = base_iw as usize;
                                    if let Some(off) = r0 {
                                        unsafe {
                                            let p = in_ptr.add(off + biw);
                                            sum += *p * w0 + *p.add(1) * w1 + *p.add(2) * w2;
                                        }
                                    }
                                    if let Some(off) = r1 {
                                        unsafe {
                                            let p = in_ptr.add(off + biw);
                                            sum += *p * w3 + *p.add(1) * w4 + *p.add(2) * w5;
                                        }
                                    }
                                    if let Some(off) = r2 {
                                        unsafe {
                                            let p = in_ptr.add(off + biw);
                                            sum += *p * w6 + *p.add(1) * w7 + *p.add(2) * w8;
                                        }
                                    }
                                } else {
                                    for kh in 0..3 {
                                        let ih = base_ih + kh as isize;
                                        if ih >= 0 && (ih as usize) < in_h {
                                            let in_row = ih as usize * in_w;
                                            let w_row = kh * 3;
                                            for kw in 0..3 {
                                                let iw = base_iw + kw as isize;
                                                if iw >= 0 && (iw as usize) < in_w {
                                                    unsafe {
                                                        sum += *in_ptr.add(in_row + iw as usize)
                                                            * *w_ptr.add(w_row + kw);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                unsafe {
                                    *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                                }
                                ow += 1;
                            }
                            continue;
                        }

                        // General stride fallback
                        for ow in 0..out_w {
                            let base_iw = (ow * stride_w) as isize - pad_left as isize;
                            let mut sum = c_bias;

                            if base_iw >= 0 && ((base_iw + 2) as usize) < in_w {
                                let biw = base_iw as usize;
                                if let Some(off) = r0 {
                                    unsafe {
                                        let p = in_ptr.add(off + biw);
                                        sum += *p * w0 + *p.add(1) * w1 + *p.add(2) * w2;
                                    }
                                }
                                if let Some(off) = r1 {
                                    unsafe {
                                        let p = in_ptr.add(off + biw);
                                        sum += *p * w3 + *p.add(1) * w4 + *p.add(2) * w5;
                                    }
                                }
                                if let Some(off) = r2 {
                                    unsafe {
                                        let p = in_ptr.add(off + biw);
                                        sum += *p * w6 + *p.add(1) * w7 + *p.add(2) * w8;
                                    }
                                }
                            } else {
                                for kh in 0..3 {
                                    let ih = base_ih + kh as isize;
                                    if ih >= 0 && (ih as usize) < in_h {
                                        let in_row = ih as usize * in_w;
                                        let w_row = kh * 3;
                                        for kw in 0..3 {
                                            let iw = base_iw + kw as isize;
                                            if iw >= 0 && (iw as usize) < in_w {
                                                unsafe {
                                                    sum += *in_ptr.add(in_row + iw as usize)
                                                        * *w_ptr.add(w_row + kw);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            unsafe {
                                *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                            }
                        }
                    }
                    return;
                }

                if is_5x5_d1_sw1 {
                    let mut w = [0.0f32; 25];
                    for i in 0..25 {
                        w[i] = unsafe { *w_ptr.add(i) };
                    }
                    let act = params.activation;

                    for oh in 0..out_h {
                        let base_ih = (oh * stride_h) as isize - pad_top as isize;
                        let mut r = [None; 5];
                        for kh in 0..5 {
                            let ih = base_ih + kh as isize;
                            if ih >= 0 && (ih as usize) < in_h {
                                r[kh] = Some(ih as usize * in_w);
                            }
                        }

                        let out_row_ptr = unsafe { out_ptr.add(oh * out_w) };
                        let mut ow = 0;

                        // Left boundary scalar
                        while ow < out_w && (ow as isize - pad_left as isize) < 0 {
                            let base_iw = ow as isize - pad_left as isize;
                            let mut sum = c_bias;
                            for kh in 0..5 {
                                if let Some(off) = r[kh] {
                                    let in_row = unsafe { in_ptr.add(off) };
                                    let w_row = kh * 5;
                                    for kw in 0..5 {
                                        let iw = base_iw + kw as isize;
                                        if iw >= 0 && (iw as usize) < in_w {
                                            unsafe {
                                                sum += *in_row.add(iw as usize) * w[w_row + kw];
                                            }
                                        }
                                    }
                                }
                            }
                            unsafe {
                                *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                            }
                            ow += 1;
                        }

                        // Interior vectorised path
                        #[cfg(target_arch = "x86_64")]
                        if crate::ops::avx2_fma_enabled() {
                            ow = unsafe {
                                dw5x5_accum_row_fma(
                                    out_row_ptr,
                                    in_ptr,
                                    ow,
                                    pad_left,
                                    in_w,
                                    out_w,
                                    w,
                                    c_bias,
                                    act,
                                    r,
                                )
                            };
                        }

                        // Remainder / Right boundary scalar
                        while ow < out_w {
                            let base_iw = ow as isize - pad_left as isize;
                            let mut sum = c_bias;
                            if base_iw >= 0 && ((base_iw + 4) as usize) < in_w {
                                let biw = base_iw as usize;
                                for kh in 0..5 {
                                    if let Some(off) = r[kh] {
                                        unsafe {
                                            let p = in_ptr.add(off + biw);
                                            sum += *p * w[kh * 5]
                                                + *p.add(1) * w[kh * 5 + 1]
                                                + *p.add(2) * w[kh * 5 + 2]
                                                + *p.add(3) * w[kh * 5 + 3]
                                                + *p.add(4) * w[kh * 5 + 4];
                                        }
                                    }
                                }
                            } else {
                                for kh in 0..5 {
                                    if let Some(off) = r[kh] {
                                        let in_row = unsafe { in_ptr.add(off) };
                                        let w_row = kh * 5;
                                        for kw in 0..5 {
                                            let iw = base_iw + kw as isize;
                                            if iw >= 0 && (iw as usize) < in_w {
                                                unsafe {
                                                    sum += *in_row.add(iw as usize) * w[w_row + kw];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            unsafe {
                                *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                            }
                            ow += 1;
                        }
                    }
                    return;
                }

                // Generic Depthwise fallback
                for oh in 0..out_h {
                    let base_ih = (oh * stride_h) as isize - pad_top as isize;
                    for ow in 0..out_w {
                        let base_iw = (ow * stride_w) as isize - pad_left as isize;
                        let mut sum = c_bias;

                        for kh in 0..k_h {
                            let ih = base_ih + (kh * dilation_h) as isize;
                            if ih >= 0 && ih < in_h as isize {
                                let in_row_offset = ih as usize * in_w;
                                let w_row_offset = kh * k_w;
                                for kw in 0..k_w {
                                    let iw = base_iw + (kw * dilation_w) as isize;
                                    if iw >= 0 && iw < in_w as isize {
                                        unsafe {
                                            let val = *in_ptr.add(in_row_offset + iw as usize);
                                            let w = *w_ptr.add(w_row_offset + kw);
                                            sum += val * w;
                                        }
                                    }
                                }
                            }
                        }

                        unsafe {
                            *out_ptr.add(oh * out_w + ow) =
                                apply_conv_activation(sum, params.activation);
                        }
                    }
                }
            };

            let num_threads = crate::tensor::parallel_degree(1, 1);
            let should_par = if out_channels >= 8 && dw_work >= 4_000 && num_threads > 1 {
                true
            } else {
                crate::tensor::should_parallelize(dw_work, 80_000)
            };
            if should_par {
                out_batch_slice
                    .par_chunks_mut(out_hw)
                    .enumerate()
                    .for_each(run_channel);
            } else {
                out_batch_slice
                    .chunks_mut(out_hw)
                    .enumerate()
                    .for_each(run_channel);
            }
        }

        return Ok(output);
    }

    // Special Fast Path: cin == 3 (RGB input image direct 3x3 convolution)
    if in_channels == 3 && group == 1 && k_h == 3 && k_w == 3 && dilation_h == 1 && dilation_w == 1
    {
        if stride_h == 2 && stride_w == 2 {
            if out_h * out_w <= 32_768 && out_h * out_w >= 128 {
                return conv2d_cin3_k3_s2_gemm(input, weight, bias, params, out_h, out_w);
            }
            return conv2d_direct_cin3_k3_s2(input, weight, bias, params, out_h, out_w);
        }
        let in_hw = in_h * in_w;
        let out_hw = out_h * out_w;

        for b in 0..batch_size {
            let in_b_off = b * in_channels * in_hw;
            let out_b_off = b * out_channels * out_hw;
            let out_batch_slice = &mut out_slice[out_b_off..out_b_off + out_channels * out_hw];
            let c0_off = in_b_off;
            let c1_off = in_b_off + in_hw;
            let c2_off = in_b_off + 2 * in_hw;

            let run_cin3 = |(oc, out_channel_chunk): (usize, &mut [f32])| {
                let w_off = oc * 27;
                let w = &weight_slice[w_off..w_off + 27];
                let b_val = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);
                let out_ptr = out_channel_chunk.as_mut_ptr();

                for oh in 0..out_h {
                    let ih_base = (oh * stride_h) as isize - pad_top as isize;
                    for ow in 0..out_w {
                        let iw_base = (ow * stride_w) as isize - pad_left as isize;
                        let mut sum = b_val;

                        for kh in 0..3 {
                            let ih = ih_base + kh as isize;
                            if ih >= 0 && ih < in_h as isize {
                                let c0_row = c0_off + ih as usize * in_w;
                                let c1_row = c1_off + ih as usize * in_w;
                                let c2_row = c2_off + ih as usize * in_w;
                                let kh_w = kh * 3;

                                for kw in 0..3 {
                                    let iw = iw_base + kw as isize;
                                    if iw >= 0 && iw < in_w as isize {
                                        let iw_u = iw as usize;
                                        sum += in_slice[c0_row + iw_u] * w[kh_w + kw];
                                        sum += in_slice[c1_row + iw_u] * w[9 + kh_w + kw];
                                        sum += in_slice[c2_row + iw_u] * w[18 + kh_w + kw];
                                    }
                                }
                            }
                        }

                        unsafe {
                            *out_ptr.add(oh * out_w + ow) =
                                apply_conv_activation(sum, params.activation);
                        }
                    }
                }
            };

            if crate::tensor::should_parallelize(out_channels, 2) {
                out_batch_slice
                    .par_chunks_mut(out_hw)
                    .enumerate()
                    .for_each(run_cin3);
            } else {
                out_batch_slice
                    .chunks_mut(out_hw)
                    .enumerate()
                    .for_each(run_cin3);
            }
        }
        return Ok(output);
    }

    // 1x1 Conv Fast Path (group == 1, k_h == 1, k_w == 1, stride == [1, 1], pads == [0, 0, 0, 0], dilation == [1, 1])
    let is_1x1 = group == 1
        && k_h == 1
        && k_w == 1
        && stride_h == 1
        && stride_w == 1
        && pad_top == 0
        && pad_left == 0
        && pad_bottom == 0
        && pad_right == 0
        && dilation_h == 1
        && dilation_w == 1;

    if is_1x1 {
        let out_spatial = out_h * out_w;
        let m = out_channels;
        let k = in_channels;
        let n = out_spatial;

        // Ultra-fast Matrix-Vector Product for Squeeze-and-Excitation / Attention blocks (N == 1)
        if n == 1 {
            for b in 0..batch_size {
                let in_batch_offset = b * k;
                let out_batch_offset = b * m;
                let in_vec = &in_slice[in_batch_offset..in_batch_offset + k];
                let out_vec = &mut out_slice[out_batch_offset..out_batch_offset + m];

                let compute_row = |i: usize| -> f32 {
                    let row = &weight_slice[i * k..(i + 1) * k];
                    let dot = unsafe { dot_product_fma(row.as_ptr(), in_vec.as_ptr(), k) }
                        + bias_slice.map(|bs| bs[i]).unwrap_or(0.0);
                    apply_conv_activation(dot, params.activation)
                };

                let flops = 2 * m * k;
                let num_threads = crate::tensor::parallel_degree(1, 1);
                if flops >= 250_000 && m >= 128 && num_threads > 1 {
                    out_vec.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                        *out_val = compute_row(i);
                    });
                } else {
                    for (i, out_val) in out_vec.iter_mut().enumerate() {
                        *out_val = compute_row(i);
                    }
                }
            }
            return Ok(output);
        }

        let num_threads = crate::tensor::parallel_degree(1, 1);
        let flops = 2 * m * k * n;

        // Multi-core parallel strategy:
        // Only parallelize when flops >= 250_000 to avoid Rayon thread scheduling overhead on small layers.
        if flops >= 250_000 && num_threads > 1 {
            // Strategy selection between:
            // - Strategy A: Column splitting (N axis). Preferred for moderate spatial widths
            //   (N <= 4096, including rec model feature maps N=720, 1440, 2880) where GEMM column
            //   microkernel amortizes best across cores.
            // - Strategy B: Output channel row splitting (M axis). Preferred when:
            //   1. N is very large (N >= 8192) and M is moderate (M <= 128, M >= 16), or N >= 16384 (M >= 16),
            //      where strided column writes across a 16MB tensor thrash CPU cache lines.
            //      Row splitting yields 100% contiguous writes per thread and 0 false sharing.
            //   2. N < 256 and M >= 8, where column splitting produces too few chunks (< num_threads).
            let use_strat_b = (n >= 16384 && m >= 16)
                || (n >= 8192 && (16..=128).contains(&m))
                || (n < 256 && m >= 8);

            // Fast Path: Native parallel GEMM for activation None (zero-overhead multi-threaded faer).
            // Avoids redundant multi-core weight packing where column-splitting produces narrow column chunks (<64)
            // or where matrix dimensions balance best with faer's 2D parallel block partitioning.
            if params.activation == ConvActivation::None
                && ((m >= 384 && k == 384) || (m >= 256 && n >= 8192))
            {
                for b in 0..batch_size {
                    let in_batch_offset = b * in_channels * out_spatial;
                    let out_batch_offset = b * out_channels * out_spatial;
                    let out_p = unsafe { out_slice.as_mut_ptr().add(out_batch_offset) };
                    let in_p = unsafe { in_slice.as_ptr().add(in_batch_offset) };

                    if let Some(bias) = bias_slice {
                        let bias_ptr = bias.as_ptr() as usize;
                        let out_p_usize = out_p as usize;
                        let chunk_m = m.div_ceil(num_threads).max(4);
                        (0..m).into_par_iter().step_by(chunk_m).for_each(|r_start| {
                            let out_p = out_p_usize as *mut f32;
                            let r_end = (r_start + chunk_m).min(m);
                            for r in r_start..r_end {
                                let b_val = unsafe { *(bias_ptr as *const f32).add(r) };
                                let ptr = unsafe { out_p.add(r * n) };
                                unsafe {
                                    let slice = std::slice::from_raw_parts_mut(ptr, n);
                                    slice.fill(b_val);
                                }
                            }
                        });

                        unsafe {
                            crate::ops::gemm_row_major_accum(
                                crate::ops::Accum::Add,
                                crate::ops::Par::rayon(num_threads),
                                m,
                                k,
                                n,
                                weight_slice.as_ptr(),
                                k,
                                in_p,
                                n,
                                out_p,
                                n,
                            );
                        }
                    } else {
                        unsafe {
                            crate::ops::gemm_row_major_accum(
                                crate::ops::Accum::Replace,
                                crate::ops::Par::rayon(num_threads),
                                m,
                                k,
                                n,
                                weight_slice.as_ptr(),
                                k,
                                in_p,
                                n,
                                out_p,
                                n,
                            );
                        }
                    }
                }
                return Ok(output);
            }

            if !use_strat_b && n >= 256 {
                let align = if n >= num_threads * 8 { 8 } else { 1 };
                let total_units = n / align;
                let rem_tail = n % align;
                let base_units = total_units / num_threads;
                let extra_units = total_units % num_threads;

                let mut col_chunks = Vec::with_capacity(num_threads);
                let mut cur = 0;
                for i in 0..num_threads {
                    let u = base_units + if i < extra_units { 1 } else { 0 };
                    let mut chunk_len = u * align;
                    if i == num_threads - 1 {
                        chunk_len += rem_tail;
                    }
                    if chunk_len > 0 {
                        col_chunks.push((cur, chunk_len));
                        cur += chunk_len;
                    }
                }

                for b in 0..batch_size {
                    let in_batch_offset = b * in_channels * out_spatial;
                    let out_batch_offset = b * out_channels * out_spatial;
                    let out_ptr_usize = out_slice.as_mut_ptr() as usize;

                    col_chunks.par_iter().for_each(|&(col_start, actual_n)| {
                        let out_p = out_ptr_usize as *mut f32;

                        unsafe {
                            gemm_row_major(
                                m,
                                k,
                                actual_n,
                                n,
                                weight_slice.as_ptr(),
                                in_slice.as_ptr().add(in_batch_offset + col_start),
                                out_p.add(out_batch_offset + col_start),
                            );
                        }

                        let act = params.activation;
                        if bias_slice.is_some() || act != ConvActivation::None {
                            #[cfg(target_arch = "x86_64")]
                            if crate::ops::avx2_fma_enabled() {
                                unsafe {
                                    apply_bias_act_fma_rows(
                                        out_p.add(out_batch_offset + col_start),
                                        m,
                                        n,
                                        actual_n,
                                        bias_slice,
                                        0,
                                        act,
                                    );
                                }
                            } else {
                                for oc in 0..m {
                                    let b_val = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);
                                    let row_start = out_batch_offset + oc * n + col_start;
                                    unsafe {
                                        let row_slice = std::slice::from_raw_parts_mut(
                                            out_p.add(row_start),
                                            actual_n,
                                        );
                                        apply_bias_act_scalar(row_slice, b_val, act);
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            for oc in 0..m {
                                let b_val = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);
                                let row_start = out_batch_offset + oc * n + col_start;
                                unsafe {
                                    let row_slice = std::slice::from_raw_parts_mut(
                                        out_p.add(row_start),
                                        actual_n,
                                    );
                                    apply_bias_act_scalar(row_slice, b_val, act);
                                }
                            }
                        }
                    });
                }

                return Ok(output);
            }

            // Strategy B: Channel (row) splitting
            if m >= 8 {
                let chunk_m = m.div_ceil(num_threads).max(4).min(m);
                let num_chunks = m.div_ceil(chunk_m);

                for b in 0..batch_size {
                    let in_batch_offset = b * in_channels * out_spatial;
                    let out_batch_offset = b * out_channels * out_spatial;
                    let out_ptr_usize = out_slice.as_mut_ptr() as usize;

                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let oc_start = chunk_idx * chunk_m;
                        let actual_m = (m - oc_start).min(chunk_m);
                        let a_offset = oc_start * k;
                        let out_p = out_ptr_usize as *mut f32;

                        unsafe {
                            gemm_row_major(
                                actual_m,
                                k,
                                n,
                                n,
                                weight_slice.as_ptr().add(a_offset),
                                in_slice.as_ptr().add(in_batch_offset),
                                out_p.add(out_batch_offset + oc_start * n),
                            );
                        }

                        let act = params.activation;
                        if bias_slice.is_some() || act != ConvActivation::None {
                            #[cfg(target_arch = "x86_64")]
                            if crate::ops::avx2_fma_enabled() {
                                unsafe {
                                    apply_bias_act_fma_rows(
                                        out_p.add(out_batch_offset + oc_start * n),
                                        actual_m,
                                        n,
                                        n,
                                        bias_slice,
                                        oc_start,
                                        act,
                                    );
                                }
                            } else {
                                for i in 0..actual_m {
                                    let b_val = bias_slice.map(|bs| bs[oc_start + i]).unwrap_or(0.0);
                                    let row_start = out_batch_offset + (oc_start + i) * n;
                                    unsafe {
                                        let row_slice =
                                            std::slice::from_raw_parts_mut(out_p.add(row_start), n);
                                        apply_bias_act_scalar(row_slice, b_val, act);
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            for i in 0..actual_m {
                                let b_val = bias_slice.map(|bs| bs[oc_start + i]).unwrap_or(0.0);
                                let row_start = out_batch_offset + (oc_start + i) * n;
                                unsafe {
                                    let row_slice =
                                        std::slice::from_raw_parts_mut(out_p.add(row_start), n);
                                    apply_bias_act_scalar(row_slice, b_val, act);
                                }
                            }
                        }
                    });
                }

                return Ok(output);
            }
        }

        // 3. Sequential fallback for tiny layers
        for b in 0..batch_size {
            let in_batch_offset = b * in_channels * out_spatial;
            let out_batch_offset = b * out_channels * out_spatial;

            unsafe {
                gemm_row_major(
                    m,
                    k,
                    n,
                    n,
                    weight_slice.as_ptr(),
                    in_slice.as_ptr().add(in_batch_offset),
                    out_slice.as_mut_ptr().add(out_batch_offset),
                );
            }

            let act = params.activation;
            if bias_slice.is_some() || act != ConvActivation::None {
                for i in 0..m {
                    let b_val = bias_slice.map(|bs| bs[i]).unwrap_or(0.0);
                    let row =
                        &mut out_slice[out_batch_offset + i * n..out_batch_offset + (i + 1) * n];
                    apply_bias_act_simd(row, b_val, act);
                }
            }
        }
        return Ok(output);
    }

    /// Row-major `c[m, n] = a[m, k] * b[k, n]`, on faer, single-threaded.
    ///
    /// `stride` is the row stride of `b` and `c` in elements, and it is **not** `n` when `n`
    /// is a column chunk: the conv path hands a worker `n` columns of a row that is `stride`
    /// wide, so both operands are strided sub-blocks of larger buffers. The probe in
    /// `examples/faer_probe.rs` passes that separately, and this had to as well -- the first
    /// version of this helper used `n` for the stride, which is right only when the chunk is the
    /// whole row. It made the parallel and sequential paths disagree on `m=384, k=192, n=144`,
    /// and `conv2d_1x1_split_strategies_agree_on_every_shape` caught it.
    ///
    /// The 1x1 convolutions are ~45% of the rec forward and are a plain GEMM in NCHW order: the
    /// weight is `[m, k]` row-major and the input is `[k, stride]` row-major. faer is faster
    /// than `matrixmultiply::sgemm` on exactly these shapes, on the same memory and with the same
    /// single-thread contract (min of 40, both serial):
    ///
    /// ```text
    /// sgemm(m,   k,  n) stride   matrixmultiply   faer    ratio
    ///     384, 192, 144    144      0.263 ms   0.160   1.64x
    ///     384, 192,  64    144      0.136 ms   0.074   1.83x
    ///     384, 192,  16    144      0.065 ms   0.022   2.93x
    ///         192, 384,  16    144      0.068 ms   0.021   3.19x
    ///         768, 384,  45    708      0.362 ms   0.223   1.62x
    ///         768, 384, 708    708      4.705 ms   3.299   1.43x
    /// ```
    ///
    /// faer reaches 105-133 GFLOP/s on a single core here, against 34-95 for `matrixmultiply`.
    /// The single-core ceiling is ~147 GFLOP/s (32 FLOP/cycle at 4.6 GHz turbo -- two 256-bit
    /// FMA units, no AVX-512: every `avx512*` bit reads false, see `examples/avx512_probe.rs`),
    /// so faer runs at 72-90% of peak.
    ///
    /// Two more things are load-bearing:
    ///
    /// * **The transposes are free.** faer is column-major, so a row-major `[r, c]` buffer is
    ///   viewed as a column-major `[c, r]` matrix with row stride 1 and column stride `c`:
    ///   element `(i, j)` lands at `i + j * c`, the original `j * c + i`. An earlier probe used
    ///   `Mat::from_fn`, which *materialises* a contiguous copy outside the timed region, handing
    ///   faer a free layout conversion the engine would never get.
    /// * **`Par::Seq`.** faer switches to rayon above `PAR_THRESHOLD_MNK = 4096` and every shape
    ///   here is above it, so the default policy would put all 16 threads inside a call whose
    ///   caller has already parallelised over output columns. That measured as a fictitious 3.84x.
    ///   The caller owns the parallelism; this kernel is one worker.
    ///
    /// # Safety
    /// `a` must be readable for `m * k` floats, `b` for `k * stride`, `c` writable for
    /// `m * stride`, and `c` must not overlap `a` or `b`.
    #[inline]
    unsafe fn gemm_row_major(
        m: usize,
        k: usize,
        n: usize,
        stride: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) {
        unsafe {
            crate::ops::gemm_row_major(m, k, n, a, k, b, stride, c, stride);
        }
    }

    // Standard Convolution & Grouped Convolution via im2col + Parallel GEMM
    let out_spatial = out_h * out_w;
    let in_channels_per_group = in_channels / group;
    let out_channels_per_group = out_channels / group;
    let k_size = in_channels_per_group * k_h * k_w;
    let num_threads = crate::tensor::parallel_degree(1, 1);

    // Fast Path: Spatial-Tiled GEMM for large general convolutions (e.g. Conv.3)
    // Eliminates 5-100MB heap allocations and keeps local tile col_buf inside L2 cache
    if group == 1 && out_spatial >= 128 && k_size * out_spatial >= 150_000 && num_threads > 1 {
        let target_tile_elements = 40_960; // ~164 KB (fits entirely in CPU 256 KB L2 cache)
        let target_cols = (target_tile_elements / k_size).max(1);

        let (tile_h, chunk_w, num_w_chunks, num_tiles) = if target_cols >= out_w {
            let mut th = (target_cols / out_w).max(1).min(out_h);
            if out_h.div_ceil(th) < num_threads {
                th = out_h.div_ceil(num_threads).max(1).min(th);
            }
            let nt = out_h.div_ceil(th);
            (th, out_w, 1, nt)
        } else {
            // Feature map is wider than target tile size: tile along W with tile_h = 1
            // Choose chunk_w as a multiple of 16 that evenly distributes out_w across chunks,
            // avoiding tiny ragged remainder tiles and maximizing AVX2 vector utilization.
            let raw_nwc = out_w.div_ceil(target_cols).max(2);
            let balanced_w = (out_w.div_ceil(raw_nwc).div_ceil(16) * 16).clamp(16, out_w);
            let nwc = out_w.div_ceil(balanced_w);
            let nt = out_h * nwc;
            (1, balanced_w, nwc, nt)
        };

        if num_tiles >= 2 {
            for b in 0..batch_size {
                let in_batch_offset = b * in_channels * in_h * in_w;
                let out_batch_offset = b * out_channels * out_spatial;
                let out_p_usize = out_slice.as_mut_ptr() as usize;

                (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                    let (oh_start, oh_end, ow_start, ow_end) = if num_w_chunks == 1 {
                        let oh_s = tile_idx * tile_h;
                        let oh_e = (oh_s + tile_h).min(out_h);
                        (oh_s, oh_e, 0, out_w)
                    } else {
                        let oh = tile_idx / num_w_chunks;
                        let w_idx = tile_idx % num_w_chunks;
                        let ow_s = w_idx * chunk_w;
                        let ow_e = (ow_s + chunk_w).min(out_w);
                        (oh, oh + 1, ow_s, ow_e)
                    };
                    let actual_h = oh_end - oh_start;
                    let actual_w = ow_end - ow_start;
                    let tile_spatial = actual_h * actual_w;

                    let mut tile_col = crate::tensor::acquire_buffer(k_size * tile_spatial);
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        tile_col.set_len(k_size * tile_spatial);
                    }

                    im2col_tile(
                        &in_slice[in_batch_offset..],
                        in_channels_per_group,
                        in_h,
                        in_w,
                        k_h,
                        k_w,
                        pad_top,
                        pad_left,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        oh_start,
                        oh_end,
                        ow_start,
                        ow_end,
                        out_w,
                        &mut tile_col,
                    );

                    let m = out_channels;
                    let k = k_size;
                    let n = tile_spatial;
                    let out_p = out_p_usize as *mut f32;
                    let c_start = out_batch_offset + oh_start * out_w + ow_start;

                    unsafe {
                        crate::ops::gemm_row_major(
                            m,
                            k,
                            n,
                            weight_slice.as_ptr(),
                            k,
                            tile_col.as_ptr(),
                            n,
                            out_p.add(c_start),
                            out_spatial,
                        );
                    }

                    let act = params.activation;
                    if bias_slice.is_some() || act != ConvActivation::None {
                        #[cfg(target_arch = "x86_64")]
                        if crate::ops::avx2_fma_enabled() {
                            unsafe {
                                apply_bias_act_fma_rows(
                                    out_p.add(c_start),
                                    m,
                                    out_spatial,
                                    tile_spatial,
                                    bias_slice,
                                    0,
                                    act,
                                );
                            }
                        } else {
                            for i in 0..m {
                                let b_val = bias_slice.map(|bs| bs[i]).unwrap_or(0.0);
                                let row_offset = c_start + i * out_spatial;
                                unsafe {
                                    let row = std::slice::from_raw_parts_mut(
                                        out_p.add(row_offset),
                                        tile_spatial,
                                    );
                                    apply_bias_act_simd(row, b_val, act);
                                }
                            }
                        }
                    }

                    crate::tensor::release_buffer(tile_col);
                });
            }
            return Ok(output);
        }
    }

    for b in 0..batch_size {
        let in_batch_offset = b * in_channels * in_h * in_w;
        let out_batch_offset = b * out_channels * out_spatial;

        for g in 0..group {
            let in_g_offset = in_batch_offset + g * in_channels_per_group * in_h * in_w;
            let out_g_offset = out_batch_offset + g * out_channels_per_group * out_spatial;
            let weight_g_offset = g * out_channels_per_group * k_size;

            // 1. im2col for this group (allocated without pre-zeroing since every element is initialized)
            let col_len = k_size * out_spatial;
            let mut col = crate::tensor::acquire_buffer(col_len);
            #[allow(clippy::uninit_vec)]
            unsafe {
                col.set_len(col_len);
            }

            im2col(
                &in_slice[in_g_offset..],
                in_channels_per_group,
                in_h,
                in_w,
                k_h,
                k_w,
                pad_top,
                pad_left,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                out_h,
                out_w,
                &mut col,
            );

            // 2. High-performance Multi-threaded GEMM: Out = Weight * Col
            let m = out_channels_per_group;
            let k = k_size;
            let n = out_spatial;
            let flops = 2 * m * k * n;
            let g_bias_offset = g * out_channels_per_group;

            // Multi-core parallel GEMM:
            // 1. Channel (row) splitting when M >= 8 and flops >= 100_000 (contiguous memory reads/writes)
            if flops >= 100_000 && m >= 8 && num_threads > 1 {
                let chunk_size = m.div_ceil(num_threads).max(4).min(m);
                let out_g_slice = &mut out_slice[out_g_offset..out_g_offset + m * n];

                out_g_slice
                    .par_chunks_mut(chunk_size * n)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let oc_start = chunk_idx * chunk_size;
                        let actual_m = chunk.len() / n;
                        let a_offset = weight_g_offset + oc_start * k;

                        unsafe {
                            crate::ops::gemm_row_major(
                                actual_m,
                                k,
                                n,
                                weight_slice.as_ptr().add(a_offset),
                                k,
                                col.as_ptr(),
                                n,
                                chunk.as_mut_ptr(),
                                n,
                            );
                        }

                        let act = params.activation;
                        if bias_slice.is_some() || act != ConvActivation::None {
                            for i in 0..actual_m {
                                let b_val = bias_slice
                                    .map(|bs| bs[g_bias_offset + oc_start + i])
                                    .unwrap_or(0.0);
                                let row = &mut chunk[i * n..(i + 1) * n];
                                apply_bias_act_simd(row, b_val, act);
                            }
                        }
                    });
            } else if flops >= 150_000 && n >= 128 && num_threads > 1 {
                // Column-splitting fallback when M < 8
                let chunk_n = n.div_ceil(num_threads).max(64).min(n);
                let num_col_chunks = n.div_ceil(chunk_n);
                let out_ptr_usize = out_slice.as_mut_ptr() as usize;

                (0..num_col_chunks).into_par_iter().for_each(|chunk_idx| {
                    let col_start = chunk_idx * chunk_n;
                    let actual_n = (n - col_start).min(chunk_n);
                    let out_p = out_ptr_usize as *mut f32;

                    unsafe {
                        crate::ops::gemm_row_major(
                            m,
                            k,
                            actual_n,
                            weight_slice.as_ptr().add(weight_g_offset),
                            k,
                            col.as_ptr().add(col_start),
                            n,
                            out_p.add(out_g_offset + col_start),
                            n,
                        );
                    }

                    let act = params.activation;
                    if bias_slice.is_some() || act != ConvActivation::None {
                        for oc in 0..m {
                            let b_val = bias_slice.map(|bs| bs[g_bias_offset + oc]).unwrap_or(0.0);
                            let row_start = out_g_offset + oc * n + col_start;
                            unsafe {
                                let row_slice =
                                    std::slice::from_raw_parts_mut(out_p.add(row_start), actual_n);
                                apply_bias_act_simd(row_slice, b_val, act);
                            }
                        }
                    }
                });
            } else {
                unsafe {
                    crate::ops::gemm_row_major(
                        m,
                        k,
                        n,
                        weight_slice.as_ptr().add(weight_g_offset),
                        k,
                        col.as_ptr(),
                        n,
                        out_slice.as_mut_ptr().add(out_g_offset),
                        n,
                    );
                }

                let act = params.activation;
                if bias_slice.is_some() || act != ConvActivation::None {
                    for oc in 0..m {
                        let b_val = bias_slice.map(|bs| bs[g_bias_offset + oc]).unwrap_or(0.0);
                        let row =
                            &mut out_slice[out_g_offset + oc * n..out_g_offset + (oc + 1) * n];
                        apply_bias_act_simd(row, b_val, act);
                    }
                }
            }

            crate::tensor::release_buffer(col);
        }
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
/// Strided (stride 2) gather of one im2col row: picks every other input element
/// and widens it into the output line. Returns the first column still needing
/// the scalar tail.
/// Fast AVX2 strided copy for `stride_w = 2, dilation_w = 1` rows.
///
/// # Safety
/// Requires AVX2 (see [`crate::ops::avx2_fma_enabled`]). `in_row_ptr` must be
/// readable for `width` elements and `out_ptr` writable for `ow_end - ow_start`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gather_stride2_row_fma(
    in_row_ptr: *const f32,
    out_ptr: *mut f32,
    ow_start: usize,
    copy_start: usize,
    copy_end: usize,
    base_iw: isize,
    width: usize,
) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let mut ow = copy_start;

        let safe_end = if width as isize >= base_iw + 16 {
            ((width as isize - base_iw - 16) / 2 + 1) as usize
        } else {
            0
        };
        let vec_end = copy_end.min(safe_end);

        while ow + 16 <= vec_end {
            let iw = (ow as isize * 2 + base_iw) as usize;
            let v0 = _mm256_loadu_ps(in_row_ptr.add(iw));
            let v1 = _mm256_loadu_ps(in_row_ptr.add(iw + 8));
            let v2 = _mm256_loadu_ps(in_row_ptr.add(iw + 16));
            let v3 = _mm256_loadu_ps(in_row_ptr.add(iw + 24));
            let shuf0 = _mm256_shuffle_ps(v0, v1, 0x88);
            let shuf1 = _mm256_shuffle_ps(v2, v3, 0x88);
            let res0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(shuf0), 0xd8));
            let res1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(shuf1), 0xd8));
            _mm256_storeu_ps(out_ptr.add(ow - ow_start), res0);
            _mm256_storeu_ps(out_ptr.add(ow - ow_start + 8), res1);
            ow += 16;
        }

        while ow + 8 <= vec_end {
            let iw = (ow as isize * 2 + base_iw) as usize;
            let v0 = _mm256_loadu_ps(in_row_ptr.add(iw));
            let v1 = _mm256_loadu_ps(in_row_ptr.add(iw + 8));
            let shuf = _mm256_shuffle_ps(v0, v1, 0x88);
            let res = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(shuf), 0xd8));
            _mm256_storeu_ps(out_ptr.add(ow - ow_start), res);
            ow += 8;
        }

        ow
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col_tile_slices(
    in_slices: &[(&[f32], usize)],
    height: usize,
    width: usize,
    k_h: usize,
    k_w: usize,
    pad_top: usize,
    pad_left: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_start: usize,
    oh_end: usize,
    ow_start: usize,
    ow_end: usize,
    out_w: usize,
    col_data: &mut [f32],
) {
    let actual_h = oh_end - oh_start;
    let actual_w = ow_end - ow_start;
    let tile_spatial = actual_h * actual_w;

    // Precompute horizontal copy bounds for each kw once for the entire tile:
    let pad_left_i = pad_left as isize;
    let mut kw_bounds = [(0usize, 0usize, 0isize); 8];
    for kw in 0..k_w.min(8) {
        let kw_i = kw as isize;
        kw_bounds[kw] = if dilation_w == 1 && stride_w == 1 {
            let full_ow_start = (pad_left_i - kw_i).max(0) as usize;
            let full_ow_end = ((width as isize + pad_left_i - kw_i).max(0) as usize).min(out_w);
            (
                ow_start.max(full_ow_start),
                ow_end.min(full_ow_end),
                kw_i - pad_left_i,
            )
        } else if dilation_w == 1 && stride_w == 2 {
            let diff = pad_left_i - kw_i;
            let full_ow_start = if diff <= 0 {
                0
            } else {
                ((diff + 1) / 2) as usize
            }
            .min(out_w);

            let max_val = width as isize + pad_left_i - kw_i;
            let full_ow_end = if max_val <= 0 {
                0
            } else {
                ((max_val + 1) / 2) as usize
            }
            .min(out_w);

            (
                ow_start.max(full_ow_start),
                ow_end.min(full_ow_end),
                kw_i - pad_left_i,
            )
        } else {
            (0, 0, 0)
        };
    }

    let mut col_chunks = col_data.chunks_exact_mut(tile_spatial);
    for &(in_data, channels) in in_slices {
        for c in 0..channels {
            let in_c_offset = c * height * width;
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let out_row_slice = col_chunks.next().unwrap();
                    let (copy_start, copy_end, base_iw) = kw_bounds[kw.min(7)];

                    for oh in oh_start..oh_end {
                        let ih = (oh * stride_h) as isize - pad_top as isize + (kh * dilation_h) as isize;
                        let tile_oh = oh - oh_start;
                        let out_row_offset = tile_oh * actual_w;

                        if ih >= 0 && (ih as usize) < height {
                            let in_row_offset = in_c_offset + (ih as usize) * width;
                            let out_line = &mut out_row_slice[out_row_offset..out_row_offset + actual_w];

                            if dilation_w == 1 && stride_w == 1 {
                                if copy_start > ow_start {
                                    let fill_len = (copy_start - ow_start).min(actual_w);
                                    out_line[..fill_len].fill(0.0);
                                }
                                if copy_start < copy_end {
                                    let iw_start = (copy_start as isize + base_iw) as usize;
                                    let copy_len = copy_end - copy_start;
                                    let dst_start = copy_start - ow_start;
                                    out_line[dst_start..dst_start + copy_len].copy_from_slice(
                                        &in_data[in_row_offset + iw_start..in_row_offset + iw_start + copy_len],
                                    );
                                }
                                if copy_end < ow_end {
                                    let dst_end = copy_end.saturating_sub(ow_start);
                                    if dst_end < actual_w {
                                        out_line[dst_end..].fill(0.0);
                                    }
                                }
                            } else if dilation_w == 1 && stride_w == 2 {
                                if copy_start > ow_start {
                                    let fill_len = (copy_start - ow_start).min(actual_w);
                                    out_line[..fill_len].fill(0.0);
                                }
                                if copy_start < copy_end {
                                    let in_row_ptr = unsafe { in_data.as_ptr().add(in_row_offset) };
                                    let out_ptr = out_line.as_mut_ptr();

                                    let mut ow = copy_start;
                                    #[cfg(target_arch = "x86_64")]
                                    if crate::ops::avx2_fma_enabled() {
                                        ow = unsafe {
                                            gather_stride2_row_fma(
                                                in_row_ptr,
                                                out_ptr,
                                                ow_start,
                                                copy_start,
                                                copy_end,
                                                base_iw,
                                                width,
                                            )
                                        };
                                    }
                                    while ow < copy_end {
                                        let iw = (ow as isize * 2 + base_iw) as usize;
                                        unsafe {
                                            *out_ptr.add(ow - ow_start) = *in_row_ptr.add(iw);
                                        }
                                        ow += 1;
                                    }
                                }
                                if copy_end < ow_end {
                                    let dst_end = copy_end.saturating_sub(ow_start);
                                    if dst_end < actual_w {
                                        out_line[dst_end..].fill(0.0);
                                    }
                                }
                            } else {
                                for ow in ow_start..ow_end {
                                    let iw = (ow * stride_w) as isize - pad_left as isize
                                        + (kw * dilation_w) as isize;
                                    out_line[ow - ow_start] = if iw >= 0 && (iw as usize) < width {
                                        in_data[in_row_offset + iw as usize]
                                    } else {
                                        0.0
                                    };
                                }
                            }
                        } else {
                            out_row_slice[out_row_offset..out_row_offset + actual_w].fill(0.0);
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col_tile(
    in_data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    k_h: usize,
    k_w: usize,
    pad_top: usize,
    pad_left: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_start: usize,
    oh_end: usize,
    ow_start: usize,
    ow_end: usize,
    out_w: usize,
    col_data: &mut [f32],
) {
    im2col_tile_slices(
        &[(in_data, channels)],
        height,
        width,
        k_h,
        k_w,
        pad_top,
        pad_left,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        oh_start,
        oh_end,
        ow_start,
        ow_end,
        out_w,
        col_data,
    );
}

/// Execute Conv2D directly over multiple input tensors concatenated along the channel axis (axis 1),
/// completely avoiding the memory allocation, buffer copy, and cache thrashing of an explicit Concat op.
pub fn conv2d_channel_sliced(
    inputs: &[&Tensor],
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2dParams,
) -> Result<Tensor> {
    if inputs.is_empty() {
        return Err(OcrError::OperatorError {
            op: "Conv2d".into(),
            msg: "inputs must not be empty".into(),
        });
    }
    if inputs.len() == 1 {
        return conv2d(inputs[0], weight, bias, params);
    }

    let batch_size = inputs[0].shape()[0];
    let in_h = inputs[0].shape()[2];
    let in_w = inputs[0].shape()[3];
    for inp in inputs {
        if inp.ndim() != 4 || inp.shape()[0] != batch_size || inp.shape()[2] != in_h || inp.shape()[3] != in_w {
            let concatted = crate::ops::concat(inputs, 1)?;
            return conv2d(&concatted, weight, bias, params);
        }
    }

    let in_channels: usize = inputs.iter().map(|inp| inp.shape()[1]).sum();
    if weight.ndim() != 4 || weight.shape()[1] != in_channels {
        let concatted = crate::ops::concat(inputs, 1)?;
        return conv2d(&concatted, weight, bias, params);
    }

    let out_channels = weight.shape()[0];
    let k_h = weight.shape()[2];
    let k_w = weight.shape()[3];
    let group = params.group;

    if group != 1 || (k_h == 1 && k_w == 1) {
        let concatted = crate::ops::concat(inputs, 1)?;
        return conv2d(&concatted, weight, bias, params);
    }

    let [stride_h, stride_w] = params.strides;
    let [dilation_h, dilation_w] = params.dilations;
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;
    let eff_k_h = (k_h - 1) * dilation_h + 1;
    let eff_k_w = (k_w - 1) * dilation_w + 1;
    let out_h = (in_h + pad_top + pad_bottom - eff_k_h) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right - eff_k_w) / stride_w + 1;
    let out_spatial = out_h * out_w;
    let k_size = in_channels * k_h * k_w;
    let num_threads = crate::tensor::parallel_degree(1, 1);

    let mut output = Tensor::empty(&[batch_size, out_channels, out_h, out_w]);
    let out_slice = output.as_mut_slice();
    let weight_slice = weight.as_slice();
    let bias_slice: Option<&[f32]> = bias.map(|b| b.as_slice());

    if out_spatial >= 128 && k_size * out_spatial >= 150_000 && num_threads > 1 {
        let target_tile_elements = 40_960;
        let target_cols = (target_tile_elements / k_size).max(1);
        let (tile_h, chunk_w, num_w_chunks, num_tiles) = if target_cols >= out_w {
            let mut th = (target_cols / out_w).max(1).min(out_h);
            if out_h.div_ceil(th) < num_threads {
                th = out_h.div_ceil(num_threads).max(1).min(th);
            }
            let nt = out_h.div_ceil(th);
            (th, out_w, 1, nt)
        } else {
            let raw_nwc = out_w.div_ceil(target_cols).max(2);
            let balanced_w = (out_w.div_ceil(raw_nwc).div_ceil(16) * 16).clamp(16, out_w);
            let nwc = out_w.div_ceil(balanced_w);
            let nt = out_h * nwc;
            (1, balanced_w, nwc, nt)
        };

        if num_tiles >= 2 {
            for b in 0..batch_size {
                let out_batch_offset = b * out_channels * out_spatial;
                let out_p_usize = out_slice.as_mut_ptr() as usize;

                let in_slices: Vec<(&[f32], usize)> = inputs
                    .iter()
                    .map(|inp| {
                        let c = inp.shape()[1];
                        let off = b * c * in_h * in_w;
                        (&inp.as_slice()[off..], c)
                    })
                    .collect();

                (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                    let (oh_start, oh_end, ow_start, ow_end) = if num_w_chunks == 1 {
                        let oh_s = tile_idx * tile_h;
                        let oh_e = (oh_s + tile_h).min(out_h);
                        (oh_s, oh_e, 0, out_w)
                    } else {
                        let oh = tile_idx / num_w_chunks;
                        let w_idx = tile_idx % num_w_chunks;
                        let ow_s = w_idx * chunk_w;
                        let ow_e = (ow_s + chunk_w).min(out_w);
                        (oh, oh + 1, ow_s, ow_e)
                    };
                    let actual_h = oh_end - oh_start;
                    let actual_w = ow_end - ow_start;
                    let tile_spatial = actual_h * actual_w;

                    let mut tile_col = crate::tensor::acquire_buffer(k_size * tile_spatial);
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        tile_col.set_len(k_size * tile_spatial);
                    }

                    im2col_tile_slices(
                        &in_slices,
                        in_h,
                        in_w,
                        k_h,
                        k_w,
                        pad_top,
                        pad_left,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        oh_start,
                        oh_end,
                        ow_start,
                        ow_end,
                        out_w,
                        &mut tile_col,
                    );

                    let m = out_channels;
                    let k = k_size;
                    let n = tile_spatial;
                    let out_p = out_p_usize as *mut f32;
                    let c_start = out_batch_offset + oh_start * out_w + ow_start;

                    unsafe {
                        crate::ops::gemm_row_major(
                            m,
                            k,
                            n,
                            weight_slice.as_ptr(),
                            k,
                            tile_col.as_ptr(),
                            n,
                            out_p.add(c_start),
                            out_spatial,
                        );
                    }

                    let act = params.activation;
                    if bias_slice.is_some() || act != ConvActivation::None {
                        #[cfg(target_arch = "x86_64")]
                        if crate::ops::avx2_fma_enabled() {
                            unsafe {
                                apply_bias_act_fma_rows(
                                    out_p.add(c_start),
                                    m,
                                    out_spatial,
                                    tile_spatial,
                                    bias_slice,
                                    0,
                                    act,
                                );
                            }
                        } else {
                            for i in 0..m {
                                let b_val = bias_slice.map(|bs| bs[i]).unwrap_or(0.0);
                                let row_offset = c_start + i * out_spatial;
                                unsafe {
                                    let row = std::slice::from_raw_parts_mut(
                                        out_p.add(row_offset),
                                        tile_spatial,
                                    );
                                    apply_bias_act_simd(row, b_val, act);
                                }
                            }
                        }
                    }

                    crate::tensor::release_buffer(tile_col);
                });
            }
            return Ok(output);
        }
    }

    let concatted = crate::ops::concat(inputs, 1)?;
    conv2d(&concatted, weight, bias, params)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn im2col(
    in_data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    k_h: usize,
    k_w: usize,
    pad_top: usize,
    pad_left: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
    col_data: &mut [f32],
) {
    im2col_tile(
        in_data, channels, height, width, k_h, k_w, pad_top, pad_left, stride_h, stride_w, dilation_h,
        dilation_w, 0, out_h, 0, out_w, out_w, col_data,
    );
}

/// One 4-wide SSE step of the `cin == 3, 3x3, stride 2` fast path: gathers the three input
/// rows, applies the 27 broadcast weights with FMA, then stores four output columns.
///
/// This lives in its own `#[target_feature(enable = "avx2,fma")]` function on purpose. An
/// `_mm_fmadd_ps` written directly in a function *without* the attribute is emitted as a
/// plain `vfmadd` instruction with no runtime check whatsoever, so the whole kernel would
/// fault on any pre-Haswell x86_64 -- see the two rules at the top of `simd.rs`.
///
/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`]. For each of the three
/// input rows, `in_p + c_off + row + iw_base` must have `4 * 2 + 2` readable floats, and
/// `out_ptr` must have room for four writable floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn cin3_k3_s2_step_avx2(
    in_p: *const f32,
    row0: usize,
    row1: usize,
    row2: usize,
    iw_base: usize,
    b_in_offset: usize,
    in_spatial: usize,
    vw: &[std::arch::x86_64::__m256; 27],
    v_bias: std::arch::x86_64::__m256,
    v_zero: std::arch::x86_64::__m256,
    v_one: std::arch::x86_64::__m256,
    v_alpha: std::arch::x86_64::__m256,
    v_beta: std::arch::x86_64::__m256,
    act: ConvActivation,
    out_ptr: *mut f32,
) {
    use std::arch::x86_64::*;

    let mut v_acc = v_bias;
    unsafe {
        macro_rules! load_row_taps_avx2 {
            ($p:expr) => {{
                let p_ptr: *const f32 = $p;
                let v0_lo = _mm_loadu_ps(p_ptr);
                let v4_lo = _mm_loadu_ps(p_ptr.add(4));
                let v8_lo = _mm_loadu_ps(p_ptr.add(6));
                let x0_lo = _mm_shuffle_ps(v0_lo, v4_lo, 0x88);
                let x1_lo = _mm_shuffle_ps(v0_lo, v4_lo, 0xDD);
                let x2_lo = _mm_shuffle_ps(_mm_loadu_ps(p_ptr.add(2)), v8_lo, 0x88);

                let p_hi = p_ptr.add(8);
                let v0_hi = _mm_loadu_ps(p_hi);
                let v4_hi = _mm_loadu_ps(p_hi.add(4));
                let v8_hi = _mm_loadu_ps(p_hi.add(6));
                let x0_hi = _mm_shuffle_ps(v0_hi, v4_hi, 0x88);
                let x1_hi = _mm_shuffle_ps(v0_hi, v4_hi, 0xDD);
                let x2_hi = _mm_shuffle_ps(_mm_loadu_ps(p_hi.add(2)), v8_hi, 0x88);

                (
                    _mm256_set_m128(x0_hi, x0_lo),
                    _mm256_set_m128(x1_hi, x1_lo),
                    _mm256_set_m128(x2_hi, x2_lo),
                )
            }};
        }

        for ic in 0..3 {
            let c_off = b_in_offset + ic * in_spatial;
            let w_base = ic * 9;

            // Row 0
            let (x0, x1, x2) = load_row_taps_avx2!(in_p.add(c_off + row0 + iw_base));
            v_acc = _mm256_fmadd_ps(x0, vw[w_base], v_acc);
            v_acc = _mm256_fmadd_ps(x1, vw[w_base + 1], v_acc);
            v_acc = _mm256_fmadd_ps(x2, vw[w_base + 2], v_acc);

            // Row 1
            let (x0, x1, x2) = load_row_taps_avx2!(in_p.add(c_off + row1 + iw_base));
            v_acc = _mm256_fmadd_ps(x0, vw[w_base + 3], v_acc);
            v_acc = _mm256_fmadd_ps(x1, vw[w_base + 4], v_acc);
            v_acc = _mm256_fmadd_ps(x2, vw[w_base + 5], v_acc);

            // Row 2
            let (x0, x1, x2) = load_row_taps_avx2!(in_p.add(c_off + row2 + iw_base));
            v_acc = _mm256_fmadd_ps(x0, vw[w_base + 6], v_acc);
            v_acc = _mm256_fmadd_ps(x1, vw[w_base + 7], v_acc);
            v_acc = _mm256_fmadd_ps(x2, vw[w_base + 8], v_acc);
        }

        let v_out = match act {
            ConvActivation::None => v_acc,
            ConvActivation::Relu => _mm256_max_ps(v_acc, v_zero),
            ConvActivation::HardSigmoid { .. } => {
                let scaled = _mm256_fmadd_ps(v_acc, v_alpha, v_beta);
                _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_one)
            }
            ConvActivation::HardSwish => avx2_hardswish(v_acc),
            ConvActivation::Gelu => avx2_gelu(v_acc),
        };

        _mm256_storeu_ps(out_ptr, v_out);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cin3_k3_s2_step_fma(
    in_p: *const f32,
    row0: usize,
    row1: usize,
    row2: usize,
    iw_base: usize,
    b_in_offset: usize,
    in_spatial: usize,
    vw: &[std::arch::x86_64::__m128; 27],
    v_bias: std::arch::x86_64::__m128,
    v_zero: std::arch::x86_64::__m128,
    v_one: std::arch::x86_64::__m128,
    v_alpha: std::arch::x86_64::__m128,
    v_beta: std::arch::x86_64::__m128,
    act: ConvActivation,
    out_ptr: *mut f32,
) {
    use std::arch::x86_64::*;

    let mut v_acc = v_bias;
    unsafe {
        for ic in 0..3 {
            let c_off = b_in_offset + ic * in_spatial;
            let w_base = ic * 9;

            // Row 0
            let p = in_p.add(c_off + row0 + iw_base);
            let v0 = _mm_loadu_ps(p);
            let v4 = _mm_loadu_ps(p.add(4));
            let v8 = _mm_loadu_ps(p.add(6));
            let x0 = _mm_shuffle_ps(v0, v4, 0x88);
            let x1 = _mm_shuffle_ps(v0, v4, 0xDD);
            let x2 = _mm_shuffle_ps(_mm_loadu_ps(p.add(2)), v8, 0x88);

            v_acc = _mm_fmadd_ps(x0, vw[w_base], v_acc);
            v_acc = _mm_fmadd_ps(x1, vw[w_base + 1], v_acc);
            v_acc = _mm_fmadd_ps(x2, vw[w_base + 2], v_acc);

            // Row 1
            let p = in_p.add(c_off + row1 + iw_base);
            let v0 = _mm_loadu_ps(p);
            let v4 = _mm_loadu_ps(p.add(4));
            let v8 = _mm_loadu_ps(p.add(6));
            let x0 = _mm_shuffle_ps(v0, v4, 0x88);
            let x1 = _mm_shuffle_ps(v0, v4, 0xDD);
            let x2 = _mm_shuffle_ps(_mm_loadu_ps(p.add(2)), v8, 0x88);

            v_acc = _mm_fmadd_ps(x0, vw[w_base + 3], v_acc);
            v_acc = _mm_fmadd_ps(x1, vw[w_base + 4], v_acc);
            v_acc = _mm_fmadd_ps(x2, vw[w_base + 5], v_acc);

            // Row 2
            let p = in_p.add(c_off + row2 + iw_base);
            let v0 = _mm_loadu_ps(p);
            let v4 = _mm_loadu_ps(p.add(4));
            let v8 = _mm_loadu_ps(p.add(6));
            let x0 = _mm_shuffle_ps(v0, v4, 0x88);
            let x1 = _mm_shuffle_ps(v0, v4, 0xDD);
            let x2 = _mm_shuffle_ps(_mm_loadu_ps(p.add(2)), v8, 0x88);

            v_acc = _mm_fmadd_ps(x0, vw[w_base + 6], v_acc);
            v_acc = _mm_fmadd_ps(x1, vw[w_base + 7], v_acc);
            v_acc = _mm_fmadd_ps(x2, vw[w_base + 8], v_acc);
        }

        let v_out = match act {
            ConvActivation::None => v_acc,
            ConvActivation::Relu => _mm_max_ps(v_acc, v_zero),
            ConvActivation::HardSigmoid { .. } => {
                let scaled = _mm_fmadd_ps(v_acc, v_alpha, v_beta);
                _mm_min_ps(_mm_max_ps(scaled, v_zero), v_one)
            }
            ConvActivation::HardSwish => {
                let scaled = _mm_fmadd_ps(v_acc, _mm_set1_ps(1.0 / 6.0), _mm_set1_ps(0.5));
                let gate = _mm_min_ps(_mm_max_ps(scaled, v_zero), v_one);
                _mm_mul_ps(v_acc, gate)
            }
            ConvActivation::Gelu => {
                let mut buf = [0.0f32; 4];
                _mm_storeu_ps(buf.as_mut_ptr(), v_acc);
                for val in &mut buf {
                    *val = apply_conv_activation(*val, act);
                }
                _mm_loadu_ps(buf.as_ptr())
            }
        };

        _mm_storeu_ps(out_ptr, v_out);
    }
}

/// Cache-resident parallel im2col + GEMM for cin==3, 3x3, stride-2 convolutions with moderate spatial dimensions (out_spatial <= 32768).
/// Employs AVX2 stride-2 gather vectorization across 27 row slices, followed by multi-threaded GEMM and in-place bias/activation.
fn conv2d_cin3_k3_s2_gemm(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2dParams,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    let in_shape = input.shape();
    let batch_size = in_shape[0];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let w_shape = weight.shape();
    let out_channels = w_shape[0];
    let out_spatial = out_h * out_w;
    let k_size = 27;

    let [pad_top, pad_left, _, _] = params.pads;
    let act = params.activation;

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

    let mut output = Tensor::empty(&[batch_size, out_channels, out_h, out_w]);
    let out_slice = output.as_mut_slice();
    let num_threads = crate::tensor::parallel_degree(1, 1);

    let pad_left_i = pad_left as isize;
    let mut kw_bounds = [(0usize, 0usize, 0isize); 3];
    for kw in 0..3 {
        let kw_i = kw as isize;
        let diff = pad_left_i - kw_i;
        let full_ow_start = if diff <= 0 {
            0
        } else {
            ((diff + 1) / 2) as usize
        }
        .min(out_w);

        let max_val = in_w as isize + pad_left_i - kw_i;
        let full_ow_end = if max_val <= 0 {
            0
        } else {
            ((max_val + 1) / 2) as usize
        }
        .min(out_w);

        kw_bounds[kw] = (full_ow_start, full_ow_end, kw_i - pad_left_i);
    }

    for b in 0..batch_size {
        let b_in_offset = b * 3 * in_h * in_w;
        let b_out_offset = b * out_channels * out_spatial;
        let in_batch_data = &in_slice[b_in_offset..];

        let mut col_buf = crate::tensor::acquire_buffer(k_size * out_spatial);
        #[allow(clippy::uninit_vec)]
        unsafe {
            col_buf.set_len(k_size * out_spatial);
        }

        col_buf
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(|(row_idx, col_row)| {
                let ic = row_idx / 9;
                let rem = row_idx % 9;
                let kh = rem / 3;
                let kw = rem % 3;
                let c_offset = ic * in_h * in_w;
                let (copy_start, copy_end, base_iw) = kw_bounds[kw];

                for oh in 0..out_h {
                    let ih = (oh * 2) as isize - pad_top as isize + kh as isize;
                    let out_offset = oh * out_w;
                    let out_line = &mut col_row[out_offset..out_offset + out_w];

                    if ih >= 0 && (ih as usize) < in_h {
                        let in_row_offset = c_offset + (ih as usize) * in_w;
                        let in_row_ptr = unsafe { in_batch_data.as_ptr().add(in_row_offset) };
                        let out_ptr = out_line.as_mut_ptr();

                        if copy_start > 0 {
                            out_line[..copy_start].fill(0.0);
                        }

                        if copy_start < copy_end {
                            let mut ow = copy_start;
                            #[cfg(target_arch = "x86_64")]
                            if crate::ops::avx2_fma_enabled() {
                                ow = unsafe {
                                    gather_stride2_row_fma(
                                        in_row_ptr,
                                        out_ptr,
                                        0,
                                        copy_start,
                                        copy_end,
                                        base_iw,
                                        in_w,
                                    )
                                };
                            }

                            while ow < copy_end {
                                let iw = (ow as isize * 2 + base_iw) as usize;
                                unsafe {
                                    *out_ptr.add(ow) = *in_row_ptr.add(iw);
                                }
                                ow += 1;
                            }
                        }

                        if copy_end < out_w {
                            out_line[copy_end..].fill(0.0);
                        }
                    } else {
                        out_line.fill(0.0);
                    }
                }
            });

        let out_p = unsafe { out_slice.as_mut_ptr().add(b_out_offset) };
        unsafe {
            crate::ops::gemm_row_major_accum(
                crate::ops::Accum::Replace,
                crate::ops::Par::rayon(num_threads),
                out_channels,
                k_size,
                out_spatial,
                weight_slice.as_ptr(),
                k_size,
                col_buf.as_ptr(),
                out_spatial,
                out_p,
                out_spatial,
            );
        }

        crate::tensor::release_buffer(col_buf);

        if bias_slice.is_some() || act != ConvActivation::None {
            let out_p_usize = out_p as usize;
            let chunk_m = out_channels.div_ceil(num_threads).max(4);
            (0..out_channels)
                .into_par_iter()
                .step_by(chunk_m)
                .for_each(|r_start| {
                    let out_p = out_p_usize as *mut f32;
                    let r_end = (r_start + chunk_m).min(out_channels);
                    let num_r = r_end - r_start;
                    let row_ptr = unsafe { out_p.add(r_start * out_spatial) };
                    #[cfg(target_arch = "x86_64")]
                    if crate::ops::avx2_fma_enabled() {
                        unsafe {
                            apply_bias_act_fma_rows(
                                row_ptr,
                                num_r,
                                out_spatial,
                                out_spatial,
                                bias_slice,
                                r_start,
                                act,
                            );
                        }
                        return;
                    }
                    for r in 0..num_r {
                        let oc = r_start + r;
                        let b_val = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);
                        let r_ptr = unsafe { row_ptr.add(r * out_spatial) };
                        let slice = unsafe { std::slice::from_raw_parts_mut(r_ptr, out_spatial) };
                        apply_bias_act_scalar(slice, b_val, act);
                    }
                });
        }
    }

    Ok(output)
}

/// Specialised 3-channel 3x3 stride-2 convolution for the first layer of every PP-OCR
/// backbone (`3 -> 16`). Scalar except for the interior-column SSE/FMA step, which lives in
/// [`cin3_k3_s2_step_fma`] and is gated on a CPUID probe hoisted out of the pixel loops.
fn conv2d_direct_cin3_k3_s2(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2dParams,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    let in_shape = input.shape();
    let batch_size = in_shape[0];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let w_shape = weight.shape();
    let out_channels = w_shape[0];

    let mut output = Tensor::empty(&[batch_size, out_channels, out_h, out_w]);
    let out_slice = output.as_mut_slice();
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

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;

    let [pad_top, pad_left, _, _] = params.pads;
    let act = params.activation;

    let out_ptr_usize = out_slice.as_mut_ptr() as usize;
    let in_ptr_usize = in_slice.as_ptr() as usize;
    let weight_ptr_usize = weight_slice.as_ptr() as usize;

    for batch in 0..batch_size {
        let b_in_offset = batch * 3 * in_spatial;
        let b_out_offset = batch * out_channels * out_spatial;

        let run_oc = |oc: usize| {
            let b_val = bias_slice.map(|bs| bs[oc]).unwrap_or(0.0);
            let out_p = out_ptr_usize as *mut f32;
            let in_p = in_ptr_usize as *const f32;
            let w_p = weight_ptr_usize as *const f32;
            let w_oc_p = unsafe { w_p.add(oc * 27) };

            let pad_l = pad_left as isize;

            // Probed once per output channel instead of once per output row: this is a
            // relaxed atomic load, and the cin==3 stem is short enough that re-probing it
            // per row was a measurable share of the kernel.
            #[cfg(target_arch = "x86_64")]
            let has_avx2_fma = crate::ops::avx2_fma_enabled();

            #[cfg(target_arch = "x86_64")]
            let (vw256, vw128, v256_bias, v128_bias, v256_zero, v128_zero, v256_one, v128_one, v256_alpha, v128_alpha, v256_beta, v128_beta) = unsafe {
                use std::arch::x86_64::*;
                let mut w256_arr = [_mm256_setzero_ps(); 27];
                let mut w128_arr = [_mm_setzero_ps(); 27];
                for (i, (s256, s128)) in w256_arr.iter_mut().zip(w128_arr.iter_mut()).enumerate() {
                    let val = *w_oc_p.add(i);
                    *s256 = _mm256_set1_ps(val);
                    *s128 = _mm_set1_ps(val);
                }
                let (a, b) = match act {
                    ConvActivation::HardSigmoid { alpha, beta } => (alpha, beta),
                    _ => (0.0, 0.0),
                };
                (
                    w256_arr,
                    w128_arr,
                    _mm256_set1_ps(b_val),
                    _mm_set1_ps(b_val),
                    _mm256_setzero_ps(),
                    _mm_setzero_ps(),
                    _mm256_set1_ps(1.0),
                    _mm_set1_ps(1.0),
                    _mm256_set1_ps(a),
                    _mm_set1_ps(a),
                    _mm256_set1_ps(b),
                    _mm_set1_ps(b),
                )
            };

            for oh in 0..out_h {
                let out_row_ptr =
                    unsafe { out_p.add(b_out_offset + oc * out_spatial + oh * out_w) };
                let ih_base = (oh * 2) as isize - pad_top as isize;

                let is_interior_h = ih_base >= 0 && (ih_base + 2) < in_h as isize;

                if is_interior_h {
                    let r0 = ih_base as usize * in_w;
                    let r1 = (ih_base as usize + 1) * in_w;
                    let r2 = (ih_base as usize + 2) * in_w;

                    let mut ow = 0;

                    // Left boundary (ow = 0)
                    while ow < out_w && ((ow * 2) as isize - pad_l) < 0 {
                        let iw_base = (ow * 2) as isize - pad_l;
                        let mut sum = b_val;
                        for ic in 0..3 {
                            let c_off = b_in_offset + ic * in_spatial;
                            let w_c_p = unsafe { w_oc_p.add(ic * 9) };
                            let p0 = unsafe { in_p.add(c_off + r0) };
                            let p1 = unsafe { in_p.add(c_off + r1) };
                            let p2 = unsafe { in_p.add(c_off + r2) };

                            for kw in 0..3 {
                                let iw = iw_base + kw as isize;
                                if iw >= 0 && (iw as usize) < in_w {
                                    unsafe {
                                        sum += *p0.add(iw as usize) * *w_c_p.add(kw);
                                        sum += *p1.add(iw as usize) * *w_c_p.add(3 + kw);
                                        sum += *p2.add(iw as usize) * *w_c_p.add(6 + kw);
                                    }
                                }
                            }
                        }
                        unsafe {
                            *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                        }
                        ow += 1;
                    }

                    let max_safe_ow = if in_w >= 2 + pad_left {
                        ((in_w - 2 + pad_left) / 2).min(out_w)
                    } else {
                        0
                    };

                    #[cfg(target_arch = "x86_64")]
                    if has_avx2_fma {
                        unsafe {
                            let avx2_safe_ow = if in_w >= 18 + pad_left {
                                ((in_w - 18 + pad_left) / 2).min(max_safe_ow)
                            } else {
                                0
                            };
                            let sse_safe_ow = if in_w >= 10 + pad_left {
                                ((in_w - 10 + pad_left) / 2).min(max_safe_ow)
                            } else {
                                0
                            };

                            while ow + 8 <= avx2_safe_ow {
                                let iw_base = ((ow * 2) as isize - pad_l) as usize;
                                cin3_k3_s2_step_avx2(
                                    in_p,
                                    r0,
                                    r1,
                                    r2,
                                    iw_base,
                                    b_in_offset,
                                    in_spatial,
                                    &vw256,
                                    v256_bias,
                                    v256_zero,
                                    v256_one,
                                    v256_alpha,
                                    v256_beta,
                                    act,
                                    out_row_ptr.add(ow),
                                );
                                ow += 8;
                            }

                            while ow + 4 <= sse_safe_ow {
                                let iw_base = ((ow * 2) as isize - pad_l) as usize;
                                cin3_k3_s2_step_fma(
                                    in_p,
                                    r0,
                                    r1,
                                    r2,
                                    iw_base,
                                    b_in_offset,
                                    in_spatial,
                                    &vw128,
                                    v128_bias,
                                    v128_zero,
                                    v128_one,
                                    v128_alpha,
                                    v128_beta,
                                    act,
                                    out_row_ptr.add(ow),
                                );
                                ow += 4;
                            }
                        }
                    }

                    while ow < out_w {
                        let iw_base = (ow * 2) as isize - pad_l;
                        let mut sum = b_val;
                        for ic in 0..3 {
                            let c_off = b_in_offset + ic * in_spatial;
                            let w_c_p = unsafe { w_oc_p.add(ic * 9) };
                            let p0 = unsafe { in_p.add(c_off + r0) };
                            let p1 = unsafe { in_p.add(c_off + r1) };
                            let p2 = unsafe { in_p.add(c_off + r2) };

                            for kw in 0..3 {
                                let iw = iw_base + kw as isize;
                                if iw >= 0 && (iw as usize) < in_w {
                                    unsafe {
                                        sum += *p0.add(iw as usize) * *w_c_p.add(kw);
                                        sum += *p1.add(iw as usize) * *w_c_p.add(3 + kw);
                                        sum += *p2.add(iw as usize) * *w_c_p.add(6 + kw);
                                    }
                                }
                            }
                        }
                        unsafe {
                            *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                        }
                        ow += 1;
                    }
                } else {
                    for ow in 0..out_w {
                        let iw_base = (ow * 2) as isize - pad_l;
                        let mut sum = b_val;
                        for ic in 0..3 {
                            let c_off = b_in_offset + ic * in_spatial;
                            let w_c_p = unsafe { w_oc_p.add(ic * 9) };
                            for kh in 0..3 {
                                let ih = ih_base + kh as isize;
                                if ih >= 0 && (ih as usize) < in_h {
                                    let row_p = unsafe { in_p.add(c_off + (ih as usize) * in_w) };
                                    for kw in 0..3 {
                                        let iw = iw_base + kw as isize;
                                        if iw >= 0 && (iw as usize) < in_w {
                                            unsafe {
                                                sum += *row_p.add(iw as usize)
                                                    * *w_c_p.add(kh * 3 + kw);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        unsafe {
                            *out_row_ptr.add(ow) = apply_conv_activation(sum, act);
                        }
                    }
                }
            }
        };

        if crate::tensor::should_parallelize(out_channels, 2) {
            (0..out_channels).into_par_iter().for_each(run_oc);
        } else {
            (0..out_channels).for_each(run_oc);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
    }

    /// Fully general f64 reference: NCHW input, `[cout, cin/group, kh, kw]` weight.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_reference(
        input: &[f32],
        batch: usize,
        cin: usize,
        ih: usize,
        iw: usize,
        weight: &[f32],
        cout: usize,
        kh: usize,
        kw: usize,
        bias: Option<&[f32]>,
        strides: [usize; 2],
        pads: [usize; 4],
        dilations: [usize; 2],
        group: usize,
    ) -> (Vec<f64>, usize, usize) {
        let cpg = cin / group;
        let ocpg = cout / group;
        let eff_kh = dilations[0] * (kh - 1) + 1;
        let eff_kw = dilations[1] * (kw - 1) + 1;
        assert!(
            ih + pads[0] + pads[2] >= eff_kh,
            "vertical extent underflow"
        );
        assert!(
            iw + pads[1] + pads[3] >= eff_kw,
            "horizontal extent underflow"
        );
        let oh = (ih + pads[0] + pads[2] - eff_kh) / strides[0] + 1;
        let ow = (iw + pads[1] + pads[3] - eff_kw) / strides[1] + 1;

        let mut out = vec![0.0f64; batch * cout * oh * ow];
        for b in 0..batch {
            for g in 0..group {
                for ocg in 0..ocpg {
                    let oc = g * ocpg + ocg;
                    for y in 0..oh {
                        for x in 0..ow {
                            let mut acc = bias.map(|b| b[oc] as f64).unwrap_or(0.0);
                            for icg in 0..cpg {
                                let ic = g * cpg + icg;
                                for ki in 0..kh {
                                    let iy = y as isize * strides[0] as isize - pads[0] as isize
                                        + (ki * dilations[0]) as isize;
                                    if iy < 0 || iy >= ih as isize {
                                        continue;
                                    }
                                    for kj in 0..kw {
                                        let ix = x as isize * strides[1] as isize
                                            - pads[1] as isize
                                            + (kj * dilations[1]) as isize;
                                        if ix < 0 || ix >= iw as isize {
                                            continue;
                                        }
                                        let w = weight[((oc * cpg + icg) * kh + ki) * kw + kj];
                                        let v = input[((b * cin + ic) * ih + iy as usize) * iw
                                            + ix as usize];
                                        acc += w as f64 * v as f64;
                                    }
                                }
                            }
                            out[((b * cout + oc) * oh + y) * ow + x] = acc;
                        }
                    }
                }
            }
        }
        (out, oh, ow)
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_conv(
        input: &[f32],
        in_shape: [usize; 4],
        weight: &[f32],
        w_shape: [usize; 4],
        bias: Option<&[f32]>,
        params: Conv2dParams,
        tol: f64,
        ctx: &str,
    ) {
        let [b, cin, ih, iw] = in_shape;
        let [cout, _cpg, kh, kw] = w_shape;
        let (want, oh, ow) = conv2d_reference(
            input,
            b,
            cin,
            ih,
            iw,
            weight,
            cout,
            kh,
            kw,
            bias,
            params.strides,
            params.pads,
            params.dilations,
            params.group,
        );

        let x = tensor(&in_shape, input);
        let w = tensor(&w_shape, weight);
        let bias_t = bias.map(|b| tensor(&[cout], b));
        let got = conv2d(&x, &w, bias_t.as_ref(), &params).expect("conv2d");

        assert_eq!(got.shape(), &[b, cout, oh, ow], "{ctx}: output shape");
        for (i, (&g, &e)) in got.as_slice().iter().zip(want.iter()).enumerate() {
            let want_val = apply_conv_activation(e as f32, params.activation) as f64;
            assert!(
                (g as f64 - want_val).abs() <= tol + tol * want_val.abs(),
                "{ctx}: index {i}: got {g}, want {want_val}"
            );
        }
    }

    #[test]
    fn conv2d_matches_reference_for_a_plain_3x3() {
        let cin = 2usize;
        let cout = 3usize;
        let (ih, iw) = (6usize, 7usize);
        let input: Vec<f32> = (0..cin * ih * iw)
            .map(|i| ((i as f32) * 0.23).sin() * 1.5)
            .collect();
        let weight: Vec<f32> = (0..cout * cin * 9)
            .map(|i| ((i as f32) * 0.11).cos() * 0.8)
            .collect();
        let bias: Vec<f32> = vec![0.1, -0.2, 0.3];

        assert_conv(
            &input,
            [1, cin, ih, iw],
            &weight,
            [cout, cin, 3, 3],
            Some(&bias),
            Conv2dParams {
                strides: [1, 1],
                pads: [1, 1, 1, 1],
                dilations: [1, 1],
                group: 1,
                activation: ConvActivation::None,
            },
            1e-4,
            "3x3 s1 p1",
        );
    }

    #[test]
    fn conv2d_matches_reference_for_the_1x1_fast_path() {
        let cin = 4usize;
        let cout = 5usize;
        let (ih, iw) = (5usize, 5usize);
        let input: Vec<f32> = (0..cin * ih * iw)
            .map(|i| (i as f32) * 0.07 - 1.2)
            .collect();
        let weight: Vec<f32> = (0..cout * cin).map(|i| ((i as f32) * 0.31).sin()).collect();

        assert_conv(
            &input,
            [1, cin, ih, iw],
            &weight,
            [cout, cin, 1, 1],
            None,
            Conv2dParams {
                activation: ConvActivation::Relu,
                ..Default::default()
            },
            1e-4,
            "1x1 + relu",
        );
    }

    /// The 1x1 path picks between splitting the output channels (m) and the output
    /// positions (n) by shape, and both are expected to produce identical results. This
    /// guards the *routing* rather than any one branch: a shape that starts taking a
    /// different branch after a threshold change must not change the numbers, and a new
    /// shape must not be routed into a branch that is only correct for the shapes it was
    /// written for.
    ///
    /// This is not a performance test, so it cannot tell a good split from a bad one -- the
    /// split axes differ by 4-6x on some of these shapes and all of them are correct. See
    /// `examples/conv1x1_split_probe.rs` for the measurement that chose the axes.
    ///
    /// The shapes straddle both gates: `n` above and below 64 (the m-split cutoff), and
    /// `n` above and below the old `n >= 256 && n >= m` gate. Large enough to clear the
    /// 250k-FLOP parallelism threshold, since a sequential run would not exercise the
    /// split at all.
    #[test]
    fn conv2d_1x1_split_strategies_agree_on_every_shape() {
        for (m, k, h, w) in [
            (384usize, 192usize, 6usize, 24usize), // the rec model's most common 1x1
            (192, 384, 6, 24),
            (96, 48, 24, 96),
            (384, 192, 3, 75),
            (64, 32, 8, 8),   // n = 64, exactly at the m-split cutoff
            (64, 32, 2, 16),  // n = 32, below the cutoff -> m-split
            (8, 16, 40, 40),  // n = 1600, n >= m
            (300, 4, 20, 20), // n = 400, m > n (the old gate rejected this to the m-split)
        ] {
            let input: Vec<f32> = (0..k * h * w)
                .map(|i| (i as f32 * 0.013).sin() * 1.5)
                .collect();
            let weight: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.021).cos()).collect();
            let bias: Vec<f32> = (0..m).map(|i| i as f32 * 0.01 - 0.3).collect();

            for act in [ConvActivation::None, ConvActivation::Relu] {
                let params = Conv2dParams {
                    activation: act,
                    ..Default::default()
                };
                let got = conv2d(
                    &tensor(&[1, k, h, w], &input),
                    &tensor(&[m, k, 1, 1], &weight),
                    Some(&tensor(&[m], &bias)),
                    &params,
                )
                .expect("1x1 conv");

                // The sequential path is the arbiter: it is what the n < 64 branch falls
                // back to, and it shares no chunking logic with the parallel branches.
                let want = {
                    let _guard = crate::tensor::ScopedDisableParallel::new();
                    conv2d(
                        &tensor(&[1, k, h, w], &input),
                        &tensor(&[m, k, 1, 1], &weight),
                        Some(&tensor(&[m], &bias)),
                        &params,
                    )
                    .expect("1x1 conv, sequential")
                };

                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "m={m} k={k} n={} act={act:?}",
                    h * w
                );
                for (i, (&g, &wt)) in got.as_slice().iter().zip(want.as_slice()).enumerate() {
                    assert!(
                        (g - wt).abs() <= 1e-4 + 1e-4 * wt.abs(),
                        "m={m} k={k} n={} act={act:?} index {i}: parallel={g} sequential={wt}",
                        h * w
                    );
                }
            }
        }
    }

    #[test]
    fn conv2d_matches_reference_for_depthwise() {
        let c = 6usize;
        let (ih, iw) = (7usize, 9usize);
        let input: Vec<f32> = (0..c * ih * iw)
            .map(|i| ((i as f32) * 0.17).cos() * 2.0)
            .collect();
        let weight: Vec<f32> = (0..c * 9)
            .map(|i| ((i as f32) * 0.41).sin() * 0.6)
            .collect();
        let bias: Vec<f32> = (0..c).map(|i| i as f32 * 0.05).collect();

        assert_conv(
            &input,
            [1, c, ih, iw],
            &weight,
            [c, 1, 3, 3],
            Some(&bias),
            Conv2dParams {
                pads: [1, 1, 1, 1],
                group: c,
                ..Default::default()
            },
            1e-4,
            "depthwise 3x3",
        );
    }

    #[test]
    fn conv2d_matches_reference_for_grouped_convolution() {
        let (cin, cout, group) = (4usize, 6usize, 2usize);
        let (ih, iw) = (6usize, 6usize);
        let input: Vec<f32> = (0..cin * ih * iw)
            .map(|i| ((i as f32) * 0.13).sin() * 1.7)
            .collect();
        let weight: Vec<f32> = (0..cout * (cin / group) * 9)
            .map(|i| ((i as f32) * 0.29).cos())
            .collect();

        assert_conv(
            &input,
            [1, cin, ih, iw],
            &weight,
            [cout, cin / group, 3, 3],
            None,
            Conv2dParams {
                pads: [1, 1, 1, 1],
                group,
                ..Default::default()
            },
            1e-4,
            "grouped 3x3 group=2",
        );
    }

    #[test]
    fn conv2d_matches_reference_for_strided_and_dilated_cases() {
        let cin = 3usize;
        let cout = 2usize;
        let (ih, iw) = (9usize, 9usize);
        let input: Vec<f32> = (0..cin * ih * iw)
            .map(|i| ((i as f32) * 0.19).sin() * 2.0)
            .collect();
        let weight: Vec<f32> = (0..cout * cin * 9)
            .map(|i| ((i as f32) * 0.37).cos() * 0.7)
            .collect();

        // Stride 2 with asymmetric padding.
        assert_conv(
            &input,
            [1, cin, ih, iw],
            &weight,
            [cout, cin, 3, 3],
            None,
            Conv2dParams {
                strides: [2, 2],
                pads: [1, 0, 0, 1],
                dilations: [1, 1],
                group: 1,
                activation: ConvActivation::None,
            },
            1e-4,
            "3x3 s2 asymmetric pads",
        );

        // Dilation 2.
        assert_conv(
            &input,
            [1, cin, ih, iw],
            &weight,
            [cout, cin, 3, 3],
            None,
            Conv2dParams {
                strides: [1, 1],
                pads: [2, 2, 2, 2],
                dilations: [2, 2],
                group: 1,
                activation: ConvActivation::None,
            },
            1e-4,
            "3x3 dil2 p2",
        );
    }

    #[test]
    fn conv2d_batches_independently() {
        let cin = 2usize;
        let cout = 2usize;
        let (ih, iw) = (4usize, 4usize);
        let batch = 3usize;
        let input: Vec<f32> = (0..batch * cin * ih * iw)
            .map(|i| ((i as f32) * 0.53).sin())
            .collect();
        let weight: Vec<f32> = (0..cout * cin * 9).map(|i| (i as f32) * 0.05).collect();

        assert_conv(
            &input,
            [batch, cin, ih, iw],
            &weight,
            [cout, cin, 3, 3],
            None,
            Conv2dParams {
                pads: [1, 1, 1, 1],
                ..Default::default()
            },
            1e-4,
            "batch=3",
        );
    }

    #[test]
    fn conv_activation_formulas_are_the_documented_ones() {
        let xs = [-8.0f32, -3.0, -1.0, 0.0, 0.5, 3.0, 8.0];

        for &x in &xs {
            assert_eq!(apply_conv_activation(x, ConvActivation::None), x);
            assert_eq!(apply_conv_activation(x, ConvActivation::Relu), x.max(0.0));
            // HardSwish(x) == x * clamp(x/6 + 0.5, 0, 1) == x * hardsigmoid_{1/6,0.5}(x)
            let hs = (x / 6.0 + 0.5f32).clamp(0.0, 1.0);
            assert!(
                (apply_conv_activation(x, ConvActivation::HardSwish) - x * hs).abs() < 1e-6,
                "HardSwish mismatch at x={x}"
            );
            let hsig = (x * (1.0 / 6.0) + 0.5).clamp(0.0, 1.0);
            let got = apply_conv_activation(
                x,
                ConvActivation::HardSigmoid {
                    alpha: 1.0 / 6.0,
                    beta: 0.5,
                },
            );
            assert!((got - hsig).abs() < 1e-6, "HardSigmoid mismatch at x={x}");
        }

        // GELU must stay within the erf-approximation error of the exact formula.
        for &x in &xs {
            let exact = 0.5 * x as f64 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2));
            let got = apply_conv_activation(x, ConvActivation::Gelu) as f64;
            assert!(
                (got - exact).abs() < 2e-3,
                "GELU at x={x}: got {got}, exact {exact}"
            );
        }
    }

    /// Abramowitz-Stegun 7.1.26, used only as an independent erf for the GELU check above.
    fn erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = ((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t
            + 0.254829592)
            * t;
        let v = 1.0 - poly * (-x * x).exp();
        if x >= 0.0 { v } else { -v }
    }

    #[test]
    fn conv2d_rejects_shape_mismatches() {
        let x = tensor(&[1, 3, 8, 8], &vec![0.0; 192]);
        let w = tensor(&[4, 2, 3, 3], &vec![0.0; 72]);
        assert!(conv2d(&x, &w, None, &Conv2dParams::default()).is_err());

        // Non-4D input.
        let flat = tensor(&[3, 8], &[0.0; 24]);
        assert!(conv2d(&flat, &w, None, &Conv2dParams::default()).is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn conv2d_avx2_matches_scalar() {
        // A depthwise case (dedicated AVX2 row kernel) and a general case, both big enough
        // to take the vectorised paths, with channel counts that are not multiples of 8 to
        // cover the scalar tails as well.
        let c = 10usize;
        let (ih, iw) = (21usize, 23usize);
        let dw_in: Vec<f32> = (0..c * ih * iw)
            .map(|i| ((i as f32) * 0.071).sin() * 3.0)
            .collect();
        let dw_w: Vec<f32> = (0..c * 9).map(|i| ((i as f32) * 0.13).cos()).collect();
        let dw_b: Vec<f32> = (0..c).map(|i| (i as f32 * 0.11).sin()).collect();
        let dw_params = Conv2dParams {
            pads: [1, 1, 1, 1],
            group: c,
            activation: ConvActivation::Relu,
            ..Default::default()
        };

        let gen_in: Vec<f32> = (0..c * ih * iw)
            .map(|i| ((i as f32) * 0.043).cos() * 2.0)
            .collect();
        let gen_w: Vec<f32> = (0..c * c * 9)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        let gen_params = Conv2dParams {
            pads: [1, 1, 1, 1],
            ..Default::default()
        };

        for (label, input, weight, bias, shape_w, params) in [
            (
                "depthwise",
                &dw_in,
                &dw_w,
                Some(&dw_b),
                [c, 1, 3, 3],
                dw_params,
            ),
            ("general", &gen_in, &gen_w, None, [c, c, 3, 3], gen_params),
        ] {
            let x = tensor(&[1, c, ih, iw], input);
            let w = tensor(&shape_w, weight);
            let b = bias.map(|b| tensor(&[c], b));

            let scalar = crate::ops::simd::with_avx2_override(false, || {
                conv2d(&x, &w, b.as_ref(), &params).expect("conv2d")
            });
            let avx2 = crate::ops::simd::with_avx2_override(true, || {
                conv2d(&x, &w, b.as_ref(), &params).expect("conv2d")
            });

            assert_eq!(avx2.shape(), scalar.shape(), "{label}: shape");
            for (i, (&a, &s)) in avx2.as_slice().iter().zip(scalar.as_slice()).enumerate() {
                assert!(
                    (a - s).abs() <= 1e-4 + 1e-4 * s.abs(),
                    "{label}: index {i}: avx2={a} scalar={s}"
                );
            }
        }
    }

    /// Regression for the unguarded-FMA bug in the `cin == 3, 3x3, stride 2` fast path.
    ///
    /// `conv2d_direct_cin3_k3_s2` used to call `_mm_fmadd_ps` from a function carrying no
    /// `#[target_feature(enable = "fma")]` and reached without consulting
    /// `avx2_fma_enabled()`. rustc emits a bare `vfmadd` in that situation, so the kernel
    /// (the first convolution of every PP-OCR backbone) faulted on any pre-Haswell x86_64,
    /// contradicting the guarantee documented in `simd.rs`. The SIMD body now lives in
    /// `cin3_k3_s2_step_fma`, behind the probe.
    ///
    /// This test cannot observe a SIGILL, but it does pin the two properties the fix relies
    /// on: the fast path is still taken, and forcing the wide path off yields the same
    /// answer. The shapes straddle the SSE block's interior/boundary split and end on a
    /// non-multiple-of-four output width, so the vectorised step *and* both scalar tails
    /// (left edge, right edge) all execute.
    /// Every input that made `conv2d` divide by zero, underflow, or index a short bias now
    /// returns a typed error. Each case used to be a panic (or, in release, a silent wrap
    /// feeding a wrongly-sized `Tensor::empty`).
    #[test]
    fn conv2d_rejects_degenerate_parameters() {
        let x = tensor(&[1, 4, 8, 8], &vec![1.0; 4 * 8 * 8]);
        let w = tensor(&[4, 4, 3, 3], &vec![1.0; 4 * 4 * 9]);
        let b = tensor(&[4], &[0.5; 4]);

        // group == 0 -> `in_channels % group` was an integer divide-by-zero panic.
        let err = conv2d(
            &x,
            &w,
            None,
            &Conv2dParams {
                group: 0,
                ..Default::default()
            },
        )
        .expect_err("group=0 must be rejected");
        assert!(matches!(err, OcrError::OperatorError { .. }), "{err:?}");

        // A zero-sized kernel underflowed `k_h - 1`: a debug panic, a silent wrap in release.
        for shape in [[4usize, 4, 0, 3], [4, 4, 3, 0]] {
            let wz = tensor(&shape, &[]);
            let r = std::panic::catch_unwind(|| {
                conv2d(
                    &x,
                    &wz,
                    None,
                    &Conv2dParams {
                        pads: [0, 0, 0, 0],
                        ..Default::default()
                    },
                )
            });
            assert!(r.is_ok(), "kernel {shape:?} must not panic");
            assert!(r.unwrap().is_err(), "kernel {shape:?} must be rejected");
        }

        // stride == 0 would divide by zero in the output-size formula.
        let err = conv2d(
            &x,
            &w,
            None,
            &Conv2dParams {
                strides: [0, 1],
                ..Default::default()
            },
        )
        .expect_err("stride=0 must be rejected");
        assert!(matches!(err, OcrError::OperatorError { .. }), "{err:?}");

        // A bias shorter than out_channels was indexed unchecked inside the kernels.
        for n in [0usize, 1, 3] {
            let short = tensor(&[n], &vec![0.0; n]);
            let r = std::panic::catch_unwind(|| {
                conv2d(
                    &x,
                    &w,
                    Some(&short),
                    &Conv2dParams {
                        pads: [1, 1, 1, 1],
                        ..Default::default()
                    },
                )
            });
            assert!(r.is_ok(), "bias of {n} must not panic");
            assert!(r.unwrap().is_err(), "bias of {n} must be rejected");
        }

        // A correct bias of the right length still works, so the check above is not just
        // rejecting everything.
        let ok = conv2d(
            &x,
            &w,
            Some(&b),
            &Conv2dParams {
                pads: [1, 1, 1, 1],
                ..Default::default()
            },
        )
        .expect("a valid bias must be accepted");
        assert_eq!(ok.shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn cin3_k3_s2_fast_path_is_gated_and_matches_scalar() {
        for (ih, iw) in [(12usize, 12usize), (10, 23), (9, 41)] {
            let out_h = ih.div_ceil(2);
            let out_w = iw.div_ceil(2);

            let x: Vec<f32> = (0..3 * ih * iw).map(|i| (i as f32 * 0.37).sin()).collect();
            let w: Vec<f32> = (0..16 * 27)
                .map(|i| (i as f32 * 0.11).cos() * 0.5)
                .collect();
            let b: Vec<f32> = (0..16).map(|i| (i as f32 * 0.21).sin()).collect();

            for (label, bias, act) in [
                ("plain", Some(b.as_slice()), ConvActivation::None),
                ("relu", Some(b.as_slice()), ConvActivation::Relu),
                ("no-bias", None, ConvActivation::None),
                (
                    "hardsigmoid",
                    Some(b.as_slice()),
                    ConvActivation::HardSigmoid {
                        alpha: 1.0 / 6.0,
                        beta: 0.5,
                    },
                ),
            ] {
                let xt = tensor(&[1, 3, ih, iw], &x);
                let wt = tensor(&[16, 3, 3, 3], &w);
                let bt = bias.map(|s| tensor(&[16], s));
                let params = Conv2dParams {
                    strides: [2, 2],
                    pads: [1, 1, 1, 1],
                    activation: act,
                    ..Default::default()
                };

                let scalar = crate::ops::simd::with_avx2_override(false, || {
                    conv2d(&xt, &wt, bt.as_ref(), &params).expect("conv2d scalar")
                });
                let wide = crate::ops::simd::with_avx2_override(true, || {
                    conv2d(&xt, &wt, bt.as_ref(), &params).expect("conv2d wide")
                });

                assert_eq!(
                    scalar.shape(),
                    &[1, 16, out_h, out_w],
                    "{label} {ih}x{iw}: unexpected output shape"
                );
                assert_eq!(wide.shape(), scalar.shape(), "{label} {ih}x{iw}: shape");
                for (i, (&a, &s)) in wide.as_slice().iter().zip(scalar.as_slice()).enumerate() {
                    // FMA and separate mul+add round differently, so allow a few ulp.
                    assert!(
                        (a - s).abs() <= 1e-5 + 1e-5 * s.abs(),
                        "{label} {ih}x{iw}: index {i}: wide={a} scalar={s}"
                    );
                }
            }
        }
    }

    /// The exact shape the PP-OCRv6 det/rec backbones feed into their first convolution
    /// (`3 -> 16`, 3x3, stride 2) must agree with a plain NCHW reference.
    #[test]
    fn cin3_k3_s2_matches_naive_reference_on_backbone_stem() {
        let (ih, iw, out_ch) = (64usize, 64usize, 16usize);
        let x: Vec<f32> = (0..3 * ih * iw).map(|i| (i as f32 * 0.017).sin()).collect();
        let w: Vec<f32> = (0..out_ch * 27).map(|i| (i as f32 * 0.023).cos()).collect();
        let b: Vec<f32> = (0..out_ch).map(|i| (i as f32 * 0.05).sin()).collect();

        let xt = tensor(&[1, 3, ih, iw], &x);
        let wt = tensor(&[out_ch, 3, 3, 3], &w);
        let bt = tensor(&[out_ch], &b);
        let params = Conv2dParams {
            strides: [2, 2],
            pads: [1, 1, 1, 1],
            ..Default::default()
        };

        let got = conv2d(&xt, &wt, Some(&bt), &params).expect("conv2d");
        let (out_h, out_w) = (ih / 2, iw / 2);
        assert_eq!(got.shape(), &[1, out_ch, out_h, out_w]);

        for oc in 0..out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = b[oc];
                    for ic in 0..3 {
                        for kh in 0..3 {
                            let ih_i = (oh * 2 + kh) as isize - 1;
                            if ih_i < 0 || ih_i >= ih as isize {
                                continue;
                            }
                            for kw in 0..3 {
                                let iw_i = (ow * 2 + kw) as isize - 1;
                                if iw_i < 0 || iw_i >= iw as isize {
                                    continue;
                                }
                                sum += x[ic * ih * iw + ih_i as usize * iw + iw_i as usize]
                                    * w[oc * 27 + ic * 9 + kh * 3 + kw];
                            }
                        }
                    }
                    let idx = oc * out_h * out_w + oh * out_w + ow;
                    let g = got.as_slice()[idx];
                    assert!(
                        (g - sum).abs() <= 1e-4 + 1e-4 * sum.abs(),
                        "oc={oc} oh={oh} ow={ow}: got={g} want={sum}"
                    );
                }
            }
        }
    }
}
