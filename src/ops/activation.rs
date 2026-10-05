use crate::tensor::Tensor;

pub fn relu(input: &Tensor) -> Tensor {
    let mut out = input.clone();
    out.relu_inplace();
    out
}

#[inline(always)]
pub fn relu_inplace(input: &mut Tensor) {
    input.relu_inplace();
}

pub fn clip(input: &Tensor, min: f32, max: f32) -> Tensor {
    let mut out = input.clone();
    out.clip_inplace(min, max);
    out
}

#[inline(always)]
pub fn clip_inplace(input: &mut Tensor, min: f32, max: f32) {
    input.clip_inplace(min, max);
}

pub fn sigmoid(input: &Tensor) -> Tensor {
    let mut out = input.clone();
    sigmoid_inplace(&mut out);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2_chunk(chunk: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        let n = chunk.len();
        let ptr = chunk.as_mut_ptr();
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let mut i = 0;
        while i + 16 <= n {
            let x0 = _mm256_loadu_ps(ptr.add(i));
            let x1 = _mm256_loadu_ps(ptr.add(i + 8));
            let neg0 = _mm256_sub_ps(zero, x0);
            let neg1 = _mm256_sub_ps(zero, x1);
            let exp0 = crate::ops::softmax::avx2_exp_ps(neg0);
            let exp1 = crate::ops::softmax::avx2_exp_ps(neg1);
            let d0 = _mm256_add_ps(one, exp0);
            let d1 = _mm256_add_ps(one, exp1);
            let s0 = _mm256_div_ps(one, d0);
            let s1 = _mm256_div_ps(one, d1);
            _mm256_storeu_ps(ptr.add(i), s0);
            _mm256_storeu_ps(ptr.add(i + 8), s1);
            i += 16;
        }
        while i + 8 <= n {
            let x = _mm256_loadu_ps(ptr.add(i));
            let neg = _mm256_sub_ps(zero, x);
            let exp = crate::ops::softmax::avx2_exp_ps(neg);
            let d = _mm256_add_ps(one, exp);
            let s = _mm256_div_ps(one, d);
            _mm256_storeu_ps(ptr.add(i), s);
            i += 8;
        }
        while i < n {
            *ptr.add(i) = 1.0 / (1.0 + (-*ptr.add(i)).exp());
            i += 1;
        }
    }
}

pub fn sigmoid_inplace(input: &mut Tensor) {
    use rayon::prelude::*;
    let slice = input.as_mut_slice();
    let n = slice.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        let chunk_size = 16_384;
        if n >= 32_768 && rayon::current_num_threads() > 1 {
            slice.par_chunks_mut(chunk_size).for_each(|chunk| {
                unsafe { sigmoid_avx2_chunk(chunk) };
            });
        } else {
            unsafe { sigmoid_avx2_chunk(slice) };
        }
        return;
    }

    input.sigmoid_inplace();
}

pub fn hardsigmoid(input: &Tensor, alpha: f32, beta: f32) -> Tensor {
    let mut out = input.clone();
    out.hardsigmoid_inplace(alpha, beta);
    out
}

#[inline(always)]
pub fn hardsigmoid_inplace(input: &mut Tensor, alpha: f32, beta: f32) {
    input.hardsigmoid_inplace(alpha, beta);
}

pub fn hardswish(input: &Tensor) -> Tensor {
    let mut out = input.clone();
    out.hardswish_inplace();
    out
}

#[inline(always)]
pub fn hardswish_inplace(input: &mut Tensor) {
    input.hardswish_inplace();
}

pub fn erf(input: &Tensor) -> Tensor {
    let mut out = input.clone();
    out.erf_inplace();
    out
}

#[inline(always)]
pub fn erf_inplace(input: &mut Tensor) {
    input.erf_inplace();
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_avx2_chunk(chunk: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        let n = chunk.len();
        let ptr = chunk.as_mut_ptr();
        let mut i = 0;
        while i + 32 <= n {
            let v0 = crate::ops::conv2d::avx2_gelu(_mm256_loadu_ps(ptr.add(i)));
            let v1 = crate::ops::conv2d::avx2_gelu(_mm256_loadu_ps(ptr.add(i + 8)));
            let v2 = crate::ops::conv2d::avx2_gelu(_mm256_loadu_ps(ptr.add(i + 16)));
            let v3 = crate::ops::conv2d::avx2_gelu(_mm256_loadu_ps(ptr.add(i + 24)));
            _mm256_storeu_ps(ptr.add(i), v0);
            _mm256_storeu_ps(ptr.add(i + 8), v1);
            _mm256_storeu_ps(ptr.add(i + 16), v2);
            _mm256_storeu_ps(ptr.add(i + 24), v3);
            i += 32;
        }
        while i + 8 <= n {
            let v = crate::ops::conv2d::avx2_gelu(_mm256_loadu_ps(ptr.add(i)));
            _mm256_storeu_ps(ptr.add(i), v);
            i += 8;
        }
        while i < n {
            let v = *ptr.add(i);
            let scaled = v * 0.707_106_77;
            let sign = if scaled >= 0.0 { 1.0 } else { -1.0 };
            let ax = scaled.abs();
            let p = 0.3275911;
            let a1 = 0.254_829_6;
            let a2 = -0.284_496_72;
            let a3 = 1.421_413_8;
            let a4 = -1.453_152_1;
            let a5 = 1.061_405_4;
            let t = 1.0 / (1.0 + p * ax);
            let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
            let erf_val = sign * (1.0 - poly * (-ax * ax).exp());
            *ptr.add(i) = 0.5 * v * (1.0 + erf_val);
            i += 1;
        }
    }
}

pub fn gelu(input: &Tensor) -> Tensor {
    let mut out = input.clone();
    gelu_inplace(&mut out);
    out
}

pub fn gelu_inplace(input: &mut Tensor) {
    let slice = input.as_mut_slice();
    let n = slice.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        use rayon::prelude::*;
        let chunk_size = 16_384;
        if n >= 32_768 && rayon::current_num_threads() > 1 {
            slice.par_chunks_mut(chunk_size).for_each(|chunk| {
                unsafe { gelu_avx2_chunk(chunk) };
            });
        } else {
            unsafe { gelu_avx2_chunk(slice) };
        }
        return;
    }

    input.gelu_inplace();
}
