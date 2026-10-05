use crate::error::{OcrError, Result};
use crate::tensor::{Tensor, compute_contiguous_strides, elementwise_op};
use rayon::prelude::*;

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        return Ok(vec_binary_same_shape(a, b, BinaryOp::Add));
    }
    if let Some(res) = try_channel_broadcast(a, b, BinaryOp::Add) {
        return Ok(res);
    }
    elementwise_op(a, b, |x, y| x + y)
}

pub fn binary_inplace(a: &mut Tensor, b: &Tensor, op: BinaryOp) -> Result<()> {
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        let n = a.numel();
        let a_slice = a.as_mut_slice();
        let b_slice = b.as_slice();
        let num_threads = crate::tensor::parallel_degree(n, 65_536);
        if num_threads > 1 {
            let chunk_size = n.div_ceil(num_threads).max(16384);
            a_slice
                .par_chunks_mut(chunk_size)
                .zip(b_slice.par_chunks(chunk_size))
                .for_each(|(a_chunk, b_chunk)| {
                    apply_binary_inplace_simd(a_chunk, b_chunk, op);
                });
        } else {
            apply_binary_inplace_simd(a_slice, b_slice, op);
        }
        return Ok(());
    }
    if try_channel_broadcast_inplace(a, b, op) {
        return Ok(());
    }
    let res = match op {
        BinaryOp::Add => add(a, b)?,
        BinaryOp::Sub => sub(a, b)?,
        BinaryOp::Mul => mul(a, b)?,
        BinaryOp::Div => div(a, b)?,
    };
    *a = res;
    Ok(())
}

pub fn add_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    binary_inplace(a, b, BinaryOp::Add)
}

pub fn sub_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    binary_inplace(a, b, BinaryOp::Sub)
}

pub fn mul_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    binary_inplace(a, b, BinaryOp::Mul)
}

pub fn div_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    binary_inplace(a, b, BinaryOp::Div)
}

#[inline]
fn apply_binary_inplace_simd(a: &mut [f32], b: &[f32], op: BinaryOp) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        unsafe { apply_binary_inplace_fma(a, b, op) };
        return;
    }
    let n = a.len();
    match op {
        BinaryOp::Add => for i in 0..n { a[i] += b[i]; },
        BinaryOp::Sub => for i in 0..n { a[i] -= b[i]; },
        BinaryOp::Mul => for i in 0..n { a[i] *= b[i]; },
        BinaryOp::Div => for i in 0..n { a[i] /= b[i]; },
    }
}

/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_binary_inplace_fma(a: &mut [f32], b: &[f32], op: BinaryOp) {
    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let a_ptr = a.as_mut_ptr();
        let b_ptr = b.as_ptr();
        let mut i = 0;

        macro_rules! unroll_inplace_loop {
            ($op_simd:ident, $op_scalar:tt) => {
                while i + 32 <= n {
                    let va0 = _mm256_loadu_ps(a_ptr.add(i));
                    let vb0 = _mm256_loadu_ps(b_ptr.add(i));
                    let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let vb1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let va2 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let vb2 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let va3 = _mm256_loadu_ps(a_ptr.add(i + 24));
                    let vb3 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    _mm256_storeu_ps(a_ptr.add(i), $op_simd(va0, vb0));
                    _mm256_storeu_ps(a_ptr.add(i + 8), $op_simd(va1, vb1));
                    _mm256_storeu_ps(a_ptr.add(i + 16), $op_simd(va2, vb2));
                    _mm256_storeu_ps(a_ptr.add(i + 24), $op_simd(va3, vb3));
                    i += 32;
                }
                while i + 8 <= n {
                    let va = _mm256_loadu_ps(a_ptr.add(i));
                    let vb = _mm256_loadu_ps(b_ptr.add(i));
                    _mm256_storeu_ps(a_ptr.add(i), $op_simd(va, vb));
                    i += 8;
                }
                while i < n {
                    *a_ptr.add(i) $op_scalar *b_ptr.add(i);
                    i += 1;
                }
            };
        }

        match op {
            BinaryOp::Add => { unroll_inplace_loop!(_mm256_add_ps, +=); }
            BinaryOp::Sub => { unroll_inplace_loop!(_mm256_sub_ps, -=); }
            BinaryOp::Mul => { unroll_inplace_loop!(_mm256_mul_ps, *=); }
            BinaryOp::Div => { unroll_inplace_loop!(_mm256_div_ps, /=); }
        }
    }
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        return Ok(vec_binary_same_shape(a, b, BinaryOp::Sub));
    }
    if let Some(res) = try_channel_broadcast(a, b, BinaryOp::Sub) {
        return Ok(res);
    }
    elementwise_op(a, b, |x, y| x - y)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        return Ok(vec_binary_same_shape(a, b, BinaryOp::Mul));
    }
    if let Some(res) = try_channel_broadcast(a, b, BinaryOp::Mul) {
        return Ok(res);
    }
    elementwise_op(a, b, |x, y| x * y)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn vec_binary_same_shape(a: &Tensor, b: &Tensor, op: BinaryOp) -> Tensor {
    let n = a.numel();
    let mut out = Tensor::empty(a.shape());
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let out_slice = out.as_mut_slice();

    let num_threads = crate::tensor::parallel_degree(n, 65_536);
    if num_threads > 1 {
        let chunk_size = n.div_ceil(num_threads).max(16384);
        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let offset = chunk_idx * chunk_size;
                let a_sub = &a_slice[offset..offset + out_chunk.len()];
                let b_sub = &b_slice[offset..offset + out_chunk.len()];
                apply_binary_simd(out_chunk, a_sub, b_sub, op);
            });
    } else {
        apply_binary_simd(out_slice, a_slice, b_slice, op);
    }

    out
}

#[inline]
fn apply_binary_simd(dst: &mut [f32], a: &[f32], b: &[f32], op: BinaryOp) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        unsafe { apply_binary_fma(dst, a, b, op) };
        return;
    }
    let n = dst.len();
    match op {
        BinaryOp::Add => for i in 0..n { dst[i] = a[i] + b[i]; },
        BinaryOp::Sub => for i in 0..n { dst[i] = a[i] - b[i]; },
        BinaryOp::Mul => for i in 0..n { dst[i] = a[i] * b[i]; },
        BinaryOp::Div => for i in 0..n { dst[i] = a[i] / b[i]; },
    }
}

/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_binary_fma(dst: &mut [f32], a: &[f32], b: &[f32], op: BinaryOp) {
    unsafe {
        use std::arch::x86_64::*;
        let n = dst.len();
        let d_ptr = dst.as_mut_ptr();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut i = 0;

        macro_rules! unroll_dst_loop {
            ($op_simd:ident, $op_scalar:tt) => {
                while i + 32 <= n {
                    let va0 = _mm256_loadu_ps(a_ptr.add(i));
                    let vb0 = _mm256_loadu_ps(b_ptr.add(i));
                    let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let vb1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let va2 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let vb2 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let va3 = _mm256_loadu_ps(a_ptr.add(i + 24));
                    let vb3 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    _mm256_storeu_ps(d_ptr.add(i), $op_simd(va0, vb0));
                    _mm256_storeu_ps(d_ptr.add(i + 8), $op_simd(va1, vb1));
                    _mm256_storeu_ps(d_ptr.add(i + 16), $op_simd(va2, vb2));
                    _mm256_storeu_ps(d_ptr.add(i + 24), $op_simd(va3, vb3));
                    i += 32;
                }
                while i + 8 <= n {
                    let va = _mm256_loadu_ps(a_ptr.add(i));
                    let vb = _mm256_loadu_ps(b_ptr.add(i));
                    _mm256_storeu_ps(d_ptr.add(i), $op_simd(va, vb));
                    i += 8;
                }
                while i < n {
                    *d_ptr.add(i) = *a_ptr.add(i) $op_scalar *b_ptr.add(i);
                    i += 1;
                }
            };
        }

        match op {
            BinaryOp::Add => { unroll_dst_loop!(_mm256_add_ps, +); }
            BinaryOp::Sub => { unroll_dst_loop!(_mm256_sub_ps, -); }
            BinaryOp::Mul => { unroll_dst_loop!(_mm256_mul_ps, *); }
            BinaryOp::Div => { unroll_dst_loop!(_mm256_div_ps, /); }
        }
    }
}

fn try_channel_broadcast(a: &Tensor, b: &Tensor, op: BinaryOp) -> Option<Tensor> {
    use std::borrow::Cow;
    let a_cont = if a.is_contiguous() {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(a.to_contiguous())
    };
    let b_cont = if b.is_contiguous() {
        Cow::Borrowed(b)
    } else {
        Cow::Owned(b.to_contiguous())
    };

    // Case 1: a is [1, C, H, W], b is [1, C, 1, 1] or [C, 1, 1]
    if a_cont.ndim() == 4 && a_cont.shape()[0] == 1 {
        let c = a_cont.shape()[1];
        let hw = a_cont.shape()[2] * a_cont.shape()[3];
        if b_cont.shape() == [1, c, 1, 1] || b_cont.shape() == [c, 1, 1] {
            let mut out = Tensor::empty(a_cont.shape());
            if hw == 0 {
                return Some(out);
            }
            let out_slice = out.as_mut_slice();
            let a_slice = a_cont.as_slice();
            let b_slice = b_cont.as_slice();

            // `par_chunks_mut` hands out disjoint `&mut` slices, so no pointer
            // laundering (which would violate the aliasing model) is needed.
            let run_channel = |(ch, dst): (usize, &mut [f32])| {
                let src = &a_slice[ch * hw..(ch + 1) * hw];
                let vb_val = b_slice[ch];

                #[cfg(target_arch = "x86_64")]
                if crate::ops::avx2_fma_enabled() {
                    unsafe { channel_broadcast_fma(dst, src, vb_val, op) };
                    return;
                }

                for (i, d) in dst.iter_mut().enumerate() {
                    *d = match op {
                        BinaryOp::Add => src[i] + vb_val,
                        BinaryOp::Sub => src[i] - vb_val,
                        BinaryOp::Mul => src[i] * vb_val,
                        BinaryOp::Div => src[i] / vb_val,
                    };
                }
            };

            if crate::tensor::should_parallelize(c * hw, 65536) {
                out_slice
                    .par_chunks_mut(hw)
                    .enumerate()
                    .for_each(run_channel);
            } else {
                out_slice.chunks_mut(hw).enumerate().for_each(run_channel);
            }
            return Some(out);
        }
    }

    // Case 2: b is [1, C, H, W], a is [1, C, 1, 1] or [C, 1, 1] (symmetric for Add/Mul)
    if (matches!(op, BinaryOp::Add | BinaryOp::Mul)) && b_cont.ndim() == 4 && b_cont.shape()[0] == 1
    {
        let c = b_cont.shape()[1];
        if a_cont.shape() == [1, c, 1, 1] || a_cont.shape() == [c, 1, 1] {
            return try_channel_broadcast(&b_cont, &a_cont, op);
        }
    }

    None
}

/// Per-channel broadcast of a scalar over `dst`, 8 lanes at a time.
///
/// # Safety
/// Requires AVX2+FMA (see [`crate::ops::avx2_fma_enabled`]). `src` must be at
/// least as long as `dst`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn channel_broadcast_fma(dst: &mut [f32], src: &[f32], vb_val: f32, op: BinaryOp) {
    let hw = dst.len();
    let mut i = 0;
    unsafe {
        use std::arch::x86_64::*;
        let vb = _mm256_set1_ps(vb_val);
        let s_ptr = src.as_ptr();
        let d_ptr = dst.as_mut_ptr();

        macro_rules! unroll_broadcast_loop {
            ($op_simd:ident, $op_scalar:tt) => {
                while i + 32 <= hw {
                    let va0 = _mm256_loadu_ps(s_ptr.add(i));
                    let va1 = _mm256_loadu_ps(s_ptr.add(i + 8));
                    let va2 = _mm256_loadu_ps(s_ptr.add(i + 16));
                    let va3 = _mm256_loadu_ps(s_ptr.add(i + 24));

                    _mm256_storeu_ps(d_ptr.add(i), $op_simd(va0, vb));
                    _mm256_storeu_ps(d_ptr.add(i + 8), $op_simd(va1, vb));
                    _mm256_storeu_ps(d_ptr.add(i + 16), $op_simd(va2, vb));
                    _mm256_storeu_ps(d_ptr.add(i + 24), $op_simd(va3, vb));
                    i += 32;
                }
                while i + 8 <= hw {
                    let va = _mm256_loadu_ps(s_ptr.add(i));
                    _mm256_storeu_ps(d_ptr.add(i), $op_simd(va, vb));
                    i += 8;
                }
                while i < hw {
                    *d_ptr.add(i) = *s_ptr.add(i) $op_scalar vb_val;
                    i += 1;
                }
            };
        }

        match op {
            BinaryOp::Add => { unroll_broadcast_loop!(_mm256_add_ps, +); }
            BinaryOp::Sub => { unroll_broadcast_loop!(_mm256_sub_ps, -); }
            BinaryOp::Mul => { unroll_broadcast_loop!(_mm256_mul_ps, *); }
            BinaryOp::Div => { unroll_broadcast_loop!(_mm256_div_ps, /); }
        }
    }
}

fn try_channel_broadcast_inplace(a: &mut Tensor, b: &Tensor, op: BinaryOp) -> bool {
    if !a.is_contiguous() || !b.is_contiguous() {
        return false;
    }
    if a.ndim() == 4 && a.shape()[0] == 1 {
        let c = a.shape()[1];
        let hw = a.shape()[2] * a.shape()[3];
        if (b.shape() == [1, c, 1, 1] || b.shape() == [c, 1, 1]) && hw > 0 {
            let b_slice = b.as_slice();
            let a_slice = a.as_mut_slice();

            let run_channel = |(ch, dst): (usize, &mut [f32])| {
                let vb_val = b_slice[ch];

                #[cfg(target_arch = "x86_64")]
                if crate::ops::avx2_fma_enabled() {
                    unsafe { channel_broadcast_inplace_fma(dst, vb_val, op) };
                    return;
                }

                for d in dst.iter_mut() {
                    match op {
                        BinaryOp::Add => *d += vb_val,
                        BinaryOp::Sub => *d -= vb_val,
                        BinaryOp::Mul => *d *= vb_val,
                        BinaryOp::Div => *d /= vb_val,
                    }
                }
            };

            let num_threads = crate::tensor::parallel_degree(a_slice.len(), 32_768);
            if num_threads > 1 {
                a_slice
                    .par_chunks_mut(hw)
                    .enumerate()
                    .for_each(run_channel);
            } else {
                a_slice.chunks_mut(hw).enumerate().for_each(run_channel);
            }
            return true;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn channel_broadcast_inplace_fma(dst: &mut [f32], vb_val: f32, op: BinaryOp) {
    let hw = dst.len();
    let mut i = 0;
    unsafe {
        use std::arch::x86_64::*;
        let vb = _mm256_set1_ps(vb_val);
        let d_ptr = dst.as_mut_ptr();

        macro_rules! unroll_broadcast_inplace_loop {
            ($op_simd:ident, $op_scalar:tt) => {
                while i + 32 <= hw {
                    let va0 = _mm256_loadu_ps(d_ptr.add(i));
                    let va1 = _mm256_loadu_ps(d_ptr.add(i + 8));
                    let va2 = _mm256_loadu_ps(d_ptr.add(i + 16));
                    let va3 = _mm256_loadu_ps(d_ptr.add(i + 24));

                    _mm256_storeu_ps(d_ptr.add(i), $op_simd(va0, vb));
                    _mm256_storeu_ps(d_ptr.add(i + 8), $op_simd(va1, vb));
                    _mm256_storeu_ps(d_ptr.add(i + 16), $op_simd(va2, vb));
                    _mm256_storeu_ps(d_ptr.add(i + 24), $op_simd(va3, vb));
                    i += 32;
                }
                while i + 8 <= hw {
                    let va = _mm256_loadu_ps(d_ptr.add(i));
                    _mm256_storeu_ps(d_ptr.add(i), $op_simd(va, vb));
                    i += 8;
                }
                while i < hw {
                    *d_ptr.add(i) $op_scalar vb_val;
                    i += 1;
                }
            };
        }

        match op {
            BinaryOp::Add => { unroll_broadcast_inplace_loop!(_mm256_add_ps, +=); }
            BinaryOp::Sub => { unroll_broadcast_inplace_loop!(_mm256_sub_ps, -=); }
            BinaryOp::Mul => { unroll_broadcast_inplace_loop!(_mm256_mul_ps, *=); }
            BinaryOp::Div => { unroll_broadcast_inplace_loop!(_mm256_div_ps, /=); }
        }
    }
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        return Ok(vec_binary_same_shape(a, b, BinaryOp::Div));
    }
    if let Some(res) = try_channel_broadcast(a, b, BinaryOp::Div) {
        return Ok(res);
    }
    elementwise_op(a, b, |x, y| x / y)
}

pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    elementwise_op(a, b, |x, y| x.powf(y))
}

pub fn sqrt(a: &Tensor) -> Tensor {
    let mut out = a.clone();
    out.sqrt_inplace();
    out
}

pub fn sqrt_inplace(a: &mut Tensor) {
    a.sqrt_inplace();
}

pub fn concat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(OcrError::OperatorError {
            op: "Concat".into(),
            msg: "Cannot concat empty tensor list".into(),
        });
    }

    let first = tensors[0];
    let ndim = first.ndim();
    let norm_axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if norm_axis >= ndim {
        return Err(OcrError::OperatorError {
            op: "Concat".into(),
            msg: format!("Axis {} out of bounds for ndim {}", axis, ndim),
        });
    }

    let mut out_shape = first.shape().to_vec();
    let mut total_axis_len = 0;

    for (i, t) in tensors.iter().enumerate() {
        if t.ndim() != ndim {
            return Err(OcrError::OperatorError {
                op: "Concat".into(),
                msg: format!(
                    "Tensor {} ndim {} does not match first tensor ndim {}",
                    i,
                    t.ndim(),
                    ndim
                ),
            });
        }
        for d in 0..ndim {
            if d != norm_axis && t.shape()[d] != first.shape()[d] {
                return Err(OcrError::OperatorError {
                    op: "Concat".into(),
                    msg: format!(
                        "Dimension mismatch at dim {}: {} vs {}",
                        d,
                        t.shape()[d],
                        first.shape()[d]
                    ),
                });
            }
        }
        total_axis_len += t.shape()[norm_axis];
    }

    out_shape[norm_axis] = total_axis_len;
    // Every output element is written by the copy loop below, so no zero-fill.
    let mut output = Tensor::empty(&out_shape);

    let outer_size: usize = out_shape[..norm_axis].iter().product();
    let inner_size: usize = out_shape[norm_axis + 1..].iter().product();
    let out_slice = output.as_mut_slice();

    let all_contiguous = tensors.iter().all(|t| t.is_contiguous());
    let contiguous_tensors: Vec<std::borrow::Cow<Tensor>> = if all_contiguous {
        Vec::new()
    } else {
        tensors
            .iter()
            .map(|t| {
                if t.is_contiguous() {
                    std::borrow::Cow::Borrowed(*t)
                } else {
                    std::borrow::Cow::Owned(t.to_contiguous())
                }
            })
            .collect()
    };

    let get_slice = |idx: usize| -> &[f32] {
        if all_contiguous {
            tensors[idx].as_slice()
        } else {
            contiguous_tensors[idx].as_slice()
        }
    };

    if outer_size == 1 && all_contiguous {
        if out_slice.len() >= 65536 && rayon::current_num_threads() > 1 {
            let num_threads = rayon::current_num_threads();
            let out_ptr = out_slice.as_mut_ptr() as usize;
            let chunk_size = (out_slice.len() / (num_threads * 2)).div_ceil(16) * 16;
            let chunk_size = chunk_size.clamp(16384, 131072);

            let mut tasks = Vec::with_capacity(num_threads * 2 + tensors.len());
            let mut offset = 0;
            for t in tensors {
                let s = t.as_slice();
                let src_base = s.as_ptr() as usize;
                let len = s.len();
                let mut c_start = 0;
                while c_start < len {
                    let c_len = (len - c_start).min(chunk_size);
                    tasks.push((src_base + c_start * 4, out_ptr + (offset + c_start) * 4, c_len));
                    c_start += c_len;
                }
                offset += len;
            }

            tasks.into_par_iter().for_each(|(src_p, dst_p, len)| {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_p as *const f32, dst_p as *mut f32, len);
                }
            });
        } else {
            let mut offset = 0;
            for t in tensors {
                let s = t.as_slice();
                out_slice[offset..offset + s.len()].copy_from_slice(s);
                offset += s.len();
            }
        }
        return Ok(output);
    }

    if outer_size == 1 && out_slice.len() >= 65536 {
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut curr = 0;
        for t in tensors {
            offsets.push(curr);
            curr += t.shape()[norm_axis] * inner_size;
        }
        let out_ptr_usize = out_slice.as_mut_ptr() as usize;
        let chunk_size = 65536;
        let mut tasks = Vec::new();
        for (i, &offset) in offsets.iter().enumerate() {
            let t_slice = get_slice(i);
            let len = t_slice.len();
            let mut c_start = 0;
            while c_start < len {
                let c_len = (len - c_start).min(chunk_size);
                tasks.push((
                    t_slice.as_ptr() as usize + c_start * 4,
                    offset + c_start,
                    c_len,
                ));
                c_start += c_len;
            }
        }

        tasks.into_par_iter().for_each(|(src_p, dst_off, c_len)| {
            unsafe {
                let src = src_p as *const f32;
                let dst = (out_ptr_usize as *mut f32).add(dst_off);
                std::ptr::copy_nonoverlapping(src, dst, c_len);
            }
        });
        return Ok(output);
    }

    for outer in 0..outer_size {
        let mut axis_offset = 0;
        for i in 0..tensors.len() {
            let t_slice = get_slice(i);
            let t_axis_len = tensors[i].shape()[norm_axis];

            let t_src_offset = outer * (t_axis_len * inner_size);
            let out_dest_offset = outer * (total_axis_len * inner_size) + axis_offset * inner_size;
            let copy_len = t_axis_len * inner_size;

            out_slice[out_dest_offset..out_dest_offset + copy_len]
                .copy_from_slice(&t_slice[t_src_offset..t_src_offset + copy_len]);

            axis_offset += t_axis_len;
        }
    }
    Ok(output)
}

pub fn reduce_mean(input: &Tensor, axes: &[isize], keepdims: bool) -> Result<Tensor> {
    let ndim = input.ndim();
    let mut norm_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim as isize + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    norm_axes.sort();
    norm_axes.dedup();

    for &a in &norm_axes {
        if a >= ndim {
            return Err(OcrError::OperatorError {
                op: "ReduceMean".into(),
                msg: format!("Axis {} out of bounds for ndim {}", a, ndim),
            });
        }
    }

    let mut out_shape = Vec::new();
    let mut reduce_count = 1usize;
    for (d, &dim) in input.shape().iter().enumerate() {
        if norm_axes.contains(&d) {
            reduce_count *= dim;
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(dim);
        }
    }

    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let mut output = Tensor::empty(&out_shape);
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let inv_count = 1.0 / (reduce_count as f32);
    let out_slice = output.as_mut_slice();

    // Fast path 1: Reduce along the last dimension [ndim - 1]
    if norm_axes == [ndim - 1] {
        let k = reduce_count;
        let m = out_slice.len();
        let num_threads = crate::tensor::parallel_degree(m * k, 262_144);

        let run_row = |(row_idx, out_val): (usize, &mut f32)| {
            let row = &in_slice[row_idx * k..(row_idx + 1) * k];
            *out_val = simd_sum_slice(row) * inv_count;
        };

        if m >= 512 && num_threads > 1 {
            let chunk_size = (m / num_threads).max(64);
            out_slice
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(c_idx, chunk)| {
                    let base = c_idx * chunk_size;
                    for (i, val) in chunk.iter_mut().enumerate() {
                        run_row((base + i, val));
                    }
                });
        } else {
            for (row_idx, val) in out_slice.iter_mut().enumerate() {
                run_row((row_idx, val));
            }
        }
        return Ok(output);
    }

    // Fast path 2: Reduce spatial dimensions [2, 3] on 4D tensor [N, C, H, W]
    if ndim == 4 && norm_axes == [2, 3] {
        let k = reduce_count;
        let m = out_slice.len();
        let num_threads = crate::tensor::parallel_degree(m * k, 262_144);

        let run_plane = |(plane_idx, out_val): (usize, &mut f32)| {
            let plane = &in_slice[plane_idx * k..(plane_idx + 1) * k];
            *out_val = simd_sum_slice(plane) * inv_count;
        };

        if m >= 512 && num_threads > 1 {
            let chunk_size = (m / num_threads).max(64);
            out_slice
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(c_idx, chunk)| {
                    let base = c_idx * chunk_size;
                    for (i, val) in chunk.iter_mut().enumerate() {
                        run_plane((base + i, val));
                    }
                });
        } else {
            for (plane_idx, val) in out_slice.iter_mut().enumerate() {
                run_plane((plane_idx, val));
            }
        }
        return Ok(output);
    }

    // General fallback with fixed stack coordinate array (zero heap allocation in inner loop)
    let in_strides = compute_contiguous_strides(input.shape());
    let out_strides = compute_contiguous_strides(&out_shape);

    let run_elem = |(out_idx, out_val): (usize, &mut f32)| {
        let mut rem = out_idx;
        let mut base_in_offset = 0;
        let mut out_dim_idx = 0;

        for (d, &in_stride) in in_strides.iter().enumerate() {
            if !norm_axes.contains(&d) {
                let coord = rem / out_strides[out_dim_idx];
                rem %= out_strides[out_dim_idx];
                base_in_offset += coord * in_stride;
                out_dim_idx += 1;
            } else if keepdims {
                out_dim_idx += 1;
            }
        }

        let mut sum = 0.0f32;
        let mut reduce_coord_stack = [0usize; 8];
        let mut reduce_coord_heap = if norm_axes.len() > 8 {
            vec![0usize; norm_axes.len()]
        } else {
            Vec::new()
        };
        let reduce_coord: &mut [usize] = if norm_axes.len() > 8 {
            &mut reduce_coord_heap
        } else {
            &mut reduce_coord_stack[..norm_axes.len()]
        };

        for _ in 0..reduce_count {
            let mut in_offset = base_in_offset;
            for (k, &axis) in norm_axes.iter().enumerate() {
                in_offset += reduce_coord[k] * in_strides[axis];
            }
            sum += in_slice[in_offset];

            for k in (0..norm_axes.len()).rev() {
                reduce_coord[k] += 1;
                if reduce_coord[k] < input.shape()[norm_axes[k]] {
                    break;
                }
                reduce_coord[k] = 0;
            }
        }

        *out_val = sum * inv_count;
    };

    if crate::tensor::should_parallelize(out_slice.len(), 65536) {
        out_slice.par_iter_mut().enumerate().for_each(run_elem);
    } else {
        out_slice.iter_mut().enumerate().for_each(run_elem);
    }

    Ok(output)
}

#[inline(always)]
fn simd_sum_slice(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::avx2_fma_enabled() {
            return unsafe { simd_sum_slice_avx2(row) };
        }
    }
    simd_sum_slice_scalar(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sum_slice_avx2(row: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let n = row.len();
        let ptr = row.as_ptr();
        let mut i = 0;
        let mut v0 = _mm256_setzero_ps();
        let mut v1 = _mm256_setzero_ps();
        let mut v2 = _mm256_setzero_ps();
        let mut v3 = _mm256_setzero_ps();

        while i + 32 <= n {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(ptr.add(i)));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(ptr.add(i + 8)));
            v2 = _mm256_add_ps(v2, _mm256_loadu_ps(ptr.add(i + 16)));
            v3 = _mm256_add_ps(v3, _mm256_loadu_ps(ptr.add(i + 24)));
            i += 32;
        }
        v0 = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));

        while i + 8 <= n {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(ptr.add(i)));
            i += 8;
        }

        let hi128 = _mm256_extractf128_ps(v0, 1);
        let lo128 = _mm256_castps256_ps128(v0);
        let sum128 = _mm_add_ps(lo128, hi128);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sum64, hi32));

        while i < n {
            sum += *ptr.add(i);
            i += 1;
        }
        sum
    }
}

#[inline(always)]
fn simd_sum_slice_scalar(row: &[f32]) -> f32 {
    row.iter().sum()
}

pub fn slice_op(
    input: &Tensor,
    starts: &[isize],
    ends: &[isize],
    axes: Option<&[isize]>,
    steps: Option<&[isize]>,
) -> Result<Tensor> {
    let ndim = input.ndim();
    let default_axes: Vec<isize> = (0..starts.len() as isize).collect();
    let axes = axes.unwrap_or(&default_axes);

    let mut cur = input.clone();

    for i in 0..axes.len() {
        let axis = if axes[i] < 0 {
            (ndim as isize + axes[i]) as usize
        } else {
            axes[i] as usize
        };

        if axis >= cur.ndim() {
            return Err(OcrError::OperatorError {
                op: "Slice".into(),
                msg: format!("Axis {} out of bounds for ndim {}", axis, cur.ndim()),
            });
        }

        let dim_len = cur.shape()[axis] as isize;
        let mut start = starts[i];
        let mut end = ends[i];
        let step = if let Some(st) = steps {
            if i < st.len() { st[i] } else { 1 }
        } else {
            1
        };

        if step <= 0 {
            return Err(OcrError::OperatorError {
                op: "Slice".into(),
                msg: "Negative or zero steps not supported".into(),
            });
        }

        if start < 0 {
            start = (dim_len + start).max(0);
        } else {
            start = start.min(dim_len);
        }

        if end < 0 {
            end = (dim_len + end).max(0);
        } else {
            end = end.min(dim_len);
        }

        if start >= end {
            return Err(OcrError::OperatorError {
                op: "Slice".into(),
                msg: format!("Empty slice: start={}, end={}, dim={}", start, end, dim_len),
            });
        }

        cur = cur.slice(axis, start as usize, end as usize, step as usize)?;
    }

    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
    }

    fn sample(shape: &[usize], seed: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5 + 0.25)
            .collect();
        tensor(shape, &data)
    }

    #[test]
    fn add_sub_mul_match_the_scalar_reference_for_equal_shapes() {
        let a = sample(&[2, 3, 4, 5], 0.0);
        let b = sample(&[2, 3, 4, 5], 7.0);

        #[allow(clippy::type_complexity)]
        let cases: [(&str, Tensor, fn(f32, f32) -> f32); 3] = [
            ("add", add(&a, &b).unwrap(), |x: f32, y: f32| x + y),
            ("sub", sub(&a, &b).unwrap(), |x: f32, y: f32| x - y),
            ("mul", mul(&a, &b).unwrap(), |x: f32, y: f32| x * y),
        ];
        for (name, got, f) in cases {
            assert_eq!(got.shape(), a.shape(), "{name}: shape");
            for (i, (&g, (&x, &y))) in got
                .as_slice()
                .iter()
                .zip(a.as_slice().iter().zip(b.as_slice()))
                .enumerate()
            {
                let want = f(x, y);
                assert!(
                    (g - want).abs() <= 1e-6 + 1e-6 * want.abs(),
                    "{name}: index {i}: {g} vs {want}"
                );
            }
        }
    }

    #[test]
    fn div_and_pow_match_the_scalar_reference() {
        let a = sample(&[3, 4], 0.0).abs_inplace_returning();
        let b = sample(&[3, 4], 3.0).abs_inplace_returning();

        let q = div(&a, &b).unwrap();
        for (i, (&g, (&x, &y))) in q
            .as_slice()
            .iter()
            .zip(a.as_slice().iter().zip(b.as_slice()))
            .enumerate()
        {
            let want = x / y;
            assert!((g - want).abs() <= 1e-6, "div index {i}: {g} vs {want}");
        }

        let p = pow(&a, &b).unwrap();
        for (i, (&g, (&x, &y))) in p
            .as_slice()
            .iter()
            .zip(a.as_slice().iter().zip(b.as_slice()))
            .enumerate()
        {
            let want = x.powf(y);
            assert!((g - want).abs() <= 1e-5, "pow index {i}: {g} vs {want}");
        }
    }

    #[test]
    fn add_supports_per_channel_broadcast() {
        // [1, C, H, W] + [1, C, 1, 1]
        let (c, h, w) = (3usize, 2usize, 2usize);
        let a = sample(&[1, c, h, w], 0.0);
        let bias = tensor(&[1, c, 1, 1], &[0.5, -1.0, 2.0]);

        let out = add(&a, &bias).unwrap();
        assert_eq!(out.shape(), &[1, c, h, w]);
        let hw = h * w;
        for ch in 0..c {
            for i in 0..hw {
                let idx = ch * hw + i;
                let want = a.as_slice()[idx] + bias.as_slice()[ch];
                assert!(
                    (out.as_slice()[idx] - want).abs() < 1e-6,
                    "channel {ch} index {i}"
                );
            }
        }
    }

    #[test]
    fn sub_supports_per_channel_broadcast() {
        // [1, C, H, W] - [1, C, 1, 1]
        let (c, h, w) = (3usize, 2usize, 2usize);
        let a = sample(&[1, c, h, w], 0.0);
        let bias = tensor(&[1, c, 1, 1], &[0.5, -1.0, 2.0]);

        let out = sub(&a, &bias).unwrap();
        assert_eq!(out.shape(), &[1, c, h, w]);
        let hw = h * w;
        for ch in 0..c {
            for i in 0..hw {
                let idx = ch * hw + i;
                let want = a.as_slice()[idx] - bias.as_slice()[ch];
                assert!(
                    (out.as_slice()[idx] - want).abs() < 1e-6,
                    "channel {ch} index {i}"
                );
            }
        }
    }

    #[test]
    fn add_inplace_matches_out_of_place_add() {
        let a = sample(&[4, 7], 1.0);
        let b = sample(&[4, 7], 2.0);
        let expected = add(&a, &b).unwrap();
        let mut acc = a.clone();
        add_inplace(&mut acc, &b).unwrap();
        for (i, (&g, &e)) in acc.as_slice().iter().zip(expected.as_slice()).enumerate() {
            assert!((g - e).abs() <= 1e-6, "index {i}: {g} vs {e}");
        }
    }

    #[test]
    fn concat_joins_along_the_requested_axis() {
        let a = tensor(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[1, 2, 2], &[5.0, 6.0, 7.0, 8.0]);

        let along_c = concat(&[&a, &b], 1).unwrap();
        assert_eq!(along_c.shape(), &[1, 4, 2]);
        assert_eq!(
            along_c.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );

        let along_w = concat(&[&a, &b], -1).unwrap();
        assert_eq!(along_w.shape(), &[1, 2, 4]);
        assert_eq!(
            along_w.as_slice(),
            &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn reduce_mean_matches_the_scalar_reference() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = tensor(&[2, 3, 4], &data);

        // Axis 2, keepdims.
        let out = reduce_mean(&t, &[2], true).unwrap();
        assert_eq!(out.shape(), &[2, 3, 1]);
        for (i, &v) in out.as_slice().iter().enumerate() {
            let want = (0..4).map(|k| (i * 4 + k) as f32).sum::<f32>() / 4.0;
            assert!((v - want).abs() < 1e-5, "index {i}: {v} vs {want}");
        }

        // Negative axis, no keepdims.
        let out2 = reduce_mean(&t, &[-1], false).unwrap();
        assert_eq!(out2.shape(), &[2, 3]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn add_and_mul_avx2_match_scalar_on_large_tensors() {
        // Above the 65536-element parallel threshold so the vectorised chunk kernels run.
        let n = 100_000usize;
        let a_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos()).collect();
        let a = tensor(&[n], &a_data);
        let b = tensor(&[n], &b_data);

        let scalar_add = crate::ops::simd::with_avx2_override(false, || add(&a, &b).unwrap());
        let avx2_add = crate::ops::simd::with_avx2_override(true, || add(&a, &b).unwrap());
        let scalar_mul = crate::ops::simd::with_avx2_override(false, || mul(&a, &b).unwrap());
        let avx2_mul = crate::ops::simd::with_avx2_override(true, || mul(&a, &b).unwrap());

        for (i, (&x, &y)) in avx2_add
            .as_slice()
            .iter()
            .zip(scalar_add.as_slice())
            .enumerate()
        {
            assert!((x - y).abs() <= 1e-6, "add index {i}: avx2={x} scalar={y}");
        }
        for (i, (&x, &y)) in avx2_mul
            .as_slice()
            .iter()
            .zip(scalar_mul.as_slice())
            .enumerate()
        {
            assert!((x - y).abs() <= 1e-6, "mul index {i}: avx2={x} scalar={y}");
        }
    }

    /// Small helper so the div/pow test has strictly positive operands without extra noise.
    trait AbsInPlace {
        fn abs_inplace_returning(self) -> Self;
    }

    impl AbsInPlace for Tensor {
        fn abs_inplace_returning(mut self) -> Self {
            for v in self.as_mut_slice() {
                *v = v.abs() + 0.5;
            }
            self
        }
    }
}
