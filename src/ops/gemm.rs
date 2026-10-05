use faer::linalg::matmul::matmul as faer_matmul;
use faer::{MatMut, MatRef};
pub use faer::{Accum, Par};

/// High-performance row-major general matrix multiplication via `faer`: C = A * B with custom Accum and Par.
///
/// # Safety
/// - `a`, `b`, and `c` must point to valid memory for the specified dimensions and strides.
/// - The memory region of `c` must not overlap with `a` or `b`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_row_major_accum(
    accum: Accum,
    par: Par,
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    a_stride: usize,
    b: *const f32,
    b_stride: usize,
    c: *mut f32,
    c_stride: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    unsafe {
        let a_t = MatRef::from_raw_parts(a, k, m, 1isize, a_stride as isize);
        let b_t = MatRef::from_raw_parts(b, n, k, 1isize, b_stride as isize);
        let mut c_t = MatMut::from_raw_parts_mut(c, n, m, 1isize, c_stride as isize);
        faer_matmul(&mut c_t, accum, &b_t, &a_t, 1.0, par);
    }
}

/// High-performance row-major general matrix multiplication via `faer`: C = A * B.
///
/// Matrix layouts (all row-major with independent row strides):
/// - `A`: [m, k], element at `r * a_stride + c`
/// - `B`: [k, n], element at `r * b_stride + c`
/// - `C`: [m, n], element at `r * c_stride + c`
///
/// Expressed in faer's column-major layout via the mathematical transpose identity:
/// `C = A * B <=> C^T = B^T * A^T`
///
/// # Safety
/// - `a`, `b`, and `c` must point to valid memory for the specified dimensions and strides.
/// - The memory region of `c` must not overlap with `a` or `b`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_row_major(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    a_stride: usize,
    b: *const f32,
    b_stride: usize,
    c: *mut f32,
    c_stride: usize,
) {
    unsafe {
        gemm_row_major_accum(
            Accum::Replace,
            Par::Seq,
            m,
            k,
            n,
            a,
            a_stride,
            b,
            b_stride,
            c,
            c_stride,
        );
    }
}

/// High-performance row-major matrix multiplication with transposed B: C = A * B^T.
///
/// Matrix layouts (all row-major with independent row strides):
/// - `A`: [m, k], element at `r * a_stride + c`
/// - `B`: [n, k], element at `r * b_stride + c`
/// - `C`: [m, n], element at `r * c_stride + c`
///
/// Expressed in faer's column-major layout via the mathematical transpose identity:
/// `C = A * B^T <=> C^T = B * A^T`
/// Zero copies: simply pass B's row stride as faer's row stride and 1 as column stride!
///
/// # Safety
/// - `a`, `b`, and `c` must point to valid memory for the specified dimensions and strides.
/// - The memory region of `c` must not overlap with `a` or `b`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_row_major_b_transposed(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    a_stride: usize,
    b: *const f32,
    b_stride: usize,
    c: *mut f32,
    c_stride: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    unsafe {
        let a_t = MatRef::from_raw_parts(a, k, m, 1isize, a_stride as isize);
        let b_mat = MatRef::from_raw_parts(b, n, k, b_stride as isize, 1isize);
        let mut c_t = MatMut::from_raw_parts_mut(c, n, m, 1isize, c_stride as isize);
        faer_matmul(&mut c_t, Accum::Replace, &b_mat, &a_t, 1.0, Par::Seq);
    }
}
