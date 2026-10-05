use crate::error::{OcrError, Result};
use crate::tensor::{Tensor, compute_contiguous_strides};
use rayon::prelude::*;

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    matmul_with_bias(a, b, None)
}

pub fn matmul_with_bias(a: &Tensor, b: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    if a_ndim < 1 || b_ndim < 1 {
        return Err(OcrError::OperatorError {
            op: "MatMul".into(),
            msg: "Operands must have at least 1 dimension".into(),
        });
    }

    let a_cont_holder;
    let a_slice = if a.is_contiguous() {
        a.as_slice()
    } else {
        a_cont_holder = a.to_contiguous();
        a_cont_holder.as_slice()
    };
    let b_cont_holder;
    let b_slice = if b.is_contiguous() {
        b.as_slice()
    } else {
        b_cont_holder = b.to_contiguous();
        b_cont_holder.as_slice()
    };

    let bias_holder = bias.and_then(|b| {
        if b.is_contiguous() {
            None
        } else {
            Some(b.to_contiguous())
        }
    });
    let bias_slice: Option<&[f32]> = bias.map(|b| {
        if let Some(ref holder) = bias_holder {
            holder.as_slice()
        } else {
            b.as_slice()
        }
    });

    // Standard 2D x 2D
    if a_ndim == 2 && b_ndim == 2 {
        let m = a.shape()[0];
        let k1 = a.shape()[1];
        let k2 = b.shape()[0];
        let n = b.shape()[1];

        if k1 != k2 {
            return Err(OcrError::OperatorError {
                op: "MatMul".into(),
                msg: format!(
                    "Incompatible matrix dimensions: [{}, {}] x [{}, {}]",
                    m, k1, k2, n
                ),
            });
        }

        let mut output = Tensor::empty(&[m, n]);
        let out_slice = output.as_mut_slice();

        unsafe {
            parallel_sgemm_2d(
                m,
                k1,
                n,
                a_slice.as_ptr(),
                b_slice.as_ptr(),
                out_slice.as_mut_ptr(),
            );
        }

        if let Some(b_data) = bias_slice {
            if b_data.len() == n {
                for row in out_slice.chunks_mut(n) {
                    add_bias_row_simd(row, b_data);
                }
            } else if b_data.len() == out_slice.len() {
                add_bias_row_simd(out_slice, b_data);
            }
        }

        return Ok(output);
    }

    // High dimensional batched MatMul
    // A: [..., M, K], B: [..., K, N]
    let m = if a_ndim == 1 {
        1
    } else {
        a.shape()[a_ndim - 2]
    };
    let k_a = a.shape()[a_ndim - 1];
    let k_b = if b_ndim == 1 {
        b.shape()[0]
    } else {
        b.shape()[b_ndim - 2]
    };
    let n = if b_ndim == 1 {
        1
    } else {
        b.shape()[b_ndim - 1]
    };

    if k_a != k_b {
        return Err(OcrError::OperatorError {
            op: "MatMul".into(),
            msg: format!("Inner dimensions must match: {} != {}", k_a, k_b),
        });
    }

    // Batch shapes
    let a_batch_shape = if a_ndim > 2 {
        &a.shape()[..a_ndim - 2]
    } else {
        &[]
    };
    let b_batch_shape = if b_ndim > 2 {
        &b.shape()[..b_ndim - 2]
    } else {
        &[]
    };

    let out_batch_shape = crate::tensor::broadcast_shapes(a_batch_shape, b_batch_shape)?;
    let mut out_shape = out_batch_shape.clone();
    out_shape.push(m);
    out_shape.push(n);

    let mut output = Tensor::empty(&out_shape);

    let batch_count: usize = out_batch_shape.iter().product();
    let batch_count = if batch_count == 0 { 1 } else { batch_count };

    let a_mat_size = m * k_a;
    let b_mat_size = k_b * n;
    let out_mat_size = m * n;

    let out_slice = output.as_mut_slice();

    if batch_count == 1 {
        unsafe {
            parallel_sgemm_2d(
                m,
                k_a,
                n,
                a_slice.as_ptr(),
                b_slice.as_ptr(),
                out_slice.as_mut_ptr(),
            );
        }

        if let Some(b_data) = bias_slice {
            if b_data.len() == n {
                for row in out_slice.chunks_mut(n) {
                    add_bias_row_simd(row, b_data);
                }
            } else if b_data.len() == out_slice.len() {
                add_bias_row_simd(out_slice, b_data);
            }
        }

        return Ok(output);
    }

    let a_batch_strides = crate::tensor::compute_broadcast_strides(a_batch_shape, &out_batch_shape);
    let b_batch_strides = crate::tensor::compute_broadcast_strides(b_batch_shape, &out_batch_shape);
    let out_batch_strides = compute_contiguous_strides(&out_batch_shape);

    let run_batch = |(b_idx, out_chunk): (usize, &mut [f32])| {
        let mut rem = b_idx;
        let mut a_offset = 0;
        let mut b_offset = 0;

        for d in 0..out_batch_shape.len() {
            let coord = rem / out_batch_strides[d];
            rem %= out_batch_strides[d];
            a_offset += coord * a_batch_strides[d];
            b_offset += coord * b_batch_strides[d];
        }

        let a_ptr = unsafe { a_slice.as_ptr().add(a_offset * a_mat_size) };
        let b_ptr = unsafe { b_slice.as_ptr().add(b_offset * b_mat_size) };
        let out_ptr = out_chunk.as_mut_ptr();

        unsafe {
            crate::ops::gemm_row_major(
                m,
                k_a,
                n,
                a_ptr,
                k_a,
                b_ptr,
                n,
                out_ptr,
                n,
            );
        }
    };

    if crate::tensor::should_parallelize(batch_count, 2) {
        out_slice
            .par_chunks_mut(out_mat_size)
            .enumerate()
            .for_each(run_batch);
    } else {
        out_slice
            .chunks_mut(out_mat_size)
            .enumerate()
            .for_each(run_batch);
    }

    if let Some(b_data) = bias_slice {
        if b_data.len() == n {
            for row in out_slice.chunks_mut(n) {
                add_bias_row_simd(row, b_data);
            }
        } else if b_data.len() == out_slice.len() {
            add_bias_row_simd(out_slice, b_data);
        }
    }

    Ok(output)
}

pub fn matmul_with_transposed_b(
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    if a_ndim < 1 || b_ndim < 1 {
        return Err(OcrError::OperatorError {
            op: "MatMul".into(),
            msg: "Operands must have at least 1 dimension".into(),
        });
    }

    if a_ndim == 2 && b_ndim == 2 {
        return matmul_transposed_b(a, b, bias);
    }

    let a_cont_holder;
    let a_slice = if a.is_contiguous() {
        a.as_slice()
    } else {
        a_cont_holder = a.to_contiguous();
        a_cont_holder.as_slice()
    };
    let b_cont_holder;
    let b_slice = if b.is_contiguous() {
        b.as_slice()
    } else {
        b_cont_holder = b.to_contiguous();
        b_cont_holder.as_slice()
    };

    let bias_holder = bias.and_then(|b| {
        if b.is_contiguous() {
            None
        } else {
            Some(b.to_contiguous())
        }
    });
    let bias_slice: Option<&[f32]> = bias.map(|b| {
        if let Some(ref holder) = bias_holder {
            holder.as_slice()
        } else {
            b.as_slice()
        }
    });

    // High dimensional batched MatMul with B transposed
    // A: [..., M, K], B: [..., N, K] -> C: [..., M, N]
    let m = if a_ndim == 1 {
        1
    } else {
        a.shape()[a_ndim - 2]
    };
    let k_a = a.shape()[a_ndim - 1];
    let k_b = b.shape()[b_ndim - 1];
    let n = if b_ndim == 1 {
        1
    } else {
        b.shape()[b_ndim - 2]
    };

    if k_a != k_b {
        return Err(OcrError::OperatorError {
            op: "MatMul".into(),
            msg: format!("Inner dimensions must match: {} != {}", k_a, k_b),
        });
    }

    // Batch shapes
    let a_batch_shape = if a_ndim > 2 {
        &a.shape()[..a_ndim - 2]
    } else {
        &[]
    };
    let b_batch_shape = if b_ndim > 2 {
        &b.shape()[..b_ndim - 2]
    } else {
        &[]
    };

    let out_batch_shape = crate::tensor::broadcast_shapes(a_batch_shape, b_batch_shape)?;
    let mut out_shape = out_batch_shape.clone();
    out_shape.push(m);
    out_shape.push(n);

    let mut output = Tensor::empty(&out_shape);
    let out_slice = output.as_mut_slice();

    let batch_count: usize = out_batch_shape.iter().product();
    let batch_count = if batch_count == 0 { 1 } else { batch_count };

    let a_mat_size = m * k_a;
    let b_mat_size = n * k_b;
    let out_mat_size = m * n;

    let a_batch_strides = crate::tensor::compute_broadcast_strides(a_batch_shape, &out_batch_shape);
    let b_batch_strides = crate::tensor::compute_broadcast_strides(b_batch_shape, &out_batch_shape);
    let out_batch_strides = compute_contiguous_strides(&out_batch_shape);

    let run_batch = |(b_idx, out_chunk): (usize, &mut [f32])| {
        let mut rem = b_idx;
        let mut a_offset = 0;
        let mut b_offset = 0;

        for d in 0..out_batch_shape.len() {
            let coord = rem / out_batch_strides[d];
            rem %= out_batch_strides[d];
            a_offset += coord * a_batch_strides[d];
            b_offset += coord * b_batch_strides[d];
        }

        let a_ptr = unsafe { a_slice.as_ptr().add(a_offset * a_mat_size) };
        let b_ptr = unsafe { b_slice.as_ptr().add(b_offset * b_mat_size) };
        let out_ptr = out_chunk.as_mut_ptr();

        unsafe {
            crate::ops::gemm_row_major_b_transposed(
                m,
                k_a,
                n,
                a_ptr,
                k_a,
                b_ptr,
                k_a,
                out_ptr,
                n,
            );
        }
    };

    if crate::tensor::should_parallelize(batch_count, 2) {
        out_slice
            .par_chunks_mut(out_mat_size)
            .enumerate()
            .for_each(run_batch);
    } else {
        out_slice
            .chunks_mut(out_mat_size)
            .enumerate()
            .for_each(run_batch);
    }

    if let Some(b_data) = bias_slice {
        if b_data.len() == n {
            for row in out_slice.chunks_mut(n) {
                add_bias_row_simd(row, b_data);
            }
        } else if b_data.len() == out_slice.len() {
            add_bias_row_simd(out_slice, b_data);
        }
    }

    Ok(output)
}

#[inline]
fn add_bias_row_simd(row: &mut [f32], bias: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() {
        unsafe { add_bias_row_fma(row, bias) };
        return;
    }
    let n = row.len().min(bias.len());
    for i in 0..n {
        row[i] += bias[i];
    }
}

/// # Safety
/// The caller must have verified [`crate::ops::avx2_fma_enabled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_bias_row_fma(row: &mut [f32], bias: &[f32]) {
    let n = row.len().min(bias.len());
    let mut i = 0;
    unsafe {
        use std::arch::x86_64::*;
        let r_ptr = row.as_mut_ptr();
        let b_ptr = bias.as_ptr();
        while i + 8 <= n {
            let vr = _mm256_loadu_ps(r_ptr.add(i));
            let vb = _mm256_loadu_ps(b_ptr.add(i));
            _mm256_storeu_ps(r_ptr.add(i), _mm256_add_ps(vr, vb));
            i += 8;
        }
    }
    while i < n {
        row[i] += bias[i];
        i += 1;
    }
}

unsafe fn parallel_sgemm_2d(
    m: usize,
    k: usize,
    n: usize,
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
) {
    let num_threads = rayon::current_num_threads();
    let flops = 2 * m * k * n;

    if n < 64 || !crate::tensor::should_parallelize(flops, 150_000) {
        unsafe {
            crate::ops::gemm_row_major(m, k, n, a_ptr, k, b_ptr, n, out_ptr, n);
        }
        return;
    }

    // When N is very large (e.g. vocabulary projection where N=6625 or 18710 >> M=43),
    // splitting along N is vastly superior: each thread packs only its own slice of B (~200KB in L2)
    // instead of all threads redundantly packing the entire 9MB matrix B.
    if ((n >= 1024 && n > 4 * m) || m < 8) && n >= 128 {
        let chunk_n = n.div_ceil(num_threads).max(64).min(n);
        let num_chunks = n.div_ceil(chunk_n);
        let out_ptr_usize = out_ptr as usize;
        let a_ptr_usize = a_ptr as usize;
        let b_ptr_usize = b_ptr as usize;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let col_start = chunk_idx * chunk_n;
            let actual_n = (n - col_start).min(chunk_n);
            unsafe {
                let a_p = a_ptr_usize as *const f32;
                let b_p = (b_ptr_usize as *const f32).add(col_start);
                let out_p = (out_ptr_usize as *mut f32).add(col_start);

                crate::ops::gemm_row_major(m, k, actual_n, a_p, k, b_p, n, out_p, n);
            }
        });
        return;
    }

    if m >= 8 {
        let chunk_size = m.div_ceil(num_threads).max(4).min(m);
        let num_chunks = m.div_ceil(chunk_size);
        let out_ptr_usize = out_ptr as usize;
        let a_ptr_usize = a_ptr as usize;
        let b_ptr_usize = b_ptr as usize;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let m_start = chunk_idx * chunk_size;
            let actual_m = (m - m_start).min(chunk_size);

            unsafe {
                let a_p = (a_ptr_usize as *const f32).add(m_start * k);
                let b_p = b_ptr_usize as *const f32;
                let out_p = (out_ptr_usize as *mut f32).add(m_start * n);

                crate::ops::gemm_row_major(actual_m, k, n, a_p, k, b_p, n, out_p, n);
            }
        });
    } else if n >= 128 {
        let chunk_n = n.div_ceil(num_threads).max(64).min(n);
        let num_chunks = n.div_ceil(chunk_n);
        let out_ptr_usize = out_ptr as usize;
        let a_ptr_usize = a_ptr as usize;
        let b_ptr_usize = b_ptr as usize;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let col_start = chunk_idx * chunk_n;
            let actual_n = (n - col_start).min(chunk_n);
            unsafe {
                let a_p = a_ptr_usize as *const f32;
                let b_p = (b_ptr_usize as *const f32).add(col_start);
                let out_p = (out_ptr_usize as *mut f32).add(col_start);

                crate::ops::gemm_row_major(m, k, actual_n, a_p, k, b_p, n, out_p, n);
            }
        });
    } else {
        unsafe {
            crate::ops::gemm_row_major(m, k, n, a_ptr, k, b_ptr, n, out_ptr, n);
        }
    }
}

pub fn matmul_transposed_b(a: &Tensor, b_t: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let a_ndim = a.ndim();
    let b_ndim = b_t.ndim();

    if a_ndim < 1 || b_ndim != 2 {
        return Err(OcrError::OperatorError {
            op: "MatMulTransposedB".into(),
            msg: "Operands must have valid dimensions".into(),
        });
    }

    let n = b_t.shape()[0];
    let k_b = b_t.shape()[1];
    let k_a = a.shape()[a_ndim - 1];
    let m = if a_ndim == 1 {
        1
    } else {
        a.shape()[a_ndim - 2]
    };

    if k_a != k_b {
        return Err(OcrError::OperatorError {
            op: "MatMulTransposedB".into(),
            msg: format!("Inner dimensions must match: {} != {}", k_a, k_b),
        });
    }

    let mut out_shape = if a_ndim > 2 {
        a.shape()[..a_ndim - 2].to_vec()
    } else {
        Vec::new()
    };
    out_shape.push(m);
    out_shape.push(n);

    let mut output = Tensor::empty(&out_shape);

    let a_cont_holder;
    let a_slice = if a.is_contiguous() {
        a.as_slice()
    } else {
        a_cont_holder = a.to_contiguous();
        a_cont_holder.as_slice()
    };

    // `as_slice()` returns the raw storage buffer, so a non-contiguous `b_t` would feed the
    // dot product the wrong elements with no error. Every other operand in the crate is put
    // through this same dance.
    let b_holder;
    let b_slice = if b_t.is_contiguous() {
        b_t.as_slice()
    } else {
        b_holder = b_t.to_contiguous();
        b_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let bias_holder = bias.and_then(|b| {
        if b.is_contiguous() {
            None
        } else {
            Some(b.to_contiguous())
        }
    });
    let bias_slice: Option<&[f32]> = bias.map(|b| {
        if let Some(ref holder) = bias_holder {
            holder.as_slice()
        } else {
            b.as_slice()
        }
    });

    // The kernels below index the bias as `bias[j]` for `j` in `0..n`, through
    // `get_unchecked` in the AVX2 path, and a length mismatch was never checked.
    if let Some(bs) = bias_slice
        && bs.len() != n
    {
        return Err(OcrError::OperatorError {
            op: "MatMulTransposedB".into(),
            msg: format!("bias has {} elements but n is {n}", bs.len()),
        });
    }

    let k = k_a;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = crate::ops::avx2_fma_enabled();

    // Respect the scoped "inner parallelism disabled" flag set by ScopedDisableParallel.
    // Without this guard, a recognition worker thread running one text line would fan
    // its inner MatMul back out onto the global Rayon pool, re-introducing the nested
    // task storm this guard exists to prevent.
    let run_row = |(r_idx, out_row): (usize, &mut [f32])| {
        let a_row = &a_slice[r_idx * k..(r_idx + 1) * k];

        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            unsafe {
                compute_transposed_b_cols_avx2(a_row, b_slice, bias_slice, out_row, 0, n, k);
            }
            return;
        }
        compute_transposed_b_cols_scalar(a_row, b_slice, bias_slice, out_row, 0, n, k);
    };

    // Register-blocked path.
    //
    // The tile accumulates `TILE_ROWS` output rows at once so each weight vector is loaded
    // once per `TILE_ROWS` rows rather than once per row (see `transposed_b_tile_fma` for
    // the load-per-FMA ratio). It is stepped over the *whole* row range of a column chunk,
    // so the thread split is still by column: with `m = 37`, row-splitting would hand only
    // 37/4 = 9 chunks to 16 threads, whereas `n = 18710` columns divide into hundreds.
    //
    // The tile steps `k` by 8 with no scalar remainder, so it is only valid when
    // `k % 8 == 0` (120 for the recognition head). Anything else keeps the one-row path,
    // which handles the remainder correctly.
    #[cfg(target_arch = "x86_64")]
    let rows_blocked = has_avx2 && k % 8 == 0;
    #[cfg(not(target_arch = "x86_64"))]
    let rows_blocked = false;

    let total_rows = out_slice.len() / n;

    // `row_offset` is the global row where this block starts in `a`; `block` is the
    // matching slice of the *output*, whose local row index is relative to that.
    let run_block = |row_offset: usize, block: &mut [f32]| {
        let rows = block.len() / n;
        let cols_full = n - n % 4;
        let rows_full = rows - rows % TILE_ROWS;
        let out_ptr = block.as_mut_ptr();
        let a_ptr = a_slice.as_ptr();
        let b_ptr = b_slice.as_ptr();

        #[cfg(target_arch = "x86_64")]
        for c in (0..cols_full).step_by(4) {
            for r in (0..rows_full).step_by(TILE_ROWS) {
                // SAFETY: the tile reads `TILE_ROWS * k` floats from `a_ptr + r*k` (in
                // bounds because r + TILE_ROWS <= rows_full <= rows, and row_offset + rows
                // <= total_rows), `4 * k` from `b_ptr + c*k` (in bounds because
                // c + 4 <= cols_full <= n), and writes only the (TILE_ROWS, 4) block at
                // `out_ptr + r*n + c`. Successive iterations take disjoint row/column
                // ranges and `par_chunks_mut` hands out disjoint blocks, so tiles never
                // overlap. `k % 8 == 0` is guaranteed by the `rows_blocked` guard.
                unsafe {
                    transposed_b_tile_fma::<TILE_ROWS>(
                        a_ptr.add((row_offset + r) * k),
                        b_ptr.add(c * k),
                        out_ptr.add(r * n + c),
                        n,
                        c,
                        bias_slice,
                        k,
                    );
                }
            }
            // Ragged last row group, if any.
            if rows > rows_full {
                // SAFETY: same as the full tiles, with a smaller `MR` and
                // `rows - rows_full` in 1..TILE_ROWS. The match covers every such value.
                unsafe {
                    match rows - rows_full {
                        1 => transposed_b_tile_fma::<1>(
                            a_ptr.add((row_offset + rows_full) * k),
                            b_ptr.add(c * k),
                            out_ptr.add(rows_full * n + c),
                            n,
                            c,
                            bias_slice,
                            k,
                        ),
                        2 => transposed_b_tile_fma::<2>(
                            a_ptr.add((row_offset + rows_full) * k),
                            b_ptr.add(c * k),
                            out_ptr.add(rows_full * n + c),
                            n,
                            c,
                            bias_slice,
                            k,
                        ),
                        3 => transposed_b_tile_fma::<3>(
                            a_ptr.add((row_offset + rows_full) * k),
                            b_ptr.add(c * k),
                            out_ptr.add(rows_full * n + c),
                            n,
                            c,
                            bias_slice,
                            k,
                        ),
                        _ => {}
                    }
                }
            }
        }

        // Tail columns of the last chunk (n % 4). At most 3 columns, so the one-row kernel
        // handles them fine.
        if cols_full < n {
            for r in 0..rows {
                // SAFETY: `cols_full < n`, so `r*n + cols_full` and the `n - cols_full`
                // elements after it are within this row of the block.
                let out_tail = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.add(r * n + cols_full), n - cols_full)
                };
                let a_row = &a_slice[(row_offset + r) * k..(row_offset + r + 1) * k];
                #[cfg(target_arch = "x86_64")]
                if has_avx2 {
                    unsafe {
                        compute_transposed_b_cols_avx2(
                            a_row,
                            b_slice,
                            bias_slice,
                            out_tail,
                            cols_full,
                            n - cols_full,
                            k,
                        );
                    }
                } else {
                    compute_transposed_b_cols_scalar(
                        a_row,
                        b_slice,
                        bias_slice,
                        out_tail,
                        cols_full,
                        n - cols_full,
                        k,
                    );
                }
            }
        }
    };

    // Fused single-row path: split the *output columns* across threads.
    //
    // The guard must be on the flattened output length, not on `m`. `m` is the *inner* row
    // count `a.shape()[a_ndim - 2]`, so for a `[B, 1, K]` input `m == 1` while the output
    // has `B * n` elements. The old guard chunked the whole flattened output by column and
    // fed it `a_slice` — the entire tensor — as if it were row 0. Consequences for B > 1:
    // the destination for batch b should be `out[b * n + c]` but the code wrote `out[c]`, so
    // the tail was never written at all (and `Tensor::empty` is genuinely uninitialised in
    // release), while `j` ran up to `B * n - 1` and `b_ptr.add(j * k)` read past the end of
    // the weight matrix. The scalar fallback panicked on the same input instead, so the
    // behaviour also depended on whether AVX2 was available.
    //
    // `src/ops/matmul.rs::tests` pins this with a `[3, 1, K] @ [K, N]` case under both paths.
    if out_slice.len() == n && crate::tensor::should_parallelize(n, 512) {
        let num_threads = rayon::current_num_threads();
        // Floor of 1, not 256: with n = 512 on 16 threads a 256-column floor yields two
        // chunks and idles 14 cores. `par_chunks_mut` is an indexed iterator, so rayon's
        // work stealing splits it further anyway.
        let col_chunk_size = (n / num_threads).clamp(1, 2048);
        out_slice
            .par_chunks_mut(col_chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * col_chunk_size;
                let col_count = out_chunk.len();
                let a_row = &a_slice[..k];
                #[cfg(target_arch = "x86_64")]
                if has_avx2 {
                    unsafe {
                        compute_transposed_b_cols_avx2(
                            a_row, b_slice, bias_slice, out_chunk, col_start, col_count, k,
                        );
                    }
                    return;
                }
                compute_transposed_b_cols_scalar(
                    a_row, b_slice, bias_slice, out_chunk, col_start, col_count, k,
                );
            });
    } else if rows_blocked && total_rows > 0 {
        if n >= 256 && crate::tensor::should_parallelize(total_rows * n, 512) {
            let num_threads = rayon::current_num_threads();
            let target_chunk = (n / (num_threads * 4)).clamp(64, 256);
            let col_chunk_size = ((target_chunk / 16) * 16).max(16);
            let num_chunks = n.div_ceil(col_chunk_size);
            let out_ptr = out_slice.as_mut_ptr() as usize;
            let a_ptr = a_slice.as_ptr() as usize;
            let b_ptr = b_slice.as_ptr() as usize;
            let rows_full = total_rows - total_rows % TILE_ROWS;

            (0..num_chunks).into_par_iter().for_each(|c_idx| {
                let col_start = c_idx * col_chunk_size;
                let col_end = (col_start + col_chunk_size).min(n);
                let cols = col_end - col_start;
                let cols_full = col_start + (cols / 4) * 4;
                let a_p = a_ptr as *const f32;
                let b_p = b_ptr as *const f32;
                let out_p = out_ptr as *mut f32;

                unsafe {
                    for c in (col_start..cols_full).step_by(4) {
                        for r in (0..rows_full).step_by(TILE_ROWS) {
                            transposed_b_tile_fma::<TILE_ROWS>(
                                a_p.add(r * k),
                                b_p.add(c * k),
                                out_p.add(r * n + c),
                                n,
                                c,
                                bias_slice,
                                k,
                            );
                        }
                        if total_rows > rows_full {
                            match total_rows - rows_full {
                                1 => transposed_b_tile_fma::<1>(
                                    a_p.add(rows_full * k),
                                    b_p.add(c * k),
                                    out_p.add(rows_full * n + c),
                                    n,
                                    c,
                                    bias_slice,
                                    k,
                                ),
                                2 => transposed_b_tile_fma::<2>(
                                    a_p.add(rows_full * k),
                                    b_p.add(c * k),
                                    out_p.add(rows_full * n + c),
                                    n,
                                    c,
                                    bias_slice,
                                    k,
                                ),
                                3 => transposed_b_tile_fma::<3>(
                                    a_p.add(rows_full * k),
                                    b_p.add(c * k),
                                    out_p.add(rows_full * n + c),
                                    n,
                                    c,
                                    bias_slice,
                                    k,
                                ),
                                _ => {}
                            }
                        }
                    }

                    if cols_full < col_end {
                        for r in 0..total_rows {
                            let out_tail = std::slice::from_raw_parts_mut(
                                out_p.add(r * n + cols_full),
                                col_end - cols_full,
                            );
                            let a_row = std::slice::from_raw_parts(a_p.add(r * k), k);
                            compute_transposed_b_cols_avx2(
                                a_row,
                                b_slice,
                                bias_slice,
                                out_tail,
                                cols_full,
                                col_end - cols_full,
                                k,
                            );
                        }
                    }
                }
            });
        } else {
            // Split into groups of `TILE_ROWS` consecutive rows so the tile has whole blocks to
            // work with. `par_chunks_mut` hands out disjoint `&mut` slices, so unlike a column
            // split this stays within the borrow checker. The last group may be short; the tile
            // dispatch inside `run_block` covers that.
            let rows_per_chunk = TILE_ROWS.min(total_rows);
            let chunk = rows_per_chunk * n;
            if crate::tensor::should_parallelize(total_rows * n, 512) {
                out_slice
                    .par_chunks_mut(chunk)
                    .enumerate()
                    .for_each(|(i, blk)| run_block(i * rows_per_chunk, blk));
            } else {
                for (i, blk) in out_slice.chunks_mut(chunk).enumerate() {
                    run_block(i * rows_per_chunk, blk);
                }
            }
        }
    } else if crate::tensor::should_parallelize(m * n, 512) {
        out_slice.par_chunks_mut(n).enumerate().for_each(run_row);
    } else {
        out_slice.chunks_mut(n).enumerate().for_each(run_row);
    }

    Ok(output)
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
unsafe fn compute_transposed_b_cols_avx2(
    a_row: &[f32],
    b_slice: &[f32],
    bias_slice: Option<&[f32]>,
    out_chunk: &mut [f32],
    col_start: usize,
    col_count: usize,
    k: usize,
) {
    unsafe {
        use std::arch::x86_64::*;
        let a_ptr = a_row.as_ptr();
        let b_ptr = b_slice.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();

        let mut c = 0;
        while c + 4 <= col_count {
            let j0 = col_start + c;
            let j1 = j0 + 1;
            let j2 = j0 + 2;
            let j3 = j0 + 3;

            let b0 = b_ptr.add(j0 * k);
            let b1 = b_ptr.add(j1 * k);
            let b2 = b_ptr.add(j2 * k);
            let b3 = b_ptr.add(j3 * k);

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut i = 0;
            while i + 8 <= k {
                let va = _mm256_loadu_ps(a_ptr.add(i));
                acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1.add(i)), acc1);
                acc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2.add(i)), acc2);
                acc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3.add(i)), acc3);
                i += 8;
            }

            let mut sum0 = hsum256(acc0);
            let mut sum1 = hsum256(acc1);
            let mut sum2 = hsum256(acc2);
            let mut sum3 = hsum256(acc3);

            while i < k {
                let a_val = *a_ptr.add(i);
                sum0 += a_val * *b0.add(i);
                sum1 += a_val * *b1.add(i);
                sum2 += a_val * *b2.add(i);
                sum3 += a_val * *b3.add(i);
                i += 1;
            }

            if let Some(bias) = bias_slice {
                *out_ptr.add(c) = sum0 + *bias.get_unchecked(j0);
                *out_ptr.add(c + 1) = sum1 + *bias.get_unchecked(j1);
                *out_ptr.add(c + 2) = sum2 + *bias.get_unchecked(j2);
                *out_ptr.add(c + 3) = sum3 + *bias.get_unchecked(j3);
            } else {
                *out_ptr.add(c) = sum0;
                *out_ptr.add(c + 1) = sum1;
                *out_ptr.add(c + 2) = sum2;
                *out_ptr.add(c + 3) = sum3;
            }

            c += 4;
        }

        while c < col_count {
            let j = col_start + c;
            let b_col = b_ptr.add(j * k);
            let dot = avx2_dot_product(a_ptr, b_col, k);
            let val = if let Some(bias) = bias_slice {
                dot + *bias.get_unchecked(j)
            } else {
                dot
            };
            *out_ptr.add(c) = val;
            c += 1;
        }
    }
}

/// Register-blocked output tile: `MR` rows x 4 columns, accumulated over `k`.
///
/// The weight is `n x k` row-major, so all `MR` output rows read the *same* `k` floats of a
/// given column. Accumulating several rows per pass therefore loads each weight vector once
/// instead of once per row, which is the whole point:
///
/// | tile        | loads per 8 values of k | FMAs | FMA per load |
/// |-------------|-------------------------|------|--------------|
/// | 1 row x 4col | 5 (1 `a`, 4 `b`)        | 4    | 0.80         |
/// | 4 rows x 4col| 8 (4 `a`, 4 `b`)        | 16   | 2.00         |
///
/// The old kernel was the first row of that table, which left the load ports as the
/// bottleneck. It also had only 4 independent accumulator chains, and covering a 4-cycle FMA
/// latency across 2 FMA ports needs about 8, so the FMA ports were idle as well.
///
/// `MR` is a const generic so the row loop unrolls into registers; the caller dispatches on
/// `m % MR` to cover a ragged last group.
///
/// # Safety
/// Requires AVX2+FMA (see [`crate::ops::avx2_fma_enabled`]). `a_ptr` must be valid for
/// `MR * k` floats, `b_ptr` for `4 * k`, and `out_ptr` for `MR * row_stride_out` with
/// `out_ptr[r * row_stride_out + c]` in bounds. The tile must not overlap another tile's
/// output. `k` must be a multiple of 8 (the caller handles the remainder).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn transposed_b_tile_fma<const MR: usize>(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    row_stride_out: usize,
    col_start: usize,
    bias: Option<&[f32]>,
    k: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        let b0 = b_ptr;
        let b1 = b_ptr.add(k);
        let b2 = b_ptr.add(2 * k);
        let b3 = b_ptr.add(3 * k);

        // Always allocate the maximum (TILE_ROWS * 4 = 16) accumulators. The loops below
        // are bounded by `MR`, so a smaller `MR` leaves the tail untouched and the optimiser
        // drops it. A `[__m256; MR * 4]` would express this more directly, but a const
        // generic cannot appear in an array length on this toolchain.
        let mut acc = [_mm256_setzero_ps(); TILE_ROWS * 4];

        let mut i = 0;
        while i < k {
            let bv0 = _mm256_loadu_ps(b0.add(i));
            let bv1 = _mm256_loadu_ps(b1.add(i));
            let bv2 = _mm256_loadu_ps(b2.add(i));
            let bv3 = _mm256_loadu_ps(b3.add(i));
            for r in 0..MR {
                let av = _mm256_loadu_ps(a_ptr.add(r * k + i));
                acc[r * 4] = _mm256_fmadd_ps(av, bv0, acc[r * 4]);
                acc[r * 4 + 1] = _mm256_fmadd_ps(av, bv1, acc[r * 4 + 1]);
                acc[r * 4 + 2] = _mm256_fmadd_ps(av, bv2, acc[r * 4 + 2]);
                acc[r * 4 + 3] = _mm256_fmadd_ps(av, bv3, acc[r * 4 + 3]);
            }
            i += 8;
        }

        let bias_vec = bias.map(|b| _mm_loadu_ps(b.as_ptr().add(col_start)));

        for r in 0..MR {
            let a0 = acc[r * 4];
            let a1 = acc[r * 4 + 1];
            let a2 = acc[r * 4 + 2];
            let a3 = acc[r * 4 + 3];

            let h01 = _mm256_hadd_ps(a0, a1);
            let h23 = _mm256_hadd_ps(a2, a3);
            let h0123 = _mm256_hadd_ps(h01, h23);

            let lo = _mm256_castps256_ps128(h0123);
            let hi = _mm256_extractf128_ps(h0123, 1);
            let mut sums = _mm_add_ps(lo, hi);

            if let Some(bv) = bias_vec {
                sums = _mm_add_ps(sums, bv);
            }

            _mm_storeu_ps(out_ptr.add(r * row_stride_out), sums);
        }
    }
}

/// Number of output rows accumulated per pass in the blocked kernel.
///
/// 3 is optimal on AVX2: MR * 4 = 12 accumulators leaves 4 vector registers free
/// for loading weights and inputs (16 vector registers total on x86_64 AVX2).
/// MR = 4 (16 accumulators) starved registers and caused heavy stack spilling.
const TILE_ROWS: usize = 3;

#[inline(always)]
fn compute_transposed_b_cols_scalar(
    a_row: &[f32],
    b_slice: &[f32],
    bias_slice: Option<&[f32]>,
    out_chunk: &mut [f32],
    col_start: usize,
    col_count: usize,
    k: usize,
) {
    for (c, out_val) in out_chunk.iter_mut().take(col_count).enumerate() {
        let j = col_start + c;
        let b_col = &b_slice[j * k..(j + 1) * k];
        let dot: f32 = a_row.iter().zip(b_col.iter()).map(|(&x, &y)| x * y).sum();
        *out_val = if let Some(bias) = bias_slice {
            dot + bias[j]
        } else {
            dot
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot_product(a: *const f32, b: *const f32, n: usize) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut i = 0;
        let mut v0 = _mm256_setzero_ps();
        let mut v1 = _mm256_setzero_ps();
        let mut v2 = _mm256_setzero_ps();
        let mut v3 = _mm256_setzero_ps();

        while i + 32 <= n {
            v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), v0);
            v1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 8)),
                _mm256_loadu_ps(b.add(i + 8)),
                v1,
            );
            v2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 16)),
                _mm256_loadu_ps(b.add(i + 16)),
                v2,
            );
            v3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 24)),
                _mm256_loadu_ps(b.add(i + 24)),
                v3,
            );
            i += 32;
        }
        v0 = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));

        while i + 8 <= n {
            v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), v0);
            i += 8;
        }

        let mut sum = hsum256(v0);

        while i < n {
            sum += *a.add(i) * *b.add(i);
            i += 1;
        }
        sum
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedWeight {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl QuantizedWeight {
    pub fn from_f32_transposed(w_t: &Tensor) -> Self {
        let n = w_t.shape()[0];
        let k = w_t.shape()[1];
        let slice = w_t.as_slice();

        let mut data = vec![0i8; n * k];
        let mut scales = vec![0.0f32; n];

        for j in 0..n {
            let row = &slice[j * k..(j + 1) * k];
            let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales[j] = scale;

            for i in 0..k {
                let q = (row[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                data[j * k + i] = q;
            }
        }

        Self {
            data,
            scales,
            rows: n,
            cols: k,
        }
    }
}

pub fn matmul_quantized_b(
    a: &Tensor,
    qw: &QuantizedWeight,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let a_ndim = a.ndim();

    if a_ndim < 1 {
        return Err(OcrError::OperatorError {
            op: "MatMulQuantizedB".into(),
            msg: "Operand must have valid dimensions".into(),
        });
    }

    let n = qw.rows;
    let k = qw.cols;
    let k_a = a.shape()[a_ndim - 1];
    let m = if a_ndim == 1 {
        1
    } else {
        a.shape()[a_ndim - 2]
    };

    if k_a != k {
        return Err(OcrError::OperatorError {
            op: "MatMulQuantizedB".into(),
            msg: format!("Inner dimensions must match: {} != {}", k_a, k),
        });
    }

    let mut out_shape = if a_ndim > 2 {
        a.shape()[..a_ndim - 2].to_vec()
    } else {
        Vec::new()
    };
    out_shape.push(m);
    out_shape.push(n);

    let mut output = Tensor::empty(&out_shape);

    let a_cont_holder;
    let a_slice = if a.is_contiguous() {
        a.as_slice()
    } else {
        a_cont_holder = a.to_contiguous();
        a_cont_holder.as_slice()
    };

    let out_slice = output.as_mut_slice();

    let bias_holder = bias.and_then(|b| {
        if b.is_contiguous() {
            None
        } else {
            Some(b.to_contiguous())
        }
    });
    let bias_slice: Option<&[f32]> = bias.map(|b| {
        if let Some(ref holder) = bias_holder {
            holder.as_slice()
        } else {
            b.as_slice()
        }
    });

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = crate::ops::avx2_fma_enabled();

    // Same scoped guard as matmul_transposed_b: keep the INT8 path from re-fanning
    // onto the global Rayon pool when called from a recognition worker thread.
    let run_row = |(r_idx, out_row): (usize, &mut [f32])| {
        let a_row = &a_slice[r_idx * k..(r_idx + 1) * k];

        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            unsafe {
                compute_quantized_b_cols_avx2(a_row, qw, bias_slice, out_row, 0, n, k);
            }
            return;
        }

        compute_quantized_b_cols_scalar(a_row, qw, bias_slice, out_row, 0, n, k);
    };

    if let Some(bs) = bias_slice
        && bs.len() != n
    {
        return Err(OcrError::OperatorError {
            op: "MatMulQuantized".into(),
            msg: format!("bias has {} elements but n is {n}", bs.len()),
        });
    }

    let total_rows = out_slice.len() / n;
    if crate::tensor::should_parallelize(total_rows * n, 512) && n >= 512 {
        let num_threads = rayon::current_num_threads();
        let col_chunk_size = (n.div_ceil(num_threads)).max(64).min(n);
        let num_chunks = n.div_ceil(col_chunk_size);
        let out_ptr_usize = out_slice.as_mut_ptr() as usize;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let col_start = chunk_idx * col_chunk_size;
            let col_count = (n - col_start).min(col_chunk_size);
            let out_p = out_ptr_usize as *mut f32;

            for r in 0..total_rows {
                let a_row = &a_slice[r * k..(r + 1) * k];
                let out_chunk = unsafe {
                    std::slice::from_raw_parts_mut(out_p.add(r * n + col_start), col_count)
                };

                #[cfg(target_arch = "x86_64")]
                if has_avx2 {
                    unsafe {
                        compute_quantized_b_cols_avx2(
                            a_row, qw, bias_slice, out_chunk, col_start, col_count, k,
                        );
                    }
                    continue;
                }
                compute_quantized_b_cols_scalar(
                    a_row, qw, bias_slice, out_chunk, col_start, col_count, k,
                );
            }
        });
    } else if crate::tensor::should_parallelize(total_rows * n, 512) {
        out_slice.par_chunks_mut(n).enumerate().for_each(run_row);
    } else {
        out_slice.chunks_mut(n).enumerate().for_each(run_row);
    }

    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_quantized_b_cols_avx2(
    a_row: &[f32],
    qw: &QuantizedWeight,
    bias_slice: Option<&[f32]>,
    out_chunk: &mut [f32],
    col_start: usize,
    col_count: usize,
    k: usize,
) {
    unsafe {
        use std::arch::x86_64::*;
        let a_ptr = a_row.as_ptr();
        let w_ptr = qw.data.as_ptr();
        let s_ptr = qw.scales.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();

        let mut c = 0;
        while c + 4 <= col_count {
            let j0 = col_start + c;
            let j1 = j0 + 1;
            let j2 = j0 + 2;
            let j3 = j0 + 3;

            let w0 = w_ptr.add(j0 * k);
            let w1 = w_ptr.add(j1 * k);
            let w2 = w_ptr.add(j2 * k);
            let w3 = w_ptr.add(j3 * k);

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut i = 0;
            while i + 8 <= k {
                let va = _mm256_loadu_ps(a_ptr.add(i));

                let b8_0 = _mm_loadl_epi64(w0.add(i) as *const __m128i);
                let bf_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8_0));
                acc0 = _mm256_fmadd_ps(va, bf_0, acc0);

                let b8_1 = _mm_loadl_epi64(w1.add(i) as *const __m128i);
                let bf_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8_1));
                acc1 = _mm256_fmadd_ps(va, bf_1, acc1);

                let b8_2 = _mm_loadl_epi64(w2.add(i) as *const __m128i);
                let bf_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8_2));
                acc2 = _mm256_fmadd_ps(va, bf_2, acc2);

                let b8_3 = _mm_loadl_epi64(w3.add(i) as *const __m128i);
                let bf_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8_3));
                acc3 = _mm256_fmadd_ps(va, bf_3, acc3);

                i += 8;
            }

            let mut sum0 = hsum256(acc0);
            let mut sum1 = hsum256(acc1);
            let mut sum2 = hsum256(acc2);
            let mut sum3 = hsum256(acc3);

            while i < k {
                let a_val = *a_ptr.add(i);
                sum0 += a_val * (*w0.add(i) as f32);
                sum1 += a_val * (*w1.add(i) as f32);
                sum2 += a_val * (*w2.add(i) as f32);
                sum3 += a_val * (*w3.add(i) as f32);
                i += 1;
            }

            let dot0 = sum0 * *s_ptr.add(j0);
            let dot1 = sum1 * *s_ptr.add(j1);
            let dot2 = sum2 * *s_ptr.add(j2);
            let dot3 = sum3 * *s_ptr.add(j3);

            if let Some(bias) = bias_slice {
                *out_ptr.add(c) = dot0 + *bias.get_unchecked(j0);
                *out_ptr.add(c + 1) = dot1 + *bias.get_unchecked(j1);
                *out_ptr.add(c + 2) = dot2 + *bias.get_unchecked(j2);
                *out_ptr.add(c + 3) = dot3 + *bias.get_unchecked(j3);
            } else {
                *out_ptr.add(c) = dot0;
                *out_ptr.add(c + 1) = dot1;
                *out_ptr.add(c + 2) = dot2;
                *out_ptr.add(c + 3) = dot3;
            }

            c += 4;
        }

        while c < col_count {
            let j = col_start + c;
            let w_col = w_ptr.add(j * k);
            let dot = avx2_dot_product_w8(a_ptr, w_col, k) * *s_ptr.add(j);
            let val = if let Some(bias) = bias_slice {
                dot + *bias.get_unchecked(j)
            } else {
                dot
            };
            *out_ptr.add(c) = val;
            c += 1;
        }
    }
}

#[inline(always)]
fn compute_quantized_b_cols_scalar(
    a_row: &[f32],
    qw: &QuantizedWeight,
    bias_slice: Option<&[f32]>,
    out_chunk: &mut [f32],
    col_start: usize,
    col_count: usize,
    k: usize,
) {
    for (c, out_val) in out_chunk.iter_mut().take(col_count).enumerate() {
        let j = col_start + c;
        let w_col = &qw.data[j * k..(j + 1) * k];
        let scale = qw.scales[j];
        let dot: f32 = a_row
            .iter()
            .zip(w_col.iter())
            .map(|(&x, &y)| x * (y as f32))
            .sum::<f32>()
            * scale;
        *out_val = if let Some(bias) = bias_slice {
            dot + bias[j]
        } else {
            dot
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot_product_w8(a: *const f32, w8: *const i8, n: usize) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut i = 0;
        let mut v0 = _mm256_setzero_ps();
        let mut v1 = _mm256_setzero_ps();

        while i + 16 <= n {
            let va0 = _mm256_loadu_ps(a.add(i));
            let va1 = _mm256_loadu_ps(a.add(i + 8));

            let b_i8_0 = _mm_loadl_epi64(w8.add(i) as *const __m128i);
            let b_f32_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_i8_0));
            v0 = _mm256_fmadd_ps(va0, b_f32_0, v0);

            let b_i8_1 = _mm_loadl_epi64(w8.add(i + 8) as *const __m128i);
            let b_f32_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_i8_1));
            v1 = _mm256_fmadd_ps(va1, b_f32_1, v1);

            i += 16;
        }
        v0 = _mm256_add_ps(v0, v1);

        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.add(i));
            let b_i8 = _mm_loadl_epi64(w8.add(i) as *const __m128i);
            let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_i8));
            v0 = _mm256_fmadd_ps(va, b_f32, v0);
            i += 8;
        }

        let mut sum = hsum256(v0);

        while i < n {
            sum += *a.add(i) * (*w8.add(i) as f32);
            i += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
    }

    /// f64 reference for `a[m,k] @ b[k,n]`.
    fn reference(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for r in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[r * k + p] as f64 * b[p * n + j] as f64;
                }
                out[r * n + j] = acc;
            }
        }
        out
    }

    /// f64 reference for one output row of `a[m,k] @ b[n,k]^T`.
    fn reference_transposed_b(a_row: &[f32], b_t: &[f32], n: usize, k: usize) -> Vec<f64> {
        (0..n)
            .map(|j| {
                a_row
                    .iter()
                    .enumerate()
                    .map(|(p, &av)| av as f64 * b_t[j * k + p] as f64)
                    .sum()
            })
            .collect()
    }

    /// Regression: the fused single-row fast path keyed on `m == 1`, but `m` is only the
    /// *inner* row count. For a `[3, 1, K]` input `m == 1` while the output has `3 * N`
    /// elements, so the path chunked the whole flattened output by column and passed the
    /// entire input tensor as if it were row 0. Batch 0 came out right, the tail was never
    /// written (`Tensor::empty` is uninitialised in release), and `b_ptr.add(j * k)` read
    /// past the end of the weight for `j >= N`. With AVX2 disabled the scalar fallback
    /// panicked on the same input, so the behaviour depended on the CPU.
    #[test]
    fn matmul_transposed_b_handles_leading_batch_dims() {
        let (batch, k, n) = (3usize, 8usize, 1024usize);

        // A transposed B: [n, k], so row j is the j-th output channel's weights.
        let bt_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).sin()).collect();
        let bt = tensor(&[n, k], &bt_data);

        // Batch `b` uses a different slice of A so a mix-up is observable.
        let a_data: Vec<f32> = (0..batch * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let a = tensor(&[batch, 1, k], &a_data);

        let got = matmul_transposed_b(&a, &bt, None).expect("batched matmul");
        assert_eq!(got.shape(), &[batch, 1, n]);

        for b in 0..batch {
            let want = reference_transposed_b(&a_data[b * k..(b + 1) * k], &bt_data, n, k);
            for (j, &w) in want.iter().enumerate() {
                let idx = b * n + j;
                let g = got.as_slice()[idx];
                assert!(
                    (g as f64 - w).abs() < 1e-4,
                    "batch {b} col {j}: got {g} want {w}"
                );
            }
        }
    }

    /// The same input must give the same answer with the wide path forced off, and the
    /// general batched case must be right too. This is what would have caught the OOB read
    /// in the AVX2 kernel had `Tensor::empty` not been uninitialised.
    #[test]
    fn batched_matmul_is_identical_on_both_paths() {
        for shape in [[3usize, 1, 8], [2, 3, 8], [4, 1, 16]] {
            let (batch, m, k) = (shape[0], shape[1], shape[2]);
            let n = 1024usize;
            let bt_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.03).cos()).collect();
            let bt = tensor(&[n, k], &bt_data);
            let a_data: Vec<f32> = (0..batch * m * k)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let a = tensor(&shape, &a_data);

            let scalar = crate::ops::simd::with_avx2_override(false, || {
                matmul_transposed_b(&a, &bt, None).expect("scalar path")
            });
            let wide = crate::ops::simd::with_avx2_override(true, || {
                matmul_transposed_b(&a, &bt, None).expect("wide path")
            });
            assert_eq!(wide.shape(), scalar.shape(), "shape {shape:?}");
            for (i, (&s, &g)) in scalar.as_slice().iter().zip(wide.as_slice()).enumerate() {
                assert!(
                    (g - s).abs() <= 1e-4 + 1e-4 * s.abs(),
                    "shape {shape:?} index {i}: wide={g} scalar={s}"
                );
            }

            // And both must match the reference.
            let rows = batch * m;
            for r in 0..rows {
                let want = reference_transposed_b(&a_data[r * k..(r + 1) * k], &bt_data, n, k);
                for (j, &w) in want.iter().enumerate() {
                    let g = scalar.as_slice()[r * n + j] as f64;
                    assert!(
                        (g - w).abs() < 1e-4,
                        "shape {shape:?} row {r} col {j}: got {g} want {w}"
                    );
                }
            }
        }
    }

    /// The register-blocked kernel has two ragged edges: `m % TILE_ROWS` rows and
    /// `n % 4` columns. The batched test above uses `n = 1024` (a multiple of 4) and
    /// `batch * m` in 3..=6, so it never reaches the column tail. The recognition head,
    /// the kernel's whole reason for existing, is `n = 18710` -> a tail of 2 columns.
    ///
    /// Every row count from 1 to `TILE_ROWS + 2` is checked, along with each column tail of
    /// 0..4, so all four `MR` instantiations and both edge branches are covered.
    #[test]
    fn blocked_kernel_matches_reference_on_ragged_edges() {
        const K: usize = 8;
        for n_tail in 0..4usize {
            let n = 4 * 6 + n_tail;
            let bt_data: Vec<f32> = (0..n * K).map(|i| (i as f32 * 0.031).cos()).collect();
            let bt = tensor(&[n, K], &bt_data);
            for rows in 1..=(TILE_ROWS + 2) {
                let a_data: Vec<f32> = (0..rows * K).map(|i| (i as f32 * 0.07).sin()).collect();
                let a = tensor(&[rows, K], &a_data);

                // With a bias, to pin the `col_start` offset used when reading it.
                for bias in [None, Some(tensor(&[n], &vec![0.5f32; n]))] {
                    let got = matmul_transposed_b(&a, &bt, bias.as_ref()).expect("blocked");
                    assert_eq!(got.shape(), &[rows, n], "rows={rows} n={n}");

                    for r in 0..rows {
                        let want =
                            reference_transposed_b(&a_data[r * K..(r + 1) * K], &bt_data, n, K);
                        let got_row = &got.as_slice()[r * n..(r + 1) * n];
                        for (j, (&w, &g)) in want.iter().zip(got_row).enumerate() {
                            let expect = if bias.is_some() { w + 0.5 } else { w };
                            let g = g as f64;
                            assert!(
                                (g - expect).abs() < 1e-4,
                                "n={n} rows={rows} bias={} row {r} col {j}: got {g} want {expect}",
                                bias.is_some()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The INT8 head has the same fused single-row path, so it needs the same coverage.
    #[test]
    fn quantized_matmul_handles_leading_batch_dims() {
        let (batch, k, n) = (3usize, 8usize, 1024usize);
        let bt_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).sin()).collect();
        let bt = tensor(&[n, k], &bt_data);
        let qw = QuantizedWeight::from_f32_transposed(&bt);

        let a_data: Vec<f32> = (0..batch * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let a = tensor(&[batch, 1, k], &a_data);

        let got = matmul_quantized_b(&a, &qw, None).expect("batched quantized matmul");
        assert_eq!(got.shape(), &[batch, 1, n]);

        // INT8 is lossy, so compare against the FP32 weight rather than an f64 reference of
        // the quantised values, and only require a relative match.
        let fp32 = matmul_transposed_b(&a, &bt, None).expect("fp32 reference");
        for i in 0..got.numel() {
            let (g, f) = (got.as_slice()[i], fp32.as_slice()[i]);
            assert!(
                (g - f).abs() <= 0.05 + 0.05 * f.abs(),
                "index {i}: int8={g} fp32={f}"
            );
        }
    }

    /// The bias is indexed as `bias[j]` for `j in 0..n`, through `get_unchecked` in the
    /// AVX2 kernel, and its length was never compared against `n`.
    #[test]
    fn a_wrong_length_bias_is_an_error_not_an_oob_read() {
        let a = tensor(&[1, 8], &(0..8).map(|i| i as f32).collect::<Vec<_>>());
        let bt = tensor(&[16, 8], &vec![1.0; 16 * 8]);
        for n_bias in [0usize, 1, 15, 17] {
            let bias = tensor(&[n_bias], &vec![0.25; n_bias]);
            let r = std::panic::catch_unwind(|| matmul_transposed_b(&a, &bt, Some(&bias)));
            assert!(r.is_ok(), "bias of {n_bias} must not panic");
            let err = match r.expect("no panic") {
                Ok(_) => panic!("bias of {n_bias} must be rejected"),
                Err(e) => e,
            };
            assert!(
                matches!(err, OcrError::OperatorError { .. }),
                "bias of {n_bias}: got {err:?}"
            );
        }
        // The correct length still works.
        let bias = tensor(&[16], &[0.25; 16]);
        matmul_transposed_b(&a, &bt, Some(&bias)).expect("a valid bias must be accepted");
    }

    #[test]
    fn matmul_matches_hand_computed_values() {
        let a = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let out = matmul(&a, &b).expect("matmul");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

        let expect = reference(a.as_slice(), 2, 3, b.as_slice(), 2);
        for (i, (&got, &want)) in out.as_slice().iter().zip(expect.iter()).enumerate() {
            assert!(
                (got as f64 - want).abs() < 1e-4,
                "index {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn matmul_with_bias_adds_a_per_column_bias() {
        let a = tensor(&[2, 3], &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0]);
        let b = tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias = tensor(&[2], &[0.5, -1.5]);
        let out = matmul_with_bias(&a, &b, Some(&bias)).expect("matmul");
        let plain = matmul(&a, &b).expect("matmul");
        assert_eq!(out.shape(), &[2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                let want = plain.as_slice()[i * 2 + j] + bias.as_slice()[j];
                let got = out.as_slice()[i * 2 + j];
                assert!((got - want).abs() < 1e-5, "({i},{j}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn matmul_batches_over_leading_dimensions() {
        let (m, k, n) = (2usize, 3usize, 4usize);
        let a_data: Vec<f32> = (0..2 * m * k).map(|i| (i as f32) * 0.13 - 1.0).collect();
        let b_data: Vec<f32> = (0..2 * k * n).map(|i| (i as f32) * -0.07 + 0.4).collect();
        let a = tensor(&[2, m, k], &a_data);
        let b = tensor(&[2, k, n], &b_data);
        let out = matmul(&a, &b).expect("matmul");
        assert_eq!(out.shape(), &[2, m, n]);

        for batch in 0..2usize {
            let expect = reference(
                &a_data[batch * m * k..(batch + 1) * m * k],
                m,
                k,
                &b_data[batch * k * n..(batch + 1) * k * n],
                n,
            );
            for (i, (&got, &want)) in out.as_slice()[batch * m * n..(batch + 1) * m * n]
                .iter()
                .zip(expect.iter())
                .enumerate()
            {
                assert!(
                    ((got as f64) - want).abs() < 1e-4,
                    "batch {batch} index {i}: {got} vs {want}"
                );
            }
        }
    }

    /// `matmul_transposed_b` takes B already transposed (`b_t` is `[N, K]`) and must be
    /// exactly equivalent to `matmul` with the untransposed `[K, N]` operand.
    #[test]
    fn matmul_transposed_b_equals_plain_matmul() {
        let (m, k, n) = (3usize, 5usize, 4usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.29).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.41).cos()).collect();

        let a = tensor(&[m, k], &a_data);
        let b = tensor(&[k, n], &b_data);
        let b_t = b.transpose(0, 1).expect("transpose").to_contiguous();

        let plain = matmul(&a, &b).expect("matmul");
        let via_t = matmul_transposed_b(&a, &b_t, None).expect("matmul_transposed_b");

        assert_eq!(via_t.shape(), plain.shape());
        for (i, (&x, &y)) in via_t.as_slice().iter().zip(plain.as_slice()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-5 + 1e-5 * y.abs(),
                "index {i}: transposed path {x} vs plain {y}"
            );
        }
    }

    #[test]
    fn matmul_quantized_b_tracks_fp32_within_quantization_error() {
        let (k, n) = (64usize, 48usize);
        let a_data: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.19).sin() * 2.0).collect();
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.07).cos() * 0.5 + (i % 11) as f32 * 0.01)
            .collect();

        let a = tensor(&[1, k], &a_data);
        // `b_t` is [N, K] i.e. the transposed weight.
        let b_t = tensor(&[n, k], &w_data);
        let qw = QuantizedWeight::from_f32_transposed(&b_t);

        let fp32 = matmul_transposed_b(&a, &b_t, None).expect("fp32");
        let int8 = matmul_quantized_b(&a, &qw, None).expect("int8");

        assert_eq!(int8.shape(), fp32.shape());
        let max_abs = fp32
            .as_slice()
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        for (i, (&q, &f)) in int8.as_slice().iter().zip(fp32.as_slice()).enumerate() {
            let tol = 1e-4 + 0.02 * f.abs() + 0.02 * max_abs;
            assert!(
                (q - f).abs() <= tol,
                "index {i}: int8={q} fp32={f} (tol {tol})"
            );
        }
    }

    #[test]
    fn matmul_rejects_incompatible_inner_dimensions() {
        let a = tensor(&[2, 3], &[0.0; 6]);
        let b = tensor(&[4, 2], &[0.0; 8]);
        assert!(matmul(&a, &b).is_err(), "3 != 4 must be rejected");
        assert!(matmul_transposed_b(&a, &b, None).is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_transposed_b_avx2_matches_scalar() {
        let (m, k, n) = (8usize, 96usize, 40usize);
        // A length that is not a multiple of the 8-wide kernel so the tail is covered.
        let a_data: Vec<f32> = (0..m * (k + 1))
            .map(|i| ((i as f32) * 0.083).sin())
            .collect();
        let a = tensor(&[m, k + 1], &a_data);
        let w_data: Vec<f32> = (0..n * (k + 1))
            .map(|i| ((i as f32) * 0.037).cos() * 1.3)
            .collect();
        let b_t = tensor(&[n, k + 1], &w_data);

        let scalar = crate::ops::simd::with_avx2_override(false, || {
            matmul_transposed_b(&a, &b_t, None).expect("matmul")
        });
        let avx2 = crate::ops::simd::with_avx2_override(true, || {
            matmul_transposed_b(&a, &b_t, None).expect("matmul")
        });

        for (i, (&x, &y)) in avx2.as_slice().iter().zip(scalar.as_slice()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 + 1e-4 * y.abs(),
                "index {i}: avx2={x} scalar={y}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_quantized_b_avx2_matches_scalar() {
        let (m, k, n) = (1usize, 123usize, 533usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.083).sin()).collect();
        let a = tensor(&[m, k], &a_data);
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.037).cos() * 1.3)
            .collect();
        let b_t = tensor(&[n, k], &w_data);
        let qw = QuantizedWeight::from_f32_transposed(&b_t);

        let scalar = crate::ops::simd::with_avx2_override(false, || {
            matmul_quantized_b(&a, &qw, None).expect("matmul scalar")
        });
        let avx2 = crate::ops::simd::with_avx2_override(true, || {
            matmul_quantized_b(&a, &qw, None).expect("matmul avx2")
        });

        for (i, (&x, &y)) in avx2.as_slice().iter().zip(scalar.as_slice()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 + 1e-4 * y.abs(),
                "index {i}: avx2={x} scalar={y}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_transposed_b_m1_large_n_matches_scalar() {
        let (m, k, n) = (1usize, 123usize, 533usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.083).sin()).collect();
        let a = tensor(&[m, k], &a_data);
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.037).cos() * 1.3)
            .collect();
        let b_t = tensor(&[n, k], &w_data);

        let scalar = crate::ops::simd::with_avx2_override(false, || {
            matmul_transposed_b(&a, &b_t, None).expect("matmul scalar")
        });
        let avx2 = crate::ops::simd::with_avx2_override(true, || {
            matmul_transposed_b(&a, &b_t, None).expect("matmul avx2")
        });

        for (i, (&x, &y)) in avx2.as_slice().iter().zip(scalar.as_slice()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 + 1e-4 * y.abs(),
                "index {i}: avx2={x} scalar={y}"
            );
        }
    }
}
