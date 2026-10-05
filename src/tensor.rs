use crate::error::{OcrError, Result};
use rayon::prelude::*;
use std::cell::{Cell, RefCell};

thread_local! {
    static BUFFER_POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
    static PARALLEL_DISABLED: Cell<bool> = const { Cell::new(false) };
}

/// Check if inner operator parallelism is disabled on the current thread.
pub fn is_parallel_disabled() -> bool {
    PARALLEL_DISABLED.get()
}

/// Set whether inner operator parallelism is disabled on the current thread.
pub fn set_parallel_disabled(disabled: bool) {
    PARALLEL_DISABLED.set(disabled);
}

/// RAII guard to disable inner parallelism on the current thread for the duration of its scope.
///
/// Nests correctly: the guard remembers the state it found and restores exactly
/// that on drop, so an inner guard cannot silently re-enable parallelism for an
/// outer one.
pub struct ScopedDisableParallel {
    previous: bool,
}

impl Default for ScopedDisableParallel {
    fn default() -> Self {
        Self::new()
    }
}

impl ScopedDisableParallel {
    pub fn new() -> Self {
        let previous = is_parallel_disabled();
        set_parallel_disabled(true);
        Self { previous }
    }
}

impl Drop for ScopedDisableParallel {
    fn drop(&mut self) {
        set_parallel_disabled(self.previous);
    }
}

/// Single decision point for "should this loop fan out to rayon?".
///
/// Every parallel site in the crate routes through here so that the
/// nested-parallelism guard ([`is_parallel_disabled`]) can never be forgotten on
/// a new code path. `work` is the number of independent units of work (floats,
/// channels, rows, ...) and `threshold` is the size below which the fork/join
/// overhead is not worth paying.
#[inline]
pub fn should_parallelize(work: usize, threshold: usize) -> bool {
    work >= threshold && !is_parallel_disabled() && rayon::current_num_threads() > 1
}

/// Like [`should_parallelize`], but for call sites that need to know the number
/// of worker threads as well (to compute chunk sizes).
#[inline]
pub fn parallel_degree(work: usize, threshold: usize) -> usize {
    if should_parallelize(work, threshold) {
        rayon::current_num_threads()
    } else {
        1
    }
}

/// Acquire a reusable buffer with at least `min_capacity` floats from the thread-local pool.
pub fn acquire_buffer(min_capacity: usize) -> Vec<f32> {
    if min_capacity < 64 {
        return Vec::with_capacity(min_capacity);
    }
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let mut best_idx = None;
        let mut best_cap = usize::MAX;
        let max_acceptable_cap = min_capacity.saturating_mul(4).max(512);
        for (i, b) in pool.iter().enumerate() {
            let cap = b.capacity();
            if cap >= min_capacity && cap <= max_acceptable_cap && cap < best_cap {
                best_cap = cap;
                best_idx = Some(i);
            }
        }
        if let Some(pos) = best_idx {
            let mut buf = pool.swap_remove(pos);
            buf.clear();
            buf
        } else {
            Vec::with_capacity(min_capacity)
        }
    })
}

/// Return a buffer to the thread-local pool for reuse.
pub fn release_buffer(mut buf: Vec<f32>) {
    let cap = buf.capacity();
    if !(64..=64 * 1024 * 1024).contains(&cap) {
        return;
    }
    buf.clear();
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 64 {
            pool.push(buf);
        }
    });
}

#[derive(Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
    contiguous: bool,
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.data.capacity() >= 64 {
            let v = std::mem::take(&mut self.data);
            release_buffer(v);
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let numel = self.data.len();
        let mut new_data = acquire_buffer(numel);
        new_data.clear();
        new_data.extend_from_slice(&self.data);
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: new_data,
            contiguous: self.contiguous,
        }
    }
}

pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl Tensor {
    pub fn new(shape: &[usize]) -> Self {
        Self::zeros(shape)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let strides = compute_contiguous_strides(shape);
        let mut data = acquire_buffer(numel);
        data.resize(numel, 0.0);
        Self {
            shape: shape.to_vec(),
            strides,
            data,
            contiguous: true,
        }
    }

    /// Allocate a tensor *without* zero-initialising it, reusing a buffer from the pool.
    ///
    /// # Caller contract
    ///
    /// Every element of the returned tensor **must be written before it is read**.
    /// In release builds the backing memory is genuinely uninitialised, so a
    /// missed write yields whatever the recycled buffer happened to contain.
    /// To make such a mistake loud instead of silent, debug builds pre-fill the
    /// buffer with `f32::NAN`: any leaked element then shows up immediately in
    /// unit tests (and in `debug_assert!`-enabled integration runs) rather than
    /// producing plausible-looking garbage.
    pub fn empty(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let strides = compute_contiguous_strides(shape);
        let mut data = acquire_buffer(numel);
        debug_assert_eq!(data.len(), 0, "pooled buffers must be handed back cleared");
        if cfg!(debug_assertions) {
            data.resize(numel, f32::NAN);
        } else {
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(numel);
            }
        }
        Self {
            shape: shape.to_vec(),
            strides,
            data,
            contiguous: true,
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(OcrError::TensorError(format!(
                "Shape {:?} requires {} elements, but provided data has {}",
                shape,
                numel,
                data.len()
            )));
        }
        let strides = compute_contiguous_strides(shape);
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            data,
            contiguous: true,
        })
    }

    pub fn from_slice(shape: &[usize], slice: &[f32]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if numel != slice.len() {
            return Err(OcrError::TensorError(format!(
                "Shape {:?} requires {} elements, but provided slice has {}",
                shape,
                numel,
                slice.len()
            )));
        }
        let mut data = acquire_buffer(numel);
        data.clear();
        data.extend_from_slice(slice);
        let strides = compute_contiguous_strides(shape);
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            data,
            contiguous: true,
        })
    }

    pub fn from_i64_slice(shape: &[usize], data: &[i64]) -> Result<Self> {
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        Self::from_vec(shape, f32_data)
    }

    pub fn from_i32_slice(shape: &[usize], data: &[i32]) -> Result<Self> {
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        Self::from_vec(shape, f32_data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if !self.contiguous {
            let cont = self.to_contiguous();
            *self = cont;
        }
        &mut self.data
    }

    pub fn to_i64_vec(&self) -> Vec<i64> {
        let cont = if self.contiguous {
            self.clone()
        } else {
            self.to_contiguous()
        };
        cont.data.iter().map(|&x| x.round() as i64).collect()
    }

    pub fn to_contiguous(&self) -> Self {
        if self.contiguous {
            return self.clone();
        }
        let total = self.shape.iter().product();
        let mut new_data = acquire_buffer(total);
        unsafe {
            new_data.set_len(total);
        }
        let new_strides = compute_contiguous_strides(&self.shape);

        // Iterate through all multi-dimensional coordinates
        let ndim = self.ndim();
        let mut coord = vec![0usize; ndim];
        for val in new_data.iter_mut() {
            // compute src offset
            let mut src_idx = 0;
            for (d, &c) in coord.iter().enumerate() {
                src_idx += c * self.strides[d];
            }
            *val = self.data[src_idx];

            // increment coord
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < self.shape[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        Self {
            shape: self.shape.clone(),
            strides: new_strides,
            data: new_data,
            contiguous: true,
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if self.numel() != new_numel {
            return Err(OcrError::TensorError(format!(
                "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                self.numel(),
                new_shape,
                new_numel
            )));
        }

        if self.contiguous {
            // Metadata-only: one buffer copy, no gather.
            let mut data = acquire_buffer(self.numel());
            data.extend_from_slice(&self.data);
            return Ok(Self {
                shape: new_shape.to_vec(),
                strides: compute_contiguous_strides(new_shape),
                data,
                contiguous: true,
            });
        }
        self.to_contiguous().into_reshape(new_shape)
    }

    /// Consuming variant of [`Tensor::reshape`].
    ///
    /// A reshape never moves elements, so when the caller owns the tensor this
    /// only rewrites the shape/strides metadata instead of copying the buffer.
    pub fn into_reshape(mut self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if self.numel() != new_numel {
            return Err(OcrError::TensorError(format!(
                "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                self.numel(),
                new_shape,
                new_numel
            )));
        }
        if !self.contiguous {
            let mut cont = self.to_contiguous();
            std::mem::swap(&mut self, &mut cont);
        }
        self.shape = new_shape.to_vec();
        self.strides = compute_contiguous_strides(new_shape);
        self.contiguous = true;
        Ok(self)
    }

    pub fn permute(&self, order: &[usize]) -> Result<Self> {
        if order.len() != self.ndim() {
            return Err(OcrError::TensorError(format!(
                "Permute order length {} must match tensor ndim {}",
                order.len(),
                self.ndim()
            )));
        }
        let mut visited = vec![false; self.ndim()];
        for &d in order {
            if d >= self.ndim() || visited[d] {
                return Err(OcrError::TensorError(format!(
                    "Invalid permute order {:?}",
                    order
                )));
            }
            visited[d] = true;
        }

        let new_shape: Vec<usize> = order.iter().map(|&d| self.shape[d]).collect();
        let new_strides: Vec<usize> = order.iter().map(|&d| self.strides[d]).collect();

        let ndim = self.ndim();
        let total: usize = new_shape.iter().product();

        if (0..ndim).zip(order.iter()).all(|(a, &b)| a == b) {
            return Ok(self.clone());
        }

        let mut data = acquire_buffer(total);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(total);
        }

        // Fast Path 1: Batched 2D matrix transpose of last 2 dimensions: [..., M, N] -> [..., N, M]
        if ndim >= 2
            && order[..ndim - 2].iter().enumerate().all(|(i, &d)| i == d)
            && order[ndim - 2] == ndim - 1
            && order[ndim - 1] == ndim - 2
            && self.is_contiguous()
        {
            let m = self.shape[ndim - 2];
            let n = self.shape[ndim - 1];
            let b_total = total / (m * n);
            let dst_p = data.as_mut_ptr() as usize;
            let src_p = self.data.as_ptr() as usize;

            let run_b = |b: usize| {
                let src_b = unsafe { (src_p as *const f32).add(b * m * n) };
                let dst_b = unsafe { (dst_p as *mut f32).add(b * n * m) };
                for i0 in (0..m).step_by(8) {
                    let i_max = (i0 + 8).min(m);
                    for j0 in (0..n).step_by(8) {
                        let j_max = (j0 + 8).min(n);
                        for i in i0..i_max {
                            for j in j0..j_max {
                                unsafe {
                                    *dst_b.add(j * m + i) = *src_b.add(i * n + j);
                                }
                            }
                        }
                    }
                }
            };

            if b_total >= 2 && total >= 8192 {
                use rayon::prelude::*;
                (0..b_total).into_par_iter().for_each(run_b);
            } else {
                for b in 0..b_total {
                    run_b(b);
                }
            }

            return Ok(Self {
                shape: new_shape.clone(),
                strides: compute_contiguous_strides(&new_shape),
                data,
                contiguous: true,
            });
        }

        // Fast Path 2: [B, M, N, L] -> [B, N, M, L] (swap middle 2 dimensions, keeping contiguous line L)
        if ndim == 4 && order == [0, 2, 1, 3] && self.is_contiguous() {
            let b_dim = self.shape[0];
            let m = self.shape[1];
            let n = self.shape[2];
            let l = self.shape[3];
            let dst_p = data.as_mut_ptr() as usize;
            let src_p = self.data.as_ptr() as usize;

            let run_b = |b: usize| {
                let src_b = unsafe { (src_p as *const f32).add(b * m * n * l) };
                let dst_b = unsafe { (dst_p as *mut f32).add(b * n * m * l) };
                for i in 0..m {
                    for j in 0..n {
                        let src_off = (i * n + j) * l;
                        let dst_off = (j * m + i) * l;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src_b.add(src_off),
                                dst_b.add(dst_off),
                                l,
                            );
                        }
                    }
                }
            };

            if b_dim >= 2 && total >= 8192 {
                use rayon::prelude::*;
                (0..b_dim).into_par_iter().for_each(run_b);
            } else {
                for b in 0..b_dim {
                    run_b(b);
                }
            }

            return Ok(Self {
                shape: new_shape.clone(),
                strides: compute_contiguous_strides(&new_shape),
                data,
                contiguous: true,
            });
        }

        // Fast Path 3: [S0, S1, S2, S3, L] -> [S2, S0, S3, S1, L] (e.g. QKV projection transpose)
        if ndim == 5 && order == [2, 0, 3, 1, 4] && self.is_contiguous() {
            let s0 = self.shape[0];
            let s1 = self.shape[1];
            let s2 = self.shape[2];
            let s3 = self.shape[3];
            let l = self.shape[4];
            let dst_p = data.as_mut_ptr() as usize;
            let src_p = self.data.as_ptr() as usize;

            let run_d2_d3 = |(d2, d3): (usize, usize)| {
                let src_p = src_p as *const f32;
                let dst_p = dst_p as *mut f32;
                for d0 in 0..s0 {
                    for d1 in 0..s1 {
                        let src_off = (((d0 * s1 + d1) * s2 + d2) * s3 + d3) * l;
                        let dst_off = (((d2 * s0 + d0) * s3 + d3) * s1 + d1) * l;
                        unsafe {
                            std::ptr::copy_nonoverlapping(src_p.add(src_off), dst_p.add(dst_off), l);
                        }
                    }
                }
            };

            if total >= 65_536 && s2 * s3 >= 4 {
                use rayon::prelude::*;
                let num_tasks = s2 * s3;
                (0..num_tasks).into_par_iter().for_each(|task| {
                    run_d2_d3((task / s3, task % s3));
                });
            } else {
                for d2 in 0..s2 {
                    for d3 in 0..s3 {
                        run_d2_d3((d2, d3));
                    }
                }
            }

            return Ok(Self {
                shape: new_shape.clone(),
                strides: compute_contiguous_strides(&new_shape),
                data,
                contiguous: true,
            });
        }

        // Fast Path 4: [B, S1, S2, S3] -> [B, S3, S1, S2] (e.g. [1, 1, 120, 120] -> [1, 120, 1, 120])
        if ndim == 4 && order == [0, 3, 1, 2] && self.is_contiguous() {
            let b_dim = self.shape[0];
            let s1 = self.shape[1];
            let s2 = self.shape[2];
            let s3 = self.shape[3];
            let dst_p = data.as_mut_ptr() as usize;
            let src_p = self.data.as_ptr() as usize;

            let run_b = |b: usize| {
                let src_p = src_p as *const f32;
                let dst_p = dst_p as *mut f32;
                for i3 in 0..s3 {
                    for i1 in 0..s1 {
                        for i2 in 0..s2 {
                            let src_off = ((b * s1 + i1) * s2 + i2) * s3 + i3;
                            let dst_off = ((b * s3 + i3) * s1 + i1) * s2 + i2;
                            unsafe {
                                *dst_p.add(dst_off) = *src_p.add(src_off);
                            }
                        }
                    }
                }
            };

            if b_dim >= 2 && total >= 8192 {
                use rayon::prelude::*;
                (0..b_dim).into_par_iter().for_each(run_b);
            } else {
                for b in 0..b_dim {
                    run_b(b);
                }
            }

            return Ok(Self {
                shape: new_shape.clone(),
                strides: compute_contiguous_strides(&new_shape),
                data,
                contiguous: true,
            });
        }

        // General fallback
        let mut coord = vec![0usize; ndim];
        for slot in data.iter_mut().take(total) {
            let mut src_idx = 0usize;
            for (j, &c) in coord.iter().enumerate() {
                src_idx += c * new_strides[j];
            }
            *slot = self.data[src_idx];

            for j in (0..ndim).rev() {
                coord[j] += 1;
                if coord[j] < new_shape[j] {
                    break;
                }
                coord[j] = 0;
            }
        }

        Ok(Self {
            shape: new_shape.clone(),
            strides: compute_contiguous_strides(&new_shape),
            data,
            contiguous: true,
        })
    }

    /// Consuming variant of [`Tensor::permute`]: identical result, but it can
    /// release the source buffer through the pool when the input is contiguous.
    pub fn into_permute(self, order: &[usize]) -> Result<Self> {
        self.permute(order)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(OcrError::TensorError(format!(
                "Transpose dims ({}, {}) out of bounds for ndim {}",
                dim0, dim1, ndim
            )));
        }
        let mut order: Vec<usize> = (0..ndim).collect();
        order.swap(dim0, dim1);
        self.permute(&order)
    }

    pub fn squeeze(&self, axes: Option<&[usize]>) -> Result<Self> {
        let mut new_shape = Vec::new();
        match axes {
            Some(target_axes) => {
                for (i, &dim) in self.shape.iter().enumerate() {
                    if target_axes.contains(&i) {
                        if dim != 1 {
                            return Err(OcrError::TensorError(format!(
                                "Cannot squeeze dimension {} of size {}",
                                i, dim
                            )));
                        }
                    } else {
                        new_shape.push(dim);
                    }
                }
            }
            None => {
                for &dim in &self.shape {
                    if dim != 1 {
                        new_shape.push(dim);
                    }
                }
            }
        }
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        self.reshape(&new_shape)
    }

    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(OcrError::TensorError(format!(
                "Unsqueeze axis {} out of bounds for ndim {}",
                axis,
                self.ndim()
            )));
        }
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        self.reshape(&new_shape)
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        self.clone().into_slice(axis, start, end, step)
    }

    /// Consuming variant of [`Tensor::slice`].
    ///
    /// When the requested range covers the whole axis this degenerates into a
    /// metadata update with zero copies.
    pub fn into_slice(
        mut self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(OcrError::TensorError(format!(
                "Slice axis {} out of bounds for ndim {}",
                axis,
                self.ndim()
            )));
        }
        let dim_size = self.shape[axis];
        let end = end.min(dim_size);
        if start >= end || step == 0 {
            return Err(OcrError::TensorError(format!(
                "Invalid slice params: start={}, end={}, step={}, dim_size={}",
                start, end, step, dim_size
            )));
        }

        if !self.contiguous {
            let mut cont = self.to_contiguous();
            std::mem::swap(&mut self, &mut cont);
        }

        if start == 0 && step == 1 && end == dim_size {
            // Selecting the whole axis moves no data at all.
            return Ok(self);
        }

        let slice_len = (end - start).div_ceil(step);
        let mut new_shape = self.shape.clone();
        new_shape[axis] = slice_len;

        let outer_size: usize = self.shape[..axis].iter().product();
        let inner_size: usize = self.shape[axis + 1..].iter().product();
        let new_total: usize = new_shape.iter().product();
        let mut new_data = acquire_buffer(new_total);
        new_data.clear();

        for outer in 0..outer_size {
            let outer_base = outer * (dim_size * inner_size);
            for s in 0..slice_len {
                let in_axis_idx = start + s * step;
                let in_offset = outer_base + in_axis_idx * inner_size;
                new_data.extend_from_slice(&self.data[in_offset..in_offset + inner_size]);
            }
        }

        Self::from_vec(&new_shape, new_data)
    }

    #[inline(always)]
    fn for_each_mut<F>(&mut self, f: F)
    where
        F: Fn(&mut f32) + Sync + Send,
    {
        let slice = self.as_mut_slice();
        if should_parallelize(slice.len(), 65536) {
            slice.par_iter_mut().for_each(f);
        } else {
            slice.iter_mut().for_each(f);
        }
    }

    pub fn clip_inplace(&mut self, min: f32, max: f32) {
        self.for_each_mut(|x| {
            *x = x.clamp(min, max);
        });
    }

    pub fn relu_inplace(&mut self) {
        self.for_each_mut(|x| {
            if *x < 0.0 {
                *x = 0.0;
            }
        });
    }

    pub fn sigmoid_inplace(&mut self) {
        self.for_each_mut(|x| {
            *x = 1.0 / (1.0 + (-*x).exp());
        });
    }

    pub fn hardsigmoid_inplace(&mut self, alpha: f32, beta: f32) {
        self.for_each_mut(|x| {
            *x = (*x * alpha + beta).clamp(0.0, 1.0);
        });
    }

    pub fn hardswish_inplace(&mut self) {
        self.for_each_mut(|x| {
            let hsigmoid = (*x + 3.0).clamp(0.0, 6.0) / 6.0;
            *x *= hsigmoid;
        });
    }

    pub fn sqrt_inplace(&mut self) {
        self.for_each_mut(|x| {
            *x = x.sqrt();
        });
    }

    #[allow(clippy::excessive_precision)]
    pub fn erf_inplace(&mut self) {
        self.for_each_mut(|x| {
            let v = *x;
            let sign = if v >= 0.0 { 1.0 } else { -1.0 };
            let ax = v.abs();
            let p = 0.3275911;
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let t = 1.0 / (1.0 + p * ax);
            let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
            *x = sign * (1.0 - poly * (-ax * ax).exp());
        });
    }

    #[allow(clippy::excessive_precision)]
    pub fn gelu_inplace(&mut self) {
        self.for_each_mut(|x| {
            let v = *x;
            let scaled = v * 0.7071067811865475; // 1 / sqrt(2)
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
            *x = 0.5 * v * (1.0 + erf_val);
        });
    }

    pub fn argmax(&self, axis: usize) -> Result<Vec<usize>> {
        if axis >= self.ndim() {
            return Err(OcrError::TensorError(format!(
                "Argmax axis {} out of bounds for ndim {}",
                axis,
                self.ndim()
            )));
        }
        let cont = if self.contiguous {
            self.clone()
        } else {
            self.to_contiguous()
        };

        let outer_size: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner_size: usize = self.shape[axis + 1..].iter().product();

        let mut result = vec![0usize; outer_size * inner_size];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;
                for a in 0..axis_size {
                    let idx = outer * (axis_size * inner_size) + a * inner_size + inner;
                    let val = cont.data[idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = a;
                    }
                }
                result[outer * inner_size + inner] = max_idx;
            }
        }
        Ok(result)
    }
}

// Broadcasting helpers
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let ndim = shape_a.len().max(shape_b.len());
    let mut out_shape = vec![0usize; ndim];

    for i in 0..ndim {
        let dim_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };
        let dim_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b {
            out_shape[ndim - 1 - i] = dim_a;
        } else if dim_a == 1 {
            out_shape[ndim - 1 - i] = dim_b;
        } else if dim_b == 1 {
            out_shape[ndim - 1 - i] = dim_a;
        } else {
            return Err(OcrError::TensorError(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                shape_a, shape_b
            )));
        }
    }

    Ok(out_shape)
}

pub fn compute_broadcast_strides(shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let orig_strides = compute_contiguous_strides(shape);
    let target_ndim = target_shape.len();
    let orig_ndim = shape.len();
    let mut strides = vec![0usize; target_ndim];

    for i in 0..target_ndim {
        let target_dim_idx = target_ndim - 1 - i;
        if i < orig_ndim {
            let orig_dim_idx = orig_ndim - 1 - i;
            if shape[orig_dim_idx] == target_shape[target_dim_idx] {
                strides[target_dim_idx] = orig_strides[orig_dim_idx];
            } else {
                strides[target_dim_idx] = 0;
            }
        } else {
            strides[target_dim_idx] = 0;
        }
    }
    strides
}

pub fn elementwise_op<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
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

    // Fast path: identical shapes
    if a_cont.shape() == b_cont.shape() {
        let n = a_cont.numel();
        let a_slice = a_cont.as_slice();
        let b_slice = b_cont.as_slice();
        let mut out = Tensor::empty(a_cont.shape());

        if should_parallelize(n, 8192) {
            out.as_mut_slice()
                .par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * 4096;
                    for (i, val) in chunk.iter_mut().enumerate() {
                        *val = op(a_slice[start + i], b_slice[start + i]);
                    }
                });
        } else {
            let out_slice = out.as_mut_slice();
            for i in 0..n {
                out_slice[i] = op(a_slice[i], b_slice[i]);
            }
        }
        return Ok(out);
    }

    // Fast path: b is scalar (1 element)
    if b_cont.numel() == 1 {
        let vb = b_cont.as_slice()[0];
        let n = a_cont.numel();
        let a_slice = a_cont.as_slice();
        let mut out = Tensor::empty(a_cont.shape());
        if should_parallelize(n, 8192) {
            out.as_mut_slice()
                .par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * 4096;
                    for (i, val) in chunk.iter_mut().enumerate() {
                        *val = op(a_slice[start + i], vb);
                    }
                });
        } else {
            let out_slice = out.as_mut_slice();
            for i in 0..n {
                out_slice[i] = op(a_slice[i], vb);
            }
        }
        return Ok(out);
    }

    // Fast path: a is scalar (1 element)
    if a_cont.numel() == 1 {
        let va = a_cont.as_slice()[0];
        let n = b_cont.numel();
        let b_slice = b_cont.as_slice();
        let mut out = Tensor::empty(b_cont.shape());
        if should_parallelize(n, 8192) {
            out.as_mut_slice()
                .par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * 4096;
                    for (i, val) in chunk.iter_mut().enumerate() {
                        *val = op(va, b_slice[start + i]);
                    }
                });
        } else {
            let out_slice = out.as_mut_slice();
            for i in 0..n {
                out_slice[i] = op(va, b_slice[i]);
            }
        }
        return Ok(out);
    }

    // Fast path: 4D Channel Broadcast [1, C, H, W] with [1, C, 1, 1] or [C, 1, 1]
    if a_cont.ndim() == 4 {
        let [n, c, h, w] = [
            a_cont.shape()[0],
            a_cont.shape()[1],
            a_cont.shape()[2],
            a_cont.shape()[3],
        ];
        let hw = h * w;
        if (b_cont.shape() == [1, c, 1, 1] || b_cont.shape() == [c, 1, 1]) && n == 1 {
            let a_slice = a_cont.as_slice();
            let b_slice = b_cont.as_slice();
            let mut out = Tensor::empty(a_cont.shape());
            let out_slice = out.as_mut_slice();
            for ch in 0..c {
                let vb = b_slice[ch];
                let src = &a_slice[ch * hw..(ch + 1) * hw];
                let dst = &mut out_slice[ch * hw..(ch + 1) * hw];
                for i in 0..hw {
                    dst[i] = op(src[i], vb);
                }
            }
            return Ok(out);
        }
    }
    if b_cont.ndim() == 4 {
        let [n, c, h, w] = [
            b_cont.shape()[0],
            b_cont.shape()[1],
            b_cont.shape()[2],
            b_cont.shape()[3],
        ];
        let hw = h * w;
        if (a_cont.shape() == [1, c, 1, 1] || a_cont.shape() == [c, 1, 1]) && n == 1 {
            let a_slice = a_cont.as_slice();
            let b_slice = b_cont.as_slice();
            let mut out = Tensor::empty(b_cont.shape());
            let out_slice = out.as_mut_slice();
            for ch in 0..c {
                let va = a_slice[ch];
                let src = &b_slice[ch * hw..(ch + 1) * hw];
                let dst = &mut out_slice[ch * hw..(ch + 1) * hw];
                for i in 0..hw {
                    dst[i] = op(va, src[i]);
                }
            }
            return Ok(out);
        }
    }

    // General broadcasting fallback with amortized O(1) coordinate stepping (zero division/modulo in inner loop)
    let out_shape = broadcast_shapes(a_cont.shape(), b_cont.shape())?;
    let mut out = Tensor::empty(&out_shape);
    let out_data = out.as_mut_slice();

    let a_strides = compute_broadcast_strides(a_cont.shape(), &out_shape);
    let b_strides = compute_broadcast_strides(b_cont.shape(), &out_shape);
    let ndim = out_shape.len();

    let a_slice = a_cont.as_slice();
    let b_slice = b_cont.as_slice();

    let mut coords = vec![0usize; ndim];
    let mut a_idx = 0usize;
    let mut b_idx = 0usize;

    for out_val in out_data.iter_mut() {
        *out_val = op(a_slice[a_idx], b_slice[b_idx]);

        for d in (0..ndim).rev() {
            coords[d] += 1;
            a_idx += a_strides[d];
            b_idx += b_strides[d];
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
            a_idx -= a_strides[d] * out_shape[d];
            b_idx -= b_strides[d] * out_shape[d];
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros_and_buffer_pool_small() {
        let t = Tensor::zeros(&[1, 96, 1, 1]);
        assert_eq!(t.numel(), 96);
        assert!(t.as_slice().iter().all(|&v| v == 0.0));
        let cap = t.data.capacity();
        drop(t);

        // Subsequent allocation of similar small size should reuse buffer
        let t2 = Tensor::zeros(&[1, 96, 1, 1]);
        assert_eq!(t2.data.capacity(), cap);
        assert!(t2.as_slice().iter().all(|&v| v == 0.0));
    }
}
