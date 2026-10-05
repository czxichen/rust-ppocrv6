use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

pub fn batch_norm2d(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    if x.ndim() < 2 {
        return Err(OcrError::OperatorError {
            op: "BatchNormalization".into(),
            msg: "Input must have at least 2 dimensions".into(),
        });
    }

    let channels = x.shape()[1];
    if scale.numel() != channels
        || bias.numel() != channels
        || mean.numel() != channels
        || var.numel() != channels
    {
        return Err(OcrError::OperatorError {
            op: "BatchNormalization".into(),
            msg: format!(
                "Channels mismatch: X has {}, scale={}, bias={}, mean={}, var={}",
                channels,
                scale.numel(),
                bias.numel(),
                mean.numel(),
                var.numel()
            ),
        });
    }

    let mut output = Tensor::empty(x.shape());
    let x_cont_holder;
    let x_slice = if x.is_contiguous() {
        x.as_slice()
    } else {
        x_cont_holder = x.to_contiguous();
        x_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let scale_s = scale.as_slice();
    let bias_s = bias.as_slice();
    let mean_s = mean.as_slice();
    let var_s = var.as_slice();

    // Precompute a and b per channel
    let mut a = vec![0.0f32; channels];
    let mut b = vec![0.0f32; channels];
    for c in 0..channels {
        let inv_std = 1.0 / (var_s[c] + epsilon).sqrt();
        a[c] = scale_s[c] * inv_std;
        b[c] = bias_s[c] - mean_s[c] * a[c];
    }

    let spatial_size: usize = x.shape()[2..].iter().product();

    // `par_chunks_mut(0)` / `chunks_mut(0)` panic, and `idx % channels` divides by zero
    // when `channels == 0`. A zero-sized spatial extent is legal ONNX, and unlike conv2d
    // nothing upstream guarantees `out_h >= 1` here.
    if spatial_size == 0 || channels == 0 {
        return Ok(output);
    }

    let run_chunk = |(idx, out_chunk): (usize, &mut [f32])| {
        let c = idx % channels;
        let in_offset = idx * spatial_size;
        let ac = a[c];
        let bc = b[c];

        for i in 0..spatial_size {
            out_chunk[i] = x_slice[in_offset + i] * ac + bc;
        }
    };

    if crate::tensor::should_parallelize(out_slice.len(), 65536) {
        out_slice
            .par_chunks_mut(spatial_size)
            .enumerate()
            .for_each(run_chunk);
    } else {
        out_slice
            .chunks_mut(spatial_size)
            .enumerate()
            .for_each(run_chunk);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
    }

    /// `par_chunks_mut(0)` panics outright, and `idx % channels` divides by zero when
    /// `channels == 0`. A zero spatial extent is legal ONNX and, unlike conv2d, nothing
    /// upstream guarantees a positive `out_h` here.
    #[test]
    fn zero_sized_spatial_extent_does_not_panic() {
        for shape in [
            vec![1usize, 2, 0, 4],
            vec![1, 2, 4, 0],
            vec![1, 2, 0, 0],
            vec![1, 0, 4, 4],
        ] {
            let n: usize = shape.iter().product();
            let x = t(&shape, &vec![1.0; n]);
            // Parameter tensors must have exactly `shape[1]` elements, so a zero-channel
            // input needs empty ones.
            let c = shape[1];
            let s = t(&[c], &vec![1.0; c]);
            let b = t(&[c], &vec![0.0; c]);
            let m = t(&[c], &vec![0.0; c]);
            let v = t(&[c], &vec![1.0; c]);

            let r = std::panic::catch_unwind(|| batch_norm2d(&x, &s, &b, &m, &v, 1e-5));
            assert!(r.is_ok(), "shape {shape:?} must not panic");
            let out = r
                .expect("no panic")
                .expect("a degenerate shape is not an error");
            assert_eq!(out.shape(), &shape[..], "shape {shape:?}");
            assert_eq!(out.numel(), 0, "shape {shape:?}");
        }
    }

    /// Control: a real tensor still normalises correctly, so the guard above is not simply
    /// short-circuiting everything.
    #[test]
    fn normal_batch_norm_matches_the_formula() {
        let x = t(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let scale = t(&[2], &[1.0, 1.0]);
        let bias = t(&[2], &[0.0, 0.0]);
        let mean = t(&[2], &[0.0, 0.0]);
        let var = t(&[2], &[1.0, 1.0]);
        let out = batch_norm2d(&x, &scale, &bias, &mean, &var, 0.0).expect("batch norm");
        // gamma = 1, beta = 0, mean = 0, var = 1 -> the identity.
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
