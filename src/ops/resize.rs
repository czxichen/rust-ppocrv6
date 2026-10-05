use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

pub fn resize_nearest_2d(input: &Tensor, target_h: usize, target_w: usize) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "ResizeNearest".into(),
            msg: "Input must be 4D tensor".into(),
        });
    }

    let [batch_size, channels, in_h, in_w] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];

    // A zero-sized spatial extent is legal in ONNX. It used to underflow `in_h - 1` into
    // `usize::MAX` in the index map below, which masks to 0 and then indexes an empty row.
    if in_h == 0 || in_w == 0 || target_h == 0 || target_w == 0 || batch_size == 0 || channels == 0
    {
        return Ok(Tensor::empty(&[batch_size, channels, target_h, target_w]));
    }

    let mut output = Tensor::empty(&[batch_size, channels, target_h, target_w]);
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let scale_h = target_h as f32 / in_h as f32;
    let scale_w = target_w as f32 / in_w as f32;
    let out_spatial = target_h * target_w;
    let in_spatial = in_h * in_w;

    let ih_map: Vec<usize> = (0..target_h)
        .map(|oh| ((oh as f32 / scale_h).floor() as usize).min(in_h - 1))
        .collect();
    let iw_map: Vec<usize> = (0..target_w)
        .map(|ow| ((ow as f32 / scale_w).floor() as usize).min(in_w - 1))
        .collect();

    let total_planes = batch_size * channels;

    let run_plane = |(c_idx, out_chunk): (usize, &mut [f32])| {
        let in_c_offset = c_idx * in_spatial;

        for (oh, &ih) in ih_map.iter().enumerate() {
            let in_row = &in_slice[in_c_offset + ih * in_w..in_c_offset + (ih + 1) * in_w];
            let out_row = &mut out_chunk[oh * target_w..(oh + 1) * target_w];

            for (ow, &iw) in iw_map.iter().enumerate() {
                out_row[ow] = in_row[iw];
            }
        }
    };

    if crate::tensor::should_parallelize(total_planes, 8) {
        out_slice
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_plane);
    } else {
        out_slice
            .chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_plane);
    }

    Ok(output)
}

pub fn resize_bilinear_2d(
    input: &Tensor,
    target_h: usize,
    target_w: usize,
    align_corners: bool,
) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(OcrError::OperatorError {
            op: "ResizeBilinear".into(),
            msg: "Input must be 4D tensor".into(),
        });
    }

    let [batch_size, channels, in_h, in_w] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];

    // See `resize_nearest_2d`. Here the failure mode is worse than an underflow: the sample
    // loop does `.clamp(0, in_w as isize - 1)`, and `clamp` panics when the bounds are
    // inverted, which is exactly what `in_w == 0` produces.
    if in_h == 0 || in_w == 0 || target_h == 0 || target_w == 0 || batch_size == 0 || channels == 0
    {
        return Ok(Tensor::empty(&[batch_size, channels, target_h, target_w]));
    }

    let mut output = Tensor::empty(&[batch_size, channels, target_h, target_w]);
    let in_cont_holder;
    let in_slice = if input.is_contiguous() {
        input.as_slice()
    } else {
        in_cont_holder = input.to_contiguous();
        in_cont_holder.as_slice()
    };
    let out_slice = output.as_mut_slice();

    let out_spatial = target_h * target_w;
    let in_spatial = in_h * in_w;

    // Precompute horizontal interpolation params (constant across all channels and heights)
    let x_params: Vec<(usize, usize, f32, f32)> = (0..target_w)
        .map(|ow| {
            let src_x = if align_corners {
                if target_w > 1 {
                    ow as f32 * (in_w - 1) as f32 / (target_w - 1) as f32
                } else {
                    0.0
                }
            } else {
                (ow as f32 + 0.5) * (in_w as f32 / target_w as f32) - 0.5
            };
            let x0 = (src_x.floor() as isize).clamp(0, in_w as isize - 1) as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let dx = (src_x - x0 as f32).clamp(0.0, 1.0);
            (x0, x1, 1.0 - dx, dx)
        })
        .collect();

    // Precompute vertical interpolation params (constant across all channels and widths)
    let y_params: Vec<(usize, usize, f32, f32)> = (0..target_h)
        .map(|oh| {
            let src_y = if align_corners {
                if target_h > 1 {
                    oh as f32 * (in_h - 1) as f32 / (target_h - 1) as f32
                } else {
                    0.0
                }
            } else {
                (oh as f32 + 0.5) * (in_h as f32 / target_h as f32) - 0.5
            };
            let y0 = (src_y.floor() as isize).clamp(0, in_h as isize - 1) as usize;
            let y1 = (y0 + 1).min(in_h - 1);
            let dy = (src_y - y0 as f32).clamp(0.0, 1.0);
            (y0, y1, 1.0 - dy, dy)
        })
        .collect();

    let total_planes = batch_size * channels;

    let run_plane = |(c_idx, out_chunk): (usize, &mut [f32])| {
        let in_c_offset = c_idx * in_spatial;

        for (oh, &(y0, y1, w_y0, w_y1)) in y_params.iter().enumerate() {
            let row0 = &in_slice[in_c_offset + y0 * in_w..in_c_offset + (y0 + 1) * in_w];
            let row1 = &in_slice[in_c_offset + y1 * in_w..in_c_offset + (y1 + 1) * in_w];
            let out_row = &mut out_chunk[oh * target_w..(oh + 1) * target_w];

            for (ow, &(x0, x1, w_x0, w_x1)) in x_params.iter().enumerate() {
                let p00 = row0[x0];
                let p01 = row0[x1];
                let p10 = row1[x0];
                let p11 = row1[x1];

                out_row[ow] = w_y0 * (w_x0 * p00 + w_x1 * p01) + w_y1 * (w_x0 * p10 + w_x1 * p11);
            }
        }
    };

    if crate::tensor::should_parallelize(total_planes, 8) {
        out_slice
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_plane);
    } else {
        out_slice
            .chunks_mut(out_spatial)
            .enumerate()
            .for_each(run_plane);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
    }

    /// f64 re-implementation of the half-pixel / align-corners bilinear sampler with
    /// edge clamping, used as the reference.
    fn bilinear_reference(
        src: &[f32],
        in_h: usize,
        in_w: usize,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
    ) -> Vec<f64> {
        let coord = |o: usize, target: usize, input: usize| -> f64 {
            if align_corners {
                if target > 1 {
                    o as f64 * (input - 1) as f64 / (target - 1) as f64
                } else {
                    0.0
                }
            } else {
                (o as f64 + 0.5) * (input as f64 / target as f64) - 0.5
            }
        };
        let mut out = vec![0.0f64; target_h * target_w];
        for oh in 0..target_h {
            let sy = coord(oh, target_h, in_h);
            let y0 = (sy.floor().max(0.0) as usize).min(in_h - 1);
            let y1 = (y0 + 1).min(in_h - 1);
            let dy = (sy - y0 as f64).clamp(0.0, 1.0);
            for ow in 0..target_w {
                let sx = coord(ow, target_w, in_w);
                let x0 = (sx.floor().max(0.0) as usize).min(in_w - 1);
                let x1 = (x0 + 1).min(in_w - 1);
                let dx = (sx - x0 as f64).clamp(0.0, 1.0);
                let p00 = src[y0 * in_w + x0] as f64;
                let p01 = src[y0 * in_w + x1] as f64;
                let p10 = src[y1 * in_w + x0] as f64;
                let p11 = src[y1 * in_w + x1] as f64;
                out[oh * target_w + ow] =
                    (1.0 - dy) * ((1.0 - dx) * p00 + dx * p01) + dy * ((1.0 - dx) * p10 + dx * p11);
            }
        }
        out
    }

    #[test]
    fn resize_nearest_upscale_replicates_pixels() {
        let src = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let out = resize_nearest_2d(&src, 4, 4).expect("resize");
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
        #[rustfmt::skip]
        let want = [
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(out.as_slice(), &want);
    }

    #[test]
    fn resize_nearest_downscale_picks_the_cell_origin() {
        // 4x4 -> 2x2: each output pixel samples the top-left corner of its 2x2 cell.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let src = tensor(&[1, 1, 4, 4], &data);
        let out = resize_nearest_2d(&src, 2, 2).expect("resize");
        assert_eq!(out.as_slice(), &[0.0, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn resize_bilinear_same_size_is_an_identity() {
        let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.31 - 3.0).collect();
        let src = tensor(&[2, 1, 3, 4], &data);
        for align_corners in [true, false] {
            let out = resize_bilinear_2d(&src, 3, 4, align_corners).expect("resize");
            assert_eq!(out.shape(), &[2, 1, 3, 4]);
            for (i, (&got, &want)) in out.as_slice().iter().zip(data.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "align_corners={align_corners} index {i}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn resize_bilinear_matches_f64_reference() {
        let (in_h, in_w) = (3usize, 4usize);
        let data: Vec<f32> = (0..in_h * in_w)
            .map(|i| ((i as f32) * 0.47).sin() * 5.0)
            .collect();
        let src = tensor(&[1, 1, in_h, in_w], &data);

        for (th, tw) in [(5usize, 7usize), (2, 3), (6, 6)] {
            for align_corners in [true, false] {
                let out = resize_bilinear_2d(&src, th, tw, align_corners).expect("resize");
                let want = bilinear_reference(&data, in_h, in_w, th, tw, align_corners);
                for (i, (&got, &w)) in out.as_slice().iter().zip(want.iter()).enumerate() {
                    assert!(
                        (got as f64 - w).abs() < 1e-5,
                        "{th}x{tw} align_corners={align_corners} index {i}: {got} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn resize_bilinear_keeps_constant_images_constant() {
        // Edge clamping must not introduce ringing at the borders.
        let src = tensor(&[1, 2, 3, 3], &[7.5f32; 18]);
        let out = resize_bilinear_2d(&src, 9, 9, false).expect("resize");
        for (i, &v) in out.as_slice().iter().enumerate() {
            assert!((v - 7.5).abs() < 1e-5, "index {i}: {v} should stay 7.5");
        }
    }

    #[test]
    fn resize_rejects_non_4d_input() {
        let t = tensor(&[2, 3], &[0.0; 6]);
        assert!(resize_nearest_2d(&t, 4, 4).is_err());
        assert!(resize_bilinear_2d(&t, 4, 4, false).is_err());
    }
    /// A zero-sized spatial extent is legal ONNX. Nearest used to underflow `in_h - 1` into
    /// `usize::MAX`; bilinear's `.clamp(0, in_w as isize - 1)` panics outright on inverted
    /// bounds, which is exactly what `in_w == 0` produces.
    #[test]
    fn zero_sized_spatial_extent_does_not_panic_or_underflow() {
        let cases: [(Vec<usize>, usize, usize); 4] = [
            (vec![1, 1, 0, 4], 2, 2),
            (vec![1, 1, 4, 0], 2, 2),
            (vec![1, 1, 0, 0], 0, 0),
            (vec![0, 1, 4, 4], 2, 2),
        ];
        for (shape, th, tw) in cases {
            let n: usize = shape.iter().product();
            let x = tensor(&shape, &vec![1.0; n]);

            let r = std::panic::catch_unwind(|| resize_nearest_2d(&x, th, tw));
            assert!(r.is_ok(), "nearest {shape:?} -> {th}x{tw} must not panic");
            r.expect("no panic").expect("nearest must not error");

            let r = std::panic::catch_unwind(|| resize_bilinear_2d(&x, th, tw, false));
            assert!(r.is_ok(), "bilinear {shape:?} -> {th}x{tw} must not panic");
            r.expect("no panic").expect("bilinear must not error");
        }
    }

    /// A zero target extent is also expressible and must not divide by zero.
    #[test]
    fn zero_target_extent_does_not_panic() {
        let x = tensor(&[1, 1, 4, 4], &[1.0; 16]);
        for (th, tw) in [(0usize, 0usize), (0, 8), (8, 0)] {
            resize_nearest_2d(&x, th, tw).expect("nearest with a zero target");
            resize_bilinear_2d(&x, th, tw, false).expect("bilinear with a zero target");
        }
    }
}
