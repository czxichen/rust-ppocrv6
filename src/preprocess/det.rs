use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use rayon::prelude::*;

pub struct DetPreprocess {
    pub max_side_len: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for DetPreprocess {
    fn default() -> Self {
        Self {
            max_side_len: 960,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

/// Round to nearest, ties to even.
///
/// PaddleOCR does the 32-alignment with `int(round(resize_h / 32) * 32)`, and Python's
/// `round` is round-half-to-**even** — not the round-half-away-from-zero that Rust's
/// `f32::round` implements. The two disagree whenever `v / 32` lands exactly on `.5`, i.e.
/// for every `v % 32 == 16`, which is ~3% of sizes. Matching it here removes an entire
/// class of "the input is one 32-step off from the reference" cases.
fn round_half_to_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Snap `v` to the nearest positive multiple of 32 the way PaddleOCR's
/// `DetResizeForTest.resize_image_type0` does: `max(int(round(v / 32) * 32), 32)`.
///
/// The explicit NaN test folds in the `max(32, ..)` *and* rejects a non-finite result, which
/// a plain `.max()` would let through as a zero-sized tensor.
fn align_to_32(v: u32) -> u32 {
    let aligned = round_half_to_even(v as f32 / 32.0) * 32.0;
    if aligned.is_nan() || aligned < 32.0 {
        32
    } else {
        aligned as u32
    }
}

impl DetPreprocess {
    pub fn new(max_side_len: u32) -> Self {
        Self {
            max_side_len,
            ..Default::default()
        }
    }

    /// Resize to a 32-aligned size bounded by `max_side_len`, then apply
    /// `NormalizeImage(order: "hwc", scale: 1/255, mean, std)`.
    ///
    /// # Channel order — deliberately RGB, pinned by `channel_order_is_rgb`
    ///
    /// The reference config shipped with this model (`models_v6/ppocrv6_det.yml`) says
    /// `DecodeImage: {img_mode: BGR}`, and `DecodeImage` only reverses the channel axis when
    /// `img_mode == 'RGB'`, so the reference's channel 0 is **B**. The `image` crate decodes
    /// to **RGB**, so this pre-processor is a straight channel swap away from the reference
    /// pipeline.
    ///
    /// That is intentional for now, not an oversight: the real behaviour of the ONNX export
    /// has never been A/B'd against the Python reference here, because
    /// `tools/compare_rust_onnx.py` feeds *both* engines a synthetic tensor and therefore
    /// cannot observe pre-processing at all. Flipping it blind would change every detection
    /// result. The test below locks the current choice in place so a future edit cannot do it
    /// by accident, and records the open question.
    ///
    /// The normalisation constants are the ImageNet ones `[0.485, 0.456, 0.406]`, which are
    /// RGB-ordered and are applied to RGB-ordered data here, so this stage is at least
    /// internally consistent.
    pub fn process(&self, img: &DynamicImage) -> Result<(Tensor, f32, f32)> {
        let (orig_w, orig_h) = img.dimensions();
        if orig_w == 0 || orig_h == 0 {
            // `scale_*` would become +/-inf and every box coordinate would come out NaN,
            // which then flows silently into the crop. Fail loudly instead.
            return Err(OcrError::ImageError(format!(
                "cannot run detection on a degenerate image of {orig_w}x{orig_h} pixels"
            )));
        }

        let max_dim = orig_w.max(orig_h);
        let ratio = if max_dim > self.max_side_len {
            self.max_side_len as f32 / max_dim as f32
        } else {
            1.0
        };

        // The reference truncates the scaled side (`int(h * ratio)`) before aligning. This
        // used to round, then round *up* to a 32 multiple.
        let resize_h = align_to_32((orig_h as f32 * ratio) as u32);
        let resize_w = align_to_32((orig_w as f32 * ratio) as u32);

        let scale_h = resize_h as f32 / orig_h as f32;
        let scale_w = resize_w as f32 / orig_w as f32;

        let resized = img.resize_exact(resize_w, resize_h, FilterType::Triangle);
        let rgb_img = resized.to_rgb8();

        let h = resize_h as usize;
        let w = resize_w as usize;
        let hw = h * w;
        let rgb_raw = rgb_img.as_raw();
        debug_assert!(
            rgb_raw.len() >= 3 * hw,
            "resized RGB buffer ({} bytes) smaller than expected ({})",
            rgb_raw.len(),
            3 * hw
        );

        let mut tensor = Tensor::empty(&[1, 3, h, w]);
        let out = tensor.as_mut_slice();
        let src = rgb_raw.as_slice();

        // `(v / (255 * std)) - mean / std`, pre-divided per channel so the pixel loop is a
        // single multiply-add. 3 * 960 * 960 = 2.7M floats, worth fanning out.
        let inv_scale = [
            1.0 / (255.0 * self.std[0]),
            1.0 / (255.0 * self.std[1]),
            1.0 / (255.0 * self.std[2]),
        ];
        let bias = [
            -self.mean[0] / self.std[0],
            -self.mean[1] / self.std[1],
            -self.mean[2] / self.std[2],
        ];

        let (r, rest) = out.split_at_mut(hw);
        let (g, b) = rest.split_at_mut(hw);

        let r_ptr_usize = r.as_mut_ptr() as usize;
        let g_ptr_usize = g.as_mut_ptr() as usize;
        let b_ptr_usize = b.as_mut_ptr() as usize;
        let src_ptr_usize = src.as_ptr() as usize;

        let inv_r = inv_scale[0];
        let inv_g = inv_scale[1];
        let inv_b = inv_scale[2];
        let bias_r = bias[0];
        let bias_g = bias[1];
        let bias_b = bias[2];

        let num_threads = crate::tensor::parallel_degree(3 * hw, 500_000);
        if num_threads > 1 {
            let chunk_size = hw.div_ceil(num_threads).max(1024);
            let num_chunks = hw.div_ceil(chunk_size);

            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let p_start = chunk_idx * chunk_size;
                let p_end = (p_start + chunk_size).min(hw);
                let p_len = p_end - p_start;

                unsafe {
                    let r_p = (r_ptr_usize as *mut f32).add(p_start);
                    let g_p = (g_ptr_usize as *mut f32).add(p_start);
                    let b_p = (b_ptr_usize as *mut f32).add(p_start);
                    let s_p = (src_ptr_usize as *const u8).add(p_start * 3);

                    let mut i = 0;
                    while i + 8 <= p_len {
                        let sp = s_p.add(i * 3);
                        for k in 0..8 {
                            *r_p.add(i + k) = (*sp.add(k * 3) as f32) * inv_r + bias_r;
                            *g_p.add(i + k) = (*sp.add(k * 3 + 1) as f32) * inv_g + bias_g;
                            *b_p.add(i + k) = (*sp.add(k * 3 + 2) as f32) * inv_b + bias_b;
                        }
                        i += 8;
                    }
                    while i < p_len {
                        let sp = s_p.add(i * 3);
                        *r_p.add(i) = (*sp.add(0) as f32) * inv_r + bias_r;
                        *g_p.add(i) = (*sp.add(1) as f32) * inv_g + bias_g;
                        *b_p.add(i) = (*sp.add(2) as f32) * inv_b + bias_b;
                        i += 1;
                    }
                }
            });
        } else {
            let r_p = r.as_mut_ptr();
            let g_p = g.as_mut_ptr();
            let b_p = b.as_mut_ptr();
            let s_p = src.as_ptr();

            unsafe {
                let mut i = 0;
                while i + 8 <= hw {
                    let sp = s_p.add(i * 3);
                    for k in 0..8 {
                        *r_p.add(i + k) = (*sp.add(k * 3) as f32) * inv_r + bias_r;
                        *g_p.add(i + k) = (*sp.add(k * 3 + 1) as f32) * inv_g + bias_g;
                        *b_p.add(i + k) = (*sp.add(k * 3 + 2) as f32) * inv_b + bias_b;
                    }
                    i += 8;
                }
                while i < hw {
                    let sp = s_p.add(i * 3);
                    *r_p.add(i) = (*sp.add(0) as f32) * inv_r + bias_r;
                    *g_p.add(i) = (*sp.add(1) as f32) * inv_g + bias_g;
                    *b_p.add(i) = (*sp.add(2) as f32) * inv_b + bias_b;
                    i += 1;
                }
            }
        }

        Ok((tensor, scale_h, scale_w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn solid(w: u32, h: u32, c: Rgb<u8>) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, c))
    }

    /// Undo `(v / (255 * std)) - mean / std` for one plane's first element.
    fn denorm(p: &DetPreprocess, out: &[f32], hw: usize, ch: usize) -> f32 {
        out[ch * hw] * (255.0 * p.std[ch]) + p.mean[ch] * 255.0
    }

    /// 32-alignment must match PaddleOCR's `max(int(round(v / 32) * 32), 32)`.
    ///
    /// `align_to_32` used to be `(v + 31) / 32 * 32`, i.e. round *up*. For the repo's own
    /// `tests/data/test_ocr.png` (400x100, ratio 1.0) that produced a 128x416 input where the
    /// reference produces 96x384 — 1.44x the pixels and a 19% different aspect ratio, which
    /// shifts the DB feature-map stride, the probabilities, the box-score masks and therefore
    /// every box coordinate.
    #[test]
    fn align_to_32_matches_paddleocr_round_to_nearest() {
        // Rounds down: 100/32 = 3.125 -> 96, 200/32 = 6.25 -> 192.
        assert_eq!(align_to_32(100), 96);
        assert_eq!(align_to_32(200), 192);
        // Exact: 128/32 = 4, 960/32 = 30.
        assert_eq!(align_to_32(128), 128);
        assert_eq!(align_to_32(960), 960);
        // Ties go to even, matching Python: 400/32 = 12.5 -> 12 -> 384, 48/32 = 1.5 -> 2 -> 64,
        // 80/32 = 2.5 -> 2 -> 64. A naive round-half-away-from-zero gives 416, 64 and 96.
        assert_eq!(align_to_32(400), 384);
        assert_eq!(align_to_32(48), 64);
        assert_eq!(align_to_32(80), 64);
        // The floor of 32 always applies.
        assert_eq!(align_to_32(0), 32);
        assert_eq!(align_to_32(1), 32);
        assert_eq!(align_to_32(16), 32);
    }

    #[test]
    fn letterbox_rounds_rather_than_ceils_to_a_32_multiple() {
        let p = DetPreprocess::default();
        let (t, scale_h, scale_w) = p
            .process(&solid(400, 100, Rgb([128, 64, 32])))
            .expect("process ok");
        // 400x100 at ratio 1.0 must land on 96x384 (H x W), the reference size. The old
        // ceil-to-32 gave 128x416.
        assert_eq!(t.shape(), &[1, 3, 96, 384]);
        assert!((scale_w - 384.0 / 400.0).abs() < 1e-6, "scale_w={scale_w}");
        assert!((scale_h - 96.0 / 100.0).abs() < 1e-6, "scale_h={scale_h}");
    }

    #[test]
    fn below_max_side_len_keeps_the_multiple_of_32_alignment() {
        let p = DetPreprocess::default();
        let (t, _, _) = p
            .process(&solid(128, 96, Rgb([10, 20, 30])))
            .expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 96, 128]);
    }

    #[test]
    fn long_side_is_capped_and_both_sides_stay_32_aligned() {
        let p = DetPreprocess::default();
        // 2000x1000: ratio = 960/2000 = 0.48 -> 480x960.
        let (t, scale_h, scale_w) = p
            .process(&solid(2000, 1000, Rgb([10, 20, 30])))
            .expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 480, 960]);
        assert_eq!(t.shape()[2] % 32, 0);
        assert_eq!(t.shape()[3] % 32, 0);
        assert!((scale_h - 480.0 / 1000.0).abs() < 1e-6, "scale_h={scale_h}");
        assert!((scale_w - 960.0 / 2000.0).abs() < 1e-6, "scale_w={scale_w}");
    }

    #[test]
    fn a_custom_max_side_len_is_respected() {
        let p = DetPreprocess::new(320);
        let (t, _, _) = p
            .process(&solid(1000, 500, Rgb([10, 20, 30])))
            .expect("process ok");
        // ratio 0.32 -> 160x320, both already 32-aligned.
        assert_eq!(t.shape(), &[1, 3, 160, 320]);
    }

    #[test]
    fn normalization_uses_the_configured_mean_and_std() {
        let p = DetPreprocess::default();
        let (t, _, _) = p
            .process(&solid(32, 32, Rgb([0, 0, 0])))
            .expect("process ok");
        let out = t.as_slice();
        let hw = 32 * 32;
        for c in 0..3 {
            let expected = -(p.mean[c] / p.std[c]);
            assert!(
                (out[c * hw] - expected).abs() < 1e-6,
                "channel {c}: got {} want {expected}",
                out[c * hw]
            );
        }
    }

    /// Pins the deliberate RGB channel order (see the note on `process`).
    ///
    /// The reference configs ask for BGR, so a red image would saturate channel 2 there. If
    /// someone "fixes" this to match, this test fails and forces the A/B discussion instead of
    /// silently changing every detection result.
    #[test]
    fn channel_order_is_rgb() {
        let p = DetPreprocess::default();
        let (t, _, _) = p
            .process(&solid(32, 32, Rgb([255, 0, 0])))
            .expect("process ok");
        let out = t.as_slice();
        let hw = 32 * 32;
        assert!(
            (denorm(&p, out, hw, 0) - 255.0).abs() < 1e-3,
            "channel 0 should be R, got {}",
            denorm(&p, out, hw, 0)
        );
        assert!(
            denorm(&p, out, hw, 1).abs() < 1e-3,
            "channel 1 should be G, got {}",
            denorm(&p, out, hw, 1)
        );
        assert!(
            denorm(&p, out, hw, 2).abs() < 1e-3,
            "channel 2 should be B, got {}",
            denorm(&p, out, hw, 2)
        );
    }

    #[test]
    fn zero_sized_image_is_an_error_not_a_nan_scale() {
        let p = DetPreprocess::default();
        let err = p
            .process(&DynamicImage::ImageRgb8(image::RgbImage::new(0, 0)))
            .expect_err("zero-sized image must be rejected");
        assert!(
            matches!(err, OcrError::ImageError(_)),
            "expected ImageError, got {err:?}"
        );
    }
}
