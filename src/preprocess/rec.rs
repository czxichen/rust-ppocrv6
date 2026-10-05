use crate::error::Result;
use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView, imageops::FilterType};

pub struct RecPreprocess {
    pub target_h: u32,
    pub max_w: u32,
}

impl Default for RecPreprocess {
    fn default() -> Self {
        Self {
            target_h: 48,
            max_w: 960,
        }
    }
}

impl RecPreprocess {
    pub fn new(target_h: u32) -> Self {
        Self {
            target_h,
            ..Default::default()
        }
    }

    pub fn with_max_w(target_h: u32, max_w: u32) -> Self {
        Self { target_h, max_w }
    }

    /// Resize a crop to `target_h` while preserving aspect ratio, without padding.
    ///
    /// # Divergence from PaddleOCR — pinned, not fixed
    ///
    /// The reference is `TextRecognizer.resize_norm_img` (PaddleOCR `release/2.7`,
    /// `tools/infer/predict_rec.py`):
    ///
    /// ```python
    /// imgW = int(imgH * max_wh_ratio)          # max_wh_ratio starts at rec_image_shape[2]/imgH
    /// if math.ceil(imgH * ratio) > imgW: resized_w = imgW
    /// else:                                   resized_w = int(math.ceil(imgH * ratio))
    /// padding_im = np.zeros((imgC, imgH, imgW)); padding_im[:, :, 0:resized_w] = resized_image
    /// ```
    ///
    /// Two differences from the above, both currently deliberate:
    ///
    /// 1. **Minimum width.** `max_wh_ratio` is seeded with `rec_image_shape[2] / imgH`, and
    ///    the bundled `models_v6/ppocrv6_rec.yml` sets `RecResizeImg: {image_shape: [3, 48,
    ///    320]}`. So the reference never produces a canvas narrower than 320 px. This stage
    ///    clamps to `[32, max_w]` instead. (The *base* of 48 is correct on both sides — only
    ///    the floor differs.)
    /// 2. **Padding.** The reference always zero-pads on the right to the canvas width; this
    ///    stage emits a tight crop with no padding. That changes the CTC time-step count
    ///    relative to the reference, so a given line can decode differently here.
    ///
    /// Neither has been A/B'd: `tools/compare_rust_onnx.py` feeds both engines a *synthetic*
    /// tensor, so pre-processing is not covered by any harness in this repo. The
    /// `wide_image_respects_max_w` / `extreme_aspect_ratio_saturates_instead_of_overflowing`
    /// tests below pin the current behaviour so a future change has to be deliberate.
    pub fn process(&self, img: &DynamicImage) -> Result<Tensor> {
        let (orig_w, orig_h) = img.dimensions();
        let orig_h = orig_h.max(1);

        // `target_h` is user-controlled (`new` / `with_max_w`); a zero height would make
        // `resize_exact` produce an empty buffer and the pixel loop below index out of
        // bounds, so clamp it to at least one pixel.
        let target_h = self.target_h.max(1);

        // The lower bound (32) must never exceed the upper bound, otherwise
        // `clamp(32, max_w)` panics (reverse interval). `max_w` is user-controlled too.
        let max_w = self.max_w.max(32);

        let aspect_ratio = orig_w as f32 / orig_h as f32;
        // `as u32` saturates on overflow/NaN, so this is already bounded.
        let target_w = ((target_h as f32 * aspect_ratio).ceil() as u32).clamp(32, max_w);

        let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
        let rgb_img = resized.to_rgb8();

        let h = target_h as usize;
        let w = target_w as usize;
        let hw = h * w;
        let rgb_raw = rgb_img.as_raw();
        debug_assert!(
            rgb_raw.len() >= 3 * hw,
            "resized RGB buffer ({} bytes) smaller than expected ({})",
            rgb_raw.len(),
            3 * hw
        );
        let mut tensor_data = crate::tensor::acquire_buffer(3 * hw);
        #[allow(clippy::uninit_vec)]
        unsafe {
            tensor_data.set_len(3 * hw);
        }

        let inv_127_5 = 1.0 / 127.5f32;
        for i in 0..hw {
            let px = i * 3;
            tensor_data[i] = (rgb_raw[px] as f32 * inv_127_5) - 1.0;
            tensor_data[hw + i] = (rgb_raw[px + 1] as f32 * inv_127_5) - 1.0;
            tensor_data[2 * hw + i] = (rgb_raw[px + 2] as f32 * inv_127_5) - 1.0;
        }

        Tensor::from_vec(&[1, 3, h, w], tensor_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([128, 64, 32])))
    }

    #[test]
    fn max_w_below_lower_bound_does_not_panic() {
        // `with_max_w(_, 8)` used to build the reverse interval `clamp(32, 8)` -> panic.
        let p = RecPreprocess::with_max_w(48, 8);
        let t = p.process(&solid(100, 10)).expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 48, 32]);
    }

    #[test]
    fn zero_target_h_is_clamped_to_one() {
        let p = RecPreprocess::with_max_w(0, 960);
        let t = p.process(&solid(64, 64)).expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 1, 32]);
    }

    #[test]
    fn wide_image_respects_max_w() {
        let p = RecPreprocess::with_max_w(48, 200);
        let t = p.process(&solid(1000, 10)).expect("process ok");
        assert_eq!(t.shape()[3], 200);
    }

    #[test]
    fn extreme_aspect_ratio_saturates_instead_of_overflowing() {
        let p = RecPreprocess::with_max_w(48, 960);
        let t = p.process(&solid(65_000, 1)).expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 48, 960]);
    }

    /// Pins the 32-px floor, which is where this stage diverges from the reference's 320
    /// (see the note on `process`). The shipped `rec_image_shape` is `[3, 48, 320]`, so
    /// PaddleOCR would never emit a canvas narrower than 320.
    #[test]
    fn floor_is_32_not_the_reference_320() {
        let p = RecPreprocess::with_max_w(48, 960);
        // aspect 0.5 -> 48 * 0.5 = 24, floored to 32 here; the reference would use 320.
        let t = p.process(&solid(50, 100)).expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 48, 32]);
    }

    /// There is no right padding: the tensor width *is* the resized width, unlike the
    /// reference which zero-pads up to the canvas width. See the note on `process`.
    #[test]
    fn output_is_a_tight_crop_with_no_padding() {
        let p = RecPreprocess::with_max_w(48, 960);
        // A black image: every column must be the normalised black value, with no trailing
        // run of mid-grey (127.5 -> 0.0) padding as the reference would produce.
        let t = p
            .process(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                400,
                40,
                image::Rgb([0, 0, 0]),
            )))
            .expect("process ok");
        assert_eq!(t.shape(), &[1, 3, 48, 480]);
        let out = t.as_slice();
        let last_plane = 2 * 48 * 480;
        for i in 0..480 {
            assert!(
                (out[last_plane + i] - (-1.0)).abs() < 1e-6,
                "column {i} is not content: {}",
                out[last_plane + i]
            );
        }
    }
}
