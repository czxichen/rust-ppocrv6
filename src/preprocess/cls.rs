use crate::error::Result;
use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView, imageops::FilterType};

pub struct ClsPreprocess {
    pub target_h: u32,
    pub target_w: u32,
}

impl Default for ClsPreprocess {
    fn default() -> Self {
        Self {
            target_h: 48,
            target_w: 192,
        }
    }
}

impl ClsPreprocess {
    pub fn new(target_h: u32, target_w: u32) -> Self {
        Self { target_h, target_w }
    }

    /// Resize a crop to fit `target_h x target_w` and right-pad it with 0.0 (grey), matching
    /// PaddleOCR's `Pad` after normalisation. Channel order is RGB; see the note on
    /// [`crate::preprocess::det::DetPreprocess::process`] for the reference's `img_mode: BGR`
    /// and why that divergence is currently pinned rather than fixed.
    pub fn process(&self, img: &DynamicImage) -> Result<Tensor> {
        let (orig_w, orig_h) = img.dimensions();
        let orig_h = orig_h.max(1);
        let orig_w = orig_w.max(1);

        // Both fields are `pub` and settable through `new`, so they are user input. A zero
        // target height produced a `[1, 3, 0, w]` tensor that then divided by zero deep
        // inside the model, and a zero target width produced an *empty* buffer that the
        // pixel loop below then indexed out of bounds on. `RecPreprocess` guards both of
        // these; this file did not.
        let target_h = self.target_h.max(1);
        let target_w = self.target_w.max(1);

        let ratio = orig_w as f32 / orig_h as f32;
        let resized_w = ((target_h as f32 * ratio).ceil() as u32)
            .min(target_w)
            .max(1);

        let resized = img.resize_exact(resized_w, target_h, FilterType::Triangle);
        let rgb_img = resized.to_rgb8();
        let rgb_raw = rgb_img.as_raw();

        let h = target_h as usize;
        let w = target_w as usize;
        let hw = h * w;

        // Padding value is 0.0 *after* normalisation, i.e. mid-grey (127.5). Not -1.0 and
        // not black -- a classic off-by-constant here.
        let mut tensor = Tensor::empty(&[1, 3, h, w]);
        let out = tensor.as_mut_slice();
        out[..3 * hw].fill(0.0);

        let inv_127_5 = 1.0 / 127.5f32;
        let rw = resized_w as usize;
        debug_assert!(
            rgb_raw.len() >= 3 * h * rw,
            "resized RGB buffer ({} bytes) smaller than expected ({})",
            rgb_raw.len(),
            3 * h * rw
        );
        for y in 0..h {
            for x in 0..rw {
                let src_idx = (y * rw + x) * 3;
                let dst_idx = y * w + x;
                out[dst_idx] = (rgb_raw[src_idx] as f32 * inv_127_5) - 1.0;
                out[hw + dst_idx] = (rgb_raw[src_idx + 1] as f32 * inv_127_5) - 1.0;
                out[2 * hw + dst_idx] = (rgb_raw[src_idx + 2] as f32 * inv_127_5) - 1.0;
            }
        }

        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn solid(w: u32, h: u32, c: Rgb<u8>) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, c))
    }

    /// `ClsPreprocess::new(48, 0).process(..)` used to write past the end of an empty
    /// `Vec`: `target_w = 0` made `hw = 0`, but `resized_w` was clamped to 1, so the pixel
    /// loop still ran once and indexed `tensor_data[0]`. Both fields are `pub`, so this was
    /// reachable straight from the public API. `RecPreprocess` has had this guard (and its
    /// tests) for a while; this file had neither a guard nor any test.
    #[test]
    fn zero_target_width_does_not_panic() {
        let p = ClsPreprocess::new(48, 0);
        let t = p
            .process(&solid(100, 40, Rgb([255, 255, 255])))
            .expect("a degenerate target width must not panic");
        // Clamped to 1x1 rather than rejected, mirroring RecPreprocess.
        assert_eq!(t.shape(), &[1, 3, 48, 1]);
    }

    #[test]
    fn zero_target_height_does_not_panic() {
        let p = ClsPreprocess::new(0, 192);
        let t = p
            .process(&solid(100, 40, Rgb([255, 255, 255])))
            .expect("a degenerate target height must not panic");
        assert_eq!(t.shape(), &[1, 3, 1, 192]);
    }

    #[test]
    fn default_target_is_48x192() {
        let p = ClsPreprocess::default();
        let t = p
            .process(&solid(400, 40, Rgb([128, 64, 32])))
            .expect("process ok");
        // The canvas is always the full target, whatever the input aspect ratio.
        assert_eq!(t.shape(), &[1, 3, 48, 192]);
    }

    /// The *content* width is `min(ceil(target_h * aspect), target_w)`; columns past it are
    /// padding. The tensor shape stays `target_h x target_w` either way, so this asserts on
    /// where the padding starts rather than on the shape.
    #[test]
    fn content_width_follows_aspect_and_leaves_padding() {
        let p = ClsPreprocess::default();
        let hw = 48 * 192;
        let denorm = |v: f32| (v + 1.0) * 127.5;

        // aspect 2.5 (100x40) -> ceil(120) = 120 columns of content, 72 of padding.
        let narrow = p
            .process(&solid(100, 40, Rgb([128, 64, 32])))
            .expect("process ok");
        assert_eq!(narrow.shape(), &[1, 3, 48, 192]);
        let out = narrow.as_slice();
        assert!((denorm(out[0]) - 128.0).abs() < 1.0, "col 0 is content");
        assert!((denorm(out[119]) - 128.0).abs() < 1.0, "col 119 is content");
        assert_eq!(out[120], 0.0, "col 120 is the first padded column");
        assert_eq!(out[191], 0.0, "col 191 is padding");

        // aspect 50 (2000x40) -> 48*50 = 2400, clamped to 192, so there is no padding at all.
        let wide = p
            .process(&solid(2000, 40, Rgb([128, 64, 32])))
            .expect("process ok");
        assert!(
            wide.as_slice()[hw - 1] != 0.0,
            "a clamped fill must not be padding"
        );
    }

    /// Padding is 0.0 *after* normalisation, i.e. mid-grey (127.5) -- not -1.0 and not black.
    #[test]
    fn padding_is_zero_after_normalisation() {
        let p = ClsPreprocess::default();
        let t = p
            .process(&solid(100, 40, Rgb([0, 0, 0])))
            .expect("process ok");
        let out = t.as_slice();
        let hw = 48 * 192;
        // Past the 120-column content width, in all three planes.
        assert_eq!(out[hw - 1], 0.0, "plane 0");
        assert_eq!(out[hw + hw - 1], 0.0, "plane 1");
        assert_eq!(out[2 * hw + hw - 1], 0.0, "plane 2");
        // A black source pixel normalises to exactly -1.0, so this also proves the real
        // content path ran and was not overwritten by the padding fill.
        assert!((out[0] - (-1.0)).abs() < 1e-6, "got {}", out[0]);
    }

    #[test]
    fn channel_order_is_rgb() {
        let p = ClsPreprocess::default();
        let t = p
            .process(&solid(2000, 40, Rgb([255, 0, 0])))
            .expect("process ok");
        let out = t.as_slice();
        // (x / 127.5) - 1.0 with x = 255 -> 1.0 in channel 0, -1.0 in channels 1 and 2.
        assert!((out[0] - 1.0).abs() < 1e-6, "channel 0: {}", out[0]);
        assert!((out[48 * 192] + 1.0).abs() < 1e-6, "channel 1");
        assert!((out[2 * 48 * 192] + 1.0).abs() < 1e-6, "channel 2");
    }
}
