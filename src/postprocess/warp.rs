use crate::postprocess::polygon::Point;
use image::{DynamicImage, ImageBuffer, Rgb};

/// Perspective transformation and crop of a 4-point quad into an upright rectangular image.
pub fn get_rotate_crop_image(img: &DynamicImage, points: &[Point; 4]) -> DynamicImage {
    match img {
        DynamicImage::ImageRgb8(rgb) => get_rotate_crop_image_rgb(rgb, points),
        other => get_rotate_crop_image_rgb(&other.to_rgb8(), points),
    }
}

/// Perspective transformation and crop of a 4-point quad from a pre-converted RgbImage.
pub fn get_rotate_crop_image_rgb(rgb_img: &image::RgbImage, points: &[Point; 4]) -> DynamicImage {
    let [p0, p1, p2, p3] = *points;

    // Crop dimensions
    let crop_w = {
        let w1 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        let w2 = ((p2[0] - p3[0]).powi(2) + (p2[1] - p3[1]).powi(2)).sqrt();
        w1.max(w2).ceil() as u32
    }
    .max(4);

    let crop_h = {
        let h1 = ((p3[0] - p0[0]).powi(2) + (p3[1] - p0[1]).powi(2)).sqrt();
        let h2 = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        h1.max(h2).ceil() as u32
    }
    .max(4);

    // Compute inverse perspective matrix M_inv
    let src = [p0, p1, p2, p3];
    let dst = [
        [0.0, 0.0],
        [crop_w as f32, 0.0],
        [crop_w as f32, crop_h as f32],
        [0.0, crop_h as f32],
    ];

    let m_inv = match get_perspective_transform(&dst, &src) {
        Some(m) => m,
        None => {
            // Degenerate quad (collinear or coincident corners): the homography is
            // singular. Sampling through a meaningless matrix produced NaN coordinates and
            // an all-black crop, so fall back to the quad's axis-aligned bounding box.
            return crop_axis_aligned(rgb_img, &src);
        }
    };

    let (orig_w, orig_h) = (rgb_img.width(), rgb_img.height());
    let raw_src = rgb_img.as_raw();

    let mut out_data = vec![0u8; crop_w as usize * crop_h as usize * 3];

    use rayon::prelude::*;
    let row_len = crop_w as usize * 3;
    let run_row = |(y, row): (usize, &mut [u8])| {
        let y_f = y as f32;
        let m21_y = m_inv[2][1] * y_f + m_inv[2][2];
        let m01_y = m_inv[0][1] * y_f + m_inv[0][2];
        let m11_y = m_inv[1][1] * y_f + m_inv[1][2];

        for x in 0..crop_w as usize {
            let x_f = x as f32;
            let z = m_inv[2][0] * x_f + m21_y;
            let z_inv = if z.abs() > 1e-8 { 1.0 / z } else { 1.0 };
            let src_x = (m_inv[0][0] * x_f + m01_y) * z_inv;
            let src_y = (m_inv[1][0] * x_f + m11_y) * z_inv;

            let pixel = bilinear_sample_raw(raw_src, src_x, src_y, orig_w, orig_h);
            let off = x * 3;
            row[off] = pixel[0];
            row[off + 1] = pixel[1];
            row[off + 2] = pixel[2];
        }
    };

    if crate::tensor::should_parallelize(crop_h as usize, 64) {
        out_data
            .par_chunks_mut(row_len)
            .enumerate()
            .for_each(run_row);
    } else {
        out_data.chunks_mut(row_len).enumerate().for_each(run_row);
    }

    let out_img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(crop_w, crop_h, out_data)
        .unwrap_or_else(|| ImageBuffer::new(crop_w, crop_h));
    let cropped = DynamicImage::ImageRgb8(out_img);

    // If text box is vertical (height >= 1.5 * width), rotate 90 degrees counter-clockwise
    // matching PaddleOCR's `np.rot90` (which rotates 90° counter-clockwise, equivalent to
    // `rotate270()` in the `image` crate) to turn top-to-bottom text into left-to-right text.
    if crop_h as f32 >= crop_w as f32 * 1.5 {
        cropped.rotate270()
    } else {
        cropped
    }
}

#[inline(always)]
fn bilinear_sample_raw(src: &[u8], x: f32, y: f32, width: u32, height: u32) -> [u8; 3] {
    if width == 0 || height == 0 || src.len() < 3 {
        return [0, 0, 0];
    }
    if x < 0.0 || x >= width as f32 - 1.0 || y < 0.0 || y >= height as f32 - 1.0 {
        let cx = (x.round() as i32).clamp(0, width as i32 - 1) as usize;
        let cy = (y.round() as i32).clamp(0, height as i32 - 1) as usize;
        let off = (cy * width as usize + cx) * 3;
        return [src[off], src[off + 1], src[off + 2]];
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width as usize - 1);
    let y1 = (y0 + 1).min(height as usize - 1);

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let off00 = (y0 * width as usize + x0) * 3;
    let off01 = (y0 * width as usize + x1) * 3;
    let off10 = (y1 * width as usize + x0) * 3;
    let off11 = (y1 * width as usize + x1) * 3;

    let w00 = (1.0 - dx) * (1.0 - dy);
    let w01 = dx * (1.0 - dy);
    let w10 = (1.0 - dx) * dy;
    let w11 = dx * dy;

    let mut rgb = [0u8; 3];
    for c in 0..3 {
        let v = w00 * src[off00 + c] as f32
            + w01 * src[off01 + c] as f32
            + w10 * src[off10 + c] as f32
            + w11 * src[off11 + c] as f32;
        rgb[c] = v.clamp(0.0, 255.0).round() as u8;
    }
    rgb
}

/// Compute the 3x3 perspective transform mapping `src`'s 4 points onto `dst`'s 4 points.
///
/// Returns `None` when the quad is degenerate (collinear or coincident corners), in which
/// case the linear system is singular and has no meaningful solution.
fn get_perspective_transform(src: &[Point; 4], dst: &[Point; 4]) -> Option<[[f32; 3]; 3]> {
    // Solve 8 linear equations for perspective projection matrix
    // [x', y', 1]^T ~ M * [x, y, 1]^T
    let mut a = [[0.0f32; 8]; 8];
    let mut b = [0.0f32; 8];

    for i in 0..4 {
        let sx = src[i][0];
        let sy = src[i][1];
        let dx = dst[i][0];
        let dy = dst[i][1];

        // row 2*i
        a[2 * i][0] = sx;
        a[2 * i][1] = sy;
        a[2 * i][2] = 1.0;
        a[2 * i][6] = -sx * dx;
        a[2 * i][7] = -sy * dx;
        b[2 * i] = dx;

        // row 2*i + 1
        a[2 * i + 1][3] = sx;
        a[2 * i + 1][4] = sy;
        a[2 * i + 1][5] = 1.0;
        a[2 * i + 1][6] = -sx * dy;
        a[2 * i + 1][7] = -sy * dy;
        b[2 * i + 1] = dy;
    }

    let h = solve_linear_8x8(&mut a, &mut b)?;

    Some([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
}

/// Axis-aligned bounding-box crop, used as the fallback for a degenerate quad.
fn crop_axis_aligned(rgb_img: &image::RgbImage, points: &[Point; 4]) -> DynamicImage {
    let (img_w, img_h) = (rgb_img.width(), rgb_img.height());
    if img_w == 0 || img_h == 0 {
        return DynamicImage::new_rgb8(4, 4);
    }

    let min_x = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_x = points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        // All-NaN quad: nothing sensible to crop.
        return DynamicImage::new_rgb8(4, 4);
    }

    let x0 = min_x.max(0.0).floor().min(img_w as f32 - 1.0) as u32;
    let y0 = min_y.max(0.0).floor().min(img_h as f32 - 1.0) as u32;
    let cw = (max_x.ceil() - x0 as f32).max(1.0).min((img_w - x0) as f32) as u32;
    let ch = (max_y.ceil() - y0 as f32).max(1.0).min((img_h - y0) as f32) as u32;

    DynamicImage::ImageRgb8(
        image::imageops::crop_imm(rgb_img, x0, y0, cw.max(1), ch.max(1)).to_image(),
    )
}

/// Gaussian elimination for an 8x8 system.
///
/// Returns `None` for a singular system (zero pivot) instead of silently continuing with a
/// partially eliminated matrix and returning garbage.
#[allow(clippy::needless_range_loop)]
fn solve_linear_8x8(a: &mut [[f32; 8]; 8], b: &mut [f32; 8]) -> Option<[f32; 8]> {
    for i in 0..8 {
        // Pivot
        let mut max_row = i;
        for k in i + 1..8 {
            if a[k][i].abs() > a[max_row][i].abs() {
                max_row = k;
            }
        }
        a.swap(i, max_row);
        b.swap(i, max_row);

        let pivot = a[i][i];
        if pivot.abs() < 1e-8 {
            return None;
        }

        for j in i..8 {
            a[i][j] /= pivot;
        }
        b[i] /= pivot;

        for k in 0..8 {
            if k != i {
                let factor = a[k][i];
                for j in i..8 {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
    }

    Some(*b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> image::RgbImage {
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
    }

    #[test]
    fn regular_quad_crops_expected_size() {
        let img = gradient(64, 64);
        let pts: [Point; 4] = [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]];
        let out = get_rotate_crop_image_rgb(&img, &pts);
        assert_eq!(out.width(), 20);
        assert_eq!(out.height(), 10);
    }

    #[test]
    fn degenerate_collinear_quad_falls_back_without_panicking() {
        // All four points on the same line -> singular homography.
        let img = gradient(32, 32);
        let pts: [Point; 4] = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]];
        let out = get_rotate_crop_image_rgb(&img, &pts);
        assert!(out.width() >= 1 && out.height() >= 1);

        // Duplicate corners -> also singular.
        let dup: [Point; 4] = [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]];
        let out2 = get_rotate_crop_image_rgb(&img, &dup);
        assert!(out2.width() >= 1 && out2.height() >= 1);

        // NaN corners must not panic either.
        let nan: [Point; 4] = [
            [f32::NAN, f32::NAN],
            [f32::NAN, f32::NAN],
            [f32::NAN, f32::NAN],
            [f32::NAN, f32::NAN],
        ];
        let out3 = get_rotate_crop_image_rgb(&img, &nan);
        assert!(out3.width() >= 1 && out3.height() >= 1);
    }

    #[test]
    fn solve_linear_8x8_reports_singular_system() {
        let mut a = [[0.0f32; 8]; 8];
        let mut b = [0.0f32; 8];
        assert!(solve_linear_8x8(&mut a, &mut b).is_none());

        let mut identity = [[0.0f32; 8]; 8];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let mut rhs = [1.0f32; 8];
        let sol = solve_linear_8x8(&mut identity, &mut rhs).expect("identity is invertible");
        assert!(sol.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn vertical_box_rotates_counter_clockwise() {
        // Create an image where top-left is red, bottom-left is blue.
        // For a vertical box (height >> width), rotating 90 degrees counter-clockwise
        // must move the top (red) to the left side and bottom (blue) to the right side.
        let mut img = image::RgbImage::new(10, 40);
        for y in 0..10 {
            for x in 0..10 {
                img.put_pixel(x, y, image::Rgb([255, 0, 0])); // Red top
            }
        }
        for y in 30..40 {
            for x in 0..10 {
                img.put_pixel(x, y, image::Rgb([0, 0, 255])); // Blue bottom
            }
        }
        let pts: [Point; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 40.0], [0.0, 40.0]];
        let out = get_rotate_crop_image_rgb(&img, &pts);
        // After counter-clockwise rotation, width should be ~40 and height ~10
        assert!(out.width() >= 38);
        assert_eq!(out.height(), 10);
        // The left edge should be red, right edge should be blue
        let rgb = out.to_rgb8();
        let left_pixel = rgb.get_pixel(2, 5);
        let right_pixel = rgb.get_pixel(out.width() - 3, 5);
        assert_eq!(left_pixel[0], 255, "left side should be red (was top)");
        assert_eq!(right_pixel[2], 255, "right side should be blue (was bottom)");
    }
}
