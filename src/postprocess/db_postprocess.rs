use crate::postprocess::polygon::{Point, min_area_rect, unclip_polygon};
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct TextBox {
    pub points: [Point; 4],
    pub score: f32,
}

pub struct DbPostProcessor {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
    pub min_size: f32,
}

impl Default for DbPostProcessor {
    fn default() -> Self {
        Self {
            thresh: 0.3,
            box_thresh: 0.6,
            max_candidates: 1000,
            unclip_ratio: 1.5,
            min_size: 3.0,
        }
    }
}

impl DbPostProcessor {
    pub fn new(thresh: f32, box_thresh: f32, unclip_ratio: f32) -> Self {
        Self {
            thresh,
            box_thresh,
            unclip_ratio,
            ..Default::default()
        }
    }

    pub fn process(&self, prob_map: &Tensor, scale_h: f32, scale_w: f32) -> Vec<TextBox> {
        // Rank check. This was the only stage in the crate that indexed `shape()[2]/[3]`
        // unguarded, so `process` panicked on a 2-D tensor. `CtcDecoder::decode` and
        // `decode_batch` both validate their rank; this now matches. A probability map that
        // is not a single-channel NCHW image is simply "no boxes", not a crash.
        let shape = prob_map.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
            return Vec::new();
        }
        let h = shape[2];
        let w = shape[3];
        if h == 0 || w == 0 {
            return Vec::new();
        }
        let prob_slice = prob_map.as_slice();

        // 1. Thresholding to create component state: 0 = background, 1 = foreground, 2 = visited
        let mut state = vec![0u8; h * w];
        for i in 0..h * w {
            if prob_slice[i] >= self.thresh {
                state[i] = 1;
            }
        }

        // 2. Connected Component Labeling (8-connectivity BFS)
        let mut components: Vec<Vec<Point>> = Vec::new();

        let dx = [-1, 0, 1, -1, 1, -1, 0, 1];
        let dy = [-1, -1, -1, 0, 0, 1, 1, 1];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if state[idx] == 1 {
                    let mut queue = vec![(x, y)];
                    state[idx] = 2;

                    let mut boundary_pts = Vec::new();

                    let mut head = 0;
                    while head < queue.len() {
                        let (cx, cy) = queue[head];
                        head += 1;

                        // Check if boundary pixel
                        let mut is_boundary = cx == 0 || cx == w - 1 || cy == 0 || cy == h - 1;

                        for dir in 0..8 {
                            let nx = cx as isize + dx[dir];
                            let ny = cy as isize + dy[dir];

                            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                                let n_idx = ny as usize * w + nx as usize;
                                let s = state[n_idx];
                                if s == 0 {
                                    is_boundary = true;
                                } else if s == 1 {
                                    state[n_idx] = 2;
                                    queue.push((nx as usize, ny as usize));
                                }
                            }
                        }

                        if is_boundary {
                            boundary_pts.push([cx as f32, cy as f32]);
                        }
                    }

                    if boundary_pts.len() >= 4 {
                        components.push(boundary_pts);
                    }
                }
            }
        }

        // 3. Replace each component with its convex hull. The hull is what both the
        //    min-area rect and the unclip operate on downstream, and computing it here means
        //    the candidate ranking can use the real area.
        //
        // 4. Cap the candidate set *before* the expensive shape fitting, keeping the
        //    `max_candidates` largest by area. PaddleOCR does
        //    `sorted(contours, key=cv2.contourArea, reverse=True)[:max_candidates]`, so area
        //    is the correct key: ranking by boundary-point count instead kept long thin
        //    slivers (many boundary pixels, little area) and discarded compact blobs.
        //
        //    `sort_by` is stable and the comparator is a descending one, which matches
        //    Python's stable `sorted(..., reverse=True)` on equal areas.
        let mut hulls: Vec<Vec<Point>> = components
            .iter()
            .map(|c| crate::postprocess::polygon::convex_hull(c))
            .collect();
        drop(components);
        if hulls.len() > self.max_candidates {
            hulls.sort_by(|a, b| {
                crate::postprocess::polygon::polygon_area(b)
                    .total_cmp(&crate::postprocess::polygon::polygon_area(a))
            });
            hulls.truncate(self.max_candidates);
        }

        // 5. Filter and unclip polygons in parallel
        use rayon::prelude::*;

        let mut boxes: Vec<TextBox> = hulls
            .into_par_iter()
            .filter_map(|hull| {
                if hull.len() < 3 {
                    return None;
                }

                // Min-area rect of the contour; PaddleOCR's `get_mini_boxes` returns the
                // *shortest side* of this rect as `sside` and rejects on that, rather than
                // on area/perimeter — a long thin sliver has a healthy perimeter but must
                // still be dropped.
                let mut rect = min_area_rect(&hull);
                if shortest_rect_side(&rect) < self.min_size {
                    return None;
                }

                // Box score: mean probability over the polygon interior.
                let score = box_score_fast(prob_slice, w, h, &rect);
                if score < self.box_thresh {
                    return None;
                }

                // Unclip
                let unclipped = unclip_polygon(&hull, self.unclip_ratio);
                if unclipped.len() < 3 {
                    return None;
                }

                // Refit after unclipping and require the same (slightly relaxed) minimum,
                // matching PaddleOCR's `min_size + 2` second check.
                rect = min_area_rect(&unclipped);
                if shortest_rect_side(&rect) < self.min_size + 2.0 {
                    return None;
                }

                // Scale back to original image size
                for pt in &mut rect {
                    pt[0] /= scale_w;
                    pt[1] /= scale_h;
                }

                Some(TextBox {
                    points: rect,
                    score,
                })
            })
            .collect();

        // 6. Order the boxes the way PaddleOCR does, so downstream row grouping sees a
        //    natural reading order.
        sort_boxes_paddle(&mut boxes);

        boxes
    }
}

/// Shortest of the two rectangle sides spanned by `[p0, p1, p2, p3]`.
///
/// Equivalent to PaddleOCR's `sside = min(bounding_box[1])`, i.e. the smaller dimension
/// of the rotated rect.
pub fn shortest_rect_side(rect: &[Point; 4]) -> f32 {
    let dist = |a: Point, b: Point| ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    dist(rect[0], rect[1]).min(dist(rect[1], rect[2]))
}

/// Mean probability of the bitmap over the *interior* of the convex polygon `poly`,
/// restricted to its integer bounding box.
///
/// Rust counterpart of PaddleOCR's `box_score_fast`:
///
/// ```python
/// mask = np.zeros(polygonShape, dtype=np.uint8)
/// cv2.fillPoly(mask, box.astype(np.int32), 1)
/// return round(float(np.mean(pred[mask == 1])), 4)
/// ```
///
/// # Coordinate convention
///
/// `poly` is in **pixel-index** space, not continuous corner space. The corners come from
/// `min_area_rect` over boundary points that were themselves built as `[cx as f32, cy as
/// f32]` — i.e. integer pixel indices — and `fillPoly` fills index `x` for every integer
/// `xmin <= x <= xmax` inclusive. This used to sample at `xx + 0.5` (pixel centres), which
/// pushed the last row and column of every axis-aligned box just outside the polygon, so
/// they were dropped: an 8x8 map with `1.0` for `x < 2` and rect `0..=3` scored 2/3 = 0.667
/// here versus PaddleOCR's 2/4 = 0.5 — a 33% relative error that biased scores *downwards*
/// and so discarded valid boxes near `box_thresh`. Sampling at integer indices fixes it, and
/// for a rotated box the integer sample is a strictly better approximation of `fillPoly`'s
/// scanline fill than a half-pixel-shifted one.
fn box_score_fast(prob: &[f32], w: usize, h: usize, poly: &[Point; 4]) -> f32 {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in poly {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return 0.0;
    }

    // The polygon spans [min_x, max_x], so the largest *index* that can be inside it is
    // `floor(max_x)`; the smallest is `ceil(min_x)`.
    let x0 = (min_x.ceil().max(0.0) as usize).min(w - 1);
    let x1 = (max_x.floor().max(0.0) as usize).min(w - 1);
    let y0 = (min_y.ceil().max(0.0) as usize).min(h - 1);
    let y1 = (max_y.floor().max(0.0) as usize).min(h - 1);

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            if point_in_convex_poly(poly, xx as f32, yy as f32) {
                sum += prob[yy * w + xx] as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Point-in-convex-polygon test: the point is inside when it lies on the same side of
/// every directed edge. `poly` is convex by construction (it comes from
/// [`min_area_rect`]), so no winding-number handling is needed.
fn point_in_convex_poly(poly: &[Point; 4], px: f32, py: f32) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = poly[i];
        let b = poly[(i + 1) % 4];
        let cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
        if cross.abs() < 1e-9 {
            continue; // on the edge -> treat as inside
        }
        if sign == 0.0 {
            sign = cross;
        } else if (cross > 0.0) != (sign > 0.0) {
            return false;
        }
    }
    true
}

/// PaddleOCR's `sorted_boxes`: order by the top edge's `y`, then bubble a box leftwards
/// while it stays within the same horizontal band (`|Δy| < 10 px`) and has a smaller `x`.
///
/// A plain sort by box-centre `y` — the previous behaviour — interleaves boxes of the same
/// visual line whenever their heights differ, scrambling the reading order.
fn sort_boxes_paddle(boxes: &mut [TextBox]) {
    boxes.sort_by(|a, b| {
        a.points[0][1]
            .total_cmp(&b.points[0][1])
            .then_with(|| a.points[0][0].total_cmp(&b.points[0][0]))
    });

    let n = boxes.len();
    if n < 2 {
        return;
    }
    for i in 0..n - 1 {
        let mut j = i;
        loop {
            let y_diff = (boxes[j + 1].points[0][1] - boxes[j].points[0][1]).abs();
            let x_out_of_order = boxes[j + 1].points[0][0] < boxes[j].points[0][0];
            if y_diff < 10.0 && x_out_of_order {
                boxes.swap(j, j + 1);
            } else {
                break;
            }
            if j == 0 {
                break;
            }
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_rect_side_picks_smaller_dimension() {
        let rect: [Point; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]];
        assert!((shortest_rect_side(&rect) - 4.0).abs() < 1e-6);
        let thin: [Point; 4] = [[0.0, 0.0], [100.0, 0.0], [100.0, 1.0], [0.0, 1.0]];
        assert!((shortest_rect_side(&thin) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_score_fast_averages_over_interior() {
        // 8x8 map: left half 1.0, right half 0.0
        let w = 8;
        let h = 8;
        let mut prob = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                prob[y * w + x] = if x < 4 { 1.0 } else { 0.0 };
            }
        }

        // Rect covering only the left (all-1.0) half, in pixel-index coordinates: indices
        // 0..=3 inclusive.
        let rect: [Point; 4] = [[0.0, 0.0], [3.0, 0.0], [3.0, 7.0], [0.0, 7.0]];
        let s = box_score_fast(&prob, w, h, &rect);
        assert!((s - 1.0).abs() < 1e-6, "expected 1.0, got {s}");

        // Rect spanning the whole map -> mean 0.5. Corner (8, 8) is one past the last
        // index, so the clamp to w-1 / h-1 gives all 8 columns.
        let full: [Point; 4] = [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]];
        let s_full = box_score_fast(&prob, w, h, &full);
        assert!((s_full - 0.5).abs() < 1e-6, "expected 0.5, got {s_full}");

        // Degenerate rect entirely outside the map must not panic.
        let outside: [Point; 4] = [
            [-50.0, -50.0],
            [-40.0, -50.0],
            [-40.0, -40.0],
            [-50.0, -40.0],
        ];
        let _ = box_score_fast(&prob, w, h, &outside);
    }

    /// Discriminates the pixel-index convention from a half-pixel-shifted one.
    ///
    /// `cv2.fillPoly` on a rect with corners at indices `0..=3` covers columns 0, 1, 2 and 3.
    /// Sampling at `xx + 0.5` instead covers only 0, 1 and 2, because `(3.5, y)` is outside
    /// a polygon whose right edge is `x = 3`. With a step edge inside the box that is
    /// observable: the reference mean is 2/4 = 0.5, the shifted one 2/3 = 0.667. Both error
    /// and its direction matter here, since a low box score is what makes DB discard a box
    /// near `box_thresh`.
    #[test]
    fn box_score_fast_covers_the_last_row_and_column() {
        let (w, h) = (8usize, 8usize);
        let mut prob = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                prob[y * w + x] = if x < 2 { 1.0 } else { 0.0 };
            }
        }
        let rect: [Point; 4] = [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]];
        let s = box_score_fast(&prob, w, h, &rect);
        assert!(
            (s - 0.5).abs() < 1e-6,
            "expected 2/4 = 0.5 (fillPoly convention), got {s}"
        );
    }

    /// A non-4D probability map used to panic on `shape()[2]`. It is the only stage in the
    /// crate that skipped the rank check `CtcDecoder::decode` performs.
    #[test]
    fn non_4d_prob_map_returns_no_boxes_instead_of_panicking() {
        let pp = DbPostProcessor::default();
        for shape in [
            vec![],
            vec![4],
            vec![4, 4],
            vec![1, 4, 4],
            vec![1, 1, 4, 4, 1],
        ] {
            let n: usize = shape.iter().product();
            let t = Tensor::from_vec(&shape, vec![1.0; n]).expect("tensor");
            assert!(
                pp.process(&t, 1.0, 1.0).is_empty(),
                "shape {shape:?} should yield no boxes"
            );
        }
    }

    /// A multi-channel or batched map is not a single probability map either.
    #[test]
    fn multi_channel_prob_map_returns_no_boxes() {
        let pp = DbPostProcessor::default();
        let t = Tensor::from_vec(&[1, 2, 4, 4], vec![1.0; 32]).expect("tensor");
        assert!(pp.process(&t, 1.0, 1.0).is_empty());
        let t = Tensor::from_vec(&[2, 1, 4, 4], vec![1.0; 32]).expect("tensor");
        assert!(pp.process(&t, 1.0, 1.0).is_empty());
    }

    /// End-to-end: a synthetic probability map must produce a box. Nothing in this file
    /// tested that `process` returns anything at all, which is why the coordinate-convention
    /// and ranking bugs above had nothing to fail against.
    #[test]
    fn a_solid_block_produces_exactly_one_box() {
        let (w, h) = (64usize, 64usize);
        let mut prob = vec![0.0f32; w * h];
        // A 40x20 solid block of 1.0, well above `thresh` with a uniform interior, so the
        // box score is 1.0 regardless of the sampling convention.
        for y in 10..30 {
            for x in 5..45 {
                prob[y * w + x] = 1.0;
            }
        }
        let t = Tensor::from_vec(&[1, 1, h, w], prob).expect("tensor");
        let boxes = DbPostProcessor::default().process(&t, 1.0, 1.0);

        assert_eq!(boxes.len(), 1, "expected exactly one box, got {boxes:?}");
        let b = &boxes[0];
        assert!(b.score > 0.99, "interior is solid, score was {}", b.score);

        // The unclip must expand the min-area rect beyond the block on every side, but not
        // run away. PaddleOCR offsets by `unclip_ratio * area / perimeter`, which for a
        // 40x20 block is 1.5 * 800 / 120 = 10 px; this crate's miter clamp differs slightly
        // from Clipper's `miterLimit = 2.0`, so the exact figure is not asserted -- only
        // that the expansion happened and stayed local.
        let xs: Vec<f32> = b.points.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = b.points.iter().map(|p| p[1]).collect();
        let (min_x, max_x) = (min_of(&xs), max_of(&xs));
        let (min_y, max_y) = (min_of(&ys), max_of(&ys));
        for (label, got, block_edge, grows_negative) in [
            ("left", min_x, 5.0f32, true),
            ("right", max_x, 45.0, false),
            ("top", min_y, 10.0, true),
            ("bottom", max_y, 30.0, false),
        ] {
            let expansion = if grows_negative {
                block_edge - got
            } else {
                got - block_edge
            };
            assert!(
                (0.5..12.0).contains(&expansion),
                "{label} edge {got} expands the block edge {block_edge} by {expansion}, \
                 expected 0.5..12.0"
            );
        }
    }

    /// The same map at two different scale factors must produce coordinates that scale
    /// exactly inversely, i.e. the output really is in original-image space.
    #[test]
    fn boxes_are_scaled_back_to_the_original_image() {
        let (w, h) = (32usize, 32usize);
        let mut prob = vec![0.0f32; w * h];
        for y in 8..20 {
            for x in 4..28 {
                prob[y * w + x] = 1.0;
            }
        }
        let t = Tensor::from_vec(&[1, 1, h, w], prob).expect("tensor");
        let pp = DbPostProcessor::default();

        let unit = pp.process(&t, 1.0, 1.0);
        let half = pp.process(&t, 0.5, 0.5);
        assert_eq!(unit.len(), 1, "unit-scale boxes: {unit:?}");
        assert_eq!(half.len(), 1, "half-scale boxes: {half:?}");

        // `pt[0] /= scale_w`, so a 0.5 scale means the map covers half the original and the
        // output coordinates come out twice as large.
        for (i, (u, s)) in unit[0].points.iter().zip(half[0].points.iter()).enumerate() {
            assert!(
                (u[0] - s[0] * 0.5).abs() < 1e-4,
                "corner {i} x: unit={} half*0.5={}",
                u[0],
                s[0] * 0.5
            );
            assert!(
                (u[1] - s[1] * 0.5).abs() < 1e-4,
                "corner {i} y: unit={} half*0.5={}",
                u[1],
                s[1] * 0.5
            );
        }
    }

    /// `max_candidates` must keep the largest components. Ranking by boundary-point count
    /// (the previous behaviour) instead of area kept long thin slivers over compact blobs.
    #[test]
    fn max_candidates_keeps_the_largest_areas() {
        let (w, h) = (64usize, 32usize);
        let mut prob = vec![0.0f32; w * h];
        // A big compact blob: 12 rows x 12 cols = 144 px, perimeter 48 -> 48 boundary px.
        for y in 2..14 {
            for x in 2..14 {
                prob[y * w + x] = 1.0;
            }
        }
        // A long thin sliver: 20 rows x 4 cols = 80 px, perimeter 48 -> 48+ boundary px,
        // so it ties or beats the blob on boundary count while being much smaller.
        for y in 18..38.min(h) {
            for x in 40..44 {
                prob[y * w + x] = 1.0;
            }
        }

        let t = Tensor::from_vec(&[1, 1, h, w], prob).expect("tensor");
        let pp = DbPostProcessor {
            max_candidates: 1,
            ..Default::default()
        };
        let boxes = pp.process(&t, 1.0, 1.0);

        assert_eq!(boxes.len(), 1, "the cap must leave one box, got {boxes:?}");
        // Whichever survived, it must be the wide one: the blob is 12 px tall, the sliver
        // 20. Assert on the surviving box's area instead of trusting the tie-break.
        let b = &boxes[0];
        let width = (b.points[1][0] - b.points[0][0]).abs();
        let height = (b.points[3][1] - b.points[0][1]).abs();
        let area = width * height;
        assert!(
            area > 144.0,
            "expected the 12x12 blob (>=144 px), got {width:.1} x {height:.1} = {area:.1}"
        );
    }

    fn min_of(v: &[f32]) -> f32 {
        v.iter().copied().fold(f32::INFINITY, f32::min)
    }

    fn max_of(v: &[f32]) -> f32 {
        v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    #[test]
    fn point_in_poly_is_correct() {
        let sq: [Point; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(point_in_convex_poly(&sq, 5.0, 5.0));
        assert!(!point_in_convex_poly(&sq, 15.0, 5.0));
        assert!(!point_in_convex_poly(&sq, 5.0, -1.0));
    }

    fn tb(x: f32, y: f32) -> TextBox {
        TextBox {
            points: [[x, y], [x + 20.0, y], [x + 20.0, y + 8.0], [x, y + 8.0]],
            score: 0.9,
        }
    }

    #[test]
    fn sort_boxes_uses_reading_order() {
        // Two visual lines. Line 2's two boxes differ in top-edge y by only 2 px, so they
        // fall inside the same 10 px band and must be ordered left-to-right even though
        // their y is not monotonic. That is precisely PaddleOCR's `sorted_boxes`
        // behaviour: a plain y-sort would scramble the reading order here.
        let mut boxes = vec![
            tb(300.0, 12.0), // line 2, right
            tb(0.0, 0.0),    // line 1, left
            tb(100.0, 14.0), // line 2, middle
            tb(200.0, 0.0),  // line 1, right
        ];
        sort_boxes_paddle(&mut boxes);

        // The lines must not interleave: both line-1 boxes first, then both line-2 boxes.
        let ys: Vec<f32> = boxes.iter().map(|b| b.points[0][1]).collect();
        assert_eq!(ys[0], 0.0, "ys={ys:?}");
        assert_eq!(ys[1], 0.0, "ys={ys:?}");
        assert!(
            ys[2] >= 10.0 && ys[3] >= 10.0,
            "lines interleaved: ys={ys:?}"
        );

        // Left-to-right within each line; the near-equal-y pair is ordered by x, not by y.
        assert!(boxes[0].points[0][0] < boxes[1].points[0][0]);
        assert_eq!(
            boxes[2].points[0][0], 100.0,
            "expected the leftmost line-2 box first"
        );
        assert_eq!(boxes[3].points[0][0], 300.0);
    }

    #[test]
    fn empty_prob_map_returns_no_boxes() {
        let pp = DbPostProcessor::default();
        let t = Tensor::from_vec(&[1, 1, 0, 0], vec![]).expect("empty tensor");
        assert!(pp.process(&t, 1.0, 1.0).is_empty());
    }
}
