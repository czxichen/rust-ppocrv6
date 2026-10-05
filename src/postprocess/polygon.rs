pub type Point = [f32; 2];

pub fn polygon_area(poly: &[Point]) -> f32 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    (area / 2.0).abs()
}

pub fn polygon_perimeter(poly: &[Point]) -> f32 {
    let n = poly.len();
    if n < 2 {
        return 0.0;
    }
    let mut peri = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = poly[j][0] - poly[i][0];
        let dy = poly[j][1] - poly[i][1];
        peri += (dx * dx + dy * dy).sqrt();
    }
    peri
}

pub fn is_clockwise(poly: &[Point]) -> bool {
    let n = poly.len();
    let mut sum = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += (poly[j][0] - poly[i][0]) * (poly[j][1] + poly[i][1]);
    }
    sum > 0.0
}

/// Unclip (expand) a polygon contour using DBNet's distance formula:
/// distance = (Area * unclip_ratio) / Perimeter
pub fn unclip_polygon(poly: &[Point], unclip_ratio: f32) -> Vec<Point> {
    let n = poly.len();
    if n < 3 {
        return poly.to_vec();
    }

    let area = polygon_area(poly);
    let peri = polygon_perimeter(poly);
    if peri < 1e-6 {
        return poly.to_vec();
    }

    let dist = (area * unclip_ratio) / peri;
    let cw = is_clockwise(poly);

    let mut offset_poly = Vec::with_capacity(n);

    for i in 0..n {
        let prev = poly[(i + n - 1) % n];
        let curr = poly[i];
        let next = poly[(i + 1) % n];

        // Edge vectors
        let v1 = [curr[0] - prev[0], curr[1] - prev[1]];
        let v2 = [next[0] - curr[0], next[1] - curr[1]];

        let len1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt().max(1e-6);
        let len2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt().max(1e-6);

        // Outward normals
        let n1 = if cw {
            [-v1[1] / len1, v1[0] / len1]
        } else {
            [v1[1] / len1, -v1[0] / len1]
        };

        let n2 = if cw {
            [-v2[1] / len2, v2[0] / len2]
        } else {
            [v2[1] / len2, -v2[0] / len2]
        };

        // Bisector vector
        let bisector = [n1[0] + n2[0], n1[1] + n2[1]];
        let b_len = (bisector[0] * bisector[0] + bisector[1] * bisector[1]).sqrt();

        if b_len < 1e-4 {
            offset_poly.push([curr[0] + n1[0] * dist, curr[1] + n1[1] * dist]);
        } else {
            let cos_half = (n1[0] * bisector[0] + n1[1] * bisector[1]) / b_len;
            let miter_len = (dist / cos_half.max(0.1)).min(dist * 3.0); // Clamp miter to avoid sharp spikes
            offset_poly.push([
                curr[0] + (bisector[0] / b_len) * miter_len,
                curr[1] + (bisector[1] / b_len) * miter_len,
            ]);
        }
    }

    offset_poly
}

/// Compute convex hull using Andrew's monotone chain algorithm
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut pts = points.to_vec();
    if pts.len() <= 3 {
        return pts;
    }

    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });

    let cross = |o: &Point, a: &Point, b: &Point| -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut lower = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(*p);
    }

    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(*p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum Area Bounding Box (Rotating Calipers)
pub fn min_area_rect(points: &[Point]) -> [Point; 4] {
    let hull = convex_hull(points);
    let n = hull.len();
    if n < 3 {
        if points.is_empty() {
            return [[0.0, 0.0]; 4];
        }
        let min_x = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max_x = points
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let max_y = points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        return [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ];
    }

    let mut min_area = f32::INFINITY;
    let mut best_box = [[0.0, 0.0]; 4];

    for i in 0..n {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % n];

        let edge = [p2[0] - p1[0], p2[1] - p1[1]];
        let edge_len = (edge[0] * edge[0] + edge[1] * edge[1]).sqrt();
        if edge_len < 1e-6 {
            continue;
        }

        let u = [edge[0] / edge_len, edge[1] / edge_len];
        let v = [-u[1], u[0]];

        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        for p in &hull {
            let proj_u = p[0] * u[0] + p[1] * u[1];
            let proj_v = p[0] * v[0] + p[1] * v[1];

            min_u = min_u.min(proj_u);
            max_u = max_u.max(proj_u);
            min_v = min_v.min(proj_v);
            max_v = max_v.max(proj_v);
        }

        let area = (max_u - min_u) * (max_v - min_v);
        if area < min_area {
            min_area = area;
            // 4 corners
            best_box = [
                [min_u * u[0] + min_v * v[0], min_u * u[1] + min_v * v[1]],
                [max_u * u[0] + min_v * v[0], max_u * u[1] + min_v * v[1]],
                [max_u * u[0] + max_v * v[0], max_u * u[1] + max_v * v[1]],
                [min_u * u[0] + max_v * v[0], min_u * u[1] + max_v * v[1]],
            ];
        }
    }

    order_points(&best_box)
}

/// Order 4 box points clockwise: [top-left, top-right, bottom-right, bottom-left]
/// Following PaddleOCR's standard ordering:
/// - top-left: smallest (x + y)
/// - bottom-right: largest (x + y)
/// - top-right: smallest (y - x)
/// - bottom-left: largest (y - x)
pub fn order_points(pts: &[Point; 4]) -> [Point; 4] {
    let mut sorted = *pts;
    // Sort by x coordinate, breaking ties with y coordinate
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));

    // Left-most 2 points and right-most 2 points
    let left1 = sorted[0];
    let left2 = sorted[1];
    let right1 = sorted[2];
    let right2 = sorted[3];

    // Top-left and bottom-left: sorted by y
    let (tl, bl) = if left1[1] <= left2[1] {
        (left1, left2)
    } else {
        (left2, left1)
    };

    // Top-right and bottom-right: sorted by y
    let (tr, br) = if right1[1] <= right2[1] {
        (right1, right2)
    } else {
        (right2, right1)
    };

    [tl, tr, br, bl]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_points_diamond_45deg() {
        let pts = [[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]];
        let ordered = order_points(&pts);
        assert_eq!(
            ordered,
            [[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]]
        );
        // All 4 points must be unique
        for i in 0..4 {
            for j in i + 1..4 {
                assert_ne!(ordered[i], ordered[j]);
            }
        }
    }

    #[test]
    fn test_order_points_standard_rectangle() {
        let pts = [[100.0, 0.0], [0.0, 0.0], [100.0, 20.0], [0.0, 20.0]];
        let ordered = order_points(&pts);
        assert_eq!(
            ordered,
            [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]]
        );
    }
}
