use std::time::Instant;
use ppocr::Tensor;

fn fast_permute(tensor: &Tensor, order: &[usize]) -> Tensor {
    let ndim = tensor.ndim();
    let shape = tensor.shape();
    let new_shape: Vec<usize> = order.iter().map(|&d| shape[d]).collect();
    let total: usize = new_shape.iter().product();

    let src = if tensor.is_contiguous() {
        std::borrow::Cow::Borrowed(tensor)
    } else {
        std::borrow::Cow::Owned(tensor.to_contiguous())
    };
    let src_slice = src.as_slice();

    let mut data = ppocr::tensor::acquire_buffer(total);
    #[allow(clippy::uninit_vec)]
    unsafe { data.set_len(total); }

    // Fast Path 1: Batched 2D transpose of last 2 dimensions: [..., M, N] -> [..., N, M]
    if ndim >= 2
        && order[..ndim - 2] == (0..ndim - 2).collect::<Vec<_>>()[..]
        && order[ndim - 2] == ndim - 1
        && order[ndim - 1] == ndim - 2
    {
        let m = shape[ndim - 2];
        let n = shape[ndim - 1];
        let b_total = total / (m * n);
        let dst_p = data.as_mut_ptr() as usize;
        let src_p = src_slice.as_ptr() as usize;

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

        return Tensor::from_vec(&new_shape, data).unwrap();
    }

    // Fast Path 2: Preserve innermost dimension if order[ndim - 1] == ndim - 1
    if ndim >= 2 && order[ndim - 1] == ndim - 1 {
        let l = shape[ndim - 1];
        let outer_total = total / l;
        let dst_p = data.as_mut_ptr() as usize;
        let src_p = src_slice.as_ptr() as usize;

        let outer_ndim = ndim - 1;
        let outer_shape: Vec<usize> = order[..outer_ndim].iter().map(|&d| shape[d]).collect();
        let src_strides = tensor.strides();
        let outer_src_strides: Vec<usize> = order[..outer_ndim].iter().map(|&d| src_strides[d]).collect();

        let run_outer = |(out_outer_idx, coord): (usize, Vec<usize>)| {
            let mut src_idx = 0usize;
            for (j, &c) in coord.iter().enumerate() {
                src_idx += c * outer_src_strides[j];
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (src_p as *const f32).add(src_idx),
                    (dst_p as *mut f32).add(out_outer_idx * l),
                    l,
                );
            }
        };

        let mut coords = Vec::with_capacity(outer_total);
        let mut coord = vec![0usize; outer_ndim];
        for _ in 0..outer_total {
            coords.push(coord.clone());
            for j in (0..outer_ndim).rev() {
                coord[j] += 1;
                if coord[j] < outer_shape[j] {
                    break;
                }
                coord[j] = 0;
            }
        }

        if outer_total >= 128 {
            use rayon::prelude::*;
            coords.into_par_iter().enumerate().for_each(run_outer);
        } else {
            for (out_outer_idx, coord) in coords.into_iter().enumerate() {
                run_outer((out_outer_idx, coord));
            }
        }

        return Tensor::from_vec(&new_shape, data).unwrap();
    }

    // General fallback
    let new_strides = tensor.strides();
    let mut coord = vec![0usize; ndim];
    let order_strides: Vec<usize> = order.iter().map(|&d| new_strides[d]).collect();

    for out_idx in 0..total {
        let mut src_idx = 0usize;
        for (j, &c) in coord.iter().enumerate() {
            src_idx += c * order_strides[j];
        }
        data[out_idx] = src_slice[src_idx];

        for j in (0..ndim).rev() {
            coord[j] += 1;
            if coord[j] < new_shape[j] {
                break;
            }
            coord[j] = 0;
        }
    }

    Tensor::from_vec(&new_shape, data).unwrap()
}

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    let test_cases = [
        (vec![1, 120, 120], vec![0, 2, 1], "Transpose.0 [1, 120, 120] -> [0, 2, 1]"),
        (vec![1, 8, 120, 120], vec![0, 1, 3, 2], "Transpose.2 [1, 8, 120, 120] -> [0, 1, 3, 2]"),
        (vec![1, 120, 8, 15], vec![0, 2, 1, 3], "Transpose.3 [1, 120, 8, 15] -> [0, 2, 1, 3]"),
        (vec![1, 3, 40, 8, 15], vec![2, 0, 3, 1, 4], "Transpose.1 [1, 3, 40, 8, 15] -> [2, 0, 3, 1, 4]"),
    ];

    for (shape, perm, label) in &test_cases {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.001).collect();
        let t = Tensor::from_vec(shape, data).unwrap();

        let ref_out = t.permute(perm).unwrap();
        let fast_out = fast_permute(&t, perm);

        let diff: f32 = ref_out.as_slice().iter().zip(fast_out.as_slice().iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert_eq!(diff, 0.0, "Diff mismatch on {label}");

        let mut best_ref = f64::INFINITY;
        for _ in 0..30 {
            let t0 = Instant::now();
            let _ = t.permute(perm).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best_ref { best_ref = ms; }
        }

        let mut best_fast = f64::INFINITY;
        for _ in 0..30 {
            let t0 = Instant::now();
            let _ = fast_permute(&t, perm);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best_fast { best_fast = ms; }
        }

        println!("{:<45}: Ref: {:6.3} ms | Fast: {:6.3} ms | Speedup: {:5.1}x",
            label, best_ref, best_fast, best_ref / best_fast);
    }
}
