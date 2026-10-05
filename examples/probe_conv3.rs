use ppocr::Tensor;
use ppocr::ops::conv2d::{ConvActivation, Conv2dParams, conv2d};
use rayon::prelude::*;

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    let input = Tensor::from_vec(
        &[1, 96, 24, 480],
        (0..1 * 96 * 24 * 480).map(|i| (i as f32 * 1e-4).sin()).collect(),
    ).unwrap();
    let weight = Tensor::from_vec(
        &[48, 96, 3, 3],
        (0..48 * 96 * 3 * 3).map(|i| (i as f32 * 1e-4).cos()).collect(),
    ).unwrap();
    let bias = Tensor::from_vec(
        &[48],
        (0..48).map(|i| i as f32 * 0.01).collect(),
    ).unwrap();
    let params = Conv2dParams {
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
        group: 1,
        activation: ConvActivation::HardSwish,
    };

    let ref_out = conv2d(&input, &weight, Some(&bias), &params).unwrap();

    let out_h = 12;
    let out_w = 240;
    let cout = 48;
    let cin = 96;
    let k_size = cin * 3 * 3;
    let out_spatial = out_h * out_w;

    let in_slice = input.as_slice();
    let weight_slice = weight.as_slice();
    let bias_slice = bias.as_slice();

    let chunk_w = 40;
    let num_w_chunks = (out_w + chunk_w - 1) / chunk_w;
    let total_tiles = out_h * num_w_chunks;

    let mut output = Tensor::empty(&[1, cout, out_h, out_w]);
    let out_p_usize = output.as_mut_slice().as_mut_ptr() as usize;

    (0..total_tiles).into_par_iter().for_each(|tile_idx| {
        let oh = tile_idx / num_w_chunks;
        let w_idx = tile_idx % num_w_chunks;
        let ow_start = w_idx * chunk_w;
        let ow_end = (ow_start + chunk_w).min(out_w);
        let actual_w = ow_end - ow_start;

        let mut tile_col = ppocr::tensor::acquire_buffer(k_size * actual_w);
        #[allow(clippy::uninit_vec)]
        unsafe {
            tile_col.set_len(k_size * actual_w);
        }

        let mut col_idx = 0;
        for c in 0..cin {
            let in_c_offset = c * 24 * 480;
            for kh in 0..3 {
                let ih = (oh * 2) as isize - 1 + kh as isize;
                if ih >= 0 && (ih as usize) < 24 {
                    let in_row = in_c_offset + (ih as usize) * 480;
                    for kw in 0..3 {
                        let col_line = &mut tile_col[col_idx * actual_w..(col_idx + 1) * actual_w];
                        for (i, ow) in (ow_start..ow_end).enumerate() {
                            let iw = (ow * 2) as isize - 1 + kw as isize;
                            col_line[i] = if iw >= 0 && (iw as usize) < 480 {
                                in_slice[in_row + iw as usize]
                            } else {
                                0.0
                            };
                        }
                        col_idx += 1;
                    }
                } else {
                    for _ in 0..3 {
                        tile_col[col_idx * actual_w..(col_idx + 1) * actual_w].fill(0.0);
                        col_idx += 1;
                    }
                }
            }
        }

        let out_p = out_p_usize as *mut f32;
        let c_start = oh * out_w + ow_start;
        unsafe {
            ppocr::ops::gemm_row_major(
                cout,
                k_size,
                actual_w,
                weight_slice.as_ptr(),
                k_size,
                tile_col.as_ptr(),
                actual_w,
                out_p.add(c_start),
                out_spatial,
            );
        }

        for i in 0..cout {
            let b_val = bias_slice[i];
            let row_offset = c_start + i * out_spatial;
            unsafe {
                for x in 0..actual_w {
                    let v = *out_p.add(row_offset + x) + b_val;
                    // HardSwish: x * relu6(x + 3) / 6
                    let hs = v * ((v + 3.0).max(0.0).min(6.0)) / 6.0;
                    *out_p.add(row_offset + x) = hs;
                }
            }
        }

        ppocr::tensor::release_buffer(tile_col);
    });

    let mut max_diff = 0.0f32;
    for (v1, v2) in ref_out.as_slice().iter().zip(output.as_slice().iter()) {
        let d = (v1 - v2).abs();
        if d > max_diff { max_diff = d; }
    }
    println!("Max diff vs reference: {:.2e}", max_diff);
}
