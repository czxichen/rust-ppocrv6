use std::time::Instant;
use ppocr::Tensor;

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    // Det Conv.2: m=24, cin=12, k_size=48, out_h=479, out_w=351
    let m: usize = 24;
    let cin: usize = 12;
    let k_size: usize = 48;
    let out_h: usize = 479;
    let out_w: usize = 351;
    let in_h: usize = 480;
    let in_w: usize = 352;
    let out_spatial = out_h * out_w;

    let input = Tensor::zeros(&[1, cin, in_h, in_w]);
    let weight = Tensor::zeros(&[m, cin, 2, 2]);
    let in_slice = input.as_slice();
    let weight_slice = weight.as_slice();

    for test_tile_h in [2, 5, 10, 11, 20, 40] {
        let num_tiles = out_h.div_ceil(test_tile_h);
        let mut output = Tensor::empty(&[1, m, out_h, out_w]);
        let out_ptr_usize = output.as_mut_slice().as_mut_ptr() as usize;

        let run_test = || {
            use rayon::prelude::*;
            (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                let oh_start = tile_idx * test_tile_h;
                let oh_end = (oh_start + test_tile_h).min(out_h);
                let actual_h = oh_end - oh_start;
                let tile_spatial = actual_h * out_w;

                let mut tile_col = ppocr::tensor::acquire_buffer(k_size * tile_spatial);
                #[allow(clippy::uninit_vec)]
                unsafe { tile_col.set_len(k_size * tile_spatial); }

                let run_row = |(col_row, out_row_slice): (usize, &mut [f32])| {
                    let kw = col_row % 2;
                    let rem = col_row / 2;
                    let kh = rem % 2;
                    let c = rem / 2;
                    let in_c_offset = c * in_h * in_w;
                    for oh in oh_start..oh_end {
                        let ih = oh + kh;
                        let tile_oh = oh - oh_start;
                        let out_row_offset = tile_oh * out_w;
                        let in_row_offset = in_c_offset + ih * in_w;
                        let out_line = &mut out_row_slice[out_row_offset..out_row_offset + out_w];
                        out_line.copy_from_slice(&in_slice[in_row_offset + kw..in_row_offset + kw + out_w]);
                    }
                };
                tile_col.chunks_mut(tile_spatial).enumerate().for_each(run_row);

                let c_start = oh_start * out_w;
                let out_p = out_ptr_usize as *mut f32;

                unsafe {
                    ppocr::ops::gemm_row_major(
                        m,
                        k_size,
                        tile_spatial,
                        weight_slice.as_ptr(),
                        k_size,
                        tile_col.as_ptr(),
                        tile_spatial,
                        out_p.add(c_start),
                        out_spatial,
                    );
                }

                ppocr::tensor::release_buffer(tile_col);
            });
        };

        for _ in 0..3 { run_test(); }
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t0 = Instant::now();
            run_test();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best { best = ms; }
        }
        println!("Conv.2 tile_h = {:2} (num_tiles = {:3}): latency = {:.3} ms",
            test_tile_h, num_tiles, best);
    }

    // Now test Conv.1: m=12, cin=24, k_size=96
    let m = 12;
    let cin = 24;
    let k_size = 96;
    let input1 = Tensor::zeros(&[1, cin, in_h, in_w]);
    let weight1 = Tensor::zeros(&[m, cin, 2, 2]);
    let in_slice1 = input1.as_slice();
    let weight_slice1 = weight1.as_slice();

    for test_tile_h in [2, 3, 4, 5, 8] {
        let num_tiles = out_h.div_ceil(test_tile_h);
        let mut output = Tensor::empty(&[1, m, out_h, out_w]);
        let out_ptr_usize = output.as_mut_slice().as_mut_ptr() as usize;

        let run_test = || {
            use rayon::prelude::*;
            (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                let oh_start = tile_idx * test_tile_h;
                let oh_end = (oh_start + test_tile_h).min(out_h);
                let actual_h = oh_end - oh_start;
                let tile_spatial = actual_h * out_w;

                let mut tile_col = ppocr::tensor::acquire_buffer(k_size * tile_spatial);
                #[allow(clippy::uninit_vec)]
                unsafe { tile_col.set_len(k_size * tile_spatial); }

                let run_row = |(col_row, out_row_slice): (usize, &mut [f32])| {
                    let kw = col_row % 2;
                    let rem = col_row / 2;
                    let kh = rem % 2;
                    let c = rem / 2;
                    let in_c_offset = c * in_h * in_w;
                    for oh in oh_start..oh_end {
                        let ih = oh + kh;
                        let tile_oh = oh - oh_start;
                        let out_row_offset = tile_oh * out_w;
                        let in_row_offset = in_c_offset + ih * in_w;
                        let out_line = &mut out_row_slice[out_row_offset..out_row_offset + out_w];
                        out_line.copy_from_slice(&in_slice1[in_row_offset + kw..in_row_offset + kw + out_w]);
                    }
                };
                tile_col.chunks_mut(tile_spatial).enumerate().for_each(run_row);

                let c_start = oh_start * out_w;
                let out_p = out_ptr_usize as *mut f32;

                unsafe {
                    ppocr::ops::gemm_row_major(
                        m,
                        k_size,
                        tile_spatial,
                        weight_slice1.as_ptr(),
                        k_size,
                        tile_col.as_ptr(),
                        tile_spatial,
                        out_p.add(c_start),
                        out_spatial,
                    );
                }

                ppocr::tensor::release_buffer(tile_col);
            });
        };

        for _ in 0..3 { run_test(); }
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t0 = Instant::now();
            run_test();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best { best = ms; }
        }
        println!("Conv.1 tile_h = {:2} (num_tiles = {:3}): latency = {:.3} ms",
            test_tile_h, num_tiles, best);
    }
}
