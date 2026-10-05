use std::time::Instant;
use ppocr::Tensor;

fn permute_5d_qkv_par(t: &Tensor) -> Vec<f32> {
    use rayon::prelude::*;
    let s0 = t.shape()[0];
    let s1 = t.shape()[1];
    let s2 = t.shape()[2];
    let s3 = t.shape()[3];
    let l = t.shape()[4];
    let total = s0 * s1 * s2 * s3 * l;
    let mut data = vec![0.0f32; total];

    let src_p = t.as_slice().as_ptr() as usize;
    let dst_p = data.as_mut_ptr() as usize;

    let num_tasks = s2 * s3;
    (0..num_tasks).into_par_iter().for_each(|task| {
        let d2 = task / s3;
        let d3 = task % s3;
        let src_p = src_p as *const f32;
        let dst_p = dst_p as *mut f32;
        for d0 in 0..s0 {
            for d1 in 0..s1 {
                let src_off = (((d0 * s1 + d1) * s2 + d2) * s3 + d3) * l;
                let dst_off = (((d2 * s0 + d0) * s3 + d3) * s1 + d1) * l;
                unsafe {
                    std::ptr::copy_nonoverlapping(src_p.add(src_off), dst_p.add(dst_off), l);
                }
            }
        }
    });
    data
}

fn permute_4d_0312(t: &Tensor) -> Vec<f32> {
    let b_dim = t.shape()[0];
    let s1 = t.shape()[1];
    let s2 = t.shape()[2];
    let s3 = t.shape()[3];
    let total = b_dim * s1 * s2 * s3;
    let mut data = vec![0.0f32; total];

    let src_p = t.as_slice().as_ptr();
    let dst_p = data.as_mut_ptr();

    for b in 0..b_dim {
        for i3 in 0..s3 {
            for i1 in 0..s1 {
                for i2 in 0..s2 {
                    let src_off = ((b * s1 + i1) * s2 + i2) * s3 + i3;
                    let dst_off = ((b * s3 + i3) * s1 + i1) * s2 + i2;
                    unsafe {
                        *dst_p.add(dst_off) = *src_p.add(src_off);
                    }
                }
            }
        }
    }
    data
}

fn main() {
    // 1. Transpose.1 shape: [1, 120, 3, 8, 15], perm: [2, 0, 3, 1, 4]
    let t1 = Tensor::from_vec(&[1, 120, 3, 8, 15], (0..1 * 120 * 3 * 8 * 15).map(|i| i as f32).collect()).unwrap();
    let iters = 200;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = t1.permute(&[2, 0, 3, 1, 4]).unwrap();
    }
    let ms1 = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Current permute [1, 120, 3, 8, 15] perm=[2, 0, 3, 1, 4]: {:.4} ms", ms1);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = permute_5d_qkv_par(&t1);
    }
    let ms1_opt = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Optimized permute_5d_qkv_par: {:.4} ms", ms1_opt);

    // Verify correctness
    let ref_out = t1.permute(&[2, 0, 3, 1, 4]).unwrap();
    let opt_out = permute_5d_qkv_par(&t1);
    assert_eq!(ref_out.as_slice(), &opt_out[..]);
    println!("permute_5d_qkv output verified identical!");

    // 2. Transpose.7 shape: [1, 1, 120, 120], perm: [0, 3, 1, 2]
    let t7 = Tensor::from_vec(&[1, 1, 120, 120], (0..14400).map(|i| i as f32).collect()).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = t7.permute(&[0, 3, 1, 2]).unwrap();
    }
    let ms7 = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Current permute [1, 1, 120, 120] perm=[0, 3, 1, 2]: {:.4} ms", ms7);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = permute_4d_0312(&t7);
    }
    let ms7_opt = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Optimized permute_4d_0312: {:.4} ms", ms7_opt);

    let ref7 = t7.permute(&[0, 3, 1, 2]).unwrap();
    let opt7 = permute_4d_0312(&t7);
    assert_eq!(ref7.as_slice(), &opt7[..]);
    println!("permute_4d_0312 output verified identical!");
}
