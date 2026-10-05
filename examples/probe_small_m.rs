use std::time::Instant;

fn main() {
    let shapes = [
        (12, 96, 1755, "Conv.1 tile (12, 96, 1755)"),
        (24, 48, 3861, "Conv.2 tile (24, 48, 3861)"),
        (12, 96, 42240, "Conv.1 full (12, 96, 42240)"),
        (24, 48, 42240, "Conv.2 full (24, 48, 42240)"),
        (24, 48, 11017, "Rec Conv.1 full (24, 48, 11017)"),
        (48, 24, 11017, "Rec Conv.2 full (48, 24, 11017)"),
    ];

    for &(m, k, n, label) in &shapes {
        let a = vec![0.01f32; m * k];
        let b = vec![0.02f32; k * n];
        let mut c_faer = vec![0.0f32; m * n];
        let mut c_mm = vec![0.0f32; m * n];

        let t0 = Instant::now();
        for _ in 0..30 {
            unsafe {
                ppocr::ops::gemm_row_major(
                    m, k, n,
                    a.as_ptr(), k,
                    b.as_ptr(), n,
                    c_faer.as_mut_ptr(), n,
                );
            }
        }
        let t_faer = t0.elapsed().as_secs_f64() * (1000.0 / 30.0);

        let t0 = Instant::now();
        for _ in 0..30 {
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n,
                    1.0,
                    a.as_ptr(), k as isize, 1,
                    b.as_ptr(), n as isize, 1,
                    0.0,
                    c_mm.as_mut_ptr(), n as isize, 1,
                );
            }
        }
        let t_mm = t0.elapsed().as_secs_f64() * (1000.0 / 30.0);

        println!("{:<32}: faer: {:6.3} ms | mm: {:6.3} ms | Speedup mm/faer: {:5.2}x",
            label, t_faer, t_mm, t_faer / t_mm);
    }
}
