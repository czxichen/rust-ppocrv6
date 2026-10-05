// Alternating A/B for the 1x1 split, because the first chunk sweep was run config-by-config
// and the host drifts.
//
// `conv1x1_chunk_probe` suggests a 2D (rows x columns) grid beats the engine's current
// column-only split on some shapes and loses on others -- most sharply 8x2 on
// m=192, k=384, n=144 (237.8 vs 166.4 GFLOP/s). That also contradicts the earlier finding
// that splitting m is always bad, which was measured with a *full-width* B in every call
// (m/16 rows x all n columns). A grid that also narrows B halves the repack per call, so
// the two experiments are not the same experiment.
//
// This runs the two candidate strategies alternately, many times, and reports the minimum
// of each. Alternation is the point: a shared host's throughput drifts by up to 2x, so
// measuring A fully then B fully can invent a difference that is not there.
//
//   cargo run --release --example conv1x1_ab_probe
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let rounds = 12;
    let iters = 25;
    let shapes = [
        (384usize, 192usize, 144usize),
        (192, 384, 144),
        (768, 384, 708),
        (384, 768, 708),
        (384, 192, 708),
        (96, 48, 2304),
    ];
    let nt = rayon::current_num_threads();
    println!("threads = {nt}, {rounds} alternating rounds x {iters} iters, min of all\n");

    println!(
        "{:<22} {:>9} {:>9} {:>7}   {}",
        "shape", "engine", "grid", "ratio", "grid shape"
    );

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1e-4).cos() * 0.01).collect();
        let mut out = vec![0.0f32; m * n];
        let p = out.as_mut_ptr() as usize;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        // The engine's current rule: split n, chunk >= 64.
        let engine_cn = n.div_ceil(nt).max(64).min(n);
        let engine_c = n.div_ceil(engine_cn);

        // The grid: narrow B to 2 column blocks, then 8 row blocks.
        let (gm, gn) = (8usize, 2usize);
        let rm = m.div_ceil(gm);
        let rn = n.div_ceil(gn);
        let g_m = m.div_ceil(rm);
        let g_n = n.div_ceil(rn);

        let run_engine = || {
            (0..engine_c).into_par_iter().for_each(|ci| {
                let cs = ci * engine_cn;
                let an = (n - cs).min(engine_cn);
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        an,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        b.as_ptr().add(cs),
                        n as isize,
                        1,
                        0.0,
                        (p as *mut f32).add(cs),
                        n as isize,
                        1,
                    );
                }
            });
        };
        let run_grid = || {
            (0..g_m * g_n).into_par_iter().for_each(|t| {
                let mi = t / g_n;
                let ni = t % g_n;
                let os = mi * rm;
                let am = (m - os).min(rm);
                let cs = ni * rn;
                let an = (n - cs).min(rn);
                unsafe {
                    matrixmultiply::sgemm(
                        am,
                        k,
                        an,
                        1.0,
                        a.as_ptr().add(os * k),
                        k as isize,
                        1,
                        b.as_ptr().add(cs),
                        n as isize,
                        1,
                        0.0,
                        (p as *mut f32).add(os * n + cs),
                        n as isize,
                        1,
                    );
                }
            });
        };

        for _ in 0..3 {
            black_box(run_engine());
            black_box(run_grid());
        }
        let (mut e, mut g) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..rounds {
            for _ in 0..iters {
                let t = Instant::now();
                black_box(run_engine());
                e = e.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            for _ in 0..iters {
                let t = Instant::now();
                black_box(run_grid());
                g = g.min(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let gf = |ms: f64| flops / (ms * 1e-3) / 1e9;
        let total = g_m * g_n;
        println!(
            "m={m} k={k} n={n:<5} {:8.3}ms {:8.3}ms {:6.2}x   {g_m}x{g_n} = {total} chunks",
            e,
            g,
            e / g,
        );
        println!(
            "                         ({:6.1} GFLOP/s vs {:6.1})",
            gf(e),
            gf(g)
        );
    }
}
