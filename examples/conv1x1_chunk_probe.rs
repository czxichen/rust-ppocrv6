// How finely should the 1x1 convolutions be split, and over which axis?
//
// The engine uses `chunk_n = n.div_ceil(threads).max(64).min(n)`. The `.max(64)` floor
// exists because a very small `sgemm` call spends more time packing than multiplying, but
// it has a cost: at n=144 on 16 threads it yields 3 chunks, so 13 of 16 threads idle.
// Those n=144 shapes (m=384, k=192) are the most frequent 1x1 in the rec model.
//
// The axis asymmetry is the thing to keep in mind. `sgemm` packs B on every call, and B is
// the *input* (k*n floats), shared by all m rows. Splitting n tiles B, so N chunks pack B
// once in total; splitting m leaves B whole in every call, so M chunks repack it M times.
// That is why the earlier fix moved these shapes off the m-split. The question here is how
// far n can be pushed before per-call packing takes over.
//
//   cargo run --release --example conv1x1_chunk_probe
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Minimum of many timed runs; this host varies 2x run to run.
fn bench_ms<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

fn main() {
    let iters = 40;
    // The real rec shapes, most frequent first.
    let shapes = [
        (384usize, 192usize, 144usize), // Conv.30/33/38/41, 6x24
        (192, 384, 144),                // Conv.31/34/39/42
        (768, 384, 708),                // Conv.49/52, 3x236
        (384, 768, 708),                // Conv.53
        (384, 192, 708),                // Conv.50
        (96, 48, 2304),                 // Conv.11
    ];
    let nt = rayon::current_num_threads();
    println!("threads = {nt}\n");

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1e-4).cos() * 0.01).collect();
        let mut out = vec![0.0f32; m * n];
        let p = out.as_mut_ptr() as usize;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let g = |ms: f64| flops / (ms * 1e-3) / 1e9;

        // Reference: one single-threaded sgemm.
        let ms1 = bench_ms(
            || unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    a.as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    p as *mut f32,
                    n as isize,
                    1,
                );
            },
            iters,
        );

        // n split into `c` equal column chunks, c from 1 to 32.
        let mut by_n: Vec<(usize, f64)> = Vec::new();
        for c in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32] {
            if c > n {
                continue;
            }
            let cn = n.div_ceil(c);
            let nc = n.div_ceil(cn);
            let ms = bench_ms(
                || {
                    (0..nc).into_par_iter().for_each(|ci| {
                        let cs = ci * cn;
                        let an = (n - cs).min(cn);
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
                },
                iters,
            );
            by_n.push((nc, ms));
        }

        // 2D grid: cm row blocks x cn column blocks. Only worth it once n is exhausted,
        // because each extra row block repacks all of B.
        let mut by_2d: Vec<(usize, usize, f64)> = Vec::new();
        for (cm, cn) in [(2usize, 8usize), (2, 16), (4, 4), (4, 8), (8, 2), (4, 16)] {
            if cm > m || cn > n {
                continue;
            }
            let rm = m.div_ceil(cm);
            let rn = n.div_ceil(cn);
            let gm = m.div_ceil(rm);
            let gn = n.div_ceil(rn);
            let ms = bench_ms(
                || {
                    (0..gm * gn).into_par_iter().for_each(|t| {
                        let mi = t / gn;
                        let ni = t % gn;
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
                },
                iters,
            );
            by_2d.push((gm * gn, cm, ms));
        }

        let engine_cn = n.div_ceil(nt).max(64).min(n);
        let engine_c = n.div_ceil(engine_cn);
        let engine_ms = by_n
            .iter()
            .find(|(c, _)| *c == engine_c)
            .map(|(_, ms)| *ms)
            .unwrap_or(f64::NAN);

        let best_n = by_n.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        let best_2d = by_2d.iter().min_by(|a, b| a.2.total_cmp(&b.2)).unwrap();

        println!("m={m} k={k} n={n}   single {ms1:7.3} ({:6.1})", g(ms1));
        print!("   n-split:");
        for (c, ms) in &by_n {
            let mark = if *c == engine_c { "*" } else { " " };
            print!("{mark}{c}:{ms:.3}({:5.1})", g(*ms));
        }
        println!();
        print!("   2d-grid:");
        for (c, cm, ms) in &by_2d {
            print!(" {}x{}:{:.3}({:5.1})", cm, c / cm, ms, g(*ms));
        }
        println!();
        println!(
            "   engine picks {engine_c} chunks -> {engine_ms:.3} ms ({:.1}); \
             best n-split is {} chunks -> {:.3} ms ({:.1}); best 2d is {} chunks -> {:.3} ms ({:.1})",
            g(engine_ms),
            best_n.0,
            best_n.1,
            g(best_n.1),
            best_2d.0,
            best_2d.2,
            g(best_2d.2)
        );
        println!();
    }
}
