// faer vs matrixmultiply, serial, on the engine's actual memory layout.
//
// What this probe has to get right, each of which was wrong in an earlier version:
//
// 1. **Serial on both sides.** The engine's 1x1 path already parallelises over output columns
//    itself (Strategy A in `src/ops/conv2d.rs`), handing each worker one GEMM. faer switches
//    to rayon above `PAR_THRESHOLD_MNK = 4096` and every shape here is above it, so the
//    default policy put all 16 threads inside a call whose caller had already parallelised.
//    That measured as a fictitious 3.84x, including 390 GFLOP/s at 1024^3 -- over 4x the
//    single-core ceiling. This pins `Par::Seq`.
//
// 2. **Identical memory, no free transpose.** faer is column-major. The first probe used
//    `Mat::from_fn`, which *copies* into contiguous column-major storage outside the timed
//    region, so faer read its preferred layout while `sgemm` read the engine's actual strided
//    buffers. This builds `MatRef`/`MatMut` views over the same row-major memory.
//
// 3. **The destination is a strided sub-block, not a compact one.** A column chunk gets
//    `actual_n` columns of a row that is `n` wide, so the output stride is `n` while the
//    chunk is narrower. With stride == chunk width the chunk looks compact, and a kernel
//    that mishandles strided output would sail through.
//
// 4. **Correctness is checked on every shape, not a subset.** The second probe gated the
//    check on `m * k <= 4096`, which skipped all nine shapes -- so it measured faer's speed
//    without ever confirming faer computed the right answer. The check is cheap next to 40
//    GEMM iterations.
//
// Ceiling note: the i7-10700 is Comet Lake, has *no* AVX-512 (every `avx512*` bit reads
// false, `examples/avx512_probe.rs`), and has two 256-bit FMA units, so AVX2 peak is
// 32 FLOP/cycle: 92.8 GFLOP/s at 2.9 GHz base, ~147 at single-core turbo.
//
//   cargo run --release --example faer_probe
use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, MatMut, MatRef, Par};
use std::hint::black_box;
use std::time::Instant;

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

/// (m, k, chunk_n, stride, label). `stride` is the destination row stride; for a column
/// chunk of an `n`-wide row it is `n`, not `chunk_n`.
const CALLS: &[(usize, usize, usize, usize, &str)] = &[
    // The chunk sequence Strategy A issues for n=144 on 16 threads:
    // chunk_n = max(144/16, 64) = 64, so 64 + 64 + 16.
    (384, 192, 144, 144, "FULL n=144 (sequential path)"),
    (384, 192, 64, 144, "chunk 1 of 144, stride 144"),
    (384, 192, 16, 144, "LAST chunk of 144, stride 144"),
    (192, 384, 144, 144, "FULL n=144 k=384 (sequential)"),
    (192, 384, 16, 144, "LAST chunk of 144 k=384, stride 144"),
    // The rest of the real rec shapes: n=2304 -> chunk 144, n=708 -> chunk 45.
    (96, 48, 144, 2304, "Conv.11/8         n=2304 chunk 144"),
    (192, 96, 144, 2304, "Conv.31          n=2304 chunk 144"),
    (768, 384, 45, 708, "Conv.49/52       n=708 chunk 45"),
    (384, 768, 45, 708, "Conv.53          n=708 chunk 45"),
    (384, 192, 45, 708, "Conv.50          n=708 chunk 45"),
    (768, 384, 708, 708, "n=708 unchunked"),
    (1024, 1024, 1024, 1024, "square 1024^3"),
];

fn main() {
    println!(
        "{:<40} {:>7} {:>8} {:>8} {:>7} {:>8} {:>8}  {}",
        "call", "MFLOP", "mm ms", "mm GF/s", "ratio", "faer ms", "faer GF/s", "check"
    );
    let mut mm_total = 0.0f64;
    let mut faer_total = 0.0f64;
    let mut wrong = 0usize;

    for &(m, k, n, stride, label) in CALLS {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
        let b: Vec<f32> = (0..k * stride).map(|i| (i as f32 * 1e-4).cos()).collect();
        let mut c = vec![0.0f32; m * stride];
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        // matrixmultiply, serial, into a preallocated strided destination.
        let ms_mm = bench_ms(
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
                    stride as isize,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    stride as isize,
                    1,
                );
            },
            40,
        );

        // faer, serial, same memory. A row-major [r, c] buffer is a column-major [c, r]
        // matrix with row stride 1 and column stride `c`: element (i, j) lands at
        // `i + j * c`, the original `j * c + i`. So the transpose is a stride change.
        let (a_t, b_t, mut c_t) = unsafe {
            (
                MatRef::from_raw_parts(a.as_ptr(), k, m, 1isize, k as isize),
                MatRef::from_raw_parts(b.as_ptr(), n, k, 1isize, stride as isize),
                MatMut::from_raw_parts_mut(c.as_mut_ptr(), n, m, 1isize, stride as isize),
            )
        };
        let ms_faer = bench_ms(
            || {
                faer_matmul(&mut c_t, Accum::Replace, &b_t, &a_t, 1.0, Par::Seq);
            },
            40,
        );

        // Correctness against a direct f32 dot product, on every shape.
        let mut worst = 0.0f64;
        let mut scale = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        for j in 0..m {
            for i in 0..n {
                let want: f32 = (0..k).map(|p| a[j * k + p] * b[p * stride + i]).sum();
                let d = (c[j * stride + i] - want).abs() as f64;
                if d > worst {
                    worst = d;
                    worst_at = (j, i);
                }
                scale = scale.max(want.abs());
            }
        }
        let rel = worst / (scale as f64).max(1e-6);
        let ok = rel < 1e-5;
        if !ok {
            wrong += 1;
            println!(
                "  MISMATCH {label}: max_abs {worst:.3e} rel {rel:.2e} at row {} col {}",
                worst_at.0, worst_at.1
            );
        }

        mm_total += ms_mm;
        faer_total += ms_faer;
        let g = |ms: f64| flops / (ms * 1e-3) / 1e9;
        println!(
            "sgemm({m:>4},{k:>4},{n:>4}) s={stride:<4}{:<6} {:7.1} {:8.3} {:8.1} {:6.2}x {:8.3} {:8.1}  {:5} {label}",
            "",
            flops / 1e6,
            ms_mm,
            g(ms_mm),
            ms_mm / ms_faer,
            ms_faer,
            g(ms_faer),
            if ok { "ok" } else { "WRONG" },
        );
    }
    println!(
        "\ntotal serial, identical memory: matrixmultiply {mm_total:.3} ms vs faer {faer_total:.3} ms = {:.2}x",
        mm_total / faer_total
    );
    println!("shapes failing the correctness check: {wrong}");
}
