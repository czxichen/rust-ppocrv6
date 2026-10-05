//! Repeatable operator-level benchmarks.
//!
//! These exist so the latency claims in `README.md` can be re-measured by anyone instead
//! of being taken on faith (see review item C6). Run them with:
//!
//! ```bash
//! cargo bench --bench ops
//! ```
//!
//! Each benchmark uses shapes taken from the real PP-OCRv6 graphs so the numbers are
//! representative of the end-to-end pipeline rather than of toy matrices.

use criterion::{Criterion, criterion_group, criterion_main};
use ppocr::Tensor;
use ppocr::ops::{self, Conv2dParams, ConvActivation, Pool2dParams};
use std::hint::black_box;

/// Deterministic, non-degenerate input data.
fn pattern(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * scale + ((i % 13) as f32) * 0.001)
        .collect()
}

fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
    Tensor::from_vec(shape, data.to_vec()).expect("valid tensor")
}

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    // A backbone-style 3x3 convolution.
    let (cin, cout, h, w) = (64usize, 64usize, 32usize, 32usize);
    let x = tensor(&[1, cin, h, w], &pattern(cin * h * w, 1.0));
    let weight = tensor(&[cout, cin, 3, 3], &pattern(cout * cin * 9, 0.05));
    let bias = tensor(&[cout], &pattern(cout, 0.1));
    let params = Conv2dParams {
        pads: [1, 1, 1, 1],
        ..Default::default()
    };
    group.bench_function("3x3_c64_32x32", |bench| {
        bench.iter(|| {
            ops::conv2d(
                black_box(&x),
                black_box(&weight),
                Some(black_box(&bias)),
                black_box(&params),
            )
            .expect("conv2d")
        })
    });

    // Depthwise 3x3, i.e. the mobile-block kernel.
    let c_ch = 128usize;
    let dw_x = tensor(&[1, c_ch, h, w], &pattern(c_ch * h * w, 1.0));
    let dw_w = tensor(&[c_ch, 1, 3, 3], &pattern(c_ch * 9, 0.3));
    let dw_params = Conv2dParams {
        pads: [1, 1, 1, 1],
        group: c_ch,
        ..Default::default()
    };
    group.bench_function("depthwise_3x3_c128_32x32", |bench| {
        bench.iter(|| {
            ops::conv2d(
                black_box(&dw_x),
                black_box(&dw_w),
                None,
                black_box(&dw_params),
            )
            .expect("conv2d")
        })
    });

    // The 1x1 pointwise kernel.
    let pw_w = tensor(&[cout, cin, 1, 1], &pattern(cout * cin, 0.05));
    group.bench_function("1x1_c64_to_c64_32x32", |bench| {
        bench.iter(|| {
            ops::conv2d(
                black_box(&x),
                black_box(&pw_w),
                Some(black_box(&bias)),
                black_box(&Conv2dParams::default()),
            )
            .expect("conv2d")
        })
    });

    group.finish();
}

fn bench_winograd(c: &mut Criterion) {
    let mut group = c.benchmark_group("winograd");

    let (cin, cout, h, w) = (64usize, 32usize, 32usize, 32usize);
    let x = tensor(&[1, cin, h, w], &pattern(cin * h * w, 1.0));
    let weight = tensor(&[cout, cin, 3, 3], &pattern(cout * cin * 9, 0.05));
    let u = ops::pretransform_weights_f23(&weight).expect("pretransform");

    group.bench_function("f23_conv_c64_to_c32_32x32", |bench| {
        bench.iter(|| {
            ops::winograd_f23_conv2d(black_box(&x), black_box(&u), None, ConvActivation::None)
                .expect("winograd")
        })
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // The rec model's vocabulary projection: [1, 120] x [18710, 120]^T.
    let (k, n) = (120usize, 18710usize);
    let a = tensor(&[1, k], &pattern(k, 1.0));
    let b_t = tensor(&[n, k], &pattern(n * k, 0.2));
    group.bench_function("transposed_b_1x120_by_18710x120", |bench| {
        bench.iter(|| {
            ops::matmul_transposed_b(black_box(&a), black_box(&b_t), None).expect("matmul")
        })
    });

    // The same projection with INT8 weights.
    let qw = ops::QuantizedWeight::from_f32_transposed(&b_t);
    group.bench_function("int8_1x120_by_18710x120", |bench| {
        bench.iter(|| ops::matmul_quantized_b(black_box(&a), black_box(&qw), None).expect("matmul"))
    });

    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // A single 18710-way row, i.e. exactly what the CTC head decodes from.
    let row = tensor(&[1, 18710], &pattern(18710, 4.0));
    group.bench_function("18710_way_row", |bench| {
        bench.iter(|| ops::softmax(black_box(&row), -1).expect("softmax"))
    });

    group.finish();
}

fn bench_layernorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm");

    let (rows, cols) = (120usize, 120usize);
    let x = tensor(&[rows, cols], &pattern(rows * cols, 2.0));
    let w = tensor(&[cols], &pattern(cols, 0.5));
    let bias = tensor(&[cols], &pattern(cols, 0.2));
    group.bench_function("120x120_with_affine", |bench| {
        bench.iter(|| {
            ops::layer_norm(
                black_box(&x),
                black_box(&[cols]),
                Some(black_box(&w)),
                Some(black_box(&bias)),
                1e-5,
            )
            .expect("layernorm")
        })
    });

    group.finish();
}

fn bench_pool_and_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_resize");

    let x = tensor(&[1, 64, 64, 64], &pattern(64 * 64 * 64, 1.0));
    let pool_params = Pool2dParams {
        kernel_shape: [3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
    };
    group.bench_function("max_pool_3x3_s2_c64_64x64", |bench| {
        bench.iter(|| ops::max_pool2d(black_box(&x), black_box(&pool_params)).expect("pool"))
    });

    group.bench_function("global_avg_pool_c64_64x64", |bench| {
        bench.iter(|| ops::global_avg_pool2d(black_box(&x)).expect("pool"))
    });

    group.bench_function("bilinear_resize_64x64_to_128x128", |bench| {
        bench.iter(|| ops::resize_bilinear_2d(black_box(&x), 128, 128, false).expect("resize"))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_conv2d,
    bench_winograd,
    bench_matmul,
    bench_softmax,
    bench_layernorm,
    bench_pool_and_resize
);
criterion_main!(benches);
