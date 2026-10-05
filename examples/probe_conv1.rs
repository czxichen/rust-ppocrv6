use std::time::Instant;
use ppocr::Tensor;
use ppocr::ops::conv2d::{ConvActivation, Conv2dParams, conv2d};

fn main() {
    let threads = 12;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();

    // Det Conv.1: [12, 24, 2, 2] in=[1, 24, 480, 352]
    let input = Tensor::zeros(&[1, 24, 480, 352]);
    let weight = Tensor::zeros(&[12, 24, 2, 2]);
    let bias = Tensor::zeros(&[12]);
    let params = Conv2dParams {
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
        group: 1,
        activation: ConvActivation::HardSwish,
    };

    // Warmup
    for _ in 0..3 {
        let _ = conv2d(&input, &weight, Some(&bias), &params).unwrap();
    }

    let mut best_current = f64::INFINITY;
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = conv2d(&input, &weight, Some(&bias), &params).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best_current { best_current = ms; }
    }
    println!("Det Conv.1 current latency: {:.3} ms", best_current);

    // Det Conv.2: [24, 12, 2, 2] in=[1, 12, 480, 352]
    let input2 = Tensor::zeros(&[1, 12, 480, 352]);
    let weight2 = Tensor::zeros(&[24, 12, 2, 2]);
    let bias2 = Tensor::zeros(&[24]);
    for _ in 0..3 {
        let _ = conv2d(&input2, &weight2, Some(&bias2), &params).unwrap();
    }
    let mut best_conv2 = f64::INFINITY;
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = conv2d(&input2, &weight2, Some(&bias2), &params).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best_conv2 { best_conv2 = ms; }
    }
    println!("Det Conv.2 current latency: {:.3} ms", best_conv2);
}
