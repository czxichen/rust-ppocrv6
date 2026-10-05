//! Isolate where the time goes for the det model's biggest hotspot.
//!
//! Findings so far:
//!   * In-model profile: `Conv.0` (3->24ch, k3s2, input 960x704) = 74.0 ms
//!   * Same op called in isolation (warm, tensor reused)  = 12.3 ms
//! This probe separates first-call / cold-tensor / steady-state behaviour.
//!
//!   cargo run --release --example op_probe

use ppocr::ops::conv2d::{Conv2dParams, ConvActivation, conv2d};
use ppocr::Tensor;
use std::time::Instant;

fn params() -> Conv2dParams {
    Conv2dParams {
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
        group: 1,
        activation: ConvActivation::None,
    }
}

fn main() {
    println!("{:>9}  {:>38}   |   {:>32}", "input", "reused (hot) input, ms", "fresh input each call, ms");
    println!("{:>9}  {:>10}{:>9}{:>9}   |   {:>15}{:>15}", "", "first", "mean", "best", "mean", "best");

    for (h, w) in [(128usize, 416usize), (480, 352), (960, 704)] {
        let weight =
            Tensor::from_vec(&[24, 3, 3, 3], vec![0.01f32; 24 * 3 * 3 * 3]).unwrap();
        let bias = Tensor::from_vec(&[24], vec![0.0f32; 24]).unwrap();
        const N: usize = 12;

        // ---- A: reused (hot) input tensor ----
        let hot = Tensor::from_vec(&[1, 3, h, w], vec![0.5f32; 3 * h * w]).unwrap();
        let mut first_hot = 0.0;
        let mut best_hot = f64::INFINITY;
        let mut sum_hot = 0.0;
        for i in 0..N {
            let t = Instant::now();
            let o = conv2d(&hot, &weight, Some(&bias), &params()).unwrap();
            let el = t.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&o);
            if i == 0 {
                first_hot = el;
            }
            sum_hot += el;
            if el < best_hot {
                best_hot = el;
            }
        }

        // ---- B: fresh input tensor every call (cache-cold working set) ----
        let mut best_cold = f64::INFINITY;
        let mut sum_cold = 0.0;
        for _ in 0..N {
            let fresh = Tensor::from_vec(&[1, 3, h, w], vec![0.5f32; 3 * h * w]).unwrap();
            let t = Instant::now();
            let o = conv2d(&fresh, &weight, Some(&bias), &params()).unwrap();
            let el = t.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&o);
            sum_cold += el;
            if el < best_cold {
                best_cold = el;
            }
        }

        println!(
            "{:>9}  {:>10.2}{:>9.2}{:>9.2}   |   {:>15.2}{:>15.2}",
            format!("{}x{}", h, w),
            first_hot,
            sum_hot / N as f64,
            best_hot,
            sum_cold / N as f64,
            best_cold
        );
    }
}
