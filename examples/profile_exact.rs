//! Profile a single model's forward pass at an exact input shape.
//!
//! The per-model head-to-head against ONNX Runtime showed the biggest gap on the
//! small rec model (rec_tiny: 33.7 ms vs ORT 4.1 ms), which is surprising because
//! the Rust MatMul kernel is individually fast. This dumps the per-node breakdown
//! for the exact shape under test so the real hot spot is visible instead of
//! inferred.
//!
//! Usage: cargo run --release --example profile_exact -- <model.safetensors> N C H W

use std::env;

use ppocr::Tensor;
use ppocr::engine::Model;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        eprintln!("用法: profile_exact <model.safetensors> N C H W");
        std::process::exit(1);
    }
    let path = &args[1];
    let shape: Vec<usize> = args[2..6].iter().map(|s| s.parse().unwrap()).collect();
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| {
            let h = i.wrapping_mul(2654435761) >> 8;
            (h % 10007) as f32 / 10007.0
        })
        .collect();

    let model = Model::load(path).expect("加载失败");
    let x = Tensor::from_vec(&shape, data).expect("构造失败");

    for _ in 0..5 {
        let _ = model.forward("x", x.clone()).unwrap();
    }

    let iters = 10;
    let mut min_profile: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for _ in 0..iters {
        let (_, profile) = model.forward_with_profile("x", x.clone()).unwrap();
        for (op, ms) in profile {
            let entry = min_profile.entry(op).or_insert(f64::INFINITY);
            if ms < *entry {
                *entry = ms;
            }
        }
    }
    let mut items: Vec<_> = min_profile.into_iter().collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let total: f64 = items.iter().map(|(_, v)| v).sum();
    println!("模型 {path}  输入 {shape:?}");
    println!("节点数 {}", items.len());
    println!("{:-<64}", "");
    println!("{:40} {:>12} {:>8}", "节点", "耗时 ms", "占比");
    for (op, ms) in items.iter() {
        println!("{:55} {:>10.3} {:6.1}%", op, ms, ms / total * 100.0);
    }
    println!("{:-<64}", "");
    println!("{:40} {total:12.3}", "合计(剖析口径)");
}
