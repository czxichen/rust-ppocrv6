//! Verify that the Rust engine and the rebuilt ONNX model compute the same thing.
//!
//! Latency comparison against ONNX Runtime is meaningless unless both sides run
//! the same graph on the same input, so this dumps Rust-side outputs to disk in
//! a raw format that tools/compare_rust_onnx.py reads back and diffs against ORT.
//!
//! Usage:
//!   cargo run --release --example dump_forward -- <model.safetensors> <out.bin> [N C H W]
//!
//! Without N C H W a deterministic pseudo-random input is generated. The same
//! generator is reproduced in Python so both sides consume identical bytes.

use std::env;
use std::fs::File;
use std::io::Write;

use ppocr::Tensor;
use ppocr::engine::Model;

/// Deterministic input identical to the NumPy generator in compare_rust_onnx.py:
///   x[i] = ((i.wrapping_mul(2654435761) >> 8) % 10007) / 10007.0
/// Reproducing the integer hash exactly matters - a float PRNG would drift
/// between languages and make the comparison meaningless.
fn synth_input(numel: usize) -> Vec<f32> {
    (0..numel)
        .map(|i| {
            let h = i.wrapping_mul(2654435761) >> 8;
            (h % 10007) as f32 / 10007.0
        })
        .collect()
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: dump_forward <model.safetensors> <out.bin> [N C H W] [--bench N]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_path = &args[2];

    let bench_iters: usize = args
        .iter()
        .position(|a| a == "--bench")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let shape: Vec<usize> = if args.len() >= 7 && args[3] != "--bench" {
        args[3..7].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![1, 3, 48, 960]
    };

    let numel: usize = shape.iter().product();
    let data = synth_input(numel);

    if bench_iters == 0 {
        println!("模型: {model_path}");
        println!("输入形状: {shape:?}  ({numel} 元素)");
    }

    let model = Model::load(model_path).expect("加载模型失败");

    // Timed loop. The first call absorbs lazy per-shape preparation (Winograd
    // weight transforms, MatMul packing), so it is warmed up separately and
    // excluded - matching how the ONNX Runtime side is measured.
    let mut times: Vec<f64> = Vec::with_capacity(bench_iters.max(1));
    let mut out = None;
    for i in 0..bench_iters.max(1) {
        let input = Tensor::from_vec(&shape, data.clone()).expect("构造输入张量失败");
        let t0 = std::time::Instant::now();
        let r = model.forward("x", input).expect("前向计算失败");
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= 1 {
            times.push(dt);
        }
        out = Some(r);
    }

    let out = out.unwrap();
    let out_shape = out.shape().to_vec();
    let out_data = out.as_slice();

    if bench_iters == 0 {
        println!("输出形状: {out_shape:?}  ({} 元素)", out_data.len());
        println!(
            "输出统计: min={:.6} max={:.6} mean={:.6}",
            out_data.iter().cloned().fold(f32::INFINITY, f32::min),
            out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            out_data.iter().sum::<f32>() / out_data.len() as f32
        );
    } else {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "RUST ms: min={:.2} p50={:.2} mean={:.2} max={:.2}  (n={}, 已跳过首帧)",
            times[0],
            percentile(&times, 0.5),
            times.iter().sum::<f64>() / times.len() as f64,
            times[times.len() - 1],
            times.len()
        );
    }

    // Raw dump: magic, ndim, dims..., then f32 payload. Kept trivial so the
    // Python side can parse it without any serialisation library.
    let mut f = File::create(out_path).expect("创建输出文件失败");
    f.write_all(b"RSFWD1\0").unwrap();
    f.write_all(&(out_shape.len() as u64).to_le_bytes())
        .unwrap();
    for d in &out_shape {
        f.write_all(&(*d as u64).to_le_bytes()).unwrap();
    }
    for v in out_data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    if bench_iters == 0 {
        println!("已写出: {out_path}");
    }
}
