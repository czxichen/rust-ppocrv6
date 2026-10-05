//! Proof-of-concept: batched recognition forward vs the current per-line loop.
//!
//! The current `OcrEngine::recognize` calls `rec_model.forward` once per text box
//! (tensor shape [1, 3, 48, W]). This example packs N crops into a single
//! [N, 3, 48, Wmax] tensor (right-padded with white) and measures the win.
//!
//!   cargo run --release --example batch_poc

use image::ImageReader;
use ppocr::{OcrEngine, Tensor};
use std::hint::black_box;
use std::time::Instant;

fn build_batch(tensors: &[Tensor]) -> (Tensor, usize) {
    let n = tensors.len();
    let h = tensors[0].shape()[2];
    let wmax = tensors.iter().map(|t| t.shape()[3]).max().unwrap();
    // rec normalisation maps white 255 -> +1.0, so pad with +1.0
    let mut data = vec![1.0f32; n * 3 * h * wmax];
    for (i, t) in tensors.iter().enumerate() {
        let s = t.as_slice();
        let w = t.shape()[3];
        for c in 0..3 {
            for y in 0..h {
                let src = c * h * w + y * w;
                let dst = i * 3 * h * wmax + c * h * wmax + y * wmax;
                data[dst..dst + w].copy_from_slice(&s[src..src + w]);
            }
        }
    }
    (Tensor::from_vec(&[n, 3, h, wmax], data).unwrap(), wmax)
}

fn main() {
    let engine = OcrEngine::new(
        "models/PP-OCRv6_small_det.safetensors",
        "models/PP-OCRv6_small_rec.safetensors",
        "models/ppocrv6_keys.txt",
    )
    .unwrap();

    let img = ImageReader::open("tests/data/bench_doc.png")
        .unwrap()
        .decode()
        .unwrap();
    let boxes = engine.detect(&img).unwrap();
    println!("检测框数量: {}", boxes.len());

    let mut tensors = Vec::new();
    for b in &boxes {
        let patch = ppocr::postprocess::get_rotate_crop_image(&img, &b.points);
        tensors.push(engine.rec_preprocess().process(&patch).unwrap());
    }
    let widths: Vec<usize> = tensors.iter().map(|t| t.shape()[3]).collect();
    println!("各文本行 rec 输入宽度: {:?}", widths);

    let rec_in = engine.rec_model().graph().inputs[0].clone();
    let warmup = 3;

    println!("\n{:<6}{:<8}{:>16}{:>16}{:>16}{:>12}", "N", "padW", "逐行耗时(ms)", "批量耗时(ms)", "逐行/行", "加速比");
    println!("{}", "-".repeat(76));

    for &n in &[1usize, 2, 4, 8, 16, 29] {
        if n > tensors.len() {
            continue;
        }
        let group = &tensors[..n];

        for _ in 0..warmup {
            for t in group {
                black_box(engine.rec_model().forward(&rec_in, t.clone()).unwrap());
            }
        }
        let t0 = Instant::now();
        let mut single_out = Vec::new();
        for t in group {
            single_out.push(engine.rec_model().forward(&rec_in, t.clone()).unwrap());
        }
        let serial = t0.elapsed().as_secs_f64() * 1000.0;

        let (batch, wmax) = build_batch(group);
        for _ in 0..warmup {
            black_box(engine.rec_model().forward(&rec_in, batch.clone()).unwrap());
        }
        let t1 = Instant::now();
        let batched = engine.rec_model().forward(&rec_in, batch).unwrap();
        let batch_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // sanity: first sample of the batch must match the single forward
        let sc = single_out[0].shape();
        let bc = batched.shape();
        let sample_len: usize = sc.iter().product();
        let diff = if bc.len() == sc.len() && bc[0] == n {
            batched.as_slice()[..sample_len]
                .iter()
                .zip(single_out[0].as_slice().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        } else {
            f32::NAN
        };

        println!(
            "{:<6}{:<8}{:>16.2}{:>16.2}{:>16.2}{:>11.2}x",
            n,
            wmax,
            serial,
            batch_ms,
            serial / n as f64,
            serial / batch_ms
        );
        println!(
            "       batch out shape {:?} vs single {:?} | 样本0 最大绝对误差 {:.3e}",
            bc, sc, diff
        );
    }
}
