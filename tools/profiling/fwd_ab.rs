//! Minimal forward-only A/B driver.
//!
//! The point of this file is that it uses **only** the tiny slice of the public
//! API that has been stable across revisions — `Model::load`, `Model::graph`,
//! `Model::forward` and `Tensor::from_vec` — so the *same source* compiles and
//! runs against both the baseline and the current tree. That makes the two
//! latency numbers a genuine A/B rather than two measurements of two different
//! programs.
//!
//! It deliberately does no preprocessing: the input is a raw `f32` blob
//! captured from the real pipeline (see `examples/onnx_ab.rs -- dump`), so both
//! revisions chew on byte-identical data and preprocessing cost cannot leak
//! into the comparison.
//!
//! ```text
//! fwd_ab <model.safetensors> <input.bin> <N,C,H,W> [--iters N] [--label L] [--out out.bin]
//! ```

use ppocr::{Model, Tensor};
use std::hint::black_box;
use std::time::Instant;

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let p50 = v[((n as f64 - 1.0) * 0.5).round() as usize];
    (v[0], p50, mean)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let model_path = a.get(1).ok_or("usage: fwd_ab <model.safetensors> <input.bin> <N,C,H,W>")?;
    let in_path = a.get(2).ok_or("missing <input.bin>")?;
    let shape: Vec<usize> = a
        .get(3)
        .ok_or("missing <N,C,H,W>")?
        .split(',')
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<Vec<usize>, _>>()?;

    let mut iters = 10usize;
    let mut label = String::from("?");
    let mut out_path: Option<String> = None;
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--iters" => {
                iters = a[i + 1].parse()?;
                i += 2;
            }
            "--label" => {
                label = a[i + 1].clone();
                i += 2;
            }
            "--out" => {
                out_path = Some(a[i + 1].clone());
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let bytes = std::fs::read(in_path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not a multiple of 4 bytes", bytes.len()).into());
    }
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let tensor = Tensor::from_vec(&shape, data)?;

    let t_load = Instant::now();
    let model = Model::load(model_path)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let name = model.graph().inputs[0].clone();
    let warmup = 3.max(iters / 4);
    for _ in 0..warmup {
        black_box(model.forward(&name, tensor.clone())?);
    }

    let mut v = Vec::with_capacity(iters);
    let mut last: Option<Tensor> = None;
    for _ in 0..iters {
        let t0 = Instant::now();
        let o = model.forward(&name, tensor.clone())?;
        v.push(t0.elapsed().as_secs_f64() * 1000.0);
        last = Some(o);
    }
    let o = last.ok_or("no output")?;
    let (mn, p50, mean) = stats(v);

    println!(
        "AB {label} in={} out={:?} load_ms={load_ms:.0} iters={iters} rayon={}",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x"),
        o.shape(),
        rayon::current_num_threads()
    );
    println!("RESULT {label} min={mn:.2} p50={p50:.2} mean={mean:.2}");

    if let Some(p) = out_path {
        let mut buf = Vec::with_capacity(o.numel() * 4);
        for x in o.as_slice() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(p, buf)?;
    }
    Ok(())
}
