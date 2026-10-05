//! Dump the FULL per-operator profile (not just top-N) for auditing kernel efficiency.
//!
//! Output format (tab separated) so it can be post-processed:
//!   CASE <name> <total_ms>
//!   <op_label> \t <ms>
//!
//!   cargo run --release --example profile_all > profile_dump.txt

use ppocr::{Model, Tensor};
use std::collections::HashMap;

fn run_case(model: &Model, name: &str, shape: &[usize], reps: usize) {
    let input_name = model.graph().inputs[0].clone();
    let mut total_ms = 0.0f64;
    let mut acc: HashMap<String, f64> = HashMap::new();

    for r in 0..reps {
        let t = Tensor::new(shape);
        let t0 = std::time::Instant::now();
        let (out, prof) = match model.forward_with_profile(&input_name, t) {
            Ok(v) => v,
            Err(e) => {
                println!("CASE\t{}\tERROR\t{}", name, e);
                return;
            }
        };
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&out);
        for (k, v) in prof {
            *acc.entry(k).or_insert(0.0) += v;
        }
        let _ = r;
    }

    println!("CASE\t{}\t{:.3}\t{}", name, total_ms / reps as f64, acc.len());
    let mut v: Vec<(String, f64)> = acc.into_iter().map(|(k, s)| (k, s / reps as f64)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (k, ms) in v {
        println!("{}\t{:.4}", k, ms);
    }
    println!("ENDCASE");
}

fn main() {
    let det_small = Model::load("models/PP-OCRv6_small_det.safetensors").unwrap();
    let rec_small = Model::load("models/PP-OCRv6_small_rec.safetensors").unwrap();
    let det_tiny = Model::load("models/PP-OCRv6_tiny_det.safetensors").unwrap();
    let rec_tiny = Model::load("models/PP-OCRv6_tiny_rec.safetensors").unwrap();
    let cls = Model::load("models/PP-OCRv6_cls.safetensors").unwrap();

    let reps = 3;

    // det: A4-class document input (1000x1400 -> 960x704) and the tiny repo test image (400x100 -> 128x416)
    run_case(&det_small, "det_small_doc", &[1, 3, 960, 704], reps);
    run_case(&det_small, "det_small_small", &[1, 3, 128, 416], reps);
    run_case(&det_tiny, "det_tiny_doc", &[1, 3, 960, 704], reps);

    // rec: a wide text line (clamped to max_w=960) and a short one
    run_case(&rec_small, "rec_small_wide", &[1, 3, 48, 960], reps);
    run_case(&rec_small, "rec_small_narrow", &[1, 3, 48, 352], reps);
    run_case(&rec_tiny, "rec_tiny_wide", &[1, 3, 48, 960], reps);

    // cls: single text line orientation
    run_case(&cls, "cls_line", &[1, 3, 48, 192], reps);
}
