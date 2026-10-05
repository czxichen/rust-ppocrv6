//! Focused A/B latency probe used to compare two builds of the engine.
//!
//! Deliberately tiny and deterministic: one model pair, one image, fixed
//! iteration count, and a checksum of the recognised text so we can prove that
//! both builds performed *identical work*.
//!
//! Usage: ab_latency <iters> <image-path>
//!
//! Wall-clock numbers alone are not trustworthy on a shared machine, so this
//! binary is meant to be run under `/usr/bin/time -l` and compared on CPU time
//! (user+sys) as well as on the *minimum* wall-clock sample.

use image::ImageReader;
use ppocr::OcrEngine;
use std::hint::black_box;
use std::time::Instant;

const DET: &str = "models/PP-OCRv6_small_det.safetensors";
const REC: &str = "models/PP-OCRv6_small_rec.safetensors";
const DICT: &str = "models/ppocrv6_keys.txt";

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let p50 = v[((n as f64 - 1.0) * 0.5).round() as usize];
    (v[0], p50, mean)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let img_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tests/data/bench_doc.png".to_string());

    let engine = OcrEngine::new(DET, REC, DICT).expect("engine init failed");
    let img = ImageReader::open(&img_path)
        .expect("open failed")
        .decode()
        .expect("decode failed");

    // Warm up: page in weights, populate the buffer pool, let caches settle.
    for _ in 0..5 {
        black_box(engine.run(&img).unwrap().len());
    }

    let probe = engine.detect(&img).unwrap();
    let nbox = probe.len();

    let mut det_v = Vec::with_capacity(iters);
    let mut rec_v = Vec::with_capacity(iters);
    let mut e2e_v = Vec::with_capacity(iters);
    let mut chars = 0usize;
    let mut sample = String::new();

    for _ in 0..iters {
        let t0 = Instant::now();
        let boxes = engine.detect(&img).unwrap();
        let t1 = Instant::now();
        let res = engine.recognize(&img, &boxes).unwrap();
        let t2 = Instant::now();

        chars += res.iter().map(|r| r.text.chars().count()).sum::<usize>();
        if sample.is_empty() {
            sample = res
                .iter()
                .map(|r| r.text.as_str())
                .collect::<Vec<_>>()
                .join("|");
        }
        black_box(res.len());

        det_v.push((t1 - t0).as_secs_f64() * 1000.0);
        rec_v.push((t2 - t1).as_secs_f64() * 1000.0);
        e2e_v.push((t2 - t0).as_secs_f64() * 1000.0);
    }

    let (dmin, dp50, dmean) = stats(det_v);
    let (rmin, rp50, rmean) = stats(rec_v);
    let (emin, ep50, emean) = stats(e2e_v);

    println!("WORK iters={iters} img={img_path} boxes={nbox} chars={chars}");
    println!("DET min={dmin:.2} p50={dp50:.2} mean={dmean:.2}");
    println!("REC min={rmin:.2} p50={rp50:.2} mean={rmean:.2}");
    println!("E2E min={emin:.2} p50={ep50:.2} mean={emean:.2}");
    println!("SAMPLE {sample}");
}
