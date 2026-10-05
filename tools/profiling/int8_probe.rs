//! FP32 vs INT8 (W8A32) recognition comparison on a full page.
//!
//! Both engines are measured in the same loop so machine drift cancels out;
//! the recognised text is compared to confirm the quantisation is lossless.
//!
//! Usage: int8_probe <iters> <image-path>

use image::ImageReader;
use ppocr::{OcrEngine, OcrEngineConfig};
use std::hint::black_box;
use std::time::Instant;

const DET: &str = "models/PP-OCRv6_small_det.safetensors";
const REC: &str = "models/PP-OCRv6_small_rec.safetensors";
const DICT: &str = "models/ppocrv6_keys.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let img_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tests/data/bench_doc.png".to_string());

    let fp32 = OcrEngine::new(DET, REC, DICT).expect("fp32 init");
    let int8 = OcrEngine::with_config(
        DET,
        REC,
        DICT,
        OcrEngineConfig {
            quantize_rec_head: true,
            ..Default::default()
        },
    )
    .expect("int8 init");

    let img = ImageReader::open(&img_path).expect("open").decode().expect("decode");
    let boxes = fp32.detect(&img).unwrap();

    let fw = fp32.rec_model().weight_footprint_bytes();
    let iw = int8.rec_model().weight_footprint_bytes();
    println!(
        "WORK boxes={} fp32_weight={:.2}MB int8_weight={:.2}MB saved={:.2}MB ({:.1}%)",
        boxes.len(),
        fw as f64 / 1e6,
        iw as f64 / 1e6,
        (fw - iw) as f64 / 1e6,
        (fw - iw) as f64 / fw as f64 * 100.0
    );

    for _ in 0..4 {
        black_box(fp32.recognize(&img, &boxes).unwrap().len());
    }
    for _ in 0..4 {
        black_box(int8.recognize(&img, &boxes).unwrap().len());
    }

    let mut a = Vec::with_capacity(iters);
    let mut b = Vec::with_capacity(iters);
    let mut ta = String::new();
    let mut tb = String::new();

    for _ in 0..iters {
        let t = Instant::now();
        let r = fp32.recognize(&img, &boxes).unwrap();
        a.push(t.elapsed().as_secs_f64() * 1000.0);
        if ta.is_empty() {
            ta = r.iter().map(|x| x.text.clone()).collect::<Vec<_>>().join("|");
        }

        let t = Instant::now();
        let r = int8.recognize(&img, &boxes).unwrap();
        b.push(t.elapsed().as_secs_f64() * 1000.0);
        if tb.is_empty() {
            tb = r.iter().map(|x| x.text.clone()).collect::<Vec<_>>().join("|");
        }
    }

    let stat = |mut v: Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v[0], v[v.len() / 2], m)
    };
    let (amin, ap50, amean) = stat(a);
    let (bmin, bp50, bmean) = stat(b);

    println!("FP32 min={amin:.1} p50={ap50:.1} mean={amean:.1}");
    println!("INT8 min={bmin:.1} p50={bp50:.1} mean={bmean:.1}");
    println!(
        "SPEEDUP mean={:.3}x p50={:.3}x min={:.3}x",
        amean / bmean, ap50 / bp50, amin / bmin
    );
    println!("TEXT_IDENTICAL={}", ta == tb);
}
