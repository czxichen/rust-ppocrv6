//! Rigorous speed benchmark for rust_ppocrv6 on Windows.
//!
//! Usage:
//!   cargo run --release --example bench_speed
//!   cargo run --release --example bench_speed -- --iters 20
//!   cargo run --release --example bench_speed -- --img tests/data/bench_doc.png --core-only
//!
//! Sections:
//!   1. model load time (Small / Tiny)
//!   2. per-image latency: det / rec / end-to-end with percentiles
//!   3. cold-start first inference
//!   4. FP32 vs INT8 quantized recognition
//!   5. Tiny vs Small end-to-end
//!   6. angle CLS single-line
//!   7. parallel vs serial multi-box recognition (thread-scaling diagnostic)

use image::{DynamicImage, ImageReader};
use ppocr::{OcrEngine, OcrEngineConfig};
use std::hint::black_box;
use std::time::{Duration, Instant};

const DET_SMALL: &str = "models/PP-OCRv6_small_det.safetensors";
const REC_SMALL: &str = "models/PP-OCRv6_small_rec.safetensors";
const DICT_SMALL: &str = "models/ppocrv6_keys.txt";
const DET_TINY: &str = "models/PP-OCRv6_tiny_det.safetensors";
const REC_TINY: &str = "models/PP-OCRv6_tiny_rec.safetensors";
const DICT_TINY: &str = "models/ppocrv6_tiny_keys.txt";

const IMAGES: &[(&str, &str)] = &[
    ("单行短文本 480x128", "tests/data/bench_single_line.png"),
    ("单行超长文本 1440x72", "tests/data/bench_long_line.png"),
    ("多行文档 1000x1400", "tests/data/bench_doc.png"),
    ("高清混排 1920x1080", "tests/data/bench_hd.png"),
    ("仓库自带测试图 400x100", "tests/data/test_ocr.png"),
];

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[derive(Clone, Copy)]
struct Stats {
    min: f64,
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

impl Stats {
    fn from(mut v: Vec<f64>) -> Self {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        let pick = |q: f64| -> f64 {
            if n == 0 {
                return 0.0;
            }
            let idx = (((n as f64 - 1.0) * q).round() as usize).min(n - 1);
            v[idx]
        };
        Stats {
            min: v.first().copied().unwrap_or(0.0),
            mean: if n == 0 { 0.0 } else { v.iter().sum::<f64>() / n as f64 },
            p50: pick(0.5),
            p90: pick(0.9),
            p99: pick(0.99),
            max: v.last().copied().unwrap_or(0.0),
        }
    }

    fn line(&self) -> String {
        format!(
            "min {:8.2} | p50 {:8.2} | mean {:8.2} | p90 {:8.2} | p99 {:8.2} | max {:8.2}",
            self.min, self.p50, self.mean, self.p90, self.p99, self.max
        )
    }

    fn json(&self) -> String {
        format!(
            "{{\"min\":{:.3},\"p50\":{:.3},\"mean\":{:.3},\"p90\":{:.3},\"p99\":{:.3},\"max\":{:.3}}}",
            self.min, self.p50, self.mean, self.p90, self.p99, self.max
        )
    }
}

struct Scenario {
    label: String,
    boxes: usize,
    det: Stats,
    rec: Stats,
    e2e: Stats,
}

fn measure_engine(
    engine: &OcrEngine,
    images: &[(String, DynamicImage)],
    warmup: usize,
    iters: usize,
) -> Vec<Scenario> {
    let mut out = Vec::new();
    for (label, img) in images {
        let mut sample = String::new();
        for _ in 0..warmup {
            let r = engine.run(img).unwrap();
            if let Some(f) = r.first() {
                sample = f.text.clone();
            }
        }

        let mut dets = Vec::with_capacity(iters);
        let mut recs = Vec::with_capacity(iters);
        let mut e2es = Vec::with_capacity(iters);
        let mut boxes_n = 0usize;

        for _ in 0..iters {
            let t0 = Instant::now();
            let boxes = engine.detect(img).unwrap();
            let t1 = Instant::now();
            let res = engine.recognize(img, &boxes).unwrap();
            let t2 = Instant::now();

            black_box(res.len());
            boxes_n = boxes.len();
            dets.push(ms(t1 - t0));
            recs.push(ms(t2 - t1));
            e2es.push(ms(t2 - t0));
        }

        let det = Stats::from(dets);
        let rec = Stats::from(recs);
        let e2e = Stats::from(e2es);
        println!("  [{}]  检测框 {} 个 | 样本文本 \"{}\"", label, boxes_n, sample);
        println!("      检测 Det  : {}", det.line());
        println!("      识别 Rec  : {}", rec.line());
        println!("      端到端 E2E: {}", e2e.line());
        println!(
            "      吞吐      : {:8.2} FPS (整图) | {:6.2} ms/行 (识别)",
            if e2e.mean > 0.0 { 1000.0 / e2e.mean } else { 0.0 },
            if boxes_n > 0 { rec.mean / boxes_n as f64 } else { 0.0 }
        );
        out.push(Scenario {
            label: label.clone(),
            boxes: boxes_n,
            det,
            rec,
            e2e,
        });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut iters = 20usize;
    let mut only_img: Vec<String> = Vec::new();
    let mut core_only = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                    iters = v.max(3);
                }
                i += 2;
            }
            "--img" => {
                if let Some(v) = args.get(i + 1) {
                    only_img.push(v.clone());
                }
                i += 2;
            }
            "--core-only" => {
                core_only = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    let warmup = 5usize;

    println!("============================================================");
    println!("      rust_ppocrv6  Windows Speed Benchmark");
    println!("============================================================");
    println!("rayon threads = {}", rayon::current_num_threads());
    println!("warmup = {}, iterations = {}", warmup, iters);

    let images: Vec<(String, DynamicImage)> = IMAGES
        .iter()
        .filter(|(_, p)| std::path::Path::new(p).exists())
        .filter(|(_, p)| only_img.is_empty() || only_img.iter().any(|f| p.ends_with(f.as_str())))
        .map(|(l, p)| (l.to_string(), ImageReader::open(p).unwrap().decode().unwrap()))
        .collect();
    if images.is_empty() {
        eprintln!("no benchmark image matched");
        std::process::exit(1);
    }

    // ---------------- 1. Model load ----------------
    println!("\n----------------- 1. 模型加载耗时 (含权重解析 + 图优化) -----------------");
    let mut small_load_v = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let e = OcrEngine::new(DET_SMALL, REC_SMALL, DICT_SMALL).unwrap();
        small_load_v.push(ms(t.elapsed()));
        black_box(&e);
        drop(e);
    }
    let small_load = Stats::from(small_load_v);
    println!("  Small det+rec+dict : {}", small_load.line());

    let mut tiny_load_v = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let e = OcrEngine::new(DET_TINY, REC_TINY, DICT_TINY).unwrap();
        tiny_load_v.push(ms(t.elapsed()));
        black_box(&e);
        drop(e);
    }
    let tiny_load = Stats::from(tiny_load_v);
    println!("  Tiny  det+rec+dict : {}", tiny_load.line());

    // ---------------- 2. Small FP32 baseline ----------------
    println!("\n----------------- 2. Small FP32 分场景延迟 -----------------");
    let engine = OcrEngine::new(DET_SMALL, REC_SMALL, DICT_SMALL).unwrap();
    let small_scenarios = measure_engine(&engine, &images, warmup, iters);

    // ---------------- 3..6 optional ----------------
    let doc_idx = images.iter().position(|(l, _)| l.contains("多行文档"));
    let mut cold_stats = None;
    let mut fp32_stats = None;
    let mut int8_stats = None;
    let mut tiny_scenarios: Vec<Scenario> = Vec::new();
    let mut cls_stats = None;

    if !core_only {
        // 3. cold start
        if let Some(di) = doc_idx {
            println!("\n----------------- 3. 冷启动首帧推理 (多行文档) -----------------");
            let mut cold = Vec::new();
            for _ in 0..5 {
                let e = OcrEngine::new(DET_SMALL, REC_SMALL, DICT_SMALL).unwrap();
                let t = Instant::now();
                let _ = e.run(&images[di].1).unwrap();
                cold.push(ms(t.elapsed()));
            }
            let s = Stats::from(cold);
            println!("  首帧 (新引擎, 未预热): {}", s.line());
            cold_stats = Some(s);

            // 4. FP32 vs INT8
            println!("\n----------------- 4. 识别模型 FP32 vs INT8 (多行文档) -----------------");
            let doc = &images[di].1;
            let doc_boxes = engine.detect(doc).unwrap();
            let quant_engine = OcrEngine::with_config(
                DET_SMALL,
                REC_SMALL,
                DICT_SMALL,
                OcrEngineConfig {
                    quantize_rec_head: true,
                    ..Default::default()
                },
            )
            .unwrap();
            for _ in 0..warmup {
                let _ = engine.recognize(doc, &doc_boxes).unwrap();
                let _ = quant_engine.recognize(doc, &doc_boxes).unwrap();
            }
            let mut a = Vec::new();
            let mut b = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let _ = engine.recognize(doc, &doc_boxes).unwrap();
                a.push(ms(t.elapsed()));
                let t = Instant::now();
                let _ = quant_engine.recognize(doc, &doc_boxes).unwrap();
                b.push(ms(t.elapsed()));
            }
            let fa = Stats::from(a);
            let fb = Stats::from(b);
            println!("  FP32 ({} 行): {}", doc_boxes.len(), fa.line());
            println!("  INT8 ({} 行): {}", doc_boxes.len(), fb.line());
            println!("  加速比       : {:.2}x", fa.mean / fb.mean.max(1e-9));
            fp32_stats = Some(fa);
            int8_stats = Some(fb);

            // 5. Tiny vs Small
            println!("\n----------------- 5. Tiny vs Small 端到端 (多行文档) -----------------");
            let tiny_engine = OcrEngine::new(DET_TINY, REC_TINY, DICT_TINY).unwrap();
            tiny_scenarios = measure_engine(&tiny_engine, &images[di..di + 1], warmup, iters);
            let se = small_scenarios[di].e2e;
            let te = tiny_scenarios[0].e2e;
            println!(
                "  Small E2E {:.2} ms  vs  Tiny E2E {:.2} ms  -> Tiny 快 {:.2}x",
                se.mean,
                te.mean,
                se.mean / te.mean.max(1e-9)
            );

            // 6. CLS
            println!("\n----------------- 6. 方向分类 CLS 单行耗时 -----------------");
            let cls_engine = OcrEngine::with_config(
                DET_SMALL,
                REC_SMALL,
                DICT_SMALL,
                OcrEngineConfig {
                    use_angle_cls: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let one_boxes = engine.detect(&images[0].1).unwrap();
            if let Some(b0) = one_boxes.first() {
                let patch = ppocr::postprocess::get_rotate_crop_image(&images[0].1, &b0.points);
                for _ in 0..warmup {
                    let _ = cls_engine.classify_angle(&patch).unwrap();
                }
                let mut v = Vec::new();
                for _ in 0..iters {
                    let t = Instant::now();
                    let r = cls_engine.classify_angle(&patch).unwrap();
                    black_box(r);
                    v.push(ms(t.elapsed()));
                }
                let s = Stats::from(v);
                println!("  CLS 单行: {}", s.line());
                cls_stats = Some(s);
            } else {
                println!("  (跳过: 首张图未检出文本框)");
            }

            // 7. parallel vs serial multi-box recognition
            println!("\n----------------- 7. 多框识别: 并行(rayon) vs 串行 -----------------");
            let doc = &images[di].1;
            let boxes = engine.detect(doc).unwrap();
            for _ in 0..warmup {
                let _ = engine.recognize(doc, &boxes).unwrap();
                for b in &boxes {
                    let _ = engine.recognize(doc, std::slice::from_ref(b)).unwrap();
                }
            }
            let mut par_v = Vec::new();
            let mut ser_v = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let r = engine.recognize(doc, &boxes).unwrap();
                par_v.push(ms(t.elapsed()));
                black_box(r.len());

                let t = Instant::now();
                for b in &boxes {
                    let r = engine.recognize(doc, std::slice::from_ref(b)).unwrap();
                    black_box(r.len());
                }
                ser_v.push(ms(t.elapsed()));
            }
            let p = Stats::from(par_v);
            let s = Stats::from(ser_v);
            println!("  {} 框 并行(par_iter): {}", boxes.len(), p.line());
            println!("  {} 框 串行(逐框调用): {}", boxes.len(), s.line());
            println!(
                "  并行加速比: {:.2}x  (理想 ~{:.1}x)  |  并行均值 {:.0} ms, 串行均值 {:.0} ms",
                s.mean / p.mean.max(1e-9),
                rayon::current_num_threads() as f64,
                p.mean,
                s.mean
            );
        }
    }

    // ---------------- JSON ----------------
    let mut js = String::from("BENCH_JSON:");
    js.push_str(&format!(
        "{{\"iters\":{},\"threads\":{},\"load_small\":{},\"load_tiny\":{}",
        iters,
        rayon::current_num_threads(),
        small_load.json(),
        tiny_load.json()
    ));
    if let Some(s) = cold_stats {
        js.push_str(&format!(",\"cold_first\":{}", s.json()));
    }
    if let Some(s) = cls_stats {
        js.push_str(&format!(",\"cls\":{}", s.json()));
    }
    if let Some(s) = fp32_stats {
        js.push_str(&format!(",\"fp32_rec_doc\":{}", s.json()));
    }
    if let Some(s) = int8_stats {
        js.push_str(&format!(",\"int8_rec_doc\":{}", s.json()));
    }
    js.push_str(",\"scenarios_small\":[");
    for (i, sc) in small_scenarios.iter().enumerate() {
        if i > 0 {
            js.push(',');
        }
        js.push_str(&format!(
            "{{\"label\":\"{}\",\"boxes\":{},\"det\":{},\"rec\":{},\"e2e\":{}}}",
            sc.label,
            sc.boxes,
            sc.det.json(),
            sc.rec.json(),
            sc.e2e.json()
        ));
    }
    js.push_str("],\"scenarios_tiny\":[");
    for (i, sc) in tiny_scenarios.iter().enumerate() {
        if i > 0 {
            js.push(',');
        }
        js.push_str(&format!(
            "{{\"label\":\"{}\",\"boxes\":{},\"det\":{},\"rec\":{},\"e2e\":{}}}",
            sc.label,
            sc.boxes,
            sc.det.json(),
            sc.rec.json(),
            sc.e2e.json()
        ));
    }
    js.push_str("]}");
    println!("\n{}", "=".repeat(60));
    println!("{}", js);
}
