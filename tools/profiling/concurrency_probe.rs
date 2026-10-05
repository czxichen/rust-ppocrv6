//! 控制变量实验：把多行识别性能的两个改动分开量化。
//!
//! 上一版（f38cc60）同时引入了两件事：
//!   (a) 外层并发限流 `rec_batch_threads`（默认 4）+ 内层 `ScopedDisableParallel`；
//!   (b) `rec_max_w` 由 960 放宽到 3200。
//!
//! 前者号称「4 线程刚好打满内存吞吐」，后者宣称「修宽行丢字」。
//! 两者都会改变识别耗时，必须分别测量才能判断各自的真实收益/代价。
//!
//! 运行：cargo run --release --example concurrency_probe
//!      cargo run --release --example concurrency_probe -- <图片> <it ers>

use image::{DynamicImage, ImageReader};
use ppocr::{OcrEngine, OcrEngineConfig};
use std::time::Instant;

const DET: &str = "models/PP-OCRv6_small_det.safetensors";
const REC: &str = "models/PP-OCRv6_small_rec.safetensors";
const DICT: &str = "models/ppocrv6_keys.txt";

fn load(path: &str) -> DynamicImage {
    ImageReader::open(path)
        .expect("open image")
        .decode()
        .expect("decode image")
}

/// 框的包围盒宽高
fn wh(points: &[[f32; 2]; 4]) -> (f32, f32) {
    let (mut x0, mut x1) = (f32::MAX, f32::MIN);
    let (mut y0, mut y1) = (f32::MAX, f32::MIN);
    for p in points.iter() {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    ((x1 - x0).max(1.0), (y1 - y0).max(1.0))
}

fn main() {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/data/bench_doc.png".to_string());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let img = load(&img_path);
    println!("图片: {} ({}x{})", img_path, img.width(), img.height());

    let base = OcrEngine::new(DET, REC, DICT).expect("engine");
    let boxes = base.detect(&img).expect("detect");
    println!("\n检测框 {} 个。前 6 个框的宽高比与预处理后宽度：", boxes.len());
    println!(
        "  {:<6} {:>8} {:>8} {:>9} {:>12} {:>12}",
        "框", "宽", "高", "宽高比", "max_w=960", "max_w=3200"
    );
    let mut px960 = 0u64;
    let mut px3200 = 0u64;
    let mut ar_sum = 0f32;
    for (i, b) in boxes.iter().enumerate() {
        let (w, h) = wh(&b.points);
        let ar = w / h;
        ar_sum += ar;
        let tw = |mw: u32| ((48.0f32 * ar).ceil() as u32).clamp(32, mw);
        px960 += 48 * tw(960) as u64;
        px3200 += 48 * tw(3200) as u64;
        if i < 6 {
            println!(
                "  #{:<5} {:>8.1} {:>8.1} {:>9.2} {:>12} {:>12}",
                i, w, h, ar, tw(960), tw(3200)
            );
        }
    }
    println!("\n  平均宽高比 {:.2}", ar_sum / boxes.len().max(1) as f32);
    println!(
        "  rec 输入总像素: max_w=960 -> {:.2} Mpx | max_w=3200 -> {:.2} Mpx  (放大 {:.2}x)",
        px960 as f64 / 1e6,
        px3200 as f64 / 1e6,
        px3200 as f64 / px960.max(1) as f64
    );

    println!("\n============ 控制变量：rec_max_w x rec_batch_threads ============");
    println!(
        "  {:<10} {:<12} {:>13} {:>11} {:>14}",
        "rec_max_w", "线程数", "Rec(mean ms)", "ms/行", "相对基线"
    );
    let mut baseline: Option<f64> = None;
    for &max_w in &[960u32, 3200u32] {
        for &threads in &[1usize, 2, 4, 8, 16] {
            let cfg = OcrEngineConfig {
                rec_max_w: max_w,
                rec_batch_threads: Some(threads),
                ..Default::default()
            };
            let engine = OcrEngine::with_config(DET, REC, DICT, cfg).expect("engine");
            let _ = engine.recognize(&img, &boxes).expect("rec"); // warmup
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                let r = engine.recognize(&img, &boxes).expect("rec");
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&r);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            if baseline.is_none() {
                baseline = Some(mean);
            }
            println!(
                "  {:<10} {:<12} {:>13.1} {:>11.1} {:>13.2}x",
                max_w,
                threads,
                mean,
                mean / boxes.len().max(1) as f64,
                mean / baseline.unwrap()
            );
        }
    }

    println!("\n注：rec_batch_threads 走专用 rayon 线程池，池内线程禁用内层算子并行；");
    println!("    默认 None 等价于 min(4, CPU 逻辑核数)。");
}
