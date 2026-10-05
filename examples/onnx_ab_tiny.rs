//! Cross-check the engine against ONNX Runtime on *byte-identical* inputs.
//!
//! Wall-clock numbers mean nothing if the two sides are not computing the same
//! thing, so this probe has two jobs:
//!
//! * `dump <image> <out_dir>` — run the engine's own preprocessing and write out
//!   the exact `f32` tensors it hands to the detection and recognition networks,
//!   plus the engine's reference outputs. An ORT session fed these tensors is
//!   doing the same arithmetic, and the dumped outputs let us prove it.
//! * `time <image> <iters>` — measure forward-only latency for det and rec, and
//!   the page-level `recognize` / `run` paths.
//!
//! Note that the recognition path is *per-box, batch = 1*: the engine parallelises
//! across boxes and disables nested parallelism inside each box. Any comparison
//! must mirror that split or it is measuring a different program.

use image::ImageReader;
use ppocr::{OcrEngine, Tensor, postprocess::get_rotate_crop_image_rgb};
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let p50 = v[((n as f64 - 1.0) * 0.5).round() as usize];
    (v[0], p50, mean)
}

fn write_f32(path: &Path, t: &Tensor) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(t.numel() * 4);
    for v in t.as_slice() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn shape_str(t: &Tensor) -> String {
    t.shape()
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Dump every tensor the ORT side needs, and the engine's outputs to check against.
fn cmd_dump(img_path: &str, out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out_dir)?;
    let engine = OcrEngine::default_tiny()?;
    let img = ImageReader::open(img_path)?.decode()?;

    let (det_in, scale_h, scale_w) = engine.det_preprocess().process(&img)?;
    let det_name = engine.det_model().graph().inputs[0].clone();
    let det_out = engine.det_model().forward(&det_name, det_in.clone())?;
    let boxes = engine.db_postprocess().process(&det_out, scale_h, scale_w);

    write_f32(&out_dir.join("det_in.bin"), &det_in)?;
    write_f32(&out_dir.join("det_out.bin"), &det_out)?;

    let mut manifest = String::new();
    let mut line = |name: &str, t: &Tensor| {
        manifest.push_str(&format!("TENSOR {name} shape={}\n", shape_str(t)));
    };
    line("det_in", &det_in);
    line("det_out", &det_out);

    let rgb = img.to_rgb8();
    let rec_name = engine.rec_model().graph().inputs[0].clone();
    for (i, b) in boxes.iter().enumerate() {
        let patch = get_rotate_crop_image_rgb(&rgb, &b.points);
        let t = engine.rec_preprocess().process(&patch)?;
        let preds = engine.rec_model().forward(&rec_name, t.clone())?;
        write_f32(&out_dir.join(format!("rec_in_{i:03}.bin")), &t)?;
        write_f32(&out_dir.join(format!("rec_out_{i:03}.bin")), &preds)?;
        line(&format!("rec_in_{i:03}"), &t);
        line(&format!("rec_out_{i:03}"), &preds);
    }

    // Full pipeline reference, for an end-to-end text check on the ORT side.
    let results = engine.run(&img)?;
    let mut texts = Vec::new();
    for r in &results {
        texts.push(format!("{}|{:.4}", r.text.replace('\n', "\\n"), r.score));
    }

    let mut fh = std::fs::File::create(out_dir.join("manifest.txt"))?;
    write!(fh, "{manifest}IMAGE {img_path}\nBOXES {}\n", boxes.len())?;
    for (i, b) in boxes.iter().enumerate() {
        let pts: Vec<String> = b
            .points
            .iter()
            .map(|p| format!("{:.3},{:.3}", p[0], p[1]))
            .collect();
        writeln!(fh, "BOX {i} {}", pts.join(" "))?;
    }
    writeln!(fh, "TEXT {}", texts.join("\t"))?;
    drop(fh);

    println!(
        "DUMP image={img_path} boxes={} det_in={} det_out={} crops={}",
        boxes.len(),
        shape_str(&det_in),
        shape_str(&det_out),
        boxes.len()
    );
    Ok(())
}

fn cmd_time(
    img_path: &str,
    iters: usize,
    threads_cfg: Option<usize>,
    light: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let load_t0 = Instant::now();
    // The recognition worker pool defaults to `available_parallelism()`, which
    // `RAYON_NUM_THREADS` does *not* influence — so a "single-threaded" run has to
    // pin it here, or the page-level numbers would still be 12-way parallel.
    let engine = match threads_cfg {
        Some(n) => {
            let cfg = ppocr::OcrEngineConfig {
                rec_batch_threads: Some(n),
                ..Default::default()
            };
            OcrEngine::with_config(
                "models/PP-OCRv6_small_det.safetensors",
                "models/PP-OCRv6_small_rec.safetensors",
                "models/ppocrv6_keys.txt",
                cfg,
            )?
        }
        None => OcrEngine::default_tiny()?,
    };
    let load_ms = load_t0.elapsed().as_secs_f64() * 1000.0;
    let img = ImageReader::open(img_path)?.decode()?;

    let (det_in, _sh, _sw) = engine.det_preprocess().process(&img)?;
    let det_name = engine.det_model().graph().inputs[0].clone();
    let rec_name = engine.rec_model().graph().inputs[0].clone();

    let det_out = engine.det_model().forward(&det_name, det_in.clone())?;
    let boxes = engine.db_postprocess().process(&det_out, 1.0, 1.0);

    let rgb = img.to_rgb8();
    let mut rec_inputs = Vec::new();
    for b in &boxes {
        let patch = get_rotate_crop_image_rgb(&rgb, &b.points);
        rec_inputs.push(engine.rec_preprocess().process(&patch)?);
    }

    // Detection and single-crop recognition are cheap, so sample them more heavily;
    // the page-level loops below dominate the wall clock and stay at `iters`.
    let light_iters = iters * 3;
    let warmup = 3.max(iters / 4);
    for _ in 0..warmup {
        black_box(engine.det_model().forward(&det_name, det_in.clone())?);
        if let Some(t) = rec_inputs.first() {
            black_box(engine.rec_model().forward(&rec_name, t.clone())?);
        }
        if !light {
            black_box(engine.run(&img)?.len());
        }
    }

    // (1) detection forward, single image
    let mut det_v = Vec::with_capacity(light_iters);
    for _ in 0..light_iters {
        let t0 = Instant::now();
        black_box(engine.det_model().forward(&det_name, det_in.clone())?);
        det_v.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // (2) recognition forward on the *first* crop, batch = 1
    let mut rec1_v = Vec::with_capacity(light_iters);
    if let Some(t0_tensor) = rec_inputs.first() {
        for _ in 0..light_iters {
            let t0 = Instant::now();
            black_box(engine.rec_model().forward(&rec_name, t0_tensor.clone())?);
            rec1_v.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    // (3) recognition forward over *every* crop, sequentially
    let mut recall_v = Vec::with_capacity(iters);
    if !light {
        for _ in 0..iters {
            let t0 = Instant::now();
            for t in &rec_inputs {
                black_box(engine.rec_model().forward(&rec_name, t.clone())?);
            }
            recall_v.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    // (4) page-level paths
    let mut recpar_v = Vec::with_capacity(iters);
    let mut run_v = Vec::with_capacity(iters);
    let mut chars = 0usize;
    if !light {
        for _ in 0..iters {
            let boxes = engine.detect(&img)?;
            let t1 = Instant::now();
            let res = engine.recognize(&img, &boxes)?;
            let t2 = Instant::now();
            chars = res.iter().map(|r| r.text.chars().count()).sum();
            black_box(res.len());
            recpar_v.push((t2 - t1).as_secs_f64() * 1000.0);

            let t3 = Instant::now();
            black_box(engine.run(&img)?.len());
            run_v.push(t3.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let (dmin, dp50, dmean) = stats(det_v);
    let (r1min, r1p50, r1mean) = stats(rec1_v);

    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    let rec_threads = match threads_cfg {
        Some(n) => n.to_string(),
        None => "default".to_string(),
    };
    let widths: Vec<String> = rec_inputs
        .iter()
        .map(|t| t.shape().last().copied().unwrap_or(0).to_string())
        .collect();

    println!(
        "WORK img={img_path} iters={iters} light={light} boxes={} chars={chars} rayon={threads} rec_pool={rec_threads}",
        rec_inputs.len()
    );
    println!("LOAD ms={load_ms:.1}");
    println!("DET_IN {}", shape_str(&det_in));
    println!("REC_W {}", widths.join(","));
    println!("DET_FWD min={dmin:.2} p50={dp50:.2} mean={dmean:.2}");
    println!("REC1_FWD min={r1min:.2} p50={r1p50:.2} mean={r1mean:.2}");
    if !light {
        let (ramin, rap50, ramean) = stats(recall_v);
        let (rpmin, rpp50, rpmean) = stats(recpar_v);
        let (umin, up50, umean) = stats(run_v);
        println!("RECALL_FWD min={ramin:.2} p50={rap50:.2} mean={ramean:.2}");
        println!("REC_PAR min={rpmin:.2} p50={rpp50:.2} mean={rpmean:.2}");
        println!("RUN min={umin:.2} p50={up50:.2} mean={umean:.2}");
    }
    Ok(())
}

fn print_profile(title: &str, prof: std::collections::HashMap<String, f64>, top: usize) {
    let mut items: Vec<(String, f64)> = prof.into_iter().collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f64 = items.iter().map(|(_, v)| *v).sum();
    println!("PROFILE {title} total={total:.2}");
    for (op, ms) in items.iter().take(top) {
        println!("  {op:44} {ms:8.2} ms  {:5.1}%", 100.0 * ms / total);
    }
}

/// Per-operator breakdown for the *real* shapes, one thread, so it can be lined up
/// against ONNX Runtime's per-node profile.
fn cmd_profile(img_path: &str, top: usize) -> Result<(), Box<dyn std::error::Error>> {
    let engine = OcrEngine::default_tiny()?;
    let img = ImageReader::open(img_path)?.decode()?;

    let (det_in, scale_h, scale_w) = engine.det_preprocess().process(&img)?;
    let det_name = engine.det_model().graph().inputs[0].clone();
    for _ in 0..2 {
        black_box(engine.det_model().forward(&det_name, det_in.clone())?);
    }
    let (det_out, prof) = engine
        .det_model()
        .forward_with_profile(&det_name, det_in.clone())?;
    println!("SHAPE det {}", shape_str(&det_in));
    print_profile("det", prof, top);

    let boxes = engine.db_postprocess().process(&det_out, scale_h, scale_w);
    let rgb = img.to_rgb8();
    if let Some(b) = boxes.first() {
        let patch = get_rotate_crop_image_rgb(&rgb, &b.points);
        let t = engine.rec_preprocess().process(&patch)?;
        let rec_name = engine.rec_model().graph().inputs[0].clone();
        for _ in 0..2 {
            black_box(engine.rec_model().forward(&rec_name, t.clone())?);
        }
        let (_, prof) = engine
            .rec_model()
            .forward_with_profile(&rec_name, t.clone())?;
        println!("SHAPE rec {}", shape_str(&t));
        print_profile("rec", prof, top);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("time");
    match mode {
        "dump" => {
            let img = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "tests/data/bench_doc.png".into());
            let out = PathBuf::from(
                args.get(3)
                    .cloned()
                    .unwrap_or_else(|| "/tmp/onnx_dump".into()),
            );
            cmd_dump(&img, &out)
        }
        "time" => {
            let img = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "tests/data/bench_doc.png".into());
            let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
            let threads: Option<usize> = args.get(4).and_then(|s| s.parse().ok());
            let light = args.iter().any(|a| a == "--light");
            cmd_time(&img, iters, threads, light)
        }
        "profile" => {
            let img = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "tests/data/bench_doc.png".into());
            let top: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(14);
            cmd_profile(&img, top)
        }
        other => {
            eprintln!("unknown mode {other:?}; expected `dump`, `time` or `profile`");
            std::process::exit(2);
        }
    }
}
