use image::ImageReader;
use ppocr::OcrEngine;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("               PP-OCR Rust Inference Benchmark              ");
    println!("============================================================");

    let det_path = "models/PP-OCRv6_small_det.safetensors";
    let rec_path = "models/PP-OCRv6_small_rec.safetensors";
    let dict_path = "models/ppocrv6_keys.txt";
    let image_path = "tests/data/test_ocr.png";

    let engine = OcrEngine::new(det_path, rec_path, dict_path).expect("Init failed");
    println!("Loaded det: {}, rec: {}", det_path, rec_path);
    let img = ImageReader::open(image_path)
        .expect("Open failed")
        .decode()
        .expect("Decode failed");

    // Warmup
    println!("Warming up 3 runs...");
    for _ in 0..3 {
        let _ = engine.run(&img).unwrap();
    }

    let iterations = 10;
    println!("Benchmarking {} iterations...", iterations);

    let mut det_times = Vec::with_capacity(iterations);
    let mut rec_times = Vec::with_capacity(iterations);
    let mut total_times = Vec::with_capacity(iterations);

    let mut det_prep_times = Vec::with_capacity(iterations);
    let mut det_fwd_times = Vec::with_capacity(iterations);
    let mut det_post_times = Vec::with_capacity(iterations);

    let det_in = &engine.det_model().graph().inputs[0];

    for _ in 0..iterations {
        let t0 = Instant::now();
        let (input_tensor, scale_h, scale_w) = engine.det_preprocess().process(&img).unwrap();
        let t_prep = Instant::now();
        let prob_map = engine.det_model().forward(det_in, input_tensor).unwrap();
        let t_fwd = Instant::now();
        let boxes = engine.db_postprocess().process(&prob_map, scale_h, scale_w);
        let t1 = Instant::now();

        let _ = engine.recognize(&img, &boxes).unwrap();
        let t2 = Instant::now();

        det_prep_times.push((t_prep - t0).as_secs_f64() * 1000.0);
        det_fwd_times.push((t_fwd - t_prep).as_secs_f64() * 1000.0);
        det_post_times.push((t1 - t_fwd).as_secs_f64() * 1000.0);

        det_times.push((t1 - t0).as_secs_f64() * 1000.0);
        rec_times.push((t2 - t1).as_secs_f64() * 1000.0);
        total_times.push((t2 - t0).as_secs_f64() * 1000.0);
        print!(".");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
    println!("\nDone!");

    let avg_prep: f64 = det_prep_times.iter().sum::<f64>() / iterations as f64;
    let avg_fwd: f64 = det_fwd_times.iter().sum::<f64>() / iterations as f64;
    let avg_post: f64 = det_post_times.iter().sum::<f64>() / iterations as f64;
    println!("  - Det Preprocess:  {:.2} ms", avg_prep);
    println!("  - Det Model Fwd:   {:.2} ms", avg_fwd);
    println!("  - Det Postprocess: {:.2} ms", avg_post);

    let avg_det: f64 = det_times.iter().sum::<f64>() / iterations as f64;
    let avg_rec: f64 = rec_times.iter().sum::<f64>() / iterations as f64;
    let avg_total: f64 = total_times.iter().sum::<f64>() / iterations as f64;

    let min_total = total_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_total = total_times
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\nBenchmark Results (over {} runs):", iterations);
    println!("  - Average Detection Latency:   {:.2} ms", avg_det);
    println!("  - Average Recognition Latency: {:.2} ms", avg_rec);
    println!("  - Average End-to-End Latency:  {:.2} ms", avg_total);
    println!("  - Best End-to-End Latency:     {:.2} ms", min_total);
    println!("  - Max End-to-End Latency:      {:.2} ms", max_total);
    println!(
        "  - Throughput:                  {:.2} FPS",
        1000.0 / avg_total
    );
    println!("============================================================");

    // Average profiled passes on Det and Rec
    let profile_iters = 5;
    let mut det_profile_accum: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    for _ in 0..profile_iters {
        let (input_tensor, _, _) = engine.det_preprocess().process(&img).unwrap();
        let (_, det_profile) = engine
            .det_model()
            .forward_with_profile(det_in, input_tensor)
            .unwrap();
        for (k, v) in det_profile {
            *det_profile_accum.entry(k).or_insert(0.0) += v / profile_iters as f64;
        }
    }
    let mut det_sorted: Vec<_> = det_profile_accum.into_iter().collect();
    det_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n[DET MODEL OPERATOR PROFILE (Average of 5 runs)]");
    for (op, ms) in det_sorted.iter().take(15) {
        println!("  {:35} : {:6.2} ms", op, ms);
    }

    let boxes = engine.detect(&img).unwrap();
    if !boxes.is_empty() {
        let cropped_patch = ppocr::postprocess::get_rotate_crop_image(&img, &boxes[0].points);
        let rec_in = &engine.rec_model().graph().inputs[0];
        let mut rec_profile_accum: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for _ in 0..profile_iters {
            let patch_tensor = engine.rec_preprocess().process(&cropped_patch).unwrap();
            let (_, rec_profile) = engine
                .rec_model()
                .forward_with_profile(rec_in, patch_tensor)
                .unwrap();
            for (k, v) in rec_profile {
                *rec_profile_accum.entry(k).or_insert(0.0) += v / profile_iters as f64;
            }
        }
        let mut rec_sorted: Vec<_> = rec_profile_accum.into_iter().collect();
        rec_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("\n[REC MODEL FP32 (Single Patch) OPERATOR PROFILE (Average of 5 runs)]");
        for (op, ms) in rec_sorted.iter().take(10) {
            println!("  {:45} : {:6.2} ms", op, ms);
        }

        // Benchmark Quantized Rec Model
        println!("\n------------------------------------------------------------");
        println!("  Testing INT8 Quantized Recognition (W8A32)");
        println!("------------------------------------------------------------");
        let quant_engine = ppocr::OcrEngine::with_config(
            det_path,
            rec_path,
            dict_path,
            ppocr::OcrEngineConfig {
                quantize_rec_head: true,
                ..Default::default()
            },
        )
        .expect("Quant engine init failed");

        // Warmup
        let _ = quant_engine.run(&img).unwrap();

        let mut q_rec_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t1 = Instant::now();
            let _ = quant_engine.recognize(&img, &boxes).unwrap();
            let t2 = Instant::now();
            q_rec_times.push((t2 - t1).as_secs_f64() * 1000.0);
        }
        let avg_q_rec: f64 = q_rec_times.iter().sum::<f64>() / iterations as f64;
        println!("  - FP32 Recognition Latency:      {:.2} ms", avg_rec);
        println!("  - INT8 Quantized Rec Latency:    {:.2} ms", avg_q_rec);
        println!(
            "  - Recognition Speedup:           {:.2}x",
            avg_rec / avg_q_rec
        );
        println!(
            "  - Total Latency (Det + Q-Rec):   {:.2} ms ({:.2} FPS)",
            avg_det + avg_q_rec,
            1000.0 / (avg_det + avg_q_rec)
        );

        let mut q_rec_profile_accum: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for _ in 0..profile_iters {
            let patch_tensor = quant_engine
                .rec_preprocess()
                .process(&cropped_patch)
                .unwrap();
            let (_, rec_profile) = quant_engine
                .rec_model()
                .forward_with_profile(rec_in, patch_tensor)
                .unwrap();
            for (k, v) in rec_profile {
                *q_rec_profile_accum.entry(k).or_insert(0.0) += v / profile_iters as f64;
            }
        }
        let mut q_rec_sorted: Vec<_> = q_rec_profile_accum.into_iter().collect();
        q_rec_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("\n[REC MODEL INT8 QUANTIZED (Single Patch) OPERATOR PROFILE]");
        for (op, ms) in q_rec_sorted.iter().take(10) {
            println!("  {:45} : {:6.2} ms", op, ms);
        }
    }
}
