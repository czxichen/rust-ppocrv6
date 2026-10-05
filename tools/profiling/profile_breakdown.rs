use ppocr::{Model, Tensor};

fn main() {
    let det_path = "models/PP-OCRv6_small_det.safetensors";
    let rec_path = "models/PP-OCRv6_small_rec.safetensors";

    println!("=== PP-OCRv6 DET PROFILING ([1, 3, 128, 416]) ===");
    let det_model = Model::load(det_path).unwrap();
    println!("Det nodes count: {}", det_model.graph().nodes.len());
    let x_det = Tensor::zeros(&[1, 3, 128, 416]);
    for _ in 0..2 {
        let _ = det_model.forward("x", x_det.clone()).unwrap();
    }
    let (_, profile_det) = det_model.forward_with_profile("x", x_det).unwrap();
    let mut det_items: Vec<_> = profile_det.into_iter().collect();
    det_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut total_det = 0.0;
    for (op, ms) in &det_items {
        println!("  {:20}: {:7.2} ms", op, ms);
        total_det += ms;
    }
    println!("  Total Det Model Time: {:7.2} ms\n", total_det);

    println!("=== PP-OCRv6 REC PROFILING ([1, 3, 48, 320]) ===");
    let rec_model = Model::load(rec_path).unwrap();
    println!("Rec nodes count: {}", rec_model.graph().nodes.len());
    let x_rec = Tensor::zeros(&[1, 3, 48, 320]);
    for _ in 0..2 {
        let _ = rec_model.forward("x", x_rec.clone()).unwrap();
    }
    let (_, profile_rec) = rec_model.forward_with_profile("x", x_rec).unwrap();
    let mut rec_items: Vec<_> = profile_rec.into_iter().collect();
    rec_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut total_rec = 0.0;
    for (op, ms) in &rec_items {
        println!("  {:20}: {:7.2} ms", op, ms);
        total_rec += ms;
    }
    println!("  Total Rec Model Time: {:7.2} ms\n", total_rec);

    println!("=== FULL PIPELINE STAGE BREAKDOWN ===");
    use image::ImageReader;
    use ppocr::postprocess::{CtcDecoder, DbPostProcessor, get_rotate_crop_image};
    use ppocr::preprocess::{DetPreprocess, RecPreprocess};
    use std::time::Instant;

    let img = ImageReader::open("tests/data/test_ocr.png")
        .unwrap()
        .decode()
        .unwrap();
    let det_prep = DetPreprocess::new(960);
    let rec_prep = RecPreprocess::new(48);
    let db_post = DbPostProcessor::default();
    let ctc = CtcDecoder::from_file("models/ppocrv6_keys.txt").unwrap();

    // Warmup
    for _ in 0..3 {
        let (t_in, sh, sw) = det_prep.process(&img).unwrap();
        let prob = det_model.forward("x", t_in).unwrap();
        let boxes = db_post.process(&prob, sh, sw);
        for b in &boxes {
            let cropped = get_rotate_crop_image(&img, &b.points);
            let r_in = rec_prep.process(&cropped).unwrap();
            let logits = rec_model.forward("x", r_in).unwrap();
            let _ = ctc.decode(&logits);
        }
    }

    let iters = 10;
    let mut t_det_prep = 0.0;
    let mut t_det_fwd = 0.0;
    let mut t_db_post = 0.0;
    let mut t_rec_crop = 0.0;
    let mut t_rec_prep = 0.0;
    let mut t_rec_fwd = 0.0;
    let mut t_ctc_dec = 0.0;

    for _ in 0..iters {
        let t0 = Instant::now();
        let (t_in, sh, sw) = det_prep.process(&img).unwrap();
        let t1 = Instant::now();
        let prob = det_model.forward("x", t_in).unwrap();
        let t2 = Instant::now();
        let boxes = db_post.process(&prob, sh, sw);
        let t3 = Instant::now();

        t_det_prep += (t1 - t0).as_secs_f64() * 1000.0;
        t_det_fwd += (t2 - t1).as_secs_f64() * 1000.0;
        t_db_post += (t3 - t2).as_secs_f64() * 1000.0;

        for b in &boxes {
            let t4 = Instant::now();
            let cropped = get_rotate_crop_image(&img, &b.points);
            let t5 = Instant::now();
            let r_in = rec_prep.process(&cropped).unwrap();
            let t6 = Instant::now();
            let logits = rec_model.forward("x", r_in).unwrap();
            let t7 = Instant::now();
            let _ = ctc.decode(&logits);
            let t8 = Instant::now();

            t_rec_crop += (t5 - t4).as_secs_f64() * 1000.0;
            t_rec_prep += (t6 - t5).as_secs_f64() * 1000.0;
            t_rec_fwd += (t7 - t6).as_secs_f64() * 1000.0;
            t_ctc_dec += (t8 - t7).as_secs_f64() * 1000.0;
        }
    }

    println!("Stage Averages over {} runs:", iters);
    println!(
        "  1. Det Preprocess (resize+norm): {:6.2} ms",
        t_det_prep / iters as f64
    );
    println!(
        "  2. Det Model Forward           : {:6.2} ms",
        t_det_fwd / iters as f64
    );
    println!(
        "  3. DB Postprocess (BFS+Polygon): {:6.2} ms",
        t_db_post / iters as f64
    );
    println!(
        "  4. Rec Crop & Warp (Bilinear)  : {:6.2} ms",
        t_rec_crop / iters as f64
    );
    println!(
        "  5. Rec Preprocess (resize+norm): {:6.2} ms",
        t_rec_prep / iters as f64
    );
    println!(
        "  6. Rec Model Forward           : {:6.2} ms",
        t_rec_fwd / iters as f64
    );
    println!(
        "  7. CTC Decode                  : {:6.2} ms",
        t_ctc_dec / iters as f64
    );
    let total =
        (t_det_prep + t_det_fwd + t_db_post + t_rec_crop + t_rec_prep + t_rec_fwd + t_ctc_dec)
            / iters as f64;
    println!("  --------------------------------------------");
    println!("  Total Pipeline Latency         : {:6.2} ms", total);
}
