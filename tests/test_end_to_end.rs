use image::ImageReader;
use ppocr::OcrEngine;
use std::path::Path;
use std::time::Instant;

/// The bundled PP-OCRv6 models are ~35 MB and are not shipped with every checkout (a fresh
/// `git clone` into CI, for instance, may not have them). Missing models must *skip* the
/// test rather than fail it, otherwise the suite cannot run anywhere without the blobs.
fn models_available(paths: &[&str]) -> bool {
    for p in paths {
        if !Path::new(p).exists() {
            eprintln!("skipping: required asset '{p}' is not present");
            return false;
        }
    }
    true
}

const SMALL_DET: &str = "models/PP-OCRv6_small_det.safetensors";
const SMALL_REC: &str = "models/PP-OCRv6_small_rec.safetensors";
const SMALL_DICT: &str = "models/ppocrv6_keys.txt";
const TINY_DET: &str = "models/PP-OCRv6_tiny_det.safetensors";
const TINY_REC: &str = "models/PP-OCRv6_tiny_rec.safetensors";
const TINY_DICT: &str = "models/ppocrv6_tiny_keys.txt";
const CLS: &str = "models/PP-OCRv6_cls.safetensors";
const TEST_IMAGE: &str = "tests/data/test_ocr.png";

#[test]
fn test_end_to_end_ocr() {
    let det_model_path = SMALL_DET;
    let rec_model_path = SMALL_REC;
    let dict_path = SMALL_DICT;
    let image_path = TEST_IMAGE;

    if !models_available(&[det_model_path, rec_model_path, dict_path, image_path]) {
        return;
    }

    println!("Initializing OcrEngine...");
    let init_start = Instant::now();
    let engine = OcrEngine::new(det_model_path, rec_model_path, dict_path)
        .expect("Failed to create OcrEngine");
    println!("Engine initialized in {:.2?}", init_start.elapsed());

    let img = ImageReader::open(image_path)
        .expect("Failed to open test image")
        .decode()
        .expect("Failed to decode test image");

    println!("Running end-to-end OCR on test image...");
    let ocr_start = Instant::now();
    let results = engine.run(&img).expect("OCR inference failed");
    let elapsed = ocr_start.elapsed();
    println!("OCR finished in {:.2?}", elapsed);

    println!("Detected {} text region(s):", results.len());
    for (i, res) in results.iter().enumerate() {
        println!(
            "  [{}] Text: '{}' (Score: {:.4}, Box: {:?})",
            i, res.text, res.score, res.points
        );
    }

    assert!(
        !results.is_empty(),
        "Expected at least 1 detected text region"
    );
    let combined_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("Combined detected text: '{}'", combined_text);
    assert!(
        combined_text.contains("深度学习") || combined_text.contains("PP-OCR"),
        "Recognized text should contain expected words: '{}'",
        combined_text
    );
}

#[test]
fn test_quantized_ocr() {
    use ppocr::OcrEngineConfig;

    let det_model_path = SMALL_DET;
    let rec_model_path = SMALL_REC;
    let dict_path = SMALL_DICT;
    let image_path = TEST_IMAGE;

    if !models_available(&[det_model_path, rec_model_path, dict_path, image_path]) {
        return;
    }

    // Standard FP32 Engine
    let fp32_engine = OcrEngine::new(det_model_path, rec_model_path, dict_path)
        .expect("Failed to create FP32 engine");

    // Quantized Engine (W8A32)
    let quant_config = OcrEngineConfig {
        quantize_rec_head: true,
        ..Default::default()
    };
    let quant_engine =
        OcrEngine::with_config(det_model_path, rec_model_path, dict_path, quant_config)
            .expect("Failed to create Quantized engine");

    let img = ImageReader::open(image_path)
        .expect("Failed to open test image")
        .decode()
        .expect("Failed to decode test image");

    let fp32_results = fp32_engine.run(&img).expect("FP32 OCR failed");
    let quant_results = quant_engine.run(&img).expect("Quantized OCR failed");

    assert_eq!(
        fp32_results.len(),
        quant_results.len(),
        "Number of detected regions must match between FP32 and Quantized"
    );

    for (i, (f_res, q_res)) in fp32_results.iter().zip(quant_results.iter()).enumerate() {
        println!(
            "Region [{}]: FP32='{}' ({:.4}) vs Quantized='{}' ({:.4})",
            i, f_res.text, f_res.score, q_res.text, q_res.score
        );
        assert_eq!(
            f_res.text, q_res.text,
            "Recognized text must match identically between FP32 and INT8 quantization"
        );
        let score_diff = (f_res.score - q_res.score).abs();
        assert!(
            score_diff < 0.05,
            "Score difference too large: {} vs {}",
            f_res.score,
            q_res.score
        );
    }
}

#[test]
fn test_tiny_end_to_end_ocr() {
    let det_model_path = TINY_DET;
    let rec_model_path = TINY_REC;
    let dict_path = TINY_DICT;
    let image_path = TEST_IMAGE;

    if !models_available(&[det_model_path, rec_model_path, dict_path, image_path]) {
        return;
    }

    println!("Initializing PP-OCRv6 Tiny OcrEngine...");
    let init_start = std::time::Instant::now();
    let engine = OcrEngine::new(det_model_path, rec_model_path, dict_path)
        .expect("Failed to create Tiny OcrEngine");
    println!("Tiny Engine initialized in {:.2?}", init_start.elapsed());

    let img = ImageReader::open(image_path)
        .expect("Failed to open test image")
        .decode()
        .expect("Failed to decode test image");

    println!("Running end-to-end OCR with Tiny models...");
    let ocr_start = std::time::Instant::now();
    let results = engine.run(&img).expect("Tiny OCR inference failed");
    let elapsed = ocr_start.elapsed();
    println!("Tiny OCR finished in {:.2?}", elapsed);

    println!("Tiny detected {} text region(s):", results.len());
    for (i, res) in results.iter().enumerate() {
        println!(
            "  [{}] Text: '{}' (Score: {:.4}, Box: {:?})",
            i, res.text, res.score, res.points
        );
    }

    assert!(
        !results.is_empty(),
        "Expected at least 1 detected text region"
    );
    let combined_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("Tiny combined detected text: '{}'", combined_text);
    assert!(
        combined_text.contains("深度学习") || combined_text.contains("PP-OCR"),
        "Recognized text should contain expected words: '{}'",
        combined_text
    );
}

#[test]
fn test_end_to_end_with_cls() {
    let image_path = TEST_IMAGE;
    if !models_available(&[SMALL_DET, SMALL_REC, SMALL_DICT, CLS, image_path]) {
        return;
    }
    let img = ImageReader::open(image_path)
        .expect("Failed to open test image")
        .decode()
        .expect("Failed to decode test image");

    println!("Initializing PP-OCRv6 Small OcrEngine with CLS enabled...");
    let engine =
        OcrEngine::default_small_with_cls().expect("Failed to create Small OcrEngine with CLS");

    // 1. Test upright image: should detect angle 0° and recognized text
    println!("Testing upright image with CLS...");
    let upright_results = engine.run(&img).expect("Upright OCR with CLS failed");
    assert!(
        !upright_results.is_empty(),
        "Expected text regions for upright image"
    );
    for res in &upright_results {
        println!("  Upright res: '{}', angle: {:?}", res.text, res.angle);
        assert_eq!(res.angle, Some(0), "Upright text should have angle 0");
    }
    let upright_text: String = upright_results.iter().map(|r| r.text.as_str()).collect();
    assert!(
        upright_text.contains("深度学习") || upright_text.contains("PP-OCR"),
        "Upright text should match expected: '{}'",
        upright_text
    );

    // 2. Test 180-degree inverted image: CLS should detect 180°, flip it, and recognize identical text!
    println!("Testing 180-degree rotated image with CLS...");
    let rotated_img = img.rotate180();
    let rotated_results = engine
        .run(&rotated_img)
        .expect("Rotated OCR with CLS failed");
    assert!(
        !rotated_results.is_empty(),
        "Expected text regions for rotated image"
    );
    for res in &rotated_results {
        println!("  Rotated res: '{}', angle: {:?}", res.text, res.angle);
        assert_eq!(
            res.angle,
            Some(180),
            "Inverted text line should have angle 180"
        );
    }
    let rotated_text: String = rotated_results.iter().map(|r| r.text.as_str()).collect();
    println!("Rotated detected text after auto-flip: '{}'", rotated_text);
    assert!(
        rotated_text.contains("深度学习") || rotated_text.contains("PP-OCR"),
        "Rotated text after auto-flip should match expected: '{}'",
        rotated_text
    );
}

#[test]
fn test_multi_line_and_wide_text_rec() {
    use image::DynamicImage;
    use ppocr::preprocess::RecPreprocess;

    // 1. Verify RecPreprocess caps width at max_w while preserving aspect ratio below it.
    // The default max_w is 960: raising it inflates recognition cost roughly linearly with
    // input pixels (measured +62% latency for +61% pixels on a 29-box page) without any
    // observed text-accuracy gain, so the default deliberately stays conservative.
    let rec_prep = RecPreprocess::default();
    assert_eq!(rec_prep.max_w, 960);

    // Wide image, aspect ratio 30.0 -> natural target_w = ceil(48 * 30) = 1440,
    // which must be clamped to max_w (960) rather than honored.
    let wide_img = DynamicImage::new_rgb8(1500, 50);
    let tensor = rec_prep
        .process(&wide_img)
        .expect("Preprocess wide image failed");
    assert_eq!(tensor.shape(), &[1, 3, 48, 960]);

    // Same aspect ratio with an explicit larger cap must be honored up to that cap,
    // so callers who genuinely need unclipped wide lines can opt in.
    let wide_prep = RecPreprocess::with_max_w(48, 3200);
    let wide_tensor = wide_prep
        .process(&wide_img)
        .expect("Preprocess wide image failed");
    assert_eq!(wide_tensor.shape(), &[1, 3, 48, 1440]);

    // 2. Verify multi-line recognition worker pool execution
    if !models_available(&[TINY_DET, TINY_REC, TINY_DICT, TEST_IMAGE]) {
        return;
    }
    let engine = OcrEngine::default_tiny().expect("Failed to load tiny OCR engine");
    let test_img = ImageReader::open(TEST_IMAGE)
        .expect("Failed to open test image")
        .decode()
        .expect("Failed to decode test image");

    let boxes = engine.detect(&test_img).expect("Detection failed");
    assert!(!boxes.is_empty());

    // Duplicate boxes to simulate multi-line document (e.g. 12 lines)
    let mut multi_boxes = Vec::new();
    for _ in 0..12 {
        multi_boxes.push(boxes[0].clone());
    }

    let start = Instant::now();
    let results = engine
        .recognize(&test_img, &multi_boxes)
        .expect("Multi-line recognize failed");
    let elapsed = start.elapsed();
    println!("Recognized 12 boxes in {:.2?}", elapsed);

    assert_eq!(results.len(), 12);
    for res in &results {
        assert!(!res.text.is_empty());
    }
}
