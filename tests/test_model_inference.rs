use ppocr::{Model, Tensor};
use std::path::Path;
use std::time::Instant;

/// The bundled models are ~35 MB and may be absent (fresh clone in CI). Missing models
/// must skip the test rather than fail it.
fn have(path: &str) -> bool {
    let ok = Path::new(path).exists();
    if !ok {
        eprintln!("skipping: required asset '{path}' is not present");
    }
    ok
}

#[test]
fn test_load_and_run_det_model() {
    let model_path = "models/PP-OCRv6_small_det.safetensors";
    if !have(model_path) {
        return;
    }

    println!("Loading PP-OCRv6 det model...");
    let start = Instant::now();
    let model = Model::load(model_path).expect("Failed to load PP-OCRv6 det model");
    println!("PP-OCRv6 Det model loaded in {:.2?}", start.elapsed());

    // Input: [1, 3, 64, 64] for quick unit test
    let input = Tensor::zeros(&[1, 3, 64, 64]);
    println!("Running forward pass on PP-OCRv6 det model...");
    let start = Instant::now();
    let output = model.forward("x", input).expect("Forward pass failed");
    println!(
        "PP-OCRv6 Det forward pass finished in {:.2?}",
        start.elapsed()
    );

    assert_eq!(output.shape(), &[1, 1, 64, 64]);
    println!("PP-OCRv6 Det output shape: {:?}", output.shape());
}

#[test]
fn test_load_and_run_rec_model() {
    let model_path = "models/PP-OCRv6_small_rec.safetensors";
    if !have(model_path) {
        return;
    }

    println!("Loading PP-OCRv6 rec model...");
    let start = Instant::now();
    let model = Model::load(model_path).expect("Failed to load PP-OCRv6 rec model");
    println!("PP-OCRv6 Rec model loaded in {:.2?}", start.elapsed());

    // Input: [1, 3, 48, 80] for quick unit test
    let input = Tensor::zeros(&[1, 3, 48, 80]);
    println!("Running forward pass on PP-OCRv6 rec model...");
    let start = Instant::now();
    let output = model.forward("x", input).expect("Forward pass failed");
    println!(
        "PP-OCRv6 Rec forward pass finished in {:.2?}",
        start.elapsed()
    );

    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[2], 18710);
    println!("PP-OCRv6 Rec output shape: {:?}", output.shape());
}

#[test]
fn test_load_and_run_tiny_det_model() {
    let model_path = "models/PP-OCRv6_tiny_det.safetensors";
    if !have(model_path) {
        return;
    }

    println!("Loading PP-OCRv6 tiny det model...");
    let start = Instant::now();
    let model = Model::load(model_path).expect("Failed to load PP-OCRv6 tiny det model");
    println!("PP-OCRv6 Tiny Det model loaded in {:.2?}", start.elapsed());

    let input = Tensor::zeros(&[1, 3, 64, 64]);
    let output = model.forward("x", input).expect("Forward pass failed");
    assert_eq!(output.shape(), &[1, 1, 64, 64]);
    println!("PP-OCRv6 Tiny Det output shape: {:?}", output.shape());
}

#[test]
fn test_load_and_run_tiny_rec_model() {
    let model_path = "models/PP-OCRv6_tiny_rec.safetensors";
    if !have(model_path) {
        return;
    }

    println!("Loading PP-OCRv6 tiny rec model...");
    let start = Instant::now();
    let model = Model::load(model_path).expect("Failed to load PP-OCRv6 tiny rec model");
    println!("PP-OCRv6 Tiny Rec model loaded in {:.2?}", start.elapsed());

    let input = Tensor::zeros(&[1, 3, 48, 80]);
    let output = model.forward("x", input).expect("Forward pass failed");
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[2], 6906);
    println!("PP-OCRv6 Tiny Rec output shape: {:?}", output.shape());
}

#[test]
fn test_load_and_run_cls_model() {
    let model_path = "models/PP-OCRv6_cls.safetensors";
    if !have(model_path) {
        return;
    }

    println!("Loading PP-OCRv6 CLS model...");
    let start = std::time::Instant::now();
    let model = Model::load(model_path).expect("Failed to load PP-OCRv6 CLS model");
    println!("PP-OCRv6 CLS model loaded in {:.2?}", start.elapsed());

    let input = Tensor::zeros(&[1, 3, 48, 192]);
    let output = model
        .forward("x", input)
        .expect("PP-OCRv6 CLS forward pass failed");
    assert_eq!(output.shape(), &[1, 2]);
    println!("PP-OCRv6 CLS dummy input probs: {:?}", output.as_slice());

    // Test with real image: upright vs rotated 180 degrees using ClsPreprocess
    let image_path = "tests/data/test_ocr.png";
    if !have(image_path) {
        return;
    }
    let img = image::ImageReader::open(image_path)
        .expect("Failed to open test image")
        .decode()
        .expect("Failed to decode test image");

    let cls_preprocess = ppocr::preprocess::ClsPreprocess::default();

    let upright_tensor = cls_preprocess.process(&img).unwrap();
    let upright_out = model.forward("x", upright_tensor).unwrap();
    let upright_probs = upright_out.as_slice();
    println!("Upright image CLS probs [0°, 180°]: {:?}", upright_probs);

    let rotated_img = img.rotate180();
    let rotated_tensor = cls_preprocess.process(&rotated_img).unwrap();
    let rotated_out = model.forward("x", rotated_tensor).unwrap();
    let rotated_probs = rotated_out.as_slice();
    println!(
        "Rotated-180 image CLS probs [0°, 180°]: {:?}",
        rotated_probs
    );

    assert!(
        upright_probs[0] > 0.9,
        "Upright image should be classified as 0 degrees"
    );
    assert!(
        rotated_probs[1] > 0.9,
        "180-rotated image should be classified as 180 degrees"
    );
}
