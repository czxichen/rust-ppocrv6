#![cfg(feature = "onnx")]

use ppocr::{Model, Tensor};
use std::path::PathBuf;

#[test]
fn onnx_loader_matches_safetensors_forward_for_rec_small() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let st_path = root.join("models/PP-OCRv6_small_rec.safetensors");
    let onnx_path = root.join("build_onnx/rec_small.onnx");
    assert!(st_path.exists(), "missing {}", st_path.display());
    assert!(onnx_path.exists(), "missing {}", onnx_path.display());

    let m_st = Model::load(&st_path).expect("load safetensors rec_small");
    let m_onnx = Model::load_onnx(&onnx_path).expect("load onnx rec_small");

    let in_name_st = m_st.graph().inputs[0].clone();
    let in_name_onnx = m_onnx.graph().inputs[0].clone();
    let make_x = || {
        let shape = [1usize, 3, 48, 320];
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * 1e-3).sin())
            .collect();
        Tensor::from_slice(&shape, &data).expect("build input tensor")
    };
    let y_st = m_st
        .forward(&in_name_st, make_x())
        .expect("safetensors forward");
    let y_onnx = m_onnx
        .forward(&in_name_onnx, make_x())
        .expect("onnx forward");

    assert_eq!(y_st.shape(), y_onnx.shape(), "output shape mismatch");
    let max_abs = y_st
        .as_slice()
        .iter()
        .zip(y_onnx.as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-4, "onnx vs safetensors max_abs={max_abs}");
}
