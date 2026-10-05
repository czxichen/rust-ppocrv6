use ppocr::{Model, Tensor};

fn main() {
    let det_path = "models/PP-OCRv6_small_det.safetensors";
    let rec_path = "models/PP-OCRv6_small_rec.safetensors";

    for (name, path, in_shape) in [
        ("DET", det_path, vec![1, 3, 128, 416]),
        ("REC", rec_path, vec![1, 3, 48, 320]),
    ] {
        println!("=== {} CONV NODE BREAKDOWN ===", name);
        let model = Model::load(path).unwrap();
        let x = Tensor::zeros(&in_shape);
        for _ in 0..2 {
            let _ = model.forward("x", x.clone()).unwrap();
        }

        let (_, profile) = model.forward_with_profile("x", x).unwrap();
        let mut conv_items: Vec<_> = profile
            .into_iter()
            .filter(|(k, _)| k.contains("Conv"))
            .collect();
        conv_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (op, time) in conv_items {
            println!("  {:40}: {:7.2} ms", op, time);
        }
    }
}
