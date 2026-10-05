// Recognition accuracy for the tiny model, against known ground truth.
//
// `tests/data/bench_*.png` were produced by `tools/profiling/gen_bench_images.py` and
// `tools/profiling/gen_cjk_stress.py`, both of which know the exact text they drew. The
// end-to-end tests only assert `contains("PP-OCR")`, which cannot tell 99% from 60%.
//
// Scoring the recognition stage in isolation: the case's crop is passed to `recognize` as a
// single explicit box, so the DBNet stage cannot contribute errors. That answers "how good
// is the rec model", which is what a change to a rec-side kernel (the head GEMM, the 1x1
// conv routing) can actually regress.
//
// `--det` runs the full detect+recognise pipeline instead, which is what a caller gets but
// which mixes in detection misses.
//
//   cargo run --release --example rec_accuracy
//   cargo run --release --example rec_accuracy -- --det
#[path = "rec_accuracy/truth.rs"]
mod truth;
use image::{DynamicImage, ImageReader, RgbImage};
use ppocr::{OcrEngine, Point, TextBox};
use std::path::Path;

const SMALL_DET: &str = "models/PP-OCRv6_tiny_det.safetensors";
const SMALL_REC: &str = "models/PP-OCRv6_tiny_rec.safetensors";
const SMALL_DICT: &str = "models/ppocrv6_tiny_keys.txt";

struct Case {
    image: &'static str,
    truth: &'static str,
    /// Box in image coordinates, or None to use the whole image.
    box_in_image: Option<[Point; 4]>,
    note: &'static str,
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> [Point; 4] {
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}

fn whole_image(w: f32, h: f32) -> [Point; 4] {
    rect(0.0, 0.0, w, h)
}

/// The `bench_*.png` fixtures and the box each is scored through.
///
/// The two whole-image fixtures need no box. The document cases are crops, because
/// `bench_doc.png` stacks 28 lines and `recognize` takes one box per call.
fn cases() -> Vec<Case> {
    // `gen_cjk_stress.py` draws this exact 37-character line, with no punctuation. Note the
    // comma-free text: an earlier hand-typed "ground truth" added two full-width commas
    // that the image does not contain, and scored a perfect case at -10%.
    const CJK: &str = "文本检测与识别端到端耗时统计包含预处理前向推理与后处理阶段以及内存带宽限制";

    // Ground truth for every `gen_bench_images.py` fixture comes from the generated table,
    // trimmed to the characters actually rasterised. Three of the eight are clipped by
    // their canvas -- the generator draws at a fixed x with no width check, so
    // `bench_long_line.png` holds 37 of its 117 characters -- and scoring against the
    // untrimmed string charges the engine for pixels that were never drawn.
    let bench_truth = |name: &str| -> &'static str {
        truth::BENCH_TRUTH
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
            .unwrap_or_else(|| panic!("no generated ground truth for case '{name}'"))
    };

    let mut out = vec![
        // Whole-image fixtures.
        Case {
            image: "tests/data/bench_single_line.png",
            truth: bench_truth("single line 36pt"),
            box_in_image: None,
            note: "single line 36pt",
        },
        Case {
            image: "tests/data/bench_long_line.png",
            truth: bench_truth("long mixed CN/EN 38pt"),
            box_in_image: None,
            note: "long mixed CN/EN 38pt",
        },
        // Squeeze sweep. `natural_rec_w = 48 * width / 56` against rec's `max_w = 960`, so
        // these are 0.89x (no squeeze) through 1.69x, the worst case rec is asked to handle.
        Case {
            image: "tests/data/bench_cjk_ref.png",
            truth: CJK,
            box_in_image: None,
            note: "CJK 26pt 0.89x no squeeze",
        },
        Case {
            image: "tests/data/bench_cjk_ar20.png",
            truth: CJK,
            box_in_image: None,
            note: "CJK 30pt 1.03x squeeze",
        },
        Case {
            image: "tests/data/bench_cjk_wide.png",
            truth: CJK,
            box_in_image: None,
            note: "CJK 44pt 1.49x squeeze",
        },
        Case {
            image: "tests/data/bench_cjk_ar33.png",
            truth: CJK,
            box_in_image: None,
            note: "CJK 50pt 1.69x squeeze worst",
        },
        // Document crops. `make_doc` draws "[%02d] %s" at (60, 90 + 45*i) in 24pt, so the
        // "[NN] " prefix is part of the ground truth rather than detector noise. The title
        // is at (60, 30) in 30pt.
        Case {
            image: "tests/data/bench_doc.png",
            truth: bench_truth("doc title 30pt"),
            box_in_image: Some(rect(40.0, 8.0, 800.0, 72.0)),
            note: "doc title 30pt",
        },
    ];

    for (i, name) in [
        "doc body 1 CN 24pt",
        "doc body 2 EN 24pt",
        "doc body 3 MIX 24pt",
        "doc body 4 CN punct 24pt",
        "doc body 5 AVX2 24pt",
    ]
    .iter()
    .enumerate()
    {
        let y = 84.0 + 45.0 * i as f32;
        out.push(Case {
            image: "tests/data/bench_doc.png",
            truth: bench_truth(name),
            box_in_image: Some(rect(50.0, y, 930.0, 36.0)),
            note: name,
        });
    }

    out.push(Case {
        image: "tests/data/test_ocr.png",
        // `gen_bench_images.py` does not produce this one; the text is read off the image.
        truth: "深度学习 PP-OCR",
        box_in_image: None,
        note: "bundled test fixture",
    });

    out
}

/// Levenshtein distance over chars, so a dropped or spurious character costs one edit.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}


fn write_f32(path: &Path, t: &ppocr::Tensor) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(t.numel() * 4);
    for v in t.as_slice() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/rec_dump_tiny".to_string());
    std::fs::create_dir_all(&out_dir)?;
    for p in [SMALL_DET, SMALL_REC, SMALL_DICT] {
        if !Path::new(p).exists() { eprintln!("missing {p}"); return Ok(()); }
    }
    let engine = OcrEngine::new(SMALL_DET, SMALL_REC, SMALL_DICT)?;
    let rec_name = engine.rec_model().graph().inputs[0].clone();
    let mut manifest = String::new();
    for (i, case) in cases().iter().enumerate() {
        let img: RgbImage = ImageReader::open(case.image)?.decode()?.to_rgb8();
        let (iw, ih) = (img.width() as f32, img.height() as f32);
        let pts = case.box_in_image.unwrap_or_else(|| whole_image(iw, ih));
        let patch = ppocr::postprocess::get_rotate_crop_image_rgb(&img, &pts);
        let t = engine.rec_preprocess().process(&patch)?;
        let preds = engine.rec_model().forward(&rec_name, t.clone())?;
        write_f32(Path::new(&format!("{out_dir}/rec_in_{i:03}.bin")), &t)?;
        write_f32(Path::new(&format!("{out_dir}/rec_out_{i:03}.bin")), &preds)?;
        manifest.push_str(&format!(
            "TENSOR rec_in_{i:03} shape={}\nTENSOR rec_out_{i:03} shape={}\n",
            t.shape().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            preds.shape().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
        ));
        manifest.push_str(&format!("CASE\t{i}\t{}\t{}\t{}\n", case.note, case.image, case.truth));
        println!("dump {i:02} {} in{:?} out{:?}", case.note, t.shape(), preds.shape());
    }
    std::fs::write(Path::new(&format!("{out_dir}/manifest.txt")), manifest)?;
    println!("wrote {} cases to {}", cases().len(), out_dir);
    Ok(())
}
