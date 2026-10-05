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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_det = std::env::args().any(|a| a == "--det");

    for p in [SMALL_DET, SMALL_REC, SMALL_DICT] {
        if !Path::new(p).exists() {
            eprintln!("skipping: required asset '{p}' is not present");
            return Ok(());
        }
    }

    let engine = OcrEngine::new(SMALL_DET, SMALL_REC, SMALL_DICT)?;
    println!(
        "tiny rec accuracy -- {}\n",
        if use_det {
            "full detect+recognise pipeline"
        } else {
            "recognition stage only (explicit box, DBNet not involved)"
        }
    );
    println!(
        "{:<26} {:>5} {:>5} {:>7} {:>7}  {}",
        "case", "chars", "edit", "acc%", "score", "output"
    );

    let mut tot_chars = 0usize;
    let mut tot_edit = 0usize;
    let mut worst = (101.0f64, String::new());
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    for case in cases() {
        let img: RgbImage = ImageReader::open(case.image)?.decode()?.to_rgb8();
        let (iw, ih) = (img.width() as f32, img.height() as f32);

        // `--det` runs the whole pipeline, which only makes sense where the image holds
        // exactly the one ground-truth line. The document cases are crops of a 28-line
        // page, so `run` returns all 28 lines and scoring that against a single line's
        // ground truth produces nonsense (it reported -1618%). Skip them with a note
        // rather than printing a number that looks like a result.
        if use_det && case.box_in_image.is_some() {
            println!(
                "{:<26} {:>5} {:>5} {:>7} {:>7}  skipped: --det on a multi-line crop",
                case.note, "-", "-", "-", "-"
            );
            continue;
        }

        let (text, score) = if use_det {
            let r = engine
                .run(&DynamicImage::ImageRgb8(img))?
                .into_iter()
                .map(|r| r.text)
                .collect::<String>();
            (r, f32::NAN)
        } else {
            let pts = case.box_in_image.unwrap_or_else(|| whole_image(iw, ih));
            let box_ = TextBox {
                points: pts,
                score: 1.0,
            };
            match engine.recognize(&DynamicImage::ImageRgb8(img), &[box_])? {
                r if r.is_empty() => (String::new(), f32::NAN),
                r => {
                    let s = r[0].score;
                    (r[0].text.clone(), s)
                }
            }
        };

        let got: Vec<char> = text.chars().collect();
        let want: Vec<char> = case.truth.chars().collect();
        let edit = edit_distance(&got, &want);
        let acc = 100.0 * (1.0 - edit as f64 / want.len().max(1) as f64);
        tot_chars += want.len();
        tot_edit += edit;
        if acc < worst.0 {
            worst = (acc, case.note.to_string());
        }
        if edit > 0 {
            fails.push(format!("  {} ({:.1}%)", case.note, acc));
        }

        let score_s = if score.is_nan() {
            "-".to_string()
        } else {
            format!("{score:.4}")
        };
        // The want/got pair goes to a UTF-8 report file rather than stdout. This is a
        // Windows console, where the CJK bytes get re-encoded (mojibake) on the way out,
        // which makes a diff against ground truth impossible to read -- and reading that
        // diff is the entire point of this tool.
        report.push_str(&format!(
            "{:<26} {:>5} chars {:>3} edits {:>6.2}%  score {}\n    want: {}\n    got : {}\n\n",
            case.note,
            want.len(),
            edit,
            acc,
            score_s,
            case.truth,
            text
        ));
        println!(
            "{:<26} {:>5} {:>5} {:>6.2}% {:>7}  (see report)",
            case.note,
            want.len(),
            edit,
            acc,
            score_s
        );
    }

    let total = 100.0 * (1.0 - tot_edit as f64 / tot_chars.max(1) as f64);
    println!("\nTOTAL {tot_chars} chars, {tot_edit} edits, {total:.2}% accurate");
    println!("worst case {:.2}% ({})", worst.0, worst.1);
    if !fails.is_empty() {
        println!("cases with edits:");
        for f in &fails {
            println!("{f}");
        }
    } else {
        println!("every case exact");
    }

    let out = std::env::args()
        .find_map(|a| a.strip_prefix("--out=").map(str::to_string))
        .unwrap_or_else(|| "rec_accuracy_report.txt".to_string());
    std::fs::write(&out, report).map_err(|e| format!("could not write '{out}': {e}"))?;
    println!("\nper-case want/got written to {out}");
    Ok(())
}
