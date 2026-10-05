//! Shared command-line front-end.
//!
//! `src/main.rs` and `examples/ocr_image.rs` used to carry two near-identical copies of
//! this logic — same banner, same hard-coded model paths, same output formatting — so every
//! change had to be made twice and the two had already drifted apart (the example was
//! missing `--tiny` and `--quantize`). Both entry points now call into this module.

use crate::error::{OcrError, Result};
use crate::pipeline::{OcrEngine, OcrEngineConfig};
use image::ImageReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Image used when the user does not name one, relative to the current directory.
pub const DEFAULT_IMAGE: &str = "tests/data/test_ocr.png";

/// Directory searched for model files when the user does not pass `--model-dir`.
pub const DEFAULT_MODEL_DIR: &str = "models";

/// Parsed command line.
#[derive(Debug, Clone)]
pub struct CliOptions {
    /// Use the Tiny model tier instead of Small.
    pub use_tiny: bool,
    /// Quantize the recognition projection head to INT8.
    pub quantize: bool,
    /// Enable the direction classifier.
    pub use_cls: bool,
    /// Directory holding the model files, so the tool is not tied to the repo root.
    pub model_dir: PathBuf,
    /// Image to run on; `None` means [`DEFAULT_IMAGE`].
    pub image_path: Option<String>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            use_tiny: false,
            quantize: false,
            use_cls: false,
            model_dir: PathBuf::from(DEFAULT_MODEL_DIR),
            image_path: None,
        }
    }
}

/// What the caller should do after parsing.
#[derive(Debug, Clone)]
pub enum CliAction {
    /// Print usage and exit successfully.
    Help,
    /// Run a full OCR pass.
    Run(Box<CliOptions>),
}

/// Parse `args` (without the program name).
pub fn parse_args(args: &[String]) -> std::result::Result<CliAction, String> {
    let mut options = CliOptions::default();
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(CliAction::Help),
            "-t" | "--tiny" => options.use_tiny = true,
            "-s" | "--small" => options.use_tiny = false,
            "-q" | "--quantize" => options.quantize = true,
            "-c" | "--cls" => options.use_cls = true,
            "--model-dir" | "-m" => {
                let value = iter
                    .next()
                    .ok_or_else(|| format!("{arg} requires a directory argument"))?;
                options.model_dir = PathBuf::from(value);
            }
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(format!("unknown option '{other}'"));
            }
            path => {
                if options.image_path.is_some() {
                    return Err(format!(
                        "more than one image given ('{}' and '{path}')",
                        options.image_path.as_deref().unwrap_or("")
                    ));
                }
                options.image_path = Some(path.to_string());
            }
        }
    }

    Ok(CliAction::Run(Box::new(options)))
}

/// Print usage information.
pub fn print_help() {
    println!("PP-OCRv6 Pure Rust Lightweight Inference Engine");
    println!();
    println!("Usage:");
    println!("  rust_ppocrv6 [OPTIONS] [IMAGE_PATH]");
    println!();
    println!("Options:");
    println!("  -t, --tiny          Use PP-OCRv6 Tiny models (fastest, lightweight ~7MB)");
    println!("  -s, --small         Use PP-OCRv6 Small models (default, high accuracy)");
    println!("  -q, --quantize      Enable INT8 weight quantization on the recognition head");
    println!("  -c, --cls           Enable text line direction classification (0° vs 180°)");
    println!("  -m, --model-dir DIR Directory holding the model files (default: models/)");
    println!("  -h, --help          Print this help message");
    println!();
    println!("Arguments:");
    println!("  [IMAGE_PATH]        Path to an image file (defaults to {DEFAULT_IMAGE})");
    println!();
    println!("Examples:");
    println!("  cargo run --release -- path/to/image.png");
    println!("  cargo run --release -- --tiny path/to/image.png");
    println!("  cargo run --release -- --cls --quantize --model-dir /opt/ocr/models img.png");
}

/// The model files for a tier, plus a human-readable tier name.
fn tier_files(dir: &Path, tiny: bool) -> (PathBuf, PathBuf, PathBuf, &'static str) {
    if tiny {
        (
            dir.join("PP-OCRv6_tiny_det.safetensors"),
            dir.join("PP-OCRv6_tiny_rec.safetensors"),
            dir.join("ppocrv6_tiny_keys.txt"),
            "PP-OCRv6 Tiny",
        )
    } else {
        (
            dir.join("PP-OCRv6_small_det.safetensors"),
            dir.join("PP-OCRv6_small_rec.safetensors"),
            dir.join("ppocrv6_keys.txt"),
            "PP-OCRv6 Small",
        )
    }
}

/// Load the models, run detection + recognition and print a report.
///
/// Every failure path is a `Result`, so callers decide how to report it — the previous
/// front-ends used `expect`/`panic!` throughout.
pub fn run(options: &CliOptions) -> Result<()> {
    let (det_path, rec_path, dict_path, tier_name) =
        tier_files(&options.model_dir, options.use_tiny);
    let cls_path = options.model_dir.join("PP-OCRv6_cls.safetensors");
    let image_path = options
        .image_path
        .clone()
        .unwrap_or_else(|| DEFAULT_IMAGE.to_string());

    // Validate everything up front so the user gets one clear message rather than an error
    // from deep inside the loader.
    let mut missing: Vec<String> = [&det_path, &rec_path, &dict_path]
        .into_iter()
        .filter(|p| !p.exists())
        .map(|p| p.display().to_string())
        .collect();
    if options.use_cls && !cls_path.exists() {
        missing.push(cls_path.display().to_string());
    }
    if !missing.is_empty() {
        return Err(OcrError::ModelError(format!(
            "model file(s) not found under '{}': {}",
            options.model_dir.display(),
            missing.join(", ")
        )));
    }
    if !Path::new(&image_path).exists() {
        return Err(OcrError::ImageError(format!(
            "image not found: {image_path}"
        )));
    }

    println!("============================================================");
    println!("       PP-OCRv6 OCR Inference        ");
    println!("============================================================");
    println!("Model Tier   : {tier_name}");
    println!(
        "Quantization : {}",
        if options.quantize {
            "Enabled (W8A32 INT8)"
        } else {
            "Disabled (FP32)"
        }
    );
    println!(
        "Orientation  : {}",
        if options.use_cls {
            "Enabled (PP-OCRv6 CLS)"
        } else {
            "Disabled"
        }
    );
    println!("Loading models from {}...", options.model_dir.display());

    let config = OcrEngineConfig {
        quantize_rec_head: options.quantize,
        use_angle_cls: options.use_cls,
        cls_model_path: options
            .use_cls
            .then(|| cls_path.to_string_lossy().into_owned()),
        ..Default::default()
    };

    let init_start = Instant::now();
    let engine = OcrEngine::with_config(&det_path, &rec_path, &dict_path, config)?;
    println!("Models loaded in {:.2?}", init_start.elapsed());

    println!("Loading image: {image_path}");
    let img = ImageReader::open(&image_path)
        .map_err(|e| OcrError::ImageError(format!("cannot open '{image_path}': {e}")))?
        .decode()
        .map_err(|e| OcrError::ImageError(format!("cannot decode '{image_path}': {e}")))?;
    println!("Image dimensions: {}x{}", img.width(), img.height());

    println!("Running text detection...");
    let det_start = Instant::now();
    let boxes = engine.detect(&img)?;
    let det_time = det_start.elapsed();
    println!(
        "Detected {} text region(s) in {:.2?}",
        boxes.len(),
        det_time
    );

    println!(
        "Running text recognition{}...",
        if options.use_cls {
            " (with angle CLS)"
        } else {
            ""
        }
    );
    let rec_start = Instant::now();
    let results = engine.recognize(&img, &boxes)?;
    let rec_time = rec_start.elapsed();

    println!("\n---------------------- OCR Results -------------------------");
    for (i, res) in results.iter().enumerate() {
        let angle_str = match res.angle {
            Some(a) => format!(", Angle: {a}°"),
            None => String::new(),
        };
        println!(
            "[{:>2}] Text: \"{}\"  (Confidence: {:.2}%{})",
            i + 1,
            res.text,
            res.score * 100.0,
            angle_str,
        );
        println!(
            "     Box: [[{:.1}, {:.1}], [{:.1}, {:.1}], [{:.1}, {:.1}], [{:.1}, {:.1}]]",
            res.points[0][0],
            res.points[0][1],
            res.points[1][0],
            res.points[1][1],
            res.points[2][0],
            res.points[2][1],
            res.points[3][0],
            res.points[3][1],
        );
    }
    println!("------------------------------------------------------------");
    println!("Performance Breakdown:");
    println!("  - Detection:   {:.2?}", det_time);
    println!("  - Recognition: {:.2?}", rec_time);
    println!("  - Total Time:  {:.2?}", det_time + rec_time);
    println!("============================================================");
    Ok(())
}

/// Parse `args`, print usage on `--help`/misuse, and run. Returns a process exit code.
pub fn main_with_args(args: &[String]) -> i32 {
    match parse_args(args) {
        Ok(CliAction::Help) => {
            print_help();
            0
        }
        Ok(CliAction::Run(options)) => match run(&options) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("Error: {e}");
                1
            }
        },
        Err(msg) => {
            eprintln!("Error: {msg}");
            print_help();
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(args: &[&str]) -> CliOptions {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        match parse_args(&owned).expect("parse should succeed") {
            CliAction::Run(o) => *o,
            CliAction::Help => panic!("expected Run, got Help"),
        }
    }

    #[test]
    fn parses_flags_and_image_path() {
        let o = opts(&["--tiny", "-q", "-c", "img.png"]);
        assert!(o.use_tiny);
        assert!(o.quantize);
        assert!(o.use_cls);
        assert_eq!(o.image_path.as_deref(), Some("img.png"));
        assert_eq!(o.model_dir, PathBuf::from("models"));
    }

    #[test]
    fn model_dir_is_configurable() {
        assert_eq!(
            opts(&["--model-dir", "/opt/ocr/models", "a.png"]).model_dir,
            PathBuf::from("/opt/ocr/models")
        );
        assert_eq!(
            opts(&["-m", "somewhere", "a.png"]).model_dir,
            PathBuf::from("somewhere")
        );
    }

    #[test]
    fn help_short_circuits_parsing() {
        assert!(matches!(
            parse_args(&["--help".to_string()]),
            Ok(CliAction::Help)
        ));
        // Even when mixed with other arguments.
        assert!(matches!(
            parse_args(&["img.png".to_string(), "-h".to_string()]),
            Ok(CliAction::Help)
        ));
    }

    #[test]
    fn rejects_unknown_options_and_missing_values() {
        assert!(parse_args(&["--nope".to_string()]).is_err());
        assert!(parse_args(&["--model-dir".to_string()]).is_err());
    }

    #[test]
    fn rejects_more_than_one_image() {
        let args = vec!["a.png".to_string(), "b.png".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn missing_models_produce_an_error_not_a_panic() {
        let o = CliOptions {
            model_dir: PathBuf::from("definitely/not/a/real/model/dir"),
            ..Default::default()
        };
        let err = run(&o).expect_err("missing model files must be reported as an error");
        assert!(
            format!("{err:?}").contains("not found"),
            "unexpected error: {err:?}"
        );
    }
}
