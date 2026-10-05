use crate::engine::Model;
use crate::error::{OcrError, Result};
use crate::postprocess::{CtcDecoder, DbPostProcessor, Point, TextBox, get_rotate_crop_image_rgb};
use crate::preprocess::{ClsPreprocess, DetPreprocess, RecPreprocess};
use crate::tensor::Tensor;
use image::DynamicImage;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct OcrResult {
    pub points: [Point; 4],
    pub text: String,
    pub score: f32,
    /// Detected text orientation in degrees (`0` or `180`), when a CLS model is active.
    pub angle: Option<u16>,
}

#[derive(Clone, Debug)]
pub struct OcrEngineConfig {
    pub max_side_len: u32,
    pub det_thresh: f32,
    pub box_thresh: f32,
    pub unclip_ratio: f32,
    pub rec_target_h: u32,
    /// Maximum width allowed for recognition image preprocessing.
    /// Default is 960, which matches the native aspect handling used by PP-OCR and avoids
    /// inflating recognition cost on high-aspect-ratio lines. Measured on a 29-box A4-class
    /// page (mean aspect ratio 32), widening this to 3200 raised total rec input pixels by
    /// 61% and recognition latency by 62%, while producing byte-identical text on every
    /// stress image tested (up to 1.67x horizontal compression). Raise it only if a
    /// concrete image proves that 960 causes dropped characters.
    pub rec_max_w: u32,
    /// Optional thread count override for multi-line recognition (rec) worker pool.
    /// Defaults to all available cores. Measured on a 29-box page, using all 16 logical
    /// cores was 1.45-1.50x faster than capping at 4, so there is no throughput ceiling
    /// to respect at 4 threads.
    pub rec_batch_threads: Option<usize>,
    /// Quantize large projection MatMul weights in the recognition model to INT8 (W8A32).
    /// Default is false for 100% bit-identical FP32 execution.
    pub quantize_rec_head: bool,
    /// Enable text line orientation classifier (0° vs 180° auto-correction).
    pub use_angle_cls: bool,
    /// Confidence threshold for rotating 180° (default: 0.9).
    pub cls_thresh: f32,
    /// Optional path to custom direction classification model (defaults to models/PP-OCRv6_cls.safetensors).
    pub cls_model_path: Option<String>,
}

impl Default for OcrEngineConfig {
    fn default() -> Self {
        Self {
            max_side_len: 960,
            det_thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.5,
            rec_target_h: 48,
            rec_max_w: 960,
            rec_batch_threads: None,
            quantize_rec_head: false,
            use_angle_cls: false,
            cls_thresh: 0.9,
            cls_model_path: None,
        }
    }
}

/// End-to-end PP-OCR pipeline: detection -> optional orientation correction ->
/// recognition -> CTC decoding.
///
/// The stages are deliberately private. Exposing them made every internal reshuffle a
/// breaking change and let callers mutate `config` behind the engine's back (for example
/// clearing `use_angle_cls` while a CLS model stayed loaded). Use the accessors below.
pub struct OcrEngine {
    det_model: Model,
    rec_model: Model,
    cls_model: Option<Model>,
    det_preprocess: DetPreprocess,
    rec_preprocess: RecPreprocess,
    cls_preprocess: ClsPreprocess,
    db_postprocess: DbPostProcessor,
    ctc_decoder: CtcDecoder,
    config: OcrEngineConfig,
    rec_pool: Option<rayon::ThreadPool>,
}

impl OcrEngine {
    /// The text-detection model.
    pub fn det_model(&self) -> &Model {
        &self.det_model
    }

    /// The text-recognition model.
    pub fn rec_model(&self) -> &Model {
        &self.rec_model
    }

    /// The direction-classification model, if one is loaded (which implies CLS is enabled).
    pub fn cls_model(&self) -> Option<&Model> {
        self.cls_model.as_ref()
    }

    /// The detection input pre-processor.
    pub fn det_preprocess(&self) -> &DetPreprocess {
        &self.det_preprocess
    }

    /// The recognition input pre-processor.
    pub fn rec_preprocess(&self) -> &RecPreprocess {
        &self.rec_preprocess
    }

    /// The DB (Differentiable Binarization) post-processor.
    pub fn db_postprocess(&self) -> &DbPostProcessor {
        &self.db_postprocess
    }

    /// The CTC decoder holding the character dictionary.
    pub fn ctc_decoder(&self) -> &CtcDecoder {
        &self.ctc_decoder
    }

    /// The active configuration.
    pub fn config(&self) -> &OcrEngineConfig {
        &self.config
    }

    /// Mutable access to the configuration.
    ///
    /// Changing `use_angle_cls` here does **not** load or unload the CLS model; prefer
    /// [`OcrEngine::set_use_angle_cls`], which keeps the model and the flag consistent.
    pub fn config_mut(&mut self) -> &mut OcrEngineConfig {
        &mut self.config
    }

    /// Load standard PP-OCRv6 Small models from the models/ directory
    pub fn default_small() -> Result<Self> {
        Self::new(
            "models/PP-OCRv6_small_det.safetensors",
            "models/PP-OCRv6_small_rec.safetensors",
            "models/ppocrv6_keys.txt",
        )
    }

    /// Load standard PP-OCRv6 Tiny models from the models/ directory
    pub fn default_tiny() -> Result<Self> {
        Self::new(
            "models/PP-OCRv6_tiny_det.safetensors",
            "models/PP-OCRv6_tiny_rec.safetensors",
            "models/ppocrv6_tiny_keys.txt",
        )
    }

    /// Load standard PP-OCRv6 Small models with direction classification (CLS) enabled
    pub fn default_small_with_cls() -> Result<Self> {
        let config = OcrEngineConfig {
            use_angle_cls: true,
            ..Default::default()
        };
        Self::with_config(
            "models/PP-OCRv6_small_det.safetensors",
            "models/PP-OCRv6_small_rec.safetensors",
            "models/ppocrv6_keys.txt",
            config,
        )
    }

    /// Load standard PP-OCRv6 Tiny models with direction classification (CLS) enabled
    pub fn default_tiny_with_cls() -> Result<Self> {
        let config = OcrEngineConfig {
            use_angle_cls: true,
            ..Default::default()
        };
        Self::with_config(
            "models/PP-OCRv6_tiny_det.safetensors",
            "models/PP-OCRv6_tiny_rec.safetensors",
            "models/ppocrv6_tiny_keys.txt",
            config,
        )
    }

    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        det_path: P1,
        rec_path: P2,
        dict_path: P3,
    ) -> Result<Self> {
        Self::with_config(det_path, rec_path, dict_path, OcrEngineConfig::default())
    }

    pub fn with_config<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        det_path: P1,
        rec_path: P2,
        dict_path: P3,
        config: OcrEngineConfig,
    ) -> Result<Self> {
        let det_model = Model::load(det_path)?;
        let mut rec_model = Model::load(rec_path)?;
        if config.quantize_rec_head {
            rec_model.quantize_matmul_weights();
        }
        let ctc_decoder = CtcDecoder::from_file(dict_path)?;

        // Supplying a CLS model path is an unambiguous request to use it. Previously this
        // loaded the model but left `use_angle_cls` at its default `false`, so inference
        // silently skipped the classifier while the model stayed resident in memory.
        let mut config = config;
        if config.cls_model_path.is_some() && !config.use_angle_cls {
            config.use_angle_cls = true;
        }

        let cls_model = if config.use_angle_cls {
            let cls_path = config
                .cls_model_path
                .as_deref()
                .unwrap_or("models/PP-OCRv6_cls.safetensors");
            Some(Model::load(cls_path)?)
        } else {
            None
        };

        let det_preprocess = DetPreprocess::new(config.max_side_len);
        let rec_preprocess = RecPreprocess::with_max_w(config.rec_target_h, config.rec_max_w);
        let cls_preprocess = ClsPreprocess::default();
        let db_postprocess =
            DbPostProcessor::new(config.det_thresh, config.box_thresh, config.unclip_ratio);

        // Default to every core the process may use. Note that
        // `available_parallelism` reports the *effective* CPU quota, so inside a
        // cgroup-limited container this can be smaller than the physical core count (and
        // with CPU-quota overcommit, larger). Set `rec_batch_threads` to pin it explicitly.
        // Measured on a 29-box page, capping this at 4 made recognition 1.45-1.50x slower
        // with no bandwidth-throughput benefit.
        let rec_threads = config.rec_batch_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        let rec_pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(rec_threads)
            .build()
        {
            Ok(pool) => Some(pool),
            Err(e) => {
                // Falling back to the global Rayon pool still works, but the thread count
                // will differ from the requested one — say so rather than hiding it.
                eprintln!(
                    "warning: could not create a {rec_threads}-thread recognition pool ({e}); \
                     falling back to the global Rayon pool"
                );
                None
            }
        };

        Ok(Self {
            det_model,
            rec_model,
            cls_model,
            det_preprocess,
            rec_preprocess,
            cls_preprocess,
            db_postprocess,
            ctc_decoder,
            config,
            rec_pool,
        })
    }

    /// Enable INT8 weight quantization for recognition projection head
    pub fn enable_rec_quantization(&mut self) {
        self.rec_model.quantize_matmul_weights();
    }

    /// Load and attach a direction classification model
    pub fn with_cls<P: AsRef<Path>>(mut self, cls_path: P) -> Result<Self> {
        self.cls_model = Some(Model::load(cls_path)?);
        self.config.use_angle_cls = true;
        Ok(self)
    }

    /// Enable or disable angle classification at runtime
    pub fn set_use_angle_cls(&mut self, enabled: bool) -> Result<()> {
        if enabled && self.cls_model.is_none() {
            let path = self
                .config
                .cls_model_path
                .as_deref()
                .unwrap_or("models/PP-OCRv6_cls.safetensors");
            self.cls_model = Some(Model::load(path)?);
        }
        self.config.use_angle_cls = enabled;
        Ok(())
    }

    /// Run the direction classifier and return its raw `[P(0°), P(180°)]` output.
    fn cls_probabilities(&self, patch: &DynamicImage) -> Result<Tensor> {
        let cls_model = self.cls_model.as_ref().ok_or_else(|| {
            OcrError::ModelError("Direction classification model not loaded".to_string())
        })?;
        let tensor = self.cls_preprocess.process(patch)?;
        let input_name = &cls_model.graph().inputs[0];
        cls_model.forward(input_name, tensor)
    }

    /// Classify orientation angle of an image crop patch, using the engine's configured
    /// `cls_thresh`.
    ///
    /// Returns `(angle_degrees, confidence)` where `angle_degrees` is `0` or `180` and
    /// `confidence` is the probability of the *chosen* class.
    pub fn classify_angle(&self, patch: &DynamicImage) -> Result<(u16, f32)> {
        self.classify_angle_with_thresh(patch, self.config.cls_thresh)
    }

    /// [`Self::classify_angle`] with an explicit decision threshold.
    ///
    /// Uses exactly the same rule as the internal orientation-correction path, so the
    /// public API can never disagree with what [`Self::run`] actually does. It previously
    /// used a bare `argmax`; with the default `cls_thresh = 0.9` that made
    /// `classify_angle` report `180°` for crops that `run()` leaves upright (for example
    /// `probs = [0.45, 0.55]`).
    pub fn classify_angle_with_thresh(
        &self,
        patch: &DynamicImage,
        thresh: f32,
    ) -> Result<(u16, f32)> {
        let out = self.cls_probabilities(patch)?;
        let probs = out.as_slice();
        if probs.len() < 2 {
            // Degenerate classifier output: no orientation signal, stay upright.
            return Ok((0, probs.first().copied().unwrap_or(1.0)));
        }
        if probs[1] >= thresh {
            Ok((180, probs[1]))
        } else {
            Ok((0, probs[0]))
        }
    }

    /// Detect text regions in the input image
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<TextBox>> {
        let (input_tensor, scale_h, scale_w) = self.det_preprocess.process(img)?;
        let det_input_name = &self.det_model.graph().inputs[0];
        let prob_map = self.det_model.forward(det_input_name, input_tensor)?;
        let boxes = self.db_postprocess.process(&prob_map, scale_h, scale_w);
        Ok(boxes)
    }

    /// Check patch orientation and rotate 180° when the classifier is confident it is
    /// inverted. Returns the effective angle (`0`/`180`), or `None` without a classifier.
    fn correct_orientation(&self, patch: &mut DynamicImage) -> Result<Option<u16>> {
        if self.cls_model.is_none() {
            return Ok(None);
        }
        let (angle, _score) = self.classify_angle(patch)?;
        if angle == 180 {
            *patch = patch.rotate180();
        }
        Ok(Some(angle))
    }

    /// Crop, optionally de-rotate, recognise and CTC-decode a single detected box.
    ///
    /// Both the single-box fast path and the parallel path funnel through this method so
    /// that their behaviour cannot drift apart — they previously duplicated the pipeline
    /// in two places.
    fn recognize_single_box(
        &self,
        rgb_img: &image::RgbImage,
        b: &TextBox,
        use_cls: bool,
        rec_input_name: &str,
    ) -> Result<Option<OcrResult>> {
        let mut cropped_patch = get_rotate_crop_image_rgb(rgb_img, &b.points);

        let detected_angle = if use_cls {
            self.correct_orientation(&mut cropped_patch)?
        } else {
            None
        };

        let patch_tensor = self.rec_preprocess.process(&cropped_patch)?;
        let preds = self.rec_model.forward(rec_input_name, patch_tensor)?;
        let (text, text_score) = self.ctc_decoder.decode(&preds)?;

        if text.is_empty() {
            return Ok(None);
        }
        Ok(Some(OcrResult {
            points: b.points,
            text,
            score: text_score,
            angle: detected_angle,
        }))
    }

    /// Recognize text within pre-detected text boxes
    pub fn recognize(&self, img: &DynamicImage, boxes: &[TextBox]) -> Result<Vec<OcrResult>> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let rec_input_name = self.rec_model.graph().inputs[0].as_str();
        let use_cls = self.config.use_angle_cls && self.cls_model.is_some();
        let rgb_img = match img {
            DynamicImage::ImageRgb8(rgb) => std::borrow::Cow::Borrowed(rgb),
            other => std::borrow::Cow::Owned(other.to_rgb8()),
        };

        if boxes.len() == 1 {
            return Ok(self
                .recognize_single_box(&rgb_img, &boxes[0], use_cls, rec_input_name)?
                .into_iter()
                .collect());
        }

        // Multi-line concurrent recognition across CPU cores with Rayon
        use rayon::prelude::*;

        let run_batch = || -> Result<Vec<Option<OcrResult>>> {
            boxes
                .par_iter()
                .map(|b| {
                    // Suppress nested Rayon parallelism in inner operators (Conv2d, MatMul, etc.)
                    // so each worker thread runs its box sequentially with optimal cache locality
                    let _guard = crate::tensor::ScopedDisableParallel::new();
                    self.recognize_single_box(&rgb_img, b, use_cls, rec_input_name)
                })
                .collect()
        };

        let results = if let Some(ref pool) = self.rec_pool {
            pool.install(run_batch)
        } else {
            run_batch()
        };

        results.map(|res| res.into_iter().flatten().collect())
    }

    /// Run full end-to-end OCR (detection + recognition)
    pub fn run(&self, img: &DynamicImage) -> Result<Vec<OcrResult>> {
        let boxes = self.detect(img)?;
        self.recognize(img, &boxes)
    }
}
