use crate::engine::graph::Graph;
use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reinterpret a safetensors byte range as `T`, or fail with a message instead of panicking.
///
/// `bytemuck::cast_slice` carries two preconditions this call site cannot promise:
///
/// * **Alignment.** `fs::read` returns a `Vec<u8>`, whose alignment is formally 1, and
///   `view.data()` hands back a sub-slice at an arbitrary byte offset into it. The load only
///   works today because the global allocator happens to return a 16-byte-aligned block for
///   these (multi-megabyte) buffers. That is an undocumented dependency, not a guarantee.
/// * **Length.** `cast_slice` panics when the byte length is not a whole number of elements.
///   In practice `safetensors::SafeTensors::deserialize` validates that upstream — probed
///   with shape/payload combinations of 2, 6, 12, 14, 20 and 28 bytes against F32/I64/I32,
///   and every inconsistent one was rejected before reaching here — so this is defence in
///   depth rather than a live crash. It stays because it costs nothing and turns a would-be
///   panic into a typed error at the crate's only untrusted-input boundary, which is what
///   `Cargo.toml`'s decision to leave `panic = "abort"` off is trying to protect.
fn reinterpret<'a, T: bytemuck::Pod>(data: &'a [u8], name: &str) -> Result<&'a [T]> {
    bytemuck::try_cast_slice(data).map_err(|_| {
        OcrError::SafeTensorsError(format!(
            "tensor '{name}': {} bytes is not a whole number of {}-byte values",
            data.len(),
            std::mem::size_of::<T>()
        ))
    })
}

pub fn load_safetensors_model<P: AsRef<Path>>(path: P) -> Result<(Graph, HashMap<String, Tensor>)> {
    let path_ref = path.as_ref();
    let buffer = fs::read(path_ref).map_err(|e| {
        OcrError::ModelError(format!("Failed to read model file {:?}: {}", path_ref, e))
    })?;

    let tensors = SafeTensors::deserialize(&buffer).map_err(|e| {
        OcrError::SafeTensorsError(format!("Failed to deserialize SafeTensors: {}", e))
    })?;

    // Read metadata
    let (_, metadata) = SafeTensors::read_metadata(&buffer)
        .map_err(|e| OcrError::SafeTensorsError(format!("Failed to read metadata: {}", e)))?;

    let meta_map = metadata
        .metadata()
        .as_ref()
        .ok_or_else(|| OcrError::ModelError("Model file missing metadata header".into()))?;

    let graph_json = meta_map
        .get("graph")
        .ok_or_else(|| OcrError::ModelError("Model metadata missing 'graph' definition".into()))?;

    let graph: Graph = serde_json::from_str(graph_json)
        .map_err(|e| OcrError::ModelError(format!("Failed to parse graph JSON: {}", e)))?;

    // Load all weight and constant tensors
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = view.dtype();
        let data = view.data();

        let tensor = match dtype {
            safetensors::Dtype::F32 => {
                let f32_slice: &[f32] = reinterpret(data, name.as_str())?;
                Tensor::from_slice(&shape, f32_slice)?
            }
            safetensors::Dtype::I64 => {
                let i64_slice: &[i64] = reinterpret(data, name.as_str())?;
                Tensor::from_i64_slice(&shape, i64_slice)?
            }
            safetensors::Dtype::I32 => {
                let i32_slice: &[i32] = reinterpret(data, name.as_str())?;
                Tensor::from_i32_slice(&shape, i32_slice)?
            }
            other => {
                return Err(OcrError::ModelError(format!(
                    "Unsupported tensor dtype {:?} for tensor {}",
                    other, name
                )));
            }
        };

        weights.insert(name.to_string(), tensor);
    }

    Ok((graph, weights))
}

/// Pick a concrete loader from the file extension. `.onnx` → ONNX parser (requires the
/// `onnx` cargo feature), everything else → safetensors bundle.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<(Graph, HashMap<String, Tensor>)> {
    let p = path.as_ref();
    match p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
    {
        Some(ext) if ext == "onnx" => {
            #[cfg(feature = "onnx")]
            {
                crate::engine::onnx_loader::load_onnx_model(p)
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(OcrError::ModelError(format!(
                    "Loading '{}' requires the `onnx` cargo feature (e.g. `--features onnx`).",
                    p.display()
                )))
            }
        }
        _ => load_safetensors_model(p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid graph JSON: one Identity node plus a matching input and output.
    const GRAPH: &str = r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"name":"id","op":"Identity","inputs":["x"],"outputs":["y"]}]}"#;

    /// Assemble a safetensors buffer in memory: an 8-byte little-endian header length, the
    /// JSON header (padded to a multiple of 8), then the tensor payload.
    ///
    /// `header_numel` is the element count written into the header and is a parameter in its
    /// own right so the negative tests can make the header disagree with the payload.
    fn build(header_numel: usize, payload: &[u8], graph: Option<&str>) -> Vec<u8> {
        let graph_field = graph
            .map(|g| format!(r#","graph":"{}""#, g.replace('"', "\\\"")))
            .unwrap_or_default();
        let header = format!(
            r#"{{"__metadata__":{{"format":"pt"{graph_field}}},"w":{{"dtype":"F32","shape":[{header_numel}],"data_offsets":[0,{n}]}}}}"#,
            n = payload.len()
        );

        let mut header_bytes = header.into_bytes();
        while header_bytes.len() % 8 != 0 {
            header_bytes.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(payload);
        out
    }

    fn f32_payload(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn write_temp(bytes: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join("ppocr_loader_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(format!(
            "m{}.safetensors",
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write temp model");
        path
    }

    /// The control case: a well-formed buffer must load. Every negative test below is only
    /// meaningful if this proves the builder produces something `safetensors` accepts.
    #[test]
    fn loads_a_well_formed_buffer() {
        let buf = build(4, &f32_payload(&[1.0, 2.0, 3.0, 4.0]), Some(GRAPH));
        let (graph, weights) = load_safetensors_model(write_temp(&buf)).expect("must load");

        assert_eq!(graph.inputs, vec!["x".to_string()]);
        assert_eq!(graph.outputs, vec!["y".to_string()]);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(weights["w"].as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// `reinterpret` is the actual guarantee under test here. `safetensors` validates
    /// payload lengths before this point (probed: F32/I64/I32 payloads of 2, 6, 12, 14, 20
    /// and 28 bytes against mismatching shapes were all rejected upstream with
    /// `MetadataIncompleteBuffer`), so a truncated file never reaches it — which is exactly
    /// why this case is tested against the helper directly instead of through a file.
    ///
    /// `bytemuck::cast_slice` would panic on every one of these; `try_cast_slice` must not.
    #[test]
    fn reinterpret_rejects_non_element_aligned_lengths_without_panicking() {
        for (label, bytes) in [
            ("f32 / 2 bytes", 2usize),
            ("f32 / 6 bytes", 6),
            ("f32 / 14 bytes", 14),
            ("i64 / 12 bytes", 12),
            ("i64 / 20 bytes", 20),
        ] {
            let data = vec![0u8; bytes];
            if label.starts_with("f32") {
                let r = std::panic::catch_unwind(|| reinterpret::<f32>(&data, "w"));
                assert!(r.is_ok(), "{label}: reinterpret must not panic");
                assert!(r.unwrap().is_err(), "{label}: must be rejected");
            } else {
                let r = std::panic::catch_unwind(|| reinterpret::<i64>(&data, "w"));
                assert!(r.is_ok(), "{label}: reinterpret must not panic");
                assert!(r.unwrap().is_err(), "{label}: must be rejected");
            }
        }
    }

    #[test]
    fn reinterpret_accepts_a_correctly_aligned_slice() {
        let payload = f32_payload(&[1.5, -2.5, 3.5]);
        let got: &[f32] = reinterpret(&payload, "w").expect("16-byte-aligned f32 slice is fine");
        assert_eq!(got, &[1.5, -2.5, 3.5]);
    }

    /// A shape that disagrees with the payload is rejected. `safetensors` catches this
    /// upstream, so this documents the boundary rather than a code path in `reinterpret`.
    #[test]
    fn shape_length_mismatch_is_an_error_not_a_panic() {
        let buf = build(8, &f32_payload(&[1.0, 2.0, 3.0, 4.0]), Some(GRAPH));
        let result = load_safetensors_model(write_temp(&buf));
        assert!(
            result.is_err(),
            "a shape/element-count mismatch must not load successfully"
        );
    }

    #[test]
    fn missing_graph_metadata_is_an_error() {
        let buf = build(1, &f32_payload(&[7.0]), None);
        let err = load_safetensors_model(write_temp(&buf)).expect_err("must be rejected");
        assert!(
            matches!(err, OcrError::ModelError(_)),
            "expected ModelError, got {err:?}"
        );
    }

    #[test]
    fn unparseable_graph_json_is_an_error() {
        let buf = build(1, &f32_payload(&[7.0]), Some("{not json"));
        let err = load_safetensors_model(write_temp(&buf)).expect_err("must be rejected");
        assert!(
            matches!(err, OcrError::ModelError(_)),
            "expected ModelError, got {err:?}"
        );
    }

    #[test]
    fn empty_buffer_is_an_error() {
        let err = load_safetensors_model(write_temp(&[])).expect_err("must be rejected");
        assert!(
            matches!(err, OcrError::SafeTensorsError(_) | OcrError::ModelError(_)),
            "expected a load error, got {err:?}"
        );
    }

    #[test]
    fn nonexistent_file_is_an_error() {
        let err = load_safetensors_model("definitely/not/here.safetensors")
            .expect_err("must be rejected");
        assert!(
            matches!(err, OcrError::ModelError(_)),
            "expected ModelError, got {err:?}"
        );
    }
}
