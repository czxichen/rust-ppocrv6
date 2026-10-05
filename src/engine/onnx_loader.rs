use crate::engine::graph::{AttributeValue, Graph, Node};
use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use protobuf::Message;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Parse a `.onnx` protobuf model into the same `(Graph, HashMap<String, Tensor>)` the
/// safetensors loader produces, so `Model::from_graph_and_weights` can take either.
///
/// Mapping rules (intentionally mirrors the Python `tools/export_onnx.py` → safetensors
/// converter used by the bundled models):
///
/// * `graph.initializer` tensors become `weights`.
/// * `Constant` nodes are folded into `weights` under their output name and dropped from
///   `graph.nodes` (so the engine sees the same node graph it sees from safetensors).
/// * Graph inputs listed again as initializers (ONNX ≤ 8) are filtered out.
/// * Node attributes are lowered to `AttributeValue` for the scalar kinds the engine
///   uses; tensor / graph payloads are skipped (they are already represented as weights).
pub fn load_onnx_model<P: AsRef<Path>>(path: P) -> Result<(Graph, HashMap<String, Tensor>)> {
    let path_ref = path.as_ref();
    let buffer = fs::read(path_ref).map_err(|e| {
        OcrError::ModelError(format!("Failed to read model file {:?}: {}", path_ref, e))
    })?;
    let model = onnx_ir::ModelProto::parse_from_bytes(&buffer).map_err(|e| {
        OcrError::ModelError(format!(
            "Failed to parse ONNX protobuf {:?}: {}",
            path_ref, e
        ))
    })?;
    let graph_p = model
        .graph
        .as_ref()
        .ok_or_else(|| OcrError::ModelError("ONNX model has no graph".into()))?;

    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for init in &graph_p.initializer {
        weights.insert(init.name.clone(), tensor_from_proto(init)?);
    }

    for node_p in &graph_p.node {
        if node_p.op_type != "Constant" {
            continue;
        }
        for attr in &node_p.attribute {
            if attr.name != "value" {
                continue;
            }
            if let Some(t) = attr.t.as_ref() {
                let out_name = node_p.output.first().cloned().unwrap_or_default();
                if out_name.is_empty() {
                    return Err(OcrError::ModelError(
                        "Constant node with 'value' attribute has no output name".into(),
                    ));
                }
                weights.insert(out_name, tensor_from_proto(t)?);
            }
        }
    }

    let initializer_names: HashSet<&str> = graph_p
        .initializer
        .iter()
        .map(|i| i.name.as_str())
        .collect();
    let inputs: Vec<String> = graph_p
        .input
        .iter()
        .filter(|v| !initializer_names.contains(v.name.as_str()))
        .map(|v| v.name.clone())
        .collect();
    let outputs: Vec<String> = graph_p.output.iter().map(|o| o.name.clone()).collect();

    let mut nodes: Vec<Node> = Vec::new();
    for node_p in &graph_p.node {
        if node_p.op_type == "Constant" {
            continue;
        }
        let mut attrs: HashMap<String, AttributeValue> = HashMap::new();
        for attr in &node_p.attribute {
            // ONNX `AttributeProto.type` numeric values; the generated enum type is
            // private to the crate, so compare `type_.value()` directly.
            const ATTR_FLOAT: i32 = 1;
            const ATTR_INT: i32 = 2;
            const ATTR_STRING: i32 = 3;
            const ATTR_FLOATS: i32 = 6;
            const ATTR_INTS: i32 = 7;
            const ATTR_STRINGS: i32 = 8;
            let value = match attr.type_.value() {
                ATTR_FLOAT => Some(AttributeValue::Float(attr.f)),
                ATTR_INT => Some(AttributeValue::Int(attr.i)),
                ATTR_STRING => match std::str::from_utf8(&attr.s) {
                    Ok(s) => Some(AttributeValue::String(s.to_string())),
                    Err(_) => None,
                },
                ATTR_FLOATS => Some(AttributeValue::Floats(attr.floats.clone())),
                ATTR_INTS => Some(AttributeValue::Ints(attr.ints.clone())),
                ATTR_STRINGS => {
                    let v: Option<Vec<String>> = attr
                        .strings
                        .iter()
                        .map(|b| std::str::from_utf8(b).ok().map(|s| s.to_string()))
                        .collect();
                    v.map(AttributeValue::Strings)
                }
                _ => None, // TENSOR / GRAPH payloads already live in `weights`
            };
            if let Some(value) = value {
                attrs.insert(attr.name.clone(), value);
            }
        }
        nodes.push(Node {
            name: if node_p.name.is_empty() {
                node_p.output.first().cloned().unwrap_or_default()
            } else {
                node_p.name.clone()
            },
            op: node_p.op_type.clone(),
            inputs: node_p.input.clone(),
            outputs: node_p.output.clone(),
            attrs,
        });
    }

    Ok((
        Graph {
            inputs,
            outputs,
            nodes,
        },
        weights,
    ))
}

/// ONNX `TensorProto.data_type` numeric values for the dtypes this engine supports.
const ONNX_FLOAT: i32 = 1;
const ONNX_INT32: i32 = 6;
const ONNX_INT64: i32 = 7;

fn tensor_from_proto(init: &onnx_ir::TensorProto) -> Result<Tensor> {
    let shape: Vec<usize> = init
        .dims
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                OcrError::ModelError(format!("tensor '{}': negative dim {}", init.name, d))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    match init.data_type {
        ONNX_FLOAT => {
            let data: Vec<f32> = if !init.float_data.is_empty() {
                init.float_data.clone()
            } else {
                decode_f32_le(&init.raw_data, &init.name)?
            };
            validate_numel(&init.name, &shape, data.len())?;
            Tensor::from_slice(&shape, &data)
        }
        ONNX_INT64 => {
            let data: Vec<i64> = if !init.int64_data.is_empty() {
                init.int64_data.clone()
            } else {
                decode_i64_le(&init.raw_data, &init.name)?
            };
            validate_numel(&init.name, &shape, data.len())?;
            Tensor::from_i64_slice(&shape, &data)
        }
        ONNX_INT32 => {
            let data: Vec<i32> = if !init.int32_data.is_empty() {
                init.int32_data.clone()
            } else {
                decode_i32_le(&init.raw_data, &init.name)?
            };
            validate_numel(&init.name, &shape, data.len())?;
            Tensor::from_i32_slice(&shape, &data)
        }
        other => Err(OcrError::ModelError(format!(
            "tensor '{}': unsupported ONNX data_type {} (only FLOAT/INT64/INT32 are supported)",
            init.name, other
        ))),
    }
}

fn validate_numel(name: &str, shape: &[usize], numel: usize) -> Result<()> {
    let expected: usize = shape.iter().product();
    if expected != numel {
        return Err(OcrError::ModelError(format!(
            "tensor '{}': shape {:?} implies {} elements but payload has {}",
            name, shape, expected, numel
        )));
    }
    Ok(())
}

fn decode_f32_le(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    decode_le(bytes, name, "f32", 4).map(|v| {
        v.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    })
}

fn decode_i64_le(bytes: &[u8], name: &str) -> Result<Vec<i64>> {
    decode_le(bytes, name, "i64", 8).map(|v| {
        v.chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    })
}

fn decode_i32_le(bytes: &[u8], name: &str) -> Result<Vec<i32>> {
    decode_le(bytes, name, "i32", 4).map(|v| {
        v.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    })
}

fn decode_le<'a>(bytes: &'a [u8], name: &str, _ty: &str, size: usize) -> Result<&'a [u8]> {
    if bytes.len() % size != 0 {
        return Err(OcrError::ModelError(format!(
            "tensor '{}': raw_data length {} is not a multiple of {}",
            name,
            bytes.len(),
            size
        )));
    }
    Ok(bytes)
}
