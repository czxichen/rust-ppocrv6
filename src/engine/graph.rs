use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Strings(Vec<String>),
}

impl AttributeValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            Self::Int(v) => Some(vec![*v]),
            Self::Ints(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Coerce a vector attribute to `usize`, rejecting negatives.
    ///
    /// The previous `x as usize` turned `-1` into `usize::MAX`, which then flowed
    /// unchecked into `strides`, `pads`, `dilations` and `kernel_shape` in
    /// `Model::forward_internal`. A model carrying `"pads": [-1, -1, -1, -1]` therefore
    /// loaded without error and produced a wrong-shaped output or an index panic deep
    /// inside an operator. The graph JSON comes from the caller's `--model-dir`, so this is
    /// untrusted input at the crate's validation boundary.
    ///
    /// Returning `None` lets the existing `.and_then(|a| a.as_usize_vec()).unwrap_or_else(..)`
    /// fall back to the default, which is the right behaviour for an attribute that is
    /// present but nonsensical; a `group` or `kernel_shape` that is genuinely zero or
    /// negative is rejected by the operators themselves.
    pub fn as_usize_vec(&self) -> Option<Vec<usize>> {
        self.as_i64_vec()?
            .into_iter()
            .map(|x| usize::try_from(x).ok())
            .collect()
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Self::Strings(s) => Some(s.as_slice()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default)]
    pub attrs: HashMap<String, AttributeValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: Vec<Node>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attr(json: &str) -> AttributeValue {
        serde_json::from_str(json).expect("attribute must deserialize")
    }

    /// `as_usize_vec` used to `as`-cast every element, so `-1` came out as
    /// `usize::MAX` and flowed unchecked into `strides` / `pads` / `dilations` /
    /// `kernel_shape`. A model carrying `"pads": [-1, -1, -1, -1]` therefore loaded without
    /// error and then produced a wrong-shaped output or an index panic inside an operator.
    #[test]
    fn as_usize_vec_rejects_negatives_instead_of_wrapping() {
        assert_eq!(attr("[1, 1]").as_usize_vec(), Some(vec![1, 1]));
        assert_eq!(attr("[0, 0, 0, 0]").as_usize_vec(), Some(vec![0, 0, 0, 0]));
        assert_eq!(attr("3").as_usize_vec(), Some(vec![3]));

        assert_eq!(attr("[-1]").as_usize_vec(), None);
        assert_eq!(attr("[1, -1]").as_usize_vec(), None);
        assert_eq!(attr("[-1, -1, -1, -1]").as_usize_vec(), None);
        // A shape/dtype mismatch is still not a vector attribute.
        assert_eq!(attr("[1.5, 2.5]").as_usize_vec(), None);
        assert_eq!(attr("\"1,1\"").as_usize_vec(), None);
    }

    /// The `untagged` enum tries variants in declaration order, so an integral JSON float
    /// must fall through `Int` to `Float` and `as_f32` must accept both spellings.
    #[test]
    fn attribute_coercion_handles_int_and_float_spellings() {
        assert_eq!(attr("1").as_i64(), Some(1));
        assert_eq!(attr("1.0").as_i64(), None);
        assert_eq!(attr("1.0").as_f32(), Some(1.0));
        assert_eq!(attr("1").as_f32(), Some(1.0));
        assert_eq!(attr("[1, 2]").as_i64_vec(), Some(vec![1, 2]));
        assert_eq!(attr("1").as_i64_vec(), Some(vec![1]));
        assert_eq!(attr("[1.0, 2.0]").as_i64_vec(), None);
        assert_eq!(attr("\"SAME_UPPER\"").as_str(), Some("SAME_UPPER"));
        assert_eq!(attr("1").as_str(), None);
    }

    /// Every node's `inputs` and `outputs` are indexed unguarded in `Model::load` and
    /// `forward_internal`, and `pipeline.rs` indexes `graph().inputs[0]`, so a graph with
    /// empty vectors must fail deserialisation-time validation rather than panic later.
    #[test]
    fn a_graph_with_empty_node_io_still_deserialises_so_load_can_reject_it() {
        // `inputs`/`outputs` on Node are required fields with no `#[serde(default)]`, so an
        // *omitted* key is already an error at parse time.
        let missing: std::result::Result<Graph, serde_json::Error> = serde_json::from_str(
            r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"name":"n","op":"Add"}]}"#,
        );
        assert!(missing.is_err(), "an omitted `inputs` must not deserialise");

        // An explicitly empty list does deserialise, which is exactly why `Model::load` now
        // has to reject it.
        let empty: Graph = serde_json::from_str(
            r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"name":"n","op":"Add","inputs":[],"outputs":[]}]}"#,
        )
        .expect("explicit empty lists deserialise");
        assert!(empty.nodes[0].inputs.is_empty());
        assert!(empty.nodes[0].outputs.is_empty());
    }
}
