use crate::engine::graph::{AttributeValue, Graph, Node};
use crate::engine::loader::load_model;
use crate::error::{OcrError, Result};
use crate::ops::{self, Conv2dParams, ConvTranspose2dParams, Pool2dParams};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub struct Model {
    graph: Graph,
    weights: HashMap<String, Tensor>,
    usage_counts: HashMap<String, usize>,
    winograd_weights: HashMap<String, Vec<Tensor>>,
    transposed_matmul_weights: HashMap<String, Tensor>,
    quantized_matmul_weights: HashMap<String, ops::QuantizedWeight>,
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (graph, weights) = load_model(path)?;
        Self::from_graph_and_weights(graph, weights)
    }

    /// Explicitly load an ONNX model (`.onnx`). Identical to [`Model::load`] when the
    /// path ends in `.onnx`; provided so callers can pick the loader instead of
    /// relying on extension detection. Requires the `onnx` cargo feature.
    #[cfg(feature = "onnx")]
    pub fn load_onnx<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (graph, weights) = crate::engine::onnx_loader::load_onnx_model(path)?;
        Self::from_graph_and_weights(graph, weights)
    }

    /// Shared entry point: validate + optimise + package a parsed `(graph, weights)`
    /// pair regardless of whether it came from a safetensors bundle or an ONNX file.
    fn from_graph_and_weights(
        mut graph: Graph,
        mut weights: HashMap<String, Tensor>,
    ) -> Result<Self> {
        // --- Structural validation ---
        //
        // The graph JSON is untrusted input (the path comes from the caller's CLI
        // `--model-dir`). Every node's `inputs`/`outputs` are indexed unguarded below and in
        // `forward_internal`, and `pipeline.rs` indexes `graph().inputs[0]`, so a node with
        // `"outputs": []` used to panic *during load* instead of returning a `ModelError` --
        // defeating the `Result`-based error handling that the `Cargo.toml` comment about
        // leaving `panic = "abort"` off is explicitly trying to preserve for embedders.
        if graph.inputs.is_empty() {
            return Err(OcrError::ModelError("graph declares no inputs".to_string()));
        }
        if graph.outputs.is_empty() {
            return Err(OcrError::ModelError(
                "graph declares no outputs".to_string(),
            ));
        }
        for node in &graph.nodes {
            if node.inputs.is_empty() || node.outputs.is_empty() {
                return Err(OcrError::ModelError(format!(
                    "node '{}' ({}) has {} inputs and {} outputs; both must be non-empty",
                    node.name,
                    node.op,
                    node.inputs.len(),
                    node.outputs.len()
                )));
            }
        }

        // --- Graph Optimization Pass 1: Identity Elimination ---
        let mut identities: HashMap<String, String> = HashMap::new();
        let mut non_identities = Vec::new();

        for n in graph.nodes {
            if n.op == "Identity" {
                let mut src = n.inputs[0].clone();
                while let Some(prev) = identities.get(&src) {
                    src = prev.clone();
                }
                identities.insert(n.outputs[0].clone(), src);
            } else {
                non_identities.push(n);
            }
        }

        for n in &mut non_identities {
            for inp in &mut n.inputs {
                if let Some(src) = identities.get(inp) {
                    *inp = src.clone();
                }
            }
        }

        for out in &mut graph.outputs {
            if let Some(src) = identities.get(out) {
                *out = src.clone();
            }
        }

        // --- Graph Optimization Pass 2: Conv / ConvTranspose / MatMul + Add(bias) Folding ---
        let mut usage: HashMap<String, usize> = HashMap::new();
        for n in &non_identities {
            for inp in &n.inputs {
                *usage.entry(inp.clone()).or_insert(0) += 1;
            }
        }
        for out in &graph.outputs {
            *usage.entry(out.clone()).or_insert(0) += 1;
        }

        let mut conv_indices: HashMap<String, usize> = HashMap::new();
        let mut conv_transpose_indices: HashMap<String, usize> = HashMap::new();
        let mut matmul_indices: HashMap<String, usize> = HashMap::new();
        let mut fused_nodes: Vec<crate::engine::graph::Node> = Vec::new();

        for n in non_identities {
            if n.op == "Add" && n.inputs.len() == 2 {
                let in0 = &n.inputs[0];
                let in1 = &n.inputs[1];
                if let Some(&conv_idx) = conv_indices.get(in0) {
                    if weights.contains_key(in1) && usage.get(in0) == Some(&1) {
                        fused_nodes[conv_idx].inputs.push(in1.clone());
                        fused_nodes[conv_idx].outputs[0] = n.outputs[0].clone();
                        continue;
                    }
                } else if let Some(&conv_idx) = conv_indices.get(in1) {
                    if weights.contains_key(in0) && usage.get(in1) == Some(&1) {
                        fused_nodes[conv_idx].inputs.push(in0.clone());
                        fused_nodes[conv_idx].outputs[0] = n.outputs[0].clone();
                        continue;
                    }
                } else if let Some(&ct_idx) = conv_transpose_indices.get(in0) {
                    if weights.contains_key(in1) && usage.get(in0) == Some(&1) {
                        fused_nodes[ct_idx].inputs.push(in1.clone());
                        fused_nodes[ct_idx].outputs[0] = n.outputs[0].clone();
                        continue;
                    }
                } else if let Some(&ct_idx) = conv_transpose_indices.get(in1) {
                    if weights.contains_key(in0) && usage.get(in1) == Some(&1) {
                        fused_nodes[ct_idx].inputs.push(in0.clone());
                        fused_nodes[ct_idx].outputs[0] = n.outputs[0].clone();
                        continue;
                    }
                } else if let Some(&mm_idx) = matmul_indices.get(in0) {
                    if weights.contains_key(in1) && usage.get(in0) == Some(&1) {
                        fused_nodes[mm_idx].inputs.push(in1.clone());
                        fused_nodes[mm_idx].outputs[0] = n.outputs[0].clone();
                        continue;
                    }
                } else if let Some(&mm_idx) = matmul_indices.get(in1)
                    && weights.contains_key(in0)
                    && usage.get(in1) == Some(&1)
                {
                    fused_nodes[mm_idx].inputs.push(in0.clone());
                    fused_nodes[mm_idx].outputs[0] = n.outputs[0].clone();
                    continue;
                }
            }

            if n.op == "Conv" && n.inputs.len() == 2 {
                let conv_idx = fused_nodes.len();
                conv_indices.insert(n.outputs[0].clone(), conv_idx);
            } else if n.op == "ConvTranspose" && n.inputs.len() == 2 {
                let ct_idx = fused_nodes.len();
                conv_transpose_indices.insert(n.outputs[0].clone(), ct_idx);
            } else if n.op == "MatMul" && n.inputs.len() == 2 {
                let mm_idx = fused_nodes.len();
                matmul_indices.insert(n.outputs[0].clone(), mm_idx);
            }

            fused_nodes.push(n);
        }

        // --- Graph Optimization Pass 3: GELU Subgraph Fusion ---
        // Fuses `Div(x, sqrt2) -> Erf -> Add(1.0) -> Mul(x, .) -> Mul(., 0.5)` into a
        // single `Gelu(x)`. A candidate is only accepted when the constant operands
        // really are sqrt(2), 1.0 and 0.5 *and* every intermediate tensor is consumed
        // exactly once; otherwise the sub-graph is left untouched instead of being
        // replaced by a node with different semantics. See `try_fuse_gelu`.
        //
        // Usage counts are recomputed here: Pass 2 rewired edges (it renamed
        // Conv/ConvTranspose/MatMul outputs when folding `Add(bias)`), so the counts
        // gathered before Pass 2 are no longer authoritative.
        let mut usage: HashMap<String, usize> = HashMap::new();
        for n in &fused_nodes {
            for inp in &n.inputs {
                *usage.entry(inp.clone()).or_insert(0) += 1;
            }
        }
        for out in &graph.outputs {
            *usage.entry(out.clone()).or_insert(0) += 1;
        }

        // --- Graph Optimization Pass 2.5: BatchNormalization folding into Conv ---
        // `BatchNormalization` applies a per-channel affine transform immediately after a
        // convolution, which composes into the Conv's own weight and bias (see
        // `fold_batch_norm_into_conv`). The cls model spends 27 of its 143 nodes on BN;
        // folding removes both the nodes and one full pass over the feature maps each.
        let mut bn_folded_nodes: Vec<Node> = Vec::with_capacity(fused_nodes.len());
        let mut produced_at: HashMap<String, usize> = HashMap::new();
        let mut bn_folded_count = 0usize;

        for n in &fused_nodes {
            if n.op == "BatchNormalization" && n.inputs.len() >= 5 {
                let x_name = n.inputs[0].clone();
                if usage.get(&x_name) == Some(&1)
                    && let Some(&conv_idx) = produced_at.get(&x_name)
                    && bn_folded_nodes[conv_idx].op == "Conv"
                {
                    let conv_snapshot = bn_folded_nodes[conv_idx].clone();
                    if let Some((w_key, b_key)) =
                        fold_batch_norm_into_conv(&conv_snapshot, n, &mut weights)
                    {
                        let conv = &mut bn_folded_nodes[conv_idx];
                        // Keep only the activation input `x`, then install the folded
                        // weight and bias. Truncating to 2 would *keep the original weight*
                        // as `inputs[1]` and leave the folded weight sitting where the
                        // engine reads the bias, turning the node into
                        // `conv(x, w_orig) + w_folded`.
                        conv.inputs.truncate(1);
                        conv.inputs.push(w_key);
                        conv.inputs.push(b_key);
                        conv.outputs[0] = n.outputs[0].clone();
                        bn_folded_count += 1;
                        continue;
                    }
                }
            }

            for o in &n.outputs {
                produced_at.insert(o.clone(), bn_folded_nodes.len());
            }
            bn_folded_nodes.push(n.clone());
        }

        if bn_folded_count > 0 {
            // The BN nodes are gone, so the reference counts gathered above are stale.
            usage.clear();
            for n in &bn_folded_nodes {
                for inp in &n.inputs {
                    *usage.entry(inp.clone()).or_insert(0) += 1;
                }
            }
            for out in &graph.outputs {
                *usage.entry(out.clone()).or_insert(0) += 1;
            }
        }

        let mut gelu_fused_nodes: Vec<Node> = Vec::with_capacity(bn_folded_nodes.len());
        let mut i = 0;
        while i < bn_folded_nodes.len() {
            if let Some(node) = try_fuse_gelu(&bn_folded_nodes[i..], i, &weights, &usage) {
                gelu_fused_nodes.push(node);
                i += 5;
            } else {
                gelu_fused_nodes.push(bn_folded_nodes[i].clone());
                i += 1;
            }
        }

        // --- Graph Optimization Pass 3.5: LayerNorm Subgraph Fusion ---
        // Fuses `ReduceMean -> Sub -> Pow(2.0) -> ReduceMean -> Add(eps) -> Sqrt
        //        -> Div -> Mul(w) -> Add(b)` into a single `LayerNorm` node.
        // Validated the same way as the GELU pass (constant symbols + usage counts).
        // See `try_fuse_layernorm`.
        let mut ln_fused_nodes: Vec<Node> = Vec::with_capacity(gelu_fused_nodes.len());
        i = 0;
        while i < gelu_fused_nodes.len() {
            if let Some(node) = try_fuse_layernorm(&gelu_fused_nodes[i..], i, &weights, &usage) {
                ln_fused_nodes.push(node);
                i += 9;
            } else {
                ln_fused_nodes.push(gelu_fused_nodes[i].clone());
                i += 1;
            }
        }

        // --- Graph Optimization Pass 3.7: HardSwish fusion ---
        // Folds `Mul(x, HardSigmoid(x))` with alpha = 1/6 and beta = 0.5 into a single
        // `HardSwish(x)` node. Runs before Pass 4 so the Conv+activation pass can then
        // absorb the HardSwish into the convolution that produces `x`.
        let mut hs_fused_nodes: Vec<Node> = Vec::with_capacity(ln_fused_nodes.len());
        i = 0;
        while i < ln_fused_nodes.len() {
            if i + 1 < ln_fused_nodes.len()
                && let Some(node) =
                    try_fuse_hardswish(&ln_fused_nodes[i], &ln_fused_nodes[i + 1], i, &usage)
            {
                hs_fused_nodes.push(node);
                i += 2;
                continue;
            }
            hs_fused_nodes.push(ln_fused_nodes[i].clone());
            i += 1;
        }

        // --- Graph Optimization Pass 4: Conv + Activation (Relu / HardSigmoid / HardSwish / Gelu) Fusion ---
        let mut usage_p4: HashMap<String, usize> = HashMap::new();
        for n in &hs_fused_nodes {
            for inp in &n.inputs {
                *usage_p4.entry(inp.clone()).or_insert(0) += 1;
            }
        }
        for out in &graph.outputs {
            *usage_p4.entry(out.clone()).or_insert(0) += 1;
        }

        let mut conv_indices: HashMap<String, usize> = HashMap::new();
        let mut conv_transpose_indices: HashMap<String, usize> = HashMap::new();
        let mut act_fused_nodes: Vec<crate::engine::graph::Node> = Vec::new();

        for n in hs_fused_nodes {
            if (n.op == "Relu" || n.op == "HardSigmoid" || n.op == "HardSwish" || n.op == "Gelu")
                && n.inputs.len() == 1
            {
                let in0 = &n.inputs[0];
                if let Some(&conv_idx) = conv_indices.get(in0) {
                    if usage_p4.get(in0) == Some(&1) {
                        let conv_node = &mut act_fused_nodes[conv_idx];
                        conv_node.attrs.insert(
                            "activation".to_string(),
                            crate::engine::graph::AttributeValue::String(n.op.clone()),
                        );
                        if n.op == "HardSigmoid" {
                            if let Some(alpha) = n.attrs.get("alpha") {
                                conv_node.attrs.insert("alpha".to_string(), alpha.clone());
                            }
                            if let Some(beta) = n.attrs.get("beta") {
                                conv_node.attrs.insert("beta".to_string(), beta.clone());
                            }
                        }
                        conv_node.outputs[0] = n.outputs[0].clone();
                        continue;
                    }
                } else if let Some(&ct_idx) = conv_transpose_indices.get(in0)
                    && usage_p4.get(in0) == Some(&1)
                {
                    let ct_node = &mut act_fused_nodes[ct_idx];
                    ct_node.attrs.insert(
                        "activation".to_string(),
                        crate::engine::graph::AttributeValue::String(n.op.clone()),
                    );
                    if n.op == "HardSigmoid" {
                        if let Some(alpha) = n.attrs.get("alpha") {
                            ct_node.attrs.insert("alpha".to_string(), alpha.clone());
                        }
                        if let Some(beta) = n.attrs.get("beta") {
                            ct_node.attrs.insert("beta".to_string(), beta.clone());
                        }
                    }
                    ct_node.outputs[0] = n.outputs[0].clone();
                    continue;
                }
            }

            if n.op == "Conv" {
                let conv_idx = act_fused_nodes.len();
                conv_indices.insert(n.outputs[0].clone(), conv_idx);
            } else if n.op == "ConvTranspose" {
                let ct_idx = act_fused_nodes.len();
                conv_transpose_indices.insert(n.outputs[0].clone(), ct_idx);
            }

            act_fused_nodes.push(n);
        }

        // --- Graph Optimization Pass 5: Fold Transpose([..., -1, -2]) into MatMul(A, B^T) ---
        let mut usage: HashMap<String, usize> = HashMap::new();
        for n in &act_fused_nodes {
            for inp in &n.inputs {
                *usage.entry(inp.clone()).or_insert(0) += 1;
            }
        }
        for out in &graph.outputs {
            *usage.entry(out.clone()).or_insert(0) += 1;
        }

        let is_last_two_swap = |perm: &[usize]| -> bool {
            let d = perm.len();
            d >= 2
                && perm[d - 2] == d - 1
                && perm[d - 1] == d - 2
                && (0..d - 2).all(|i| perm[i] == i)
        };

        let mut transposes_to_fold: HashMap<String, String> = HashMap::new();
        for n in &act_fused_nodes {
            if n.op == "Transpose" && n.inputs.len() == 1 && n.outputs.len() == 1 {
                let out_name = &n.outputs[0];
                if usage.get(out_name) == Some(&1) {
                    if let Some(perm) = n.attrs.get("perm").and_then(|a| a.as_usize_vec()) {
                        if is_last_two_swap(&perm) {
                            transposes_to_fold.insert(out_name.clone(), n.inputs[0].clone());
                        }
                    }
                }
            }
        }

        let mut folded_transposes: HashSet<String> = HashSet::new();
        for n in &act_fused_nodes {
            if n.op == "MatMul" && n.inputs.len() >= 2 {
                let in1 = &n.inputs[1];
                if transposes_to_fold.contains_key(in1) {
                    folded_transposes.insert(in1.clone());
                }
            }
        }

        let mut final_nodes: Vec<Node> = Vec::with_capacity(act_fused_nodes.len());
        for mut n in act_fused_nodes {
            if n.op == "Transpose"
                && n.outputs.len() == 1
                && folded_transposes.contains(&n.outputs[0])
            {
                // Drop this Transpose; it is folded into the following MatMul
                continue;
            }
            if n.op == "MatMul" && n.inputs.len() >= 2 && folded_transposes.contains(&n.inputs[1]) {
                let orig_in1 = n.inputs[1].clone();
                let actual_src = transposes_to_fold.get(&orig_in1).unwrap().clone();
                n.inputs[1] = actual_src;
                n.attrs.insert(
                    "transposed_b".to_string(),
                    crate::engine::graph::AttributeValue::Int(1),
                );
            }
            final_nodes.push(n);
        }

        // --- Graph Optimization Pass 6: Fold Concat(axis=1) into Conv2d ---
        let mut usage_p6: HashMap<String, usize> = HashMap::new();
        for n in &final_nodes {
            for inp in &n.inputs {
                *usage_p6.entry(inp.clone()).or_insert(0) += 1;
            }
        }
        for out in &graph.outputs {
            *usage_p6.entry(out.clone()).or_insert(0) += 1;
        }

        let mut concats_to_fold: HashMap<String, Vec<String>> = HashMap::new();
        for n in &final_nodes {
            if n.op == "Concat" && n.inputs.len() >= 2 && n.outputs.len() == 1 {
                let axis = n.attrs.get("axis").and_then(|a| a.as_i64()).unwrap_or(1);
                if axis == 1 && usage_p6.get(&n.outputs[0]) == Some(&1) {
                    concats_to_fold.insert(n.outputs[0].clone(), n.inputs.clone());
                }
            }
        }

        let mut folded_concats: HashSet<String> = HashSet::new();
        for n in &final_nodes {
            if n.op == "Conv" && n.inputs.len() >= 2 {
                let in0 = &n.inputs[0];
                if concats_to_fold.contains_key(in0) {
                    let group = group_attr(n).unwrap_or(1);
                    if group == 1 {
                        let w_name = &n.inputs[1];
                        let is_1x1 = if let Some(w) = weights.get(w_name) {
                            w.ndim() == 4 && w.shape()[2] == 1 && w.shape()[3] == 1
                        } else {
                            false
                        };
                        if !is_1x1 {
                            folded_concats.insert(in0.clone());
                        }
                    }
                }
            }
        }

        let mut p6_nodes: Vec<Node> = Vec::with_capacity(final_nodes.len());
        for mut n in final_nodes {
            if n.op == "Concat" && n.outputs.len() == 1 && folded_concats.contains(&n.outputs[0]) {
                // Drop this Concat; its inputs are read directly by the following Conv
                continue;
            }
            if n.op == "Conv" && n.inputs.len() >= 2 && folded_concats.contains(&n.inputs[0]) {
                let concat_out = &n.inputs[0];
                let concat_inps = concats_to_fold.get(concat_out).unwrap().clone();
                n.attrs.insert(
                    "concat_inputs".to_string(),
                    crate::engine::graph::AttributeValue::Strings(concat_inps),
                );
            }
            p6_nodes.push(n);
        }

        graph.nodes = p6_nodes;

        // Precompute usage counts for eagerly pruned intermediate activations
        let mut usage_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph.nodes {
            if let Some(concat_inps) = node.attrs.get("concat_inputs").and_then(|a| a.as_strings())
            {
                for inp in concat_inps {
                    if !weights.contains_key(inp) {
                        *usage_counts.entry(inp.clone()).or_insert(0) += 1;
                    }
                }
                for inp in &node.inputs[1..] {
                    if !weights.contains_key(inp) {
                        *usage_counts.entry(inp.clone()).or_insert(0) += 1;
                    }
                }
            } else {
                for inp in &node.inputs {
                    if !weights.contains_key(inp) {
                        *usage_counts.entry(inp.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        for out in &graph.outputs {
            *usage_counts.entry(out.clone()).or_insert(0) += 1;
        }

        // --- Weight Pre-packing: Winograd F(2x2, 3x3) Pretransformation ---
        let mut winograd_weights: HashMap<String, Vec<Tensor>> = HashMap::new();
        for node in &graph.nodes {
            if node.op == "Conv" && node.inputs.len() >= 2 {
                let w_name = &node.inputs[1];
                if let Some(w) = weights.get(w_name) {
                    let shape = w.shape();
                    // Kept as `i64` and compared against 1 without a `usize` cast: a
                    // negative `group` would become `usize::MAX` and, while that still
                    // fails the `group == 1` test below, relying on the wrap for control
                    // flow is how the bug got there in the first place. The operator itself
                    // rejects a bad `group` with a proper error.
                    let group = node
                        .attrs
                        .get("group")
                        .and_then(|a| a.as_i64())
                        .unwrap_or(1);
                    let strides = node
                        .attrs
                        .get("strides")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let pads = node
                        .attrs
                        .get("pads")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![0, 0, 0, 0]);
                    let dilations = node
                        .attrs
                        .get("dilations")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);

                    let is_3x3_stride1 = shape.len() == 4
                        && shape[2] == 3
                        && shape[3] == 3
                        && group == 1
                        && shape[1] >= 16
                        && strides == vec![1, 1]
                        && pads == vec![1, 1, 1, 1]
                        && dilations == vec![1, 1];

                    if is_3x3_stride1 && let Ok(u) = ops::pretransform_weights_f23(w) {
                        winograd_weights.insert(w_name.clone(), u);
                    }
                }
            }
        }

        // --- Weight Pre-packing: MatMul Weight Transpose for Large Vocabulary Projection ---
        let mut transposed_matmul_weights: HashMap<String, Tensor> = HashMap::new();
        // Names whose FP32 original can be dropped once packing is done.
        let mut releasable_weights: Vec<String> = Vec::new();
        // Exact reference counts over the final graph: a weight may only be released when
        // the packed node is its sole reader.
        let mut ref_counts: HashMap<&str, usize> = HashMap::new();
        for node in &graph.nodes {
            for inp in &node.inputs {
                *ref_counts.entry(inp.as_str()).or_insert(0) += 1;
            }
        }
        for out in &graph.outputs {
            *ref_counts.entry(out.as_str()).or_insert(0) += 1;
        }

        for node in &graph.nodes {
            if node.op == "MatMul" && node.inputs.len() >= 2 {
                let w_name = &node.inputs[1];
                if let Some(w) = weights.get(w_name) {
                    let shape = w.shape();
                    if shape.len() == 2
                        && shape[1] >= 1024
                        && let Ok(w_t) = w.transpose(0, 1)
                    {
                        transposed_matmul_weights.insert(w_name.clone(), w_t.to_contiguous());
                        // `MatMul` prefers `transposed_matmul_weights` and never falls back to
                        // `self.weights` for this name, so keeping the original FP32 copy
                        // resident doubled the footprint of the vocabulary projection
                        // (8.98 MB duplicated for the 18710-class rec head).
                        if ref_counts.get(w_name.as_str()) == Some(&1) {
                            releasable_weights.push(w_name.clone());
                        }
                    }
                }
            }
        }

        for name in releasable_weights {
            weights.remove(&name);
        }

        let quantized_matmul_weights = HashMap::new();

        Ok(Self {
            graph,
            weights,
            usage_counts,
            winograd_weights,
            transposed_matmul_weights,
            quantized_matmul_weights,
        })
    }

    /// Quantize large projection MatMul weights to INT8 (W8A32) with AVX2 acceleration.
    /// This reduces weight memory footprint by 4x and fits into CPU L3 cache.
    ///
    /// The FP32 copies are released as we go. `MatMul` looks up
    /// `quantized_matmul_weights` before `transposed_matmul_weights` before `self.weights`,
    /// so once a name is quantized neither of the FP32 tensors is reachable any more.
    /// Without this the INT8 weights were *added* to the existing footprint rather than
    /// replacing it (the 8.98 MB rec head went from 2 x 8.98 MB to 20.21 MB instead of
    /// dropping to 2.25 MB).
    pub fn quantize_matmul_weights(&mut self) {
        let names: Vec<String> = self.transposed_matmul_weights.keys().cloned().collect();
        for name in names {
            let qw =
                ops::QuantizedWeight::from_f32_transposed(&self.transposed_matmul_weights[&name]);
            self.quantized_matmul_weights.insert(name.clone(), qw);
            // Both FP32 forms are now shadowed by the quantized entry.
            self.transposed_matmul_weights.remove(&name);
            self.weights.remove(&name);
        }
    }

    /// Whether any MatMul weights are quantized in this model.
    pub fn is_matmul_quantized(&self) -> bool {
        !self.quantized_matmul_weights.is_empty()
    }

    /// Total bytes held by all resident weight tensors: FP32 originals,
    /// Winograd-pretransformed kernels, transposed MatMul copies and INT8-quantized head
    /// weights.
    ///
    /// Exposed so tests and callers can verify the memory numbers quoted in the README
    /// instead of taking them on faith.
    pub fn weight_footprint_bytes(&self) -> usize {
        let f32_bytes = |t: &Tensor| t.numel() * std::mem::size_of::<f32>();
        let original: usize = self.weights.values().map(f32_bytes).sum();
        let winograd: usize = self
            .winograd_weights
            .values()
            .flat_map(|v| v.iter())
            .map(f32_bytes)
            .sum();
        let transposed: usize = self.transposed_matmul_weights.values().map(f32_bytes).sum();
        let quantized: usize = self
            .quantized_matmul_weights
            .values()
            .map(|q| q.data.len() + q.scales.len() * std::mem::size_of::<f32>())
            .sum();
        original + winograd + transposed + quantized
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    pub fn forward(&self, input_name: &str, input_tensor: Tensor) -> Result<Tensor> {
        let (out, _) = self.forward_internal(input_name, input_tensor, false)?;
        Ok(out)
    }

    pub fn forward_with_profile(
        &self,
        input_name: &str,
        input_tensor: Tensor,
    ) -> Result<(Tensor, HashMap<String, f64>)> {
        self.forward_internal(input_name, input_tensor, true)
    }

    fn forward_internal(
        &self,
        input_name: &str,
        input_tensor: Tensor,
        profile: bool,
    ) -> Result<(Tensor, HashMap<String, f64>)> {
        let mut env: HashMap<&str, Tensor> = HashMap::with_capacity(64);
        env.insert(input_name, input_tensor);
        let mut usage_count: HashMap<&str, usize> = self
            .usage_counts
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        let mut profile_data: HashMap<String, f64> = HashMap::new();

        for node in &self.graph.nodes {
            let t_start = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let out_name = &node.outputs[0];

            let get_tensor = |name: &str| -> Result<&Tensor> {
                if let Some(t) = env.get(name) {
                    Ok(t)
                } else if let Some(t) = self.weights.get(name) {
                    Ok(t)
                } else {
                    Err(OcrError::OperatorError {
                        op: node.op.clone(),
                        msg: format!("Input tensor '{}' not found for node '{}'", name, node.name),
                    })
                }
            };

            let out_tensor = match node.op.as_str() {
                "Conv" => {
                    let concat_inps_opt =
                        node.attrs.get("concat_inputs").and_then(|a| a.as_strings());
                    let (x_ref, inps_holder) = if let Some(concat_inps) = concat_inps_opt {
                        let mut inps = Vec::with_capacity(concat_inps.len());
                        for name in concat_inps {
                            inps.push(get_tensor(name)?);
                        }
                        let first = inps[0];
                        (first, Some(inps))
                    } else {
                        (get_tensor(&node.inputs[0])?, None)
                    };
                    let w = get_tensor(&node.inputs[1])?;
                    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                        Some(get_tensor(&node.inputs[2])?)
                    } else {
                        None
                    };

                    let strides = node
                        .attrs
                        .get("strides")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let dilations = node
                        .attrs
                        .get("dilations")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let group = group_attr(node)?;

                    let pads = if let Some(auto_pad) =
                        node.attrs.get("auto_pad").and_then(|a| a.as_str())
                    {
                        if auto_pad == "SAME_UPPER" {
                            compute_same_upper_pads(
                                x_ref.shape()[2],
                                x_ref.shape()[3],
                                w.shape()[2],
                                w.shape()[3],
                                strides[0],
                                strides[1],
                                dilations[0],
                                dilations[1],
                            )
                        } else {
                            [0, 0, 0, 0]
                        }
                    } else {
                        let p = node
                            .attrs
                            .get("pads")
                            .and_then(|a| a.as_usize_vec())
                            .unwrap_or_else(|| vec![0, 0, 0, 0]);
                        [p[0], p[1], p[2], p[3]]
                    };

                    let activation = if let Some(act_str) =
                        node.attrs.get("activation").and_then(|a| a.as_str())
                    {
                        match act_str {
                            "Relu" => ops::ConvActivation::Relu,
                            "HardSigmoid" => {
                                let alpha = node
                                    .attrs
                                    .get("alpha")
                                    .and_then(|a| a.as_f32())
                                    .unwrap_or(0.2);
                                let beta = node
                                    .attrs
                                    .get("beta")
                                    .and_then(|a| a.as_f32())
                                    .unwrap_or(0.5);
                                ops::ConvActivation::HardSigmoid { alpha, beta }
                            }
                            "HardSwish" => ops::ConvActivation::HardSwish,
                            "Gelu" => ops::ConvActivation::Gelu,
                            _ => ops::ConvActivation::None,
                        }
                    } else {
                        ops::ConvActivation::None
                    };

                    let w_name = &node.inputs[1];
                    let params = Conv2dParams {
                        strides: [strides[0], strides[1]],
                        pads,
                        dilations: [dilations[0], dilations[1]],
                        group,
                        activation,
                    };

                    if let Some(ref inps) = inps_holder {
                        ops::conv2d_channel_sliced(inps.as_slice(), w, bias, &params)?
                    } else if let Some(u_weights) = self.winograd_weights.get(w_name) {
                        ops::winograd_f23_conv2d(x_ref, u_weights, bias, activation)?
                    } else {
                        ops::conv2d(x_ref, w, bias, &params)?
                    }
                }

                "ConvTranspose" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let w = get_tensor(&node.inputs[1])?;
                    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                        Some(get_tensor(&node.inputs[2])?)
                    } else {
                        None
                    };

                    let strides = node
                        .attrs
                        .get("strides")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let pads = node
                        .attrs
                        .get("pads")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![0, 0, 0, 0]);
                    let dilations = node
                        .attrs
                        .get("dilations")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let group = group_attr(node)?;

                    let activation = if let Some(act_str) =
                        node.attrs.get("activation").and_then(|a| a.as_str())
                    {
                        match act_str {
                            "Relu" => ops::ConvActivation::Relu,
                            "HardSigmoid" => {
                                let alpha = node
                                    .attrs
                                    .get("alpha")
                                    .and_then(|a| a.as_f32())
                                    .unwrap_or(0.2);
                                let beta = node
                                    .attrs
                                    .get("beta")
                                    .and_then(|a| a.as_f32())
                                    .unwrap_or(0.5);
                                ops::ConvActivation::HardSigmoid { alpha, beta }
                            }
                            "HardSwish" => ops::ConvActivation::HardSwish,
                            "Gelu" => ops::ConvActivation::Gelu,
                            _ => ops::ConvActivation::None,
                        }
                    } else {
                        ops::ConvActivation::None
                    };

                    let params = ConvTranspose2dParams {
                        strides: [strides[0], strides[1]],
                        pads: [pads[0], pads[1], pads[2], pads[3]],
                        dilations: [dilations[0], dilations[1]],
                        group,
                        activation,
                    };

                    ops::conv_transpose2d(x, w, bias, &params)?
                }

                "BatchNormalization" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let scale = get_tensor(&node.inputs[1])?;
                    let bias = get_tensor(&node.inputs[2])?;
                    let mean = get_tensor(&node.inputs[3])?;
                    let var = get_tensor(&node.inputs[4])?;
                    let epsilon = node
                        .attrs
                        .get("epsilon")
                        .and_then(|a| a.as_f32())
                        .unwrap_or(1e-5);

                    ops::batch_norm2d(x, scale, bias, mean, var, epsilon)?
                }

                "Relu" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::relu_inplace(&mut x);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::relu(x)
                    }
                }

                "Clip" => {
                    let min_val = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                        let t = get_tensor(&node.inputs[1])?;
                        t.as_slice()[0]
                    } else {
                        node.attrs
                            .get("min")
                            .and_then(|a| a.as_f32())
                            .unwrap_or(f32::NEG_INFINITY)
                    };

                    let max_val = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                        let t = get_tensor(&node.inputs[2])?;
                        t.as_slice()[0]
                    } else {
                        node.attrs
                            .get("max")
                            .and_then(|a| a.as_f32())
                            .unwrap_or(f32::INFINITY)
                    };

                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::clip_inplace(&mut x, min_val, max_val);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::clip(x, min_val, max_val)
                    }
                }

                "Sigmoid" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::sigmoid_inplace(&mut x);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::sigmoid(x)
                    }
                }

                "HardSigmoid" => {
                    let alpha = node
                        .attrs
                        .get("alpha")
                        .and_then(|a| a.as_f32())
                        .unwrap_or(0.2);
                    let beta = node
                        .attrs
                        .get("beta")
                        .and_then(|a| a.as_f32())
                        .unwrap_or(0.5);

                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::hardsigmoid_inplace(&mut x, alpha, beta);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::hardsigmoid(x, alpha, beta)
                    }
                }

                "Add" => {
                    let in0 = node.inputs[0].as_str();
                    let in1 = node.inputs[1].as_str();

                    if usage_count.get(in0) == Some(&1) && env.contains_key(in0) {
                        let mut a = env.remove(in0).unwrap();
                        usage_count.remove(in0);
                        let b = if let Some(t) = env.get(in1) {
                            t
                        } else if let Some(t) = self.weights.get(in1) {
                            t
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Add".into(),
                                msg: format!("Input '{}' not found", in1),
                            });
                        };
                        if ops::add_inplace(&mut a, b).is_ok() {
                            a
                        } else {
                            ops::add(&a, b)?
                        }
                    } else if usage_count.get(in1) == Some(&1) && env.contains_key(in1) {
                        let mut b = env.remove(in1).unwrap();
                        usage_count.remove(in1);
                        let a = if let Some(t) = env.get(in0) {
                            t
                        } else if let Some(t) = self.weights.get(in0) {
                            t
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Add".into(),
                                msg: format!("Input '{}' not found", in0),
                            });
                        };
                        if ops::add_inplace(&mut b, a).is_ok() {
                            b
                        } else {
                            ops::add(a, &b)?
                        }
                    } else {
                        let a = get_tensor(in0)?;
                        let b = get_tensor(in1)?;
                        ops::add(a, b)?
                    }
                }

                "Sub" => {
                    let in0 = node.inputs[0].as_str();
                    let in1 = node.inputs[1].as_str();
                    if usage_count.get(in0) == Some(&1) && env.contains_key(in0) {
                        let mut a = env.remove(in0).unwrap();
                        usage_count.remove(in0);
                        let b = if let Some(t) = env.get(in1) {
                            t
                        } else if let Some(t) = self.weights.get(in1) {
                            t
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Sub".into(),
                                msg: format!("Input '{}' not found", in1),
                            });
                        };
                        if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
                            ops::sub_inplace(&mut a, b)?;
                            a
                        } else {
                            ops::sub(&a, b)?
                        }
                    } else {
                        let a = get_tensor(in0)?;
                        let b = get_tensor(in1)?;
                        ops::sub(a, b)?
                    }
                }

                "Mul" => {
                    let in0 = node.inputs[0].as_str();
                    let in1 = node.inputs[1].as_str();
                    if usage_count.get(in0) == Some(&1) && env.contains_key(in0) {
                        let mut a = env.remove(in0).unwrap();
                        usage_count.remove(in0);
                        let b = if let Some(t) = env.get(in1) {
                            t
                        } else if let Some(t) = self.weights.get(in1) {
                            t
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Mul".into(),
                                msg: format!("Input '{}' not found", in1),
                            });
                        };
                        if ops::mul_inplace(&mut a, b).is_ok() {
                            a
                        } else {
                            ops::mul(&a, b)?
                        }
                    } else if usage_count.get(in1) == Some(&1) && env.contains_key(in1) {
                        let mut b = env.remove(in1).unwrap();
                        usage_count.remove(in1);
                        let a = if let Some(t) = env.get(in0) {
                            t
                        } else if let Some(t) = self.weights.get(in0) {
                            t
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Mul".into(),
                                msg: format!("Input '{}' not found", in0),
                            });
                        };
                        if ops::mul_inplace(&mut b, a).is_ok() {
                            b
                        } else {
                            ops::mul(a, &b)?
                        }
                    } else {
                        let a = get_tensor(in0)?;
                        let b = get_tensor(in1)?;
                        ops::mul(a, b)?
                    }
                }

                "Div" => {
                    let in0 = node.inputs[0].as_str();
                    let in1 = node.inputs[1].as_str();
                    if usage_count.get(in0) == Some(&1) && env.contains_key(in0) {
                        let mut a = env.remove(in0).unwrap();
                        usage_count.remove(in0);
                        let b = if let Some(t) = env.get(in1) {
                            t
                        } else if let Some(t) = self.weights.get(in1) {
                            t
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Div".into(),
                                msg: format!("Input '{}' not found", in1),
                            });
                        };
                        if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
                            ops::div_inplace(&mut a, b)?;
                            a
                        } else {
                            ops::div(&a, b)?
                        }
                    } else {
                        let a = get_tensor(in0)?;
                        let b = get_tensor(in1)?;
                        ops::div(a, b)?
                    }
                }

                "Pow" => {
                    let a = get_tensor(&node.inputs[0])?;
                    let b = get_tensor(&node.inputs[1])?;
                    ops::pow(a, b)?
                }

                "Sqrt" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::sqrt_inplace(&mut x);
                        x
                    } else {
                        let a = get_tensor(in_name)?;
                        ops::sqrt(a)
                    }
                }

                "GlobalAveragePool" => {
                    let x = get_tensor(&node.inputs[0])?;
                    ops::global_avg_pool2d(x)?
                }

                "AveragePool" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let kernel_shape = node
                        .attrs
                        .get("kernel_shape")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![2, 2]);
                    let strides = node
                        .attrs
                        .get("strides")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let pads = node
                        .attrs
                        .get("pads")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![0, 0, 0, 0]);
                    let count_include_pad = node
                        .attrs
                        .get("count_include_pad")
                        .and_then(|a| a.as_i64())
                        .unwrap_or(0)
                        != 0;

                    let params = Pool2dParams {
                        kernel_shape: [kernel_shape[0], kernel_shape[1]],
                        strides: [strides[0], strides[1]],
                        pads: [pads[0], pads[1], pads[2], pads[3]],
                    };

                    ops::avg_pool2d(x, &params, count_include_pad)?
                }

                "MaxPool" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let kernel_shape = node
                        .attrs
                        .get("kernel_shape")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![2, 2]);
                    let strides = node
                        .attrs
                        .get("strides")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_else(|| vec![1, 1]);
                    let pads = if let Some(auto_pad) =
                        node.attrs.get("auto_pad").and_then(|a| a.as_str())
                    {
                        if auto_pad == "SAME_UPPER" {
                            compute_same_upper_pads(
                                x.shape()[2],
                                x.shape()[3],
                                kernel_shape[0],
                                kernel_shape[1],
                                strides[0],
                                strides[1],
                                1,
                                1,
                            )
                        } else {
                            [0, 0, 0, 0]
                        }
                    } else {
                        let p = node
                            .attrs
                            .get("pads")
                            .and_then(|a| a.as_usize_vec())
                            .unwrap_or_else(|| vec![0, 0, 0, 0]);
                        [p[0], p[1], p[2], p[3]]
                    };

                    let params = Pool2dParams {
                        kernel_shape: [kernel_shape[0], kernel_shape[1]],
                        strides: [strides[0], strides[1]],
                        pads,
                    };

                    ops::max_pool2d(x, &params)?
                }

                "MatMul" => {
                    let a = get_tensor(&node.inputs[0])?;
                    let w_name = &node.inputs[1];
                    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                        Some(get_tensor(&node.inputs[2])?)
                    } else {
                        None
                    };

                    let transposed_b = node
                        .attrs
                        .get("transposed_b")
                        .and_then(|a| a.as_i64())
                        .unwrap_or(0)
                        != 0;

                    if let Some(qw) = self.quantized_matmul_weights.get(w_name) {
                        ops::matmul_quantized_b(a, qw, bias)?
                    } else if let Some(w_t) = self.transposed_matmul_weights.get(w_name) {
                        ops::matmul_transposed_b(a, w_t, bias)?
                    } else if transposed_b {
                        let b = get_tensor(w_name)?;
                        ops::matmul_with_transposed_b(a, b, bias)?
                    } else {
                        let b = get_tensor(w_name)?;
                        ops::matmul_with_bias(a, b, bias)?
                    }
                }

                "LayerNorm" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let w = get_tensor(&node.inputs[1])?;
                    let b = get_tensor(&node.inputs[2])?;
                    let eps = node
                        .attrs
                        .get("eps")
                        .and_then(|a| a.as_f32())
                        .unwrap_or(1e-5);
                    ops::layer_norm(x, w.shape(), Some(w), Some(b), eps)?
                }

                "Softmax" => {
                    let in_name = node.inputs[0].as_str();
                    let axis = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.as_i64())
                        .unwrap_or(-1) as isize;
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::softmax_inplace(&mut x, axis)?;
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::softmax(x, axis)?
                    }
                }

                "ReduceMean" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                        let t = get_tensor(&node.inputs[1])?;
                        t.to_i64_vec().into_iter().map(|v| v as isize).collect()
                    } else if let Some(a) = node.attrs.get("axes").and_then(|a| a.as_i64_vec()) {
                        a.into_iter().map(|v| v as isize).collect()
                    } else {
                        vec![-1]
                    };

                    let keepdims = node
                        .attrs
                        .get("keepdims")
                        .and_then(|a| a.as_i64())
                        .unwrap_or(1)
                        != 0;

                    ops::reduce_mean(x, &axes, keepdims)?
                }

                "Transpose" => {
                    let in_name = node.inputs[0].as_str();
                    let perm = node
                        .attrs
                        .get("perm")
                        .and_then(|a| a.as_usize_vec())
                        .ok_or_else(|| OcrError::OperatorError {
                            op: "Transpose".into(),
                            msg: "Missing 'perm' attribute".into(),
                        })?;
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        x.into_permute(&perm)?
                    } else {
                        let x = get_tensor(in_name)?;
                        x.permute(&perm)?
                    }
                }

                "Reshape" => {
                    let in_name = node.inputs[0].as_str();
                    let shape_tensor = get_tensor(&node.inputs[1])?;
                    let target_shape = shape_tensor.to_i64_vec();

                    let (in_ndim, in_numel, in_shape) = {
                        let x = get_tensor(in_name)?;
                        (x.ndim(), x.numel(), x.shape().to_vec())
                    };

                    let mut resolved_shape = Vec::with_capacity(target_shape.len());
                    let mut infer_idx = None;
                    // Use a checked product: a malformed model with a 0/huge dimension used
                    // to cause a silent `0` and a division by zero panic below.
                    let mut known_prod = 1usize;

                    for (i, &s) in target_shape.iter().enumerate() {
                        if s == -1 {
                            if infer_idx.is_some() {
                                return Err(OcrError::OperatorError {
                                    op: "Reshape".into(),
                                    msg: format!(
                                        "at most one -1 dimension is allowed, got {:?}",
                                        target_shape
                                    ),
                                });
                            }
                            infer_idx = Some(i);
                            resolved_shape.push(0);
                        } else if s == 0 {
                            // ONNX semantics: copy the corresponding input dimension.
                            let orig_dim =
                                *in_shape.get(i).ok_or_else(|| OcrError::OperatorError {
                                    op: "Reshape".into(),
                                    msg: format!(
                                        "dimension 0 at index {} refers to input dim {} but the \
                                         input has rank {} (shape {:?})",
                                        i, i, in_ndim, in_shape
                                    ),
                                })?;
                            known_prod = known_prod.checked_mul(orig_dim).ok_or_else(|| {
                                OcrError::OperatorError {
                                    op: "Reshape".into(),
                                    msg: format!("shape product overflows: {:?}", target_shape),
                                }
                            })?;
                            resolved_shape.push(orig_dim);
                        } else if s > 0 {
                            let dim = s as usize;
                            known_prod = known_prod.checked_mul(dim).ok_or_else(|| {
                                OcrError::OperatorError {
                                    op: "Reshape".into(),
                                    msg: format!("shape product overflows: {:?}", target_shape),
                                }
                            })?;
                            resolved_shape.push(dim);
                        } else {
                            return Err(OcrError::OperatorError {
                                op: "Reshape".into(),
                                msg: format!("invalid dimension {} in {:?}", s, target_shape),
                            });
                        }
                    }

                    if let Some(idx) = infer_idx {
                        if known_prod == 0 || in_numel % known_prod != 0 {
                            return Err(OcrError::OperatorError {
                                op: "Reshape".into(),
                                msg: format!(
                                    "cannot infer the -1 dimension: {} elements are not divisible \
                                     by the known product {} of {:?}",
                                    in_numel, known_prod, target_shape
                                ),
                            });
                        }
                        resolved_shape[idx] = in_numel / known_prod;
                    }

                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        x.into_reshape(&resolved_shape)?
                    } else {
                        let x = get_tensor(in_name)?;
                        x.reshape(&resolved_shape)?
                    }
                }

                "Shape" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let shape_vec: Vec<i64> = x.shape().iter().map(|&s| s as i64).collect();
                    Tensor::from_i64_slice(&[shape_vec.len()], &shape_vec)?
                }

                "Slice" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let (starts, ends, axes, steps) = if node.inputs.len() >= 3 {
                        let starts = get_tensor(&node.inputs[1])?.to_i64_vec();
                        let ends = get_tensor(&node.inputs[2])?.to_i64_vec();
                        let axes = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                            Some(get_tensor(&node.inputs[3])?.to_i64_vec())
                        } else {
                            None
                        };
                        let steps = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
                            Some(get_tensor(&node.inputs[4])?.to_i64_vec())
                        } else {
                            None
                        };
                        (starts, ends, axes, steps)
                    } else {
                        let starts = node
                            .attrs
                            .get("starts")
                            .and_then(|a| a.as_i64_vec())
                            .unwrap_or_default();
                        let ends = node
                            .attrs
                            .get("ends")
                            .and_then(|a| a.as_i64_vec())
                            .unwrap_or_default();
                        let axes = node.attrs.get("axes").and_then(|a| a.as_i64_vec());
                        let steps = node.attrs.get("steps").and_then(|a| a.as_i64_vec());
                        (starts, ends, axes, steps)
                    };

                    let starts_isize: Vec<isize> = starts.into_iter().map(|v| v as isize).collect();
                    let ends_isize: Vec<isize> = ends.into_iter().map(|v| v as isize).collect();
                    let axes_isize: Option<Vec<isize>> =
                        axes.map(|v| v.into_iter().map(|x| x as isize).collect());
                    let steps_isize: Option<Vec<isize>> =
                        steps.map(|v| v.into_iter().map(|x| x as isize).collect());

                    ops::slice_op(
                        x,
                        &starts_isize,
                        &ends_isize,
                        axes_isize.as_deref(),
                        steps_isize.as_deref(),
                    )?
                }

                "Concat" => {
                    let mut input_tensors = Vec::new();
                    for inp in &node.inputs {
                        input_tensors.push(get_tensor(inp)?);
                    }
                    let axis =
                        node.attrs.get("axis").and_then(|a| a.as_i64()).unwrap_or(1) as isize;

                    ops::concat(&input_tensors, axis)?
                }

                "Squeeze" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                        Some(
                            get_tensor(&node.inputs[1])?
                                .as_slice()
                                .iter()
                                .map(|&v| v as usize)
                                .collect(),
                        )
                    } else {
                        node.attrs.get("axes").and_then(|a| a.as_usize_vec())
                    };

                    x.squeeze(axes.as_deref())?
                }

                "Resize" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let mode = node
                        .attrs
                        .get("mode")
                        .and_then(|a| a.as_str())
                        .unwrap_or("nearest");

                    let in_h = *x.shape().get(2).ok_or_else(|| OcrError::OperatorError {
                        op: "Resize".into(),
                        msg: format!("expected a 4D NCHW input, got shape {:?}", x.shape()),
                    })?;
                    let in_w = *x.shape().get(3).ok_or_else(|| OcrError::OperatorError {
                        op: "Resize".into(),
                        msg: format!("expected a 4D NCHW input, got shape {:?}", x.shape()),
                    })?;

                    // ONNX `Resize` provides `scales` (inputs[2]) *or* `sizes` (inputs[3]);
                    // the unused one is left empty. Output dims must end up >= 1, otherwise
                    // the resize below divides by zero.
                    let (target_h, target_w) =
                        if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                            let scales = get_tensor(&node.inputs[2])?;
                            let s = scales.as_slice();
                            if s.len() >= 4 {
                                (
                                    (in_h as f32 * s[2]).floor().max(1.0) as usize,
                                    (in_w as f32 * s[3]).floor().max(1.0) as usize,
                                )
                            } else {
                                (in_h * 2, in_w * 2)
                            }
                        } else if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                            let sizes = get_tensor(&node.inputs[3])?;
                            let s = sizes.as_slice();
                            // Entries may legitimately be -1 for axes that come from
                            // `scales`; the `as usize` cast saturates those to 0.
                            let th = s.get(2).copied().unwrap_or(0.0);
                            let tw = s.get(3).copied().unwrap_or(0.0);
                            if th < 1.0 || tw < 1.0 {
                                return Err(OcrError::OperatorError {
                                    op: "Resize".into(),
                                    msg: format!(
                                        "`sizes` must provide positive H/W, got [{}, {}]",
                                        th, tw
                                    ),
                                });
                            }
                            (th as usize, tw as usize)
                        } else {
                            (in_h * 2, in_w * 2)
                        };

                    if mode == "linear" || mode == "bilinear" {
                        ops::resize_bilinear_2d(x, target_h, target_w, false)?
                    } else {
                        ops::resize_nearest_2d(x, target_h, target_w)?
                    }
                }

                "Cast" | "Identity" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let t = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        t
                    } else {
                        let x = get_tensor(in_name)?;
                        x.clone()
                    }
                }

                "Erf" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::erf_inplace(&mut x);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::erf(x)
                    }
                }

                "Gelu" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::gelu_inplace(&mut x);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::gelu(x)
                    }
                }

                "HardSwish" => {
                    let in_name = node.inputs[0].as_str();
                    if usage_count.get(in_name) == Some(&1) && env.contains_key(in_name) {
                        let mut x = env.remove(in_name).unwrap();
                        usage_count.remove(in_name);
                        ops::hardswish_inplace(&mut x);
                        x
                    } else {
                        let x = get_tensor(in_name)?;
                        ops::hardswish(x)
                    }
                }

                "Unsqueeze" => {
                    let x = get_tensor(&node.inputs[0])?;
                    let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                        get_tensor(&node.inputs[1])?
                            .as_slice()
                            .iter()
                            .map(|&v| v as usize)
                            .collect()
                    } else if let Some(a) = node.attrs.get("axes").and_then(|a| a.as_usize_vec()) {
                        a
                    } else {
                        vec![0]
                    };

                    if x.numel() == 1 && axes == vec![0] && x.ndim() == 1 && x.shape() == [1] {
                        // Squeezed scalar already in 1D form [1]
                        x.clone()
                    } else {
                        let mut cur = x.clone();
                        for &axis in &axes {
                            cur = cur.unsqueeze(axis)?;
                        }
                        cur
                    }
                }

                other => {
                    return Err(OcrError::OperatorError {
                        op: other.to_string(),
                        msg: format!("Unsupported operator '{}' in node '{}'", other, node.name),
                    });
                }
            };

            if let Some(t0) = t_start {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let op_key = if node.op == "Conv" {
                    let w_name = &node.inputs[1];
                    let w = self.weights.get(w_name);
                    let shape = w.map(|t| t.shape()).unwrap_or(&[]);
                    let group = node
                        .attrs
                        .get("group")
                        .and_then(|a| a.as_i64())
                        .unwrap_or(1);
                    let in_shape = if let Some(concat_inps) =
                        node.attrs.get("concat_inputs").and_then(|a| a.as_strings())
                    {
                        if let Some(t0) = env.get(concat_inps[0].as_str()) {
                            let mut s = t0.shape().to_vec();
                            s[1] = concat_inps
                                .iter()
                                .filter_map(|n| env.get(n.as_str()))
                                .map(|t| t.shape()[1])
                                .sum();
                            s
                        } else {
                            vec![]
                        }
                    } else {
                        env.get(node.inputs[0].as_str())
                            .map(|t| t.shape().to_vec())
                            .unwrap_or_default()
                    };
                    if shape.len() >= 4 && shape[2] == 1 && shape[3] == 1 && group == 1 {
                        let act = node
                            .attrs
                            .get("activation")
                            .and_then(|a| a.as_str())
                            .unwrap_or("None");
                        format!(
                            "Conv1x1: {} ({}x{}) act={} in={:?}",
                            node.name, shape[0], shape[1], act, in_shape
                        )
                    } else if shape.len() >= 4 && group > 1 && shape[1] == 1 {
                        let strides = node
                            .attrs
                            .get("strides")
                            .and_then(|a| a.as_usize_vec())
                            .unwrap_or_default();
                        format!(
                            "ConvDW: {} {:?} s={:?} in={:?}",
                            node.name, shape, strides, in_shape
                        )
                    } else {
                        let is_wino = self.winograd_weights.contains_key(w_name);
                        let strides = node
                            .attrs
                            .get("strides")
                            .and_then(|a| a.as_usize_vec())
                            .unwrap_or_default();
                        let pads = node
                            .attrs
                            .get("pads")
                            .and_then(|a| a.as_usize_vec())
                            .unwrap_or_default();
                        format!(
                            "Conv: {} {:?} s={:?} p={:?} in={:?} wino={}",
                            node.name, shape, strides, pads, in_shape, is_wino
                        )
                    }
                } else if node.op == "MaxPool" {
                    let k = node
                        .attrs
                        .get("kernel_shape")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_default();
                    let s = node
                        .attrs
                        .get("strides")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_default();
                    let p = node
                        .attrs
                        .get("pads")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_default();
                    format!("MaxPool: {} k={:?} s={:?} p={:?}", node.name, k, s, p)
                } else if node.op == "MatMul" {
                    let s0 = env
                        .get(node.inputs[0].as_str())
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let s1 = env
                        .get(node.inputs[1].as_str())
                        .or_else(|| self.weights.get(node.inputs[1].as_str()))
                        .or_else(|| self.transposed_matmul_weights.get(node.inputs[1].as_str()))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let is_q = self.quantized_matmul_weights.contains_key(&node.inputs[1]);
                    let is_t = self.transposed_matmul_weights.contains_key(&node.inputs[1]);
                    format!(
                        "MatMul: {} {:?} x {:?} (quantized={}, transposed={})",
                        node.name, s0, s1, is_q, is_t
                    )
                } else if node.op == "Transpose" {
                    let perm = node
                        .attrs
                        .get("perm")
                        .and_then(|a| a.as_usize_vec())
                        .unwrap_or_default();
                    format!(
                        "Transpose: {} out={:?} perm={:?}",
                        node.name,
                        out_tensor.shape(),
                        perm
                    )
                } else if node.op == "Concat" {
                    let axis = node.attrs.get("axis").and_then(|a| a.as_i64()).unwrap_or(0);
                    let cont: Vec<bool> = node
                        .inputs
                        .iter()
                        .map(|inp| {
                            env.get(inp.as_str())
                                .map(|t| t.is_contiguous())
                                .unwrap_or(true)
                        })
                        .collect();
                    format!(
                        "Concat: {} axis={} in={:?} cont={:?} out={:?}",
                        node.name,
                        axis,
                        node.inputs,
                        cont,
                        out_tensor.shape()
                    )
                } else {
                    format!("{}: {}", node.op, node.name)
                };
                *profile_data.entry(op_key).or_insert(0.0) += ms;
            }

            env.insert(out_name.as_str(), out_tensor);

            // Eager garbage collection for intermediate tensors
            if let Some(concat_inps) = node.attrs.get("concat_inputs").and_then(|a| a.as_strings())
            {
                for inp in concat_inps {
                    let inp_str = inp.as_str();
                    if let Some(cnt) = usage_count.get_mut(inp_str) {
                        *cnt -= 1;
                        if *cnt == 0 {
                            env.remove(inp_str);
                        }
                    }
                }
                for inp in &node.inputs[1..] {
                    let inp_str = inp.as_str();
                    if let Some(cnt) = usage_count.get_mut(inp_str) {
                        *cnt -= 1;
                        if *cnt == 0 {
                            env.remove(inp_str);
                        }
                    }
                }
            } else {
                for inp in &node.inputs {
                    let inp_str = inp.as_str();
                    if let Some(cnt) = usage_count.get_mut(inp_str) {
                        *cnt -= 1;
                        if *cnt == 0 {
                            env.remove(inp_str);
                        }
                    }
                }
            }
        }

        let final_out_name = &self.graph.outputs[0];
        let final_tensor = env.remove(final_out_name.as_str()).ok_or_else(|| {
            OcrError::ModelError(format!(
                "Output tensor '{}' not found after graph execution",
                final_out_name
            ))
        })?;

        Ok((final_tensor, profile_data))
    }
}

/// Read the `group` attribute of `node` as a `usize`, rejecting zero and negatives.
///
/// A zero divides by zero inside the operator, and a negative value becomes a huge `usize`
/// through a plain `as` cast. Since the graph JSON is untrusted input, neither should reach
/// an operator.
fn group_attr(node: &Node) -> Result<usize> {
    let g = node
        .attrs
        .get("group")
        .and_then(|a| a.as_i64())
        .unwrap_or(1);
    usize::try_from(g)
        .ok()
        .filter(|&v| v >= 1)
        .ok_or_else(|| OcrError::OperatorError {
            op: node.op.clone(),
            msg: format!("node '{}' has an invalid `group` attribute: {g}", node.name),
        })
}

#[allow(clippy::too_many_arguments)]
fn compute_same_upper_pads(
    in_h: usize,
    in_w: usize,
    k_h: usize,
    k_w: usize,
    s_h: usize,
    s_w: usize,
    d_h: usize,
    d_w: usize,
) -> [usize; 4] {
    let eff_k_h = d_h * (k_h - 1) + 1;
    let eff_k_w = d_w * (k_w - 1) + 1;
    let out_h = in_h.div_ceil(s_h);
    let out_w = in_w.div_ceil(s_w);
    let pad_needed_h = ((out_h - 1) * s_h + eff_k_h).saturating_sub(in_h);
    let pad_needed_w = ((out_w - 1) * s_w + eff_k_w).saturating_sub(in_w);
    let pad_top = pad_needed_h / 2;
    let pad_bottom = pad_needed_h - pad_top;
    let pad_left = pad_needed_w / 2;
    let pad_right = pad_needed_w - pad_left;
    [pad_top, pad_left, pad_bottom, pad_right]
}

// ---------------------------------------------------------------------------
// Graph-fusion helpers
// ---------------------------------------------------------------------------

/// `sqrt(2)` as stored by the Paddle/ONNX exporter in PP-OCR GELU sub-graphs.
const GELU_SQRT2: f32 = std::f32::consts::SQRT_2;

/// Relative tolerance used when matching a graph constant against an expected value.
const CONST_RTOL: f32 = 1e-5;

/// Returns `true` when `name` is a constant tensor (present in `weights`) whose first
/// element is within [`CONST_RTOL`] of `expected`.
///
/// Used to validate fusion preconditions: a sub-graph is only rewritten when its
/// constant operands really have the expected value, so a structurally-similar but
/// semantically different pattern is never replaced by a wrong fused node.
fn is_scalar_const(weights: &HashMap<String, Tensor>, name: &str, expected: f32) -> bool {
    weights
        .get(name)
        .and_then(|t| t.as_slice().first().copied())
        .map(|v| (v - expected).abs() <= CONST_RTOL * expected.abs().max(1.0))
        .unwrap_or(false)
}

/// Returns the single element of `inputs` that is not equal to `exclude`.
///
/// Returns `None` when there is no such element *or* when more than one exists, so an
/// ambiguous operand list is treated as "does not match" instead of silently picking an
/// arbitrary operand (which is what the previous `find(..).unwrap()` did — and which
/// panicked when every operand was `exclude`).
fn other_operand<'a>(inputs: &'a [String], exclude: &str) -> Option<&'a String> {
    let mut found: Option<&String> = None;
    for inp in inputs {
        if inp != exclude {
            if found.is_some() {
                return None;
            }
            found = Some(inp);
        }
    }
    found
}

/// Fold a `BatchNormalization` into the `Conv` that produces its input.
///
/// BatchNorm applies the per-channel affine transform
/// `y[c] = (x[c] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]`, which composes
/// cleanly with the convolution's own affine parameters:
///
/// ```text
/// a[c]  = gamma[c] / sqrt(var[c] + eps)
/// W'[c] = W[c] * a[c]
/// b'[c] = (b[c] - mean[c]) * a[c] + beta[c]      (b defaults to 0 when the Conv has none)
/// ```
///
/// The folded tensors are stored under fresh names so the originals remain available (they
/// may be shared with other nodes).
///
/// Returns `None` — leaving `weights` untouched — when the fold is not applicable: a BN
/// parameter that is not a constant, a non-4D Conv weight, a channel-count mismatch, or a
/// non-finite scale factor.
fn fold_batch_norm_into_conv(
    conv: &Node,
    bn: &Node,
    weights: &mut HashMap<String, Tensor>,
) -> Option<(String, String)> {
    // BN inputs are [X, scale (gamma), B (beta), mean, var].
    let w_name = conv.inputs.get(1)?;
    let w = weights.get(w_name)?;
    if w.ndim() != 4 {
        return None;
    }
    let out_c = w.shape()[0];
    if out_c == 0 {
        return None;
    }

    let scale = weights.get(bn.inputs.get(1)?)?;
    let beta = weights.get(bn.inputs.get(2)?)?;
    let mean = weights.get(bn.inputs.get(3)?)?;
    let var = weights.get(bn.inputs.get(4)?)?;
    if scale.numel() != out_c
        || beta.numel() != out_c
        || mean.numel() != out_c
        || var.numel() != out_c
    {
        return None;
    }

    let orig_bias = match conv.inputs.get(2) {
        Some(name) if !name.is_empty() => {
            let b = weights.get(name)?;
            if b.numel() != out_c {
                return None;
            }
            Some(b.as_slice().to_vec())
        }
        _ => None,
    };

    let eps = bn
        .attrs
        .get("epsilon")
        .and_then(|a| a.as_f32())
        .unwrap_or(1e-5);

    let (sc, be, mu, va) = (
        scale.as_slice(),
        beta.as_slice(),
        mean.as_slice(),
        var.as_slice(),
    );

    let mut new_w = w.clone();
    let mut new_b = vec![0.0f32; out_c];
    {
        let nw = new_w.as_mut_slice();
        let per_channel = nw.len() / out_c;
        for c in 0..out_c {
            let denom = (va[c] + eps).max(0.0).sqrt();
            if denom <= 0.0 || !denom.is_finite() {
                return None;
            }
            let a = sc[c] / denom;
            if !a.is_finite() {
                return None;
            }
            for k in 0..per_channel {
                nw[c * per_channel + k] *= a;
            }
            let b0 = orig_bias.as_ref().map(|b| b[c]).unwrap_or(0.0);
            new_b[c] = (b0 - mu[c]) * a + be[c];
        }
    }

    let w_key = format!("{}__bnfold_w", w_name);
    let b_key = format!("{}__bnfold_b", w_name);
    weights.insert(w_key.clone(), new_w);
    weights.insert(b_key.clone(), Tensor::from_vec(&[out_c], new_b).ok()?);
    Some((w_key, b_key))
}

/// Try to fuse `HardSigmoid(alpha = 1/6, beta = 0.5)` followed by `Mul(x, hs_out)` into a
/// single `HardSwish(x)` node fed directly by `x`.
///
/// `HardSwish(x) = x * clamp(x/6 + 0.5, 0, 1)`, which is exactly `x * HardSigmoid(x)` for
/// that parameter pair — so any other alpha/beta is rejected, as is a `Mul` whose second
/// operand is not the very same tensor the HardSigmoid consumed (that pattern is a plain
/// gating multiply, not HardSwish).
fn try_fuse_hardswish(
    hs: &Node,
    mul: &Node,
    idx: usize,
    usage: &HashMap<String, usize>,
) -> Option<Node> {
    if hs.op != "HardSigmoid" || mul.op != "Mul" || hs.inputs.len() != 1 {
        return None;
    }
    let alpha = hs.attrs.get("alpha").and_then(|a| a.as_f32())?;
    let beta = hs.attrs.get("beta").and_then(|a| a.as_f32())?;
    if (alpha - 1.0 / 6.0).abs() > CONST_RTOL || (beta - 0.5).abs() > CONST_RTOL {
        return None;
    }

    let x = hs.inputs.first()?;
    let hs_out = hs.outputs.first()?;
    let y = mul.outputs.first()?;

    if !mul.inputs.contains(hs_out) || !mul.inputs.contains(x) {
        return None;
    }
    if usage.get(hs_out) != Some(&1) {
        return None;
    }

    Some(Node {
        name: format!("Fused_HardSwish_{}", idx),
        op: "HardSwish".to_string(),
        inputs: vec![x.clone()],
        outputs: vec![y.clone()],
        attrs: HashMap::new(),
    })
}

/// Try to fuse the GELU sub-graph
/// `Div(x, sqrt2) -> Erf -> Add(1.0) -> Mul(x, ·) -> Mul(·, 0.5)` starting at `window[0]`.
///
/// Returns `None` unless *all* of the following hold, in which case the original nodes
/// are kept as-is:
/// * the five ops appear in the expected order and are wired to each other,
/// * the `Div` divisor is the constant sqrt(2),
/// * the `Add` addend is the constant `1.0`,
/// * the outer `Mul` factor is the constant `0.5`,
/// * every intermediate tensor (`Div`, `Erf`, `Add`, inner `Mul` output) is consumed
///   exactly once, i.e. nothing outside the sub-graph reads it.
fn try_fuse_gelu(
    window: &[Node],
    idx: usize,
    weights: &HashMap<String, Tensor>,
    usage: &HashMap<String, usize>,
) -> Option<Node> {
    if window.len() < 5 {
        return None;
    }
    let (n0, n1, n2, n3, n4) = (&window[0], &window[1], &window[2], &window[3], &window[4]);
    if n0.op != "Div" || n1.op != "Erf" || n2.op != "Add" || n3.op != "Mul" || n4.op != "Mul" {
        return None;
    }

    let x = n0.inputs.first()?;
    let div_out = n0.outputs.first()?;
    let erf_out = n1.outputs.first()?;
    let add_out = n2.outputs.first()?;
    let mul1_out = n3.outputs.first()?;
    let gelu_out = n4.outputs.first()?;

    // --- Structural wiring ---
    if n1.inputs.first() != Some(div_out)
        || !n2.inputs.contains(erf_out)
        || !n3.inputs.contains(x)
        || !n3.inputs.contains(add_out)
        || !n4.inputs.contains(mul1_out)
    {
        return None;
    }

    // --- Constant-symbol validation ---
    let div_const = n0.inputs.get(1).filter(|c| *c != x)?;
    let add_const = other_operand(&n2.inputs, erf_out)?;
    let mul_const = other_operand(&n4.inputs, mul1_out)?;
    if !is_scalar_const(weights, div_const, GELU_SQRT2)
        || !is_scalar_const(weights, add_const, 1.0)
        || !is_scalar_const(weights, mul_const, 0.5)
    {
        return None;
    }

    // --- Single-use validation ---
    for t in [div_out, erf_out, add_out, mul1_out] {
        if usage.get(t) != Some(&1) {
            return None;
        }
    }

    Some(Node {
        name: format!("Fused_Gelu_{}", idx),
        op: "Gelu".to_string(),
        inputs: vec![x.clone()],
        outputs: vec![gelu_out.clone()],
        attrs: HashMap::new(),
    })
}

/// Try to fuse the LayerNorm sub-graph
/// `ReduceMean -> Sub -> Pow(2.0) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul(w) -> Add(b)`
/// starting at `window[0]`.
///
/// Returns `None` unless *all* of the following hold, in which case the original nodes
/// are kept as-is:
/// * the nine ops appear in the expected order and are wired to each other,
/// * the `Pow` exponent is the constant `2.0`,
/// * the `Add(eps)` operand is a real constant present in `weights` (no silent `1e-5`
///   fallback, which previously masked a mis-detected sub-graph),
/// * `eps`, `w` and `b` are each unambiguous single operands,
/// * usage counts match the expected shape: `diff` is read exactly twice (by `Pow` and
///   by `Div`, both inside the sub-graph) and every other intermediate exactly once.
fn try_fuse_layernorm(
    window: &[Node],
    idx: usize,
    weights: &HashMap<String, Tensor>,
    usage: &HashMap<String, usize>,
) -> Option<Node> {
    if window.len() < 9 {
        return None;
    }
    let (n0, n1, n2, n3) = (&window[0], &window[1], &window[2], &window[3]);
    let (n4, n5, n6, n7, n8) = (&window[4], &window[5], &window[6], &window[7], &window[8]);
    if n0.op != "ReduceMean"
        || n1.op != "Sub"
        || n2.op != "Pow"
        || n3.op != "ReduceMean"
        || n4.op != "Add"
        || n5.op != "Sqrt"
        || n6.op != "Div"
        || n7.op != "Mul"
        || n8.op != "Add"
    {
        return None;
    }

    let x = n0.inputs.first()?;
    let mean = n0.outputs.first()?;
    let diff = n1.outputs.first()?;
    let pow = n2.outputs.first()?;
    let var = n3.outputs.first()?;
    let eps_add = n4.outputs.first()?;
    let std = n5.outputs.first()?;
    let norm = n6.outputs.first()?;
    let mul_out = n7.outputs.first()?;
    let final_out = n8.outputs.first()?;

    // --- Structural wiring ---
    if !n1.inputs.contains(x)
        || !n1.inputs.contains(mean)
        || n2.inputs.first() != Some(diff)
        || n3.inputs.first() != Some(pow)
        || !n4.inputs.contains(var)
        || n5.inputs.first() != Some(eps_add)
        || n6.inputs.first() != Some(diff)
        || n6.inputs.get(1) != Some(std)
        || !n7.inputs.contains(norm)
        || !n8.inputs.contains(mul_out)
    {
        return None;
    }

    // --- Constant-symbol validation: exponent must be 2.0 ---
    let pow_const = n2.inputs.get(1).filter(|c| *c != diff)?;
    if !is_scalar_const(weights, pow_const, 2.0) {
        return None;
    }

    // --- Usage-count validation ---
    if usage.get(mean) != Some(&1)
        || usage.get(diff) != Some(&2)
        || usage.get(pow) != Some(&1)
        || usage.get(var) != Some(&1)
        || usage.get(eps_add) != Some(&1)
        || usage.get(std) != Some(&1)
        || usage.get(norm) != Some(&1)
        || usage.get(mul_out) != Some(&1)
    {
        return None;
    }

    // --- Operand extraction (unambiguous) ---
    let eps_name = other_operand(&n4.inputs, var)?;
    let weight_name = other_operand(&n7.inputs, norm)?;
    let bias_name = other_operand(&n8.inputs, mul_out)?;

    // `eps` must be a real constant; a missing one would change the semantics.
    let eps_val = weights
        .get(eps_name)
        .and_then(|t| t.as_slice().first().copied())?;

    let mut attrs = HashMap::new();
    attrs.insert("eps".to_string(), AttributeValue::Float(eps_val));

    Some(Node {
        name: format!("Fused_LayerNorm_{}", idx),
        op: "LayerNorm".to_string(),
        inputs: vec![x.clone(), weight_name.clone(), bias_name.clone()],
        outputs: vec![final_out.clone()],
        attrs,
    })
}

#[cfg(test)]
mod fusion_tests {
    use super::*;
    use crate::tensor::Tensor;

    fn node(name: &str, op: &str, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            name: name.to_string(),
            op: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: HashMap::new(),
        }
    }

    fn consts(items: &[(&str, f32)]) -> HashMap<String, Tensor> {
        items
            .iter()
            .map(|(n, v)| {
                (
                    n.to_string(),
                    Tensor::from_vec(&[1], vec![*v]).expect("valid scalar tensor"),
                )
            })
            .collect()
    }

    fn use_once(names: &[&str]) -> HashMap<String, usize> {
        names.iter().map(|n| (n.to_string(), 1usize)).collect()
    }

    fn gelu_window() -> Vec<Node> {
        vec![
            node("d", "Div", &["x", "sqrt2"], &["d_out"]),
            node("e", "Erf", &["d_out"], &["e_out"]),
            node("a", "Add", &["e_out", "one"], &["a_out"]),
            node("m1", "Mul", &["x", "a_out"], &["m1_out"]),
            node("m2", "Mul", &["m1_out", "half"], &["y"]),
        ]
    }

    #[test]
    fn gelu_fuses_when_constants_and_usage_match() {
        let w = consts(&[("sqrt2", GELU_SQRT2), ("one", 1.0), ("half", 0.5)]);
        let u = use_once(&["d_out", "e_out", "a_out", "m1_out"]);
        let fused = try_fuse_gelu(&gelu_window(), 0, &w, &u).expect("should fuse");
        assert_eq!(fused.op, "Gelu");
        assert_eq!(fused.inputs, vec!["x".to_string()]);
        assert_eq!(fused.outputs, vec!["y".to_string()]);
    }

    #[test]
    fn gelu_rejects_wrong_constant() {
        // divisor is 2.0 instead of sqrt(2) -> must NOT fuse
        let w = consts(&[("sqrt2", 2.0), ("one", 1.0), ("half", 0.5)]);
        let u = use_once(&["d_out", "e_out", "a_out", "m1_out"]);
        assert!(try_fuse_gelu(&gelu_window(), 0, &w, &u).is_none());
    }

    #[test]
    fn gelu_rejects_multi_use_intermediate() {
        let w = consts(&[("sqrt2", GELU_SQRT2), ("one", 1.0), ("half", 0.5)]);
        let mut u = use_once(&["d_out", "e_out", "a_out", "m1_out"]);
        u.insert("d_out".to_string(), 2); // read by something outside as well
        assert!(try_fuse_gelu(&gelu_window(), 0, &w, &u).is_none());
    }

    #[test]
    fn gelu_rejects_short_window() {
        let w = consts(&[("sqrt2", GELU_SQRT2), ("one", 1.0), ("half", 0.5)]);
        let u = HashMap::new();
        let mut win = gelu_window();
        win.pop();
        assert!(try_fuse_gelu(&win, 0, &w, &u).is_none());
    }

    fn ln_window() -> Vec<Node> {
        vec![
            node("rm", "ReduceMean", &["x"], &["mean"]),
            node("sub", "Sub", &["x", "mean"], &["diff"]),
            node("pow", "Pow", &["diff", "two"], &["pow"]),
            node("rm2", "ReduceMean", &["pow"], &["var"]),
            node("adde", "Add", &["var", "eps"], &["eps_add"]),
            node("sqrt", "Sqrt", &["eps_add"], &["std"]),
            node("div", "Div", &["diff", "std"], &["norm"]),
            node("mul", "Mul", &["norm", "w"], &["mul_out"]),
            node("addb", "Add", &["mul_out", "b"], &["y"]),
        ]
    }

    #[test]
    fn layernorm_fuses_when_constants_and_usage_match() {
        let w = consts(&[("two", 2.0), ("eps", 1e-5)]);
        let mut u = use_once(&["mean", "pow", "var", "eps_add", "std", "norm", "mul_out"]);
        u.insert("diff".to_string(), 2); // Pow + Div
        let fused = try_fuse_layernorm(&ln_window(), 0, &w, &u).expect("should fuse");
        assert_eq!(fused.op, "LayerNorm");
        assert_eq!(fused.inputs, vec!["x", "w", "b"]);
        assert!(matches!(
            fused.attrs.get("eps"),
            Some(AttributeValue::Float(v)) if (*v - 1e-5).abs() < 1e-9
        ));
    }

    #[test]
    fn layernorm_rejects_wrong_exponent() {
        let w = consts(&[("two", 3.0), ("eps", 1e-5)]);
        let mut u = use_once(&["mean", "pow", "var", "eps_add", "std", "norm", "mul_out"]);
        u.insert("diff".to_string(), 2);
        assert!(try_fuse_layernorm(&ln_window(), 0, &w, &u).is_none());
    }

    #[test]
    fn layernorm_rejects_missing_eps_const() {
        // `eps` is not a constant at all -> previously silently defaulted to 1e-5.
        let w = consts(&[("two", 2.0)]);
        let mut u = use_once(&["mean", "pow", "var", "eps_add", "std", "norm", "mul_out"]);
        u.insert("diff".to_string(), 2);
        assert!(try_fuse_layernorm(&ln_window(), 0, &w, &u).is_none());
    }

    #[test]
    fn layernorm_rejects_extra_diff_consumer() {
        let w = consts(&[("two", 2.0), ("eps", 1e-5)]);
        let mut u = use_once(&["mean", "pow", "var", "eps_add", "std", "norm", "mul_out"]);
        u.insert("diff".to_string(), 3);
        assert!(try_fuse_layernorm(&ln_window(), 0, &w, &u).is_none());
    }

    #[test]
    fn other_operand_is_strict() {
        let two = vec!["a".to_string(), "b".to_string()];
        assert_eq!(other_operand(&two, "a"), Some(&"b".to_string()));
        let only = vec!["a".to_string()];
        assert!(other_operand(&only, "a").is_none());
        let dup = vec!["a".to_string(), "a".to_string()];
        assert!(other_operand(&dup, "a").is_none());
        let ambiguous = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!(other_operand(&ambiguous, "a").is_none());
    }
}

#[cfg(test)]
mod graph_opt_tests {
    use super::*;

    fn mknode(
        op: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: &[(&str, AttributeValue)],
    ) -> Node {
        Node {
            name: format!("{}_{}", op, outputs[0]),
            op: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape, data).expect("valid tensor")
    }

    #[test]
    fn batch_norm_fold_matches_unfused_math() {
        // Two output channels of a 2x2 kernel -> 4 weights per channel.
        let w = tensor(
            &[2, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0],
        );
        let b = tensor(&[2], vec![0.5, -1.0]);
        let gamma = tensor(&[2], vec![2.0, 0.25]);
        let beta = tensor(&[2], vec![0.1, -0.2]);
        let mean = tensor(&[2], vec![1.0, -0.5]);
        let var = tensor(&[2], vec![4.0, 0.25]);
        let eps = 1e-5f32;

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        for (k, t) in [
            ("w", &w),
            ("b", &b),
            ("gamma", &gamma),
            ("beta", &beta),
            ("mean", &mean),
            ("var", &var),
        ] {
            weights.insert(k.to_string(), t.clone());
        }

        let conv = mknode("Conv", &["x", "w", "b"], &["conv_out"], &[]);
        let bn = mknode(
            "BatchNormalization",
            &["conv_out", "gamma", "beta", "mean", "var"],
            &["y"],
            &[("epsilon", AttributeValue::Float(eps))],
        );

        let (w_key, b_key) =
            fold_batch_norm_into_conv(&conv, &bn, &mut weights).expect("fold should apply");
        let fw = weights.get(&w_key).expect("folded weight");
        let fb = weights.get(&b_key).expect("folded bias");

        for c in 0..2usize {
            let a = gamma.as_slice()[c] / (var.as_slice()[c] + eps).sqrt();
            for k in 0..4usize {
                let expected = w.as_slice()[c * 4 + k] * a;
                assert!(
                    (fw.as_slice()[c * 4 + k] - expected).abs() < 1e-6,
                    "weight mismatch at c={c} k={k}"
                );
            }
            let expected_b = (b.as_slice()[c] - mean.as_slice()[c]) * a + beta.as_slice()[c];
            assert!(
                (fb.as_slice()[c] - expected_b).abs() < 1e-6,
                "bias mismatch at c={c}"
            );
        }

        // The folded parameters must reproduce the original two-step result for any input.
        for &x in &[-3.0f32, -0.25, 0.0, 1.7, 8.0] {
            for c in 0..2usize {
                let pre = x * w.as_slice()[c * 4] + b.as_slice()[c];
                let unfused = (pre - mean.as_slice()[c]) / (var.as_slice()[c] + eps).sqrt()
                    * gamma.as_slice()[c]
                    + beta.as_slice()[c];
                let fused = x * fw.as_slice()[c * 4] + fb.as_slice()[c];
                assert!(
                    (unfused - fused).abs() < 1e-4,
                    "c={c} x={x}: unfused={unfused} fused={fused}"
                );
            }
        }
    }

    #[test]
    fn batch_norm_fold_creates_bias_when_conv_has_none() {
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("w".to_string(), tensor(&[1, 1, 1, 1], vec![2.0]));
        weights.insert("gamma".to_string(), tensor(&[1], vec![1.0]));
        weights.insert("beta".to_string(), tensor(&[1], vec![0.0]));
        weights.insert("mean".to_string(), tensor(&[1], vec![0.0]));
        weights.insert("var".to_string(), tensor(&[1], vec![1.0]));

        let conv = mknode("Conv", &["x", "w"], &["conv_out"], &[]);
        let bn = mknode(
            "BatchNormalization",
            &["conv_out", "gamma", "beta", "mean", "var"],
            &["y"],
            &[("epsilon", AttributeValue::Float(0.0))],
        );
        let (w_key, b_key) =
            fold_batch_norm_into_conv(&conv, &bn, &mut weights).expect("fold should apply");
        assert_eq!(weights.get(&w_key).map(|t| t.numel()), Some(1));
        assert_eq!(weights.get(&b_key).map(|t| t.numel()), Some(1));
    }

    #[test]
    fn batch_norm_fold_rejects_non_constant_parameters() {
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("w".to_string(), tensor(&[1, 1, 1, 1], vec![1.0]));
        // `gamma` deliberately absent: it does not resolve to a constant.
        weights.insert("beta".to_string(), tensor(&[1], vec![0.0]));
        weights.insert("mean".to_string(), tensor(&[1], vec![0.0]));
        weights.insert("var".to_string(), tensor(&[1], vec![1.0]));

        let conv = mknode("Conv", &["x", "w"], &["conv_out"], &[]);
        let bn = mknode(
            "BatchNormalization",
            &["conv_out", "gamma", "beta", "mean", "var"],
            &["y"],
            &[],
        );
        let before = weights.len();
        assert!(fold_batch_norm_into_conv(&conv, &bn, &mut weights).is_none());
        assert_eq!(
            weights.len(),
            before,
            "weights must be left untouched when the fold is rejected"
        );
    }

    #[test]
    fn hardswish_fold_requires_matching_constants_and_same_operand() {
        let hs = mknode(
            "HardSigmoid",
            &["x"],
            &["hs_out"],
            &[
                ("alpha", AttributeValue::Float(1.0 / 6.0)),
                ("beta", AttributeValue::Float(0.5)),
            ],
        );
        let mul = mknode("Mul", &["hs_out", "x"], &["y"], &[]);

        let mut ok = HashMap::new();
        ok.insert("hs_out".to_string(), 1usize);
        let fused = try_fuse_hardswish(&hs, &mul, 0, &ok).expect("should fuse");
        assert_eq!(fused.op, "HardSwish");
        assert_eq!(fused.inputs, vec!["x".to_string()]);
        assert_eq!(fused.outputs, vec!["y".to_string()]);

        let mut multi = HashMap::new();
        multi.insert("hs_out".to_string(), 2usize);
        assert!(try_fuse_hardswish(&hs, &mul, 0, &multi).is_none());

        // Different alpha -> a plain gating multiply, not HardSwish.
        let hs_bad = mknode(
            "HardSigmoid",
            &["x"],
            &["hs_out"],
            &[
                ("alpha", AttributeValue::Float(0.25)),
                ("beta", AttributeValue::Float(0.5)),
            ],
        );
        assert!(try_fuse_hardswish(&hs_bad, &mul, 0, &ok).is_none());

        // Second operand is a different tensor.
        let mul_other = mknode("Mul", &["hs_out", "z"], &["y"], &[]);
        assert!(try_fuse_hardswish(&hs, &mul_other, 0, &ok).is_none());
    }

    #[test]
    fn hardswish_activation_matches_scalar_formula() {
        for i in -300..=300 {
            let x = i as f32 / 20.0; // -15.0 ..= 15.0
            let expected = x * ((x + 3.0).clamp(0.0, 6.0) / 6.0);
            let got =
                crate::ops::conv2d::apply_conv_activation(x, crate::ops::ConvActivation::HardSwish);
            assert!(
                (got - expected).abs() < 1e-6,
                "x={x}: got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn cls_graph_folds_batch_norm_and_hardswish() {
        let path = "models/PP-OCRv6_cls.safetensors";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let model = Model::load(path).expect("load cls model");
        let nodes = &model.graph().nodes;
        let count = |op: &str| nodes.iter().filter(|n| n.op == op).count();

        assert_eq!(
            count("BatchNormalization"),
            0,
            "BN must be folded into Conv"
        );
        assert_eq!(
            count("HardSwish"),
            0,
            "HardSwish must be absorbed as a Conv activation"
        );

        let hs_convs = nodes
            .iter()
            .filter(|n| {
                n.op == "Conv"
                    && n.attrs.get("activation").and_then(|a| a.as_str()) == Some("HardSwish")
            })
            .count();
        assert!(
            hs_convs >= 20,
            "expected at least 20 Conv+HardSwish fusions, got {hs_convs}"
        );
        assert!(
            nodes.len() < 143 - 50,
            "cls graph should shrink substantially, got {} nodes",
            nodes.len()
        );
    }

    /// Regression for the BatchNorm-fold rewiring bug.
    ///
    /// The fold used to `truncate(2)` the Conv inputs, which dropped only the bias and kept
    /// the *original* weight in `inputs[1]`; the two pushed tensors therefore landed in
    /// `inputs[2]`/`inputs[3]`. The engine reads `inputs[1]` as the weight and `inputs[2]` as
    /// the bias, so every folded Conv silently evaluated `conv(x, w_orig) + w_folded` —
    /// which garbled recognition for every model containing a Conv→BN pair.
    #[test]
    fn batch_norm_fold_rewires_conv_weight_and_bias_slots() {
        let mut total = 0usize;
        for path in [
            "models/PP-OCRv6_cls.safetensors",
            "models/PP-OCRv6_small_rec.safetensors",
        ] {
            if !std::path::Path::new(path).exists() {
                continue;
            }
            let model = Model::load(path).expect("load model");
            let mut checked = 0usize;
            for n in model.graph().nodes.iter().filter(|n| n.op == "Conv") {
                let Some(w) = n.inputs.get(1) else { continue };
                if !w.ends_with("__bnfold_w") {
                    continue;
                }
                checked += 1;
                assert_eq!(
                    n.inputs.len(),
                    3,
                    "{path}: folded Conv '{}' must be [x, w_folded, b_folded], got {:?}",
                    n.name,
                    n.inputs
                );
                let b = &n.inputs[2];
                assert!(
                    b.ends_with("__bnfold_b"),
                    "{path}: folded Conv '{}' has '{b}' in the bias slot — the folded weight \
                     must never be read as a bias",
                    n.name
                );
                assert!(
                    model.weights().contains_key(w),
                    "{path}: missing folded weight {w}"
                );
                assert!(
                    model.weights().contains_key(b),
                    "{path}: missing folded bias {b}"
                );
            }
            assert!(checked > 0, "{path}: expected at least one folded Conv");
            total += checked;
        }
        // Silently skipping both models would make this test vacuous.
        if std::path::Path::new("models/PP-OCRv6_cls.safetensors").exists() {
            assert!(
                total >= 20,
                "expected the CLS/REC models to expose folded Convs"
            );
        }
    }

    /// Regression for the missing structural validation in `Model::load`.
    ///
    /// A node with empty `inputs`/`outputs` used to panic *during load* (`n.inputs[0]` at the
    /// Identity-elimination pass, `n.outputs[0]` right after it), so a malformed model file
    /// crashed instead of returning a `ModelError` — defeating the `Result`-based boundary
    /// that `Cargo.toml` deliberately preserves by not setting `panic = "abort"`.
    ///
    /// These build a real safetensors model in memory so the whole `Model::load` path runs,
    /// including the loader and the graph optimisation passes.
    #[test]
    fn load_rejects_malformed_graphs_instead_of_panicking() {
        for (label, graph_json) in [
            (
                "node with empty outputs",
                r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"name":"n","op":"Add","inputs":["x"],"outputs":[]}]}"#,
            ),
            (
                "node with empty inputs",
                r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"name":"n","op":"Relu","inputs":[],"outputs":["y"]}]}"#,
            ),
            (
                "graph with no inputs",
                r#"{"inputs":[],"outputs":["y"],"nodes":[{"name":"n","op":"Identity","inputs":["x"],"outputs":["y"]}]}"#,
            ),
            (
                "graph with no outputs",
                r#"{"inputs":["x"],"outputs":[],"nodes":[{"name":"n","op":"Identity","inputs":["x"],"outputs":["y"]}]}"#,
            ),
        ] {
            let path = write_model_with_graph(graph_json, &[(1usize, 0.5f32)]);
            // `Model` is not `UnwindSafe` (it owns raw weight buffers), hence the wrapper.
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Model::load(&path)));
            assert!(result.is_ok(), "{label}: Model::load must not panic");
            // `Model` has no `Debug`, so `expect_err` is unavailable; match instead.
            let err = match result.expect("no panic") {
                Ok(_) => panic!("{label} must be rejected"),
                Err(e) => e,
            };
            assert!(
                matches!(err, OcrError::ModelError(_)),
                "{label}: expected ModelError, got {err:?}"
            );
        }
    }

    /// A well-formed graph must still load, so the negative test above is only meaningful if
    /// the builder produces something valid. The node is a `Relu` rather than an `Identity`
    /// because the first optimisation pass strips every `Identity` node.
    #[test]
    fn load_accepts_a_well_formed_minimal_graph() {
        let path = write_model_with_graph(
            r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"name":"n","op":"Relu","inputs":["x"],"outputs":["y"]}]}"#,
            &[(1usize, 0.5f32)],
        );
        let model = Model::load(&path).expect("well-formed graph must load");
        assert_eq!(model.graph().nodes.len(), 1);
        assert_eq!(model.graph().nodes[0].op, "Relu");
    }

    /// A negative `group` used to become `usize::MAX` through an `as` cast, and a zero
    /// divided by zero inside the operator. Both must be a clean error now.
    #[test]
    fn invalid_group_attribute_is_rejected() {
        for g in [-1i64, 0] {
            let node = Node {
                name: "bad".to_string(),
                op: "Conv".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                attrs: HashMap::from([("group".to_string(), AttributeValue::Int(g))]),
            };
            let err = group_attr(&node).expect_err(&format!("group={g} must be rejected"));
            assert!(
                matches!(err, OcrError::OperatorError { .. }),
                "group={g}: expected OperatorError, got {err:?}"
            );
        }

        // Valid values still pass through, including the default.
        let ok = Node {
            name: "good".to_string(),
            op: "Conv".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: HashMap::new(),
        };
        assert_eq!(group_attr(&ok).expect("no attribute -> 1"), 1);
        for g in [1i64, 2, 64] {
            let node = Node {
                attrs: HashMap::from([("group".to_string(), AttributeValue::Int(g))]),
                ..ok.clone()
            };
            assert_eq!(group_attr(&node).expect("valid group"), g as usize);
        }
    }

    // --- helper for the load tests above ---

    /// Write a single-tensor safetensors model whose metadata carries `graph_json`.
    fn write_model_with_graph(graph_json: &str, values: &[(usize, f32)]) -> std::path::PathBuf {
        let mut payload: Vec<u8> = Vec::new();
        for (_, v) in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let header = format!(
            r#"{{"__metadata__":{{"format":"pt","graph":"{}"}},"w":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{}]}}}}"#,
            graph_json.replace('"', "\\\""),
            values.len(),
            payload.len()
        );
        let mut hb = header.into_bytes();
        while hb.len() % 8 != 0 {
            hb.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&payload);

        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join("ppocr_model_tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!(
            "m{}.safetensors",
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, out).expect("write temp model");
        path
    }

    #[test]
    fn rec_graph_folds_batch_norm() {
        let path = "models/PP-OCRv6_small_rec.safetensors";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let model = Model::load(path).expect("load rec model");
        let bn = model
            .graph()
            .nodes
            .iter()
            .filter(|n| n.op == "BatchNormalization")
            .count();
        assert_eq!(bn, 0, "rec BN nodes should be folded into their Conv");
    }

    #[test]
    fn packing_releases_fp32_matmul_originals() {
        let path = "models/PP-OCRv6_small_rec.safetensors";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let model = Model::load(path).expect("load rec model");

        assert!(
            model
                .transposed_matmul_weights
                .keys()
                .any(|k| k.contains("linear_8")),
            "expected the 18710-class rec head to have a transposed copy"
        );
        let leaked: Vec<&String> = model
            .transposed_matmul_weights
            .keys()
            .filter(|k| model.weights.contains_key(*k))
            .collect();
        assert!(
            leaked.is_empty(),
            "FP32 originals shadowed by the transposed copy must be released, still held: {leaked:?}"
        );

        // Quantization must shrink the resident footprint, not grow it.
        let packed: Vec<String> = model.transposed_matmul_weights.keys().cloned().collect();
        assert!(!packed.is_empty(), "expected packed MatMul weights");

        let before = model.weight_footprint_bytes();
        let mut quantized = Model::load(path).expect("load rec model");

        // Report the projection head specifically: this is the tensor the README quotes.
        let head_name = model
            .transposed_matmul_weights
            .keys()
            .find(|k| k.contains("linear_8"))
            .cloned()
            .expect("rec head weight");
        let head_raw = model
            .weights
            .get(&head_name)
            .map(|t| t.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let head_packed =
            model.transposed_matmul_weights[&head_name].numel() * std::mem::size_of::<f32>();

        let base = quantized.weight_footprint_bytes();
        quantized.quantize_matmul_weights();
        let after = quantized.weight_footprint_bytes();
        let head_int8 = quantized
            .quantized_matmul_weights
            .get(&head_name)
            .map(|q| q.data.len() + q.scales.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);

        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        println!(
            "[mem] rec model resident weights: raw={:.2} MB -> after packing={:.2} MB -> \
             after INT8={:.2} MB",
            mb(base),
            mb(before),
            mb(after),
        );
        println!(
            "[mem] head '{head_name}': raw FP32 original={:.2} MB, live after packing={:.2} MB, \
             INT8 ({:.2} MB data + scales)={:.2} MB",
            mb(head_raw),
            mb(head_packed),
            head_int8 as f64 / (1024.0 * 1024.0),
            mb(head_int8),
        );
        assert!(
            head_raw == 0,
            "the head's FP32 original must be released once the transposed copy exists"
        );
        // 4x for the i8 payload, minus the per-output-row f32 scales (0.8 MB of the 8.98 MB),
        // so the honest figure is ~3.87x rather than a round 4x.
        assert!(
            head_int8 * 7 < head_packed * 2,
            "INT8 must cut the live head weights by at least 3.5x: {head_packed} -> {head_int8}"
        );
        assert!(
            after < before,
            "INT8 must reduce the footprint: {before} -> {after}"
        );
        assert!(
            after + 5 * 1024 * 1024 < before,
            "expected at least a 5 MB reduction, got {before} -> {after}"
        );

        // After quantizing, neither FP32 form may still be resident.
        assert!(
            quantized.transposed_matmul_weights.is_empty(),
            "the transposed FP32 copies must be released once quantized"
        );
        for name in &packed {
            assert!(
                quantized.quantized_matmul_weights.contains_key(name),
                "{name} should be quantized"
            );
            assert!(
                !quantized.weights.contains_key(name),
                "the FP32 original {name} must be released once quantized"
            );
        }
    }
}
