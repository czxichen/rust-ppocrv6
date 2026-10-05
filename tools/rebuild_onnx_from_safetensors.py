#!/usr/bin/env python3
"""Rebuild a runnable .onnx model from a rust_ppocrv6 safetensors bundle.

The safetensors files produced by tools/convert_onnx_to_safetensors.py keep the
original ONNX graph topology in __metadata__["graph"], so a faithful ONNX model
can be reconstructed. This exists purely to enable apples-to-apples latency
comparison against ONNX Runtime: the Rust engine and ORT then execute the exact
same graph with the exact same weights.

Usage:
    python tools/rebuild_onnx_from_safetensors.py <in.safetensors> <out.onnx>
"""

import json
import struct
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def read_safetensors(path):
    """Return (header_dict, tensor_name -> np.ndarray)."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        body = fh.read()

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = info["dtype"]
        shape = info["shape"]
        if dtype == "F32":
            arr = np.frombuffer(body[start:end], dtype="<f4")
        elif dtype == "I64":
            arr = np.frombuffer(body[start:end], dtype="<i8")
        elif dtype == "I32":
            arr = np.frombuffer(body[start:end], dtype="<i4")
        elif dtype == "U8":
            arr = np.frombuffer(body[start:end], dtype="<u1")
        elif dtype == "BOOL":
            arr = np.frombuffer(body[start:end], dtype="?")
        else:
            raise ValueError(f"{name}: unsupported dtype {dtype}")
        tensors[name] = arr.reshape(shape).copy()
    return header.get("__metadata__", {}), tensors


def numpy_to_onnx_dtype(arr):
    return {
        np.dtype("float32"): TensorProto.FLOAT,
        np.dtype("int64"): TensorProto.INT64,
        np.dtype("int32"): TensorProto.INT32,
        np.dtype("uint8"): TensorProto.UINT8,
        np.dtype("bool"): TensorProto.BOOL,
    }[arr.dtype]


def infer_opset(graph):
    """Pick the ONNX opset whose semantics match how this graph was serialised.

    The converter dumped attributes verbatim from the original model, so the
    dialect is detectable from what it emitted:
      - BatchNormalization carrying `spatial` means opset < 9, since 9 removed
        the attribute (spatial=1, its default, was the only supported mode).
      - Squeeze/Unsqueeze carrying an `axes` attribute means opset < 13,
        because 13 moved axes to an input tensor.
      - Resize taking only (X, scales) means opset < 11, because 11 added
        roi and moved scales to the 3rd input.
    Getting this wrong makes ORT either reject the model or silently mis-read
    the Resize signature, so it must be derived rather than hard-coded.
    """
    for nd in graph["nodes"]:
        attrs = nd.get("attrs") or {}
        if nd["op"] == "BatchNormalization" and "spatial" in attrs:
            return 9
        if nd["op"] in ("Squeeze", "Unsqueeze") and "axes" in attrs:
            return 11
        if nd["op"] == "Resize" and len(nd["inputs"]) == 2:
            return 10
    return 13


# Attributes that were valid in older opsets but were dropped later. Keeping
# them makes onnx.checker reject the rebuilt model, and their values are the
# only ones the newer spec supports anyway, so dropping them is lossless.
OBSOLETE_ATTRS = {
    "BatchNormalization": {"spatial"},
}


def build(graph, tensors, opset=None):
    if opset is None:
        opset = infer_opset(graph)
    nodes = []
    for nd in graph["nodes"]:
        attrs = dict(nd.get("attrs") or {})
        for stale in OBSOLETE_ATTRS.get(nd["op"], ()):
            attrs.pop(stale, None)
        # Every non-Constant node in this graph takes its values from
        # initializers, which we register separately; there are no Constant nodes
        # to inline because the converter pre-extracted them into tensors.
        nodes.append(
            helper.make_node(
                nd["op"],
                nd["inputs"],
                nd["outputs"],
                name=nd.get("name") or nd["outputs"][0],
                **attrs,
            )
        )

    initializers = []
    for name, arr in tensors.items():
        if arr.dtype == np.dtype("O"):
            continue
        initializers.append(
            numpy_helper.from_array(np.ascontiguousarray(arr), name=name)
        )

    # Graph inputs: only the real runtime input. Everything else the graph
    # consumes (Shape/Slice helpers, Resize sizes/scales) is an initializer.
    init_names = {i.name for i in initializers}
    graph_inputs = []
    for nm in graph["inputs"]:
        if nm in init_names:
            continue
        # NCHW float input; spatial dims are dynamic (the engine feeds whatever
        # the preprocessing produced).
        graph_inputs.append(
            helper.make_tensor_value_info(nm, TensorProto.FLOAT, ["N", 3, "H", "W"])
        )

    # The engine's graph emits untyped outputs; declare them as float tensors
    # with symbolic dims so the checker accepts the model.
    graph_outputs = []
    for o in graph["outputs"]:
        graph_outputs.append(
            helper.make_tensor_value_info(o, TensorProto.FLOAT, ["N", "C", "H", "W"])
        )

    made = helper.make_graph(
        nodes, "ppocrv6", graph_inputs, graph_outputs, initializer=initializers
    )
    model = helper.make_model(
        made, opset_imports=[helper.make_opsetid("", opset)], producer_name="rebuild"
    )
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    return model


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]

    meta, tensors = read_safetensors(src)
    graph = json.loads(meta["graph"])
    opset = infer_opset(graph)
    print(
        f"{src}: {len(graph['nodes'])} nodes, {len(tensors)} tensors, "
        f"source={meta.get('source')}, opset={opset}"
    )

    model = build(graph, tensors, opset=opset)
    onnx.save(model, dst)
    size_mb = onnx.load(dst).ByteSize() / 1e6
    print(f"wrote {dst} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
