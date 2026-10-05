#!/usr/bin/env python3
"""
Convert ONNX models (PP-OCR Det, Rec, Cls) to a unified SafeTensors format.
The SafeTensors file contains:
  1. All weight tensors and constants (float32 and int64).
  2. The computation graph topology and attributes in the metadata header (__metadata__["graph"]).
"""

import sys
import os
import json
import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

def convert_onnx_to_safetensors(onnx_path: str, output_path: str):
    print(f"Loading {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph

    tensors = {}
    nodes = []

    # 1. Extract graph initializers
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        # Ensure contiguous memory layout
        arr = np.ascontiguousarray(arr)
        tensors[init.name] = arr

    # 2. Extract nodes and Constant node values
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    arr = np.ascontiguousarray(arr)
                    tensors[node.output[0]] = arr
        else:
            attrs = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    attrs[attr.name] = int(attr.i)
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs[attr.name] = [int(x) for x in attr.ints]
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs[attr.name] = float(attr.f)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attrs[attr.name] = [float(x) for x in attr.floats]
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs[attr.name] = attr.s.decode("utf-8")
                elif attr.type == onnx.AttributeProto.STRINGS:
                    attrs[attr.name] = [s.decode("utf-8") for s in attr.strings]

            nodes.append({
                "name": node.name if node.name else node.output[0],
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": attrs
            })

    # Graph inputs and outputs
    inputs = [inp.name for inp in graph.input if inp.name not in tensors]
    outputs = [out.name for out in graph.output]

    graph_meta = {
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes
    }

    metadata = {
        "graph": json.dumps(graph_meta),
        "source": os.path.basename(onnx_path),
        "num_nodes": str(len(nodes)),
        "num_tensors": str(len(tensors))
    }

    print(f"Saving to {output_path} (Tensors: {len(tensors)}, Executable nodes: {len(nodes)})...")
    save_file(tensors, output_path, metadata=metadata)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: convert_onnx_to_safetensors.py <input.onnx> <output.safetensors>")
        sys.exit(1)
    convert_onnx_to_safetensors(sys.argv[1], sys.argv[2])
