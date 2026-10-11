"""Verify CUDA ONNX graphs, permitting only proven CPU shape bookkeeping."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path


def verified_cuda_graph(session):
    """Return evidence only while its native session and no-fallback policy hold."""
    report = getattr(session, "_edumind_cuda_graph", None)
    native = getattr(session, "_sess", None)
    if (
        isinstance(report, dict)
        and report.get("verification") == "optimized-ort-graph-v1"
        and native is not None
        and getattr(session, "_edumind_verified_native_session", None) is native
        and not getattr(session, "_enable_fallback", True)
    ):
        return report
    return None


def cuda_shape_session(original):
    """Create and inspect the exact session used for inference, without a probe."""
    import onnxruntime as ort
    from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession import (
        InferenceSession,
    )

    options = original.get_session_options()
    # CPU graph nodes are permitted only after inspecting their actual dataflow.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    options.add_session_config_entry("session.save_model_format", "ORT")
    with tempfile.TemporaryDirectory(prefix="onnx-placement-") as directory:
        path = Path(directory) / "graph.ort"
        options.optimized_model_filepath = str(path)
        session = ort.InferenceSession(
            original._model_path,
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )
        session.disable_fallback()  # Never retry a failed CUDA run on CPU.
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX CUDA provider did not initialize")
        data = bytearray(path.read_bytes())
        if not InferenceSession.InferenceSessionBufferHasIdentifier(data, 0):
            raise RuntimeError("ONNX CUDA placement graph is unavailable or invalid")
        graph = InferenceSession.GetRootAsInferenceSession(data, 0).Model().Graph()
        nodes, constants, outputs = _describe_graph(graph)
        report = verify_shape_only_cpu_nodes(nodes, constants, outputs)
        report["graph_sha256"] = hashlib.sha256(data).hexdigest()
        report["model_file"] = Path(original._model_path).name
    session._edumind_cuda_graph = report
    # Bind the evidence to the native session: set_providers() recreates it.
    session._edumind_verified_native_session = session._sess
    return session


def _describe_graph(graph):
    from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.AttributeType import (
        AttributeType,
    )
    from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.TensorDataType import (
        TensorDataType,
    )

    nodes = []
    for index in range(graph.NodesLength()):
        node = graph.Nodes(index)
        if any(
            node.Attributes(i).Type() in (AttributeType.GRAPH, AttributeType.GRAPHS)
            for i in range(node.AttributesLength())
        ):
            raise RuntimeError("ONNX CUDA placement cannot verify nested graphs")
        nodes.append(
            {
                "name": node.Name().decode(),
                "op_type": node.OpType().decode(),
                "domain": (node.Domain() or b"").decode(),
                "provider": (node.ExecutionProviderType() or b"").decode(),
                "inputs": [node.Inputs(i).decode() for i in range(node.InputsLength())],
                "outputs": [
                    node.Outputs(i).decode() for i in range(node.OutputsLength())
                ],
            }
        )
    constants = {
        tensor.Name().decode()
        for i in range(graph.InitializersLength())
        if (tensor := graph.Initializers(i)).DataType()
        in (TensorDataType.INT32, TensorDataType.INT64)
        and tensor.DimsLength() <= 1
    }
    outputs = {graph.Outputs(i).decode() for i in range(graph.OutputsLength())}
    return nodes, constants, outputs


def verify_shape_only_cpu_nodes(nodes, integer_constants, graph_outputs):
    """Prove shape provenance AND use; operator names alone are insufficient.

    Deliberately conservative: unfamiliar shape operators/consumers fail closed.
    Constants cannot establish shape provenance without a Shape-derived input.
    """
    producers = {output: node for node in nodes for output in node["outputs"]}
    shape_values = {}

    def is_shape(value, visiting=frozenset()):
        if value in shape_values:
            return shape_values[value]
        if value in visiting or value not in producers:
            return False
        node = producers[value]
        inputs, op = node["inputs"], node["op_type"]
        following = visiting | {value}
        proven = False
        if node["domain"] in ("", "ai.onnx"):
            if op == "Shape":
                proven = True  # Reads dimensions, not tensor values or weights.
            elif op == "Slice" and len(inputs) >= 3:
                proven = is_shape(inputs[0], following) and all(
                    not item or item in integer_constants for item in inputs[1:]
                )
            elif op == "Concat" and inputs:
                proven = all(
                    item in integer_constants or is_shape(item, following)
                    for item in inputs
                ) and any(is_shape(item, following) for item in inputs)
        shape_values[value] = proven
        return proven

    cuda_count, cpu_nodes = 0, []
    for node in nodes:
        provider = node["provider"]
        if provider == "CUDAExecutionProvider":
            cuda_count += 1
            continue
        if provider != "CPUExecutionProvider" or not node["outputs"]:
            raise RuntimeError(f"Unverifiable ONNX CUDA placement: {node['name']}")
        if not all(is_shape(value) for value in node["outputs"]):
            raise RuntimeError(
                f"CPU model computation is prohibited: {node['name']} ({node['op_type']})"
            )
        for value in node["outputs"]:
            if value in graph_outputs:
                raise RuntimeError(f"CPU shape value is a model output: {value}")
            for consumer in nodes:
                for index, argument in enumerate(consumer["inputs"]):
                    if argument != value:
                        continue
                    shape_consumer = bool(consumer["outputs"]) and all(
                        is_shape(v) for v in consumer["outputs"]
                    )
                    reshape_argument = (
                        consumer["domain"] in ("", "ai.onnx")
                        and consumer["op_type"] == "Reshape"
                        and index == 1
                        and consumer["provider"] == "CUDAExecutionProvider"
                    )
                    if not (shape_consumer or reshape_argument):
                        raise RuntimeError(
                            f"CPU shape value feeds model data: {value} -> {consumer['name']}"
                        )
        cpu_nodes.append(
            {key: node[key] for key in ("name", "op_type", "inputs", "outputs")}
        )
    if not cuda_count:
        raise RuntimeError("ONNX graph contains no verified CUDA computation")
    return {
        "verification": "optimized-ort-graph-v1",
        "node_count": len(nodes),
        "cuda_node_count": cuda_count,
        "cpu_shape_nodes": cpu_nodes,
    }
