"""CPU shape exceptions must never permit feature processing or learned layers."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from edumind.common.onnx_placement import _describe_graph, verify_shape_only_cpu_nodes


def node(name, op, inputs, outputs, provider="CUDAExecutionProvider", domain=""):
    return {
        "name": name,
        "op_type": op,
        "inputs": inputs,
        "outputs": outputs,
        "provider": provider,
        "domain": domain,
    }


@pytest.fixture
def graph():
    return [
        node("features", "MatMul", ["image", "weights"], ["features"]),
        node("shape", "Shape", ["features"], ["dimensions"]),
        node(
            "slice",
            "Slice",
            ["dimensions", "start", "end"],
            ["tail"],
            "CPUExecutionProvider",
        ),
        node(
            "concat",
            "Concat",
            ["tail", "batch"],
            ["target"],
            "CPUExecutionProvider",
        ),
        node("reshape", "Reshape", ["features", "target"], ["prediction"]),
    ]


def verify(graph):
    return verify_shape_only_cpu_nodes(graph, {"start", "end", "batch"}, {"prediction"})


def test_cpu_shape_chain_is_allowed_and_reported(graph):
    report = verify(graph)
    assert report["cuda_node_count"] == 3
    assert report["node_count"] == 5
    assert [n["name"] for n in report["cpu_shape_nodes"]] == ["slice", "concat"]


def test_cpu_shape_root_is_also_permitted(graph):
    graph[1]["provider"] = "CPUExecutionProvider"
    assert [n["name"] for n in verify(graph)["cpu_shape_nodes"]] == [
        "shape",
        "slice",
        "concat",
    ]


def test_graph_reader_excludes_float_and_matrix_initializers():
    tensors = [
        SimpleNamespace(
            Name=lambda: b"integer", DataType=lambda: 7, DimsLength=lambda: 1
        ),
        SimpleNamespace(
            Name=lambda: b"weight", DataType=lambda: 1, DimsLength=lambda: 1
        ),
        SimpleNamespace(
            Name=lambda: b"matrix", DataType=lambda: 7, DimsLength=lambda: 2
        ),
    ]
    graph = SimpleNamespace(
        NodesLength=lambda: 0,
        InitializersLength=lambda: len(tensors),
        Initializers=lambda i: tensors[i],
        OutputsLength=lambda: 0,
    )
    assert _describe_graph(graph) == ([], {"integer"}, set())


def test_graph_reader_rejects_uninspected_nested_graphs():
    graph = SimpleNamespace(
        NodesLength=lambda: 1,
        Nodes=lambda i: SimpleNamespace(
            AttributesLength=lambda: 1,
            Attributes=lambda j: SimpleNamespace(
                Type=lambda: 5
            ),  # ORT GRAPH attribute.
        ),
    )
    with pytest.raises(RuntimeError, match="nested graphs"):
        _describe_graph(graph)


@pytest.mark.parametrize(
    "mutation",
    [
        "cpu_layer",
        "slice_features",
        "concat_features",
        "only_constants",
        "shape_as_features",
        "shape_as_output",
        "unknown_provider",
        "custom_shape",
    ],
)
def test_cpu_model_computation_and_unproven_shape_uses_are_rejected(graph, mutation):
    graph = deepcopy(graph)
    if mutation == "cpu_layer":
        graph[0]["provider"] = "CPUExecutionProvider"
    elif mutation == "slice_features":
        graph[2]["inputs"][0] = "features"
    elif mutation == "concat_features":
        graph[3]["inputs"][1] = "features"
    elif mutation == "only_constants":
        graph[3]["inputs"] = ["start", "end"]
    elif mutation == "shape_as_features":
        graph[4] = node(
            "fake_features", "MatMul", ["target", "weights"], ["prediction"]
        )
    elif mutation == "shape_as_output":
        graph[3]["outputs"] = ["prediction"]
    elif mutation == "unknown_provider":
        graph[0]["provider"] = ""
    elif mutation == "custom_shape":
        graph[1]["domain"] = "custom"
    with pytest.raises(RuntimeError):
        verify(graph)


def test_entirely_cpu_graph_does_not_qualify_as_cuda():
    with pytest.raises(RuntimeError, match="no verified CUDA"):
        verify_shape_only_cpu_nodes(
            [node("shape", "Shape", ["input"], ["dims"], "CPUExecutionProvider")],
            set(),
            set(),
        )


def test_cuda_only_graph_needs_no_cpu_exceptions(graph):
    for entry in graph:
        entry["provider"] = "CUDAExecutionProvider"
    assert verify(graph)["cpu_shape_nodes"] == []
