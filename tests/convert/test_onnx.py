import pytest

onnx = pytest.importorskip("onnx")

from fastretrieval.convert.onnx import fix_cast_nodes  # noqa: E402


def _graph_with_cast_in_subgraph():
    """Đồ thị có một node Cast(to=FLOAT) nằm trong subgraph của If."""
    from onnx import TensorProto, helper

    inner_cast = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT)
    inner = helper.make_graph(
        [inner_cast],
        "inner",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    outer_cast = helper.make_node("Cast", ["a"], ["b"], to=TensorProto.FLOAT)
    if_node = helper.make_node("If", ["cond"], ["z"], then_branch=inner, else_branch=inner)
    return helper.make_graph(
        [outer_cast, if_node],
        "outer",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])],
    )


def _all_cast_targets(graph):
    from onnx import GraphProto

    out = []
    for node in graph.node:
        if node.op_type == "Cast":
            out += [a.i for a in node.attribute if a.name == "to"]
        for attr in node.attribute:
            if attr.g and isinstance(attr.g, GraphProto):
                out += _all_cast_targets(attr.g)
    return out


def test_fix_cast_nodes_reaches_into_subgraphs():
    from onnx import TensorProto

    graph = _graph_with_cast_in_subgraph()
    assert TensorProto.FLOAT in _all_cast_targets(graph)

    fix_cast_nodes(graph)

    targets = _all_cast_targets(graph)
    assert targets, "no Cast nodes found, the fixture is wrong"
    assert all(t == TensorProto.FLOAT16 for t in targets), (
        "a Cast(to=FLOAT) survived, most likely one nested in an If/Loop subgraph"
    )


def test_unknown_variant_is_rejected():
    from fastretrieval.convert.onnx import convert_onnx

    with pytest.raises(ValueError, match="unknown variant"):
        convert_onnx("acme/tiny", "/tmp/out", variants=["int8", "bf8"])
