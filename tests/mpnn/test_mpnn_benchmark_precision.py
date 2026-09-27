"""The MPNN comparison's precision label must match its actual tensors/execution."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("family", ["message", "edge_mlp", "edge_tail", "edge_layernorm", "node_message", "relative_position", "edge_dropout"])
@pytest.mark.parametrize("precision", ["bf16", "bf16-mixed"])
def test_comparison_uses_requested_parameter_precision(family, precision):
    from benchmarks.runners.mpnn_compare import make_case

    forward, leaves, _, dtypes = make_case(family, 64, "pytorch", True, precision)
    assert not torch.is_autocast_enabled("cuda")
    result = forward(0.0)
    result.float().square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in leaves)
    norm_count = 2 if family in {"edge_layernorm", "edge_tail"} else 0
    expected_fp32 = (
        norm_count
        if precision == "bf16"
        else {
            "message": 2,
            "edge_mlp": 4,
            "edge_tail": 7,
            "edge_layernorm": 3,
            "node_message": 3,
            "relative_position": 2,
            "edge_dropout": 0,
        }[family]
    )
    assert sum(t.dtype == torch.float32 for t in leaves) == expected_fp32
    if precision == "bf16":
        assert dtypes["autocast_enabled"] is False
        if family not in {"edge_layernorm", "edge_dropout"}:
            assert "bfloat16" in dtypes["parameter_dtype"]
    else:
        assert dtypes["autocast_enabled"] is True
        assert dtypes["parameter_dtype"] == (
            "" if family == "edge_dropout" else "float32"
        )


@pytest.mark.parametrize("training", [False, True])
def test_native_comparison_observes_compile_and_graph(training):
    from benchmarks.runners.measurement import benchmark_source_hash
    from benchmarks.runners.mpnn_compare import evaluate

    result = evaluate(
        "edge_tail",
        64,
        "pytorch",
        training,
        1,
        True,
        benchmark_source_hash(),
        "time",
        "bf16",
    )
    assert result["status"] == "ok"
    assert result["accuracy_pass"]
    assert result["parameter_dtype"] == "bfloat16+float32"
    assert result["autocast_enabled"] is False
    sample = result["samples"][0]
    assert sample["compiled"] is True
    assert sample["cudagraph"] == ("disabled" if training else "manual")
