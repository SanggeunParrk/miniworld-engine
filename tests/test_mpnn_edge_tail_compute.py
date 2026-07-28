"""Regression tests for the compute-oriented edge tail's gradient accumulators.

The bug these exist for: ``reset_to_zero`` clears whatever buffer a launch is handed,
and Triton applies it only when that launch AUTOTUNES. Two launches of the same kernel
sharing one accumulator are therefore fine right up until something changes the autotune
key so the second launch stops hitting the first one's cache entry -- at which point it
tunes, resets, and silently wipes the gradient the first launch had just accumulated.
``grad_b2`` went to exactly zero that way.

Two tests, deliberately at different levels:

* the structural one runs anywhere and states the invariant directly, so the bug class
  cannot come back through a different launch;
* the numerical one needs a GPU AND more than one autotune config, because with a single
  config Triton skips tuning entirely and never resets -- which is precisely why running
  these kernels under the CPU interpreter cannot see this failure.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch.nn.functional as F

import miniworld_kernels.kernels.mpnn_edge_tail.triton.compute as compute

_WIDTH = 128
_GRADIENT_NAMES = (
    "edge", "query", "table", "w1", "w2", "b2", "w3", "b3", "gamma", "beta",
)


def _reset_buffers_per_launch() -> list[tuple[int, str, tuple[str, ...]]]:
    """(line, kernel, buffers) for every autotuned launch inside ``_launch_backward``."""
    source = pathlib.Path(compute.__file__).read_text()
    tree = ast.parse(source)

    resets: dict[str, tuple[list[str], list[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "attr", "") == "autotune"
            ):
                continue
            for keyword in decorator.keywords:
                if keyword.arg == "reset_to_zero":
                    resets[node.name] = (
                        [ast.literal_eval(e) for e in keyword.value.elts],
                        [a.arg for a in node.args.args],
                    )

    launches = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_launch_backward"):
            continue
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript)):
                continue
            kernel = getattr(call.func.value, "id", "")
            if kernel not in resets:
                continue
            names, params = resets[kernel]
            position = {p: i for i, p in enumerate(params)}
            launches.append((
                call.lineno,
                kernel,
                tuple(
                    ast.unparse(call.args[position[n]])
                    for n in names
                    if position[n] < len(call.args)
                ),
            ))
    return launches


def test_no_reset_to_zero_buffer_is_shared_between_launches() -> None:
    launches = _reset_buffers_per_launch()
    assert launches, "found no autotuned launches to check -- the AST walk is stale"

    seen: dict[str, tuple[int, str]] = {}
    for line, kernel, buffers in launches:
        for buffer in buffers:
            if buffer in seen:
                other_line, other_kernel = seen[buffer]
                pytest.fail(
                    f"{buffer!r} is passed as a reset_to_zero accumulator to two "
                    f"launches: {other_kernel} at line {other_line} and {kernel} at "
                    f"line {line}. Whichever of the two autotunes will zero what the "
                    f"other accumulated. Give each launch its own buffer."
                )
            seen[buffer] = (line, kernel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_every_gradient_survives_autotuning(monkeypatch: pytest.MonkeyPatch) -> None:
    """More than one config, so every launch really tunes and really resets."""
    two = [
        triton_config
        for triton_config in _two_configs()
    ]
    for kernel in (
        compute._project_edge, compute._project_hidden, compute._project_output,
        compute._project_backward, compute._edge_backward,
    ):
        monkeypatch.setattr(kernel, "configs", two, raising=False)
        monkeypatch.setattr(kernel, "cache", {}, raising=False)
    monkeypatch.setattr(compute._norm_backward, "configs", _two_row_configs())
    monkeypatch.setattr(compute._norm_backward, "cache", {})

    device, dtype, eps = "cuda", torch.bfloat16, 1e-5
    nodes, neighbors = 64, 8
    rows = nodes * neighbors
    torch.manual_seed(0)
    scale = _WIDTH**-0.5

    def weight() -> torch.Tensor:
        return torch.randn(_WIDTH, _WIDTH, device=device) * scale

    params = (
        weight(), weight(), torch.zeros(_WIDTH, device=device), weight(),
        torch.zeros(_WIDTH, device=device), torch.ones(_WIDTH, device=device),
        torch.zeros(_WIDTH, device=device),
    )
    edge = torch.randn(rows, _WIDTH, device=device, dtype=dtype)
    query = torch.randn(nodes, _WIDTH, device=device, dtype=dtype)
    table = torch.randn(nodes, _WIDTH, device=device, dtype=dtype)
    index = torch.arange(rows, device=device) % nodes
    groups = torch.arange(rows, device=device) // neighbors
    dy = torch.randn(rows, _WIDTH, device=device, dtype=dtype)

    def leaves() -> list[torch.Tensor]:
        return [
            t.clone().requires_grad_(True) for t in (edge, query, table, *params)
        ]

    def reference(e, q, t, w1, w2, b2, w3, b3, gamma, beta):  # noqa: ANN001
        pre = (
            F.linear(e, w1.to(dtype)).float() + q[groups].float() + t[index].float()
        ).to(dtype)
        hidden = (
            F.linear(F.gelu(pre.float()).to(dtype), w2.to(dtype)).float() + b2
        ).to(dtype)
        update = F.linear(F.gelu(hidden.float()).to(dtype), w3.to(dtype)).float() + b3
        values = (e.float() + update).to(dtype)
        return F.layer_norm(values.float(), (_WIDTH,), gamma, beta, eps).to(dtype)

    expected_leaves = leaves()
    reference(*expected_leaves).backward(dy)
    actual_leaves = leaves()
    compute.edge_tail_compute(
        *actual_leaves[:3], index, *actual_leaves[3:], eps, 0.0
    ).backward(dy)

    for name, actual, expected in zip(
        _GRADIENT_NAMES, actual_leaves, expected_leaves, strict=True
    ):
        got, want = actual.grad, expected.grad
        assert got is not None, f"{name} received no gradient at all"
        # The sharp form of the bug: a wiped accumulator is identically zero while the
        # reference's is not. Checked before the tolerance so the failure names itself.
        assert not (got.abs().max() == 0 and want.abs().max() > 0), (
            f"{name} gradient is identically zero but the reference's is not -- an "
            f"accumulator was reset after it was filled"
        )
        relative = (
            (got.double() - want.double()).norm() / want.double().norm().clamp_min(1e-30)
        ).item()
        assert relative < 5e-2, f"{name} gradient differs by {relative:.3e}"


def _two_configs() -> list:
    import triton

    return [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    ]


def _two_row_configs() -> list:
    import triton

    return [
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=2),
    ]
