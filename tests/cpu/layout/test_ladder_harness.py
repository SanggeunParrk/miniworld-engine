"""The performance-ladder harness (`benchmarks/runners/ladder.py`): its grid names real bench targets, its verdict logic is the invariant.

No GPU: the ladder module is stdlib-only, `bench.py` is read with `ast` (it cannot be imported without a GPU).
"""
from __future__ import annotations

import ast
import re

import pytest
from benchmarks.runners import ladder
from paths import ROOT

BENCH = ROOT / "benchmarks" / "runners" / "bench.py"


def _module_targets() -> set[str]:
    for node in ast.walk(ast.parse(BENCH.read_text())):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) \
                and any(isinstance(t, ast.Name) and t.id == "MODULE_TARGETS" for t in node.targets):
            return {k.value for k in node.value.keys}
    raise AssertionError("MODULE_TARGETS not found")


def test_group_ids_are_unique_and_target_real_bench_modules() -> None:
    ids = [g.id for g in ladder.GROUPS]
    assert len(ids) == len(set(ids))
    assert {g.target for g in ladder.GROUPS} <= _module_targets()


def test_every_group_has_a_cuda_and_a_pytorch_rung_and_overrides_are_hydra_overrides() -> None:
    for g in ladder.GROUPS:
        assert {"pytorch", "cuda"} <= set(g.rungs), g.id
        assert all(re.match(r"\+?[A-Za-z_][\w.]*=", o) for o in g.overrides), g.id
        assert g.tier in ladder.TIERS and g.weight in (1, 2, 3)
        assert set(g.modes) <= {"inference", "training"}


def test_every_group_override_names_a_benchconfig_field_or_yaml_key() -> None:
    src = BENCH.read_text()
    fields = set(re.findall(r"^    (\w+): ", src.split("class BenchConfig")[1].split("def _resolve_module_augmentation")[0], re.M))
    for g in ladder.GROUPS:
        for o in g.overrides:
            key = o.lstrip("+").split("=")[0]
            assert key in fields, f"{g.id}: {o} is not a BenchConfig field"


def _row(cuda, portable, pytorch, kc=("a",), kp=("b",), kt=("c",)):
    return {
        "cuda_ms": cuda, "portable_ms": portable, "pytorch_ms": pytorch,
        "cuda_over_portable": cuda / portable if cuda and portable else None,
        "portable_over_pytorch": portable / pytorch if portable and pytorch else None,
        "cuda_over_pytorch": cuda / pytorch if cuda and pytorch else None,
        "noise": 0.0, "errors": {},
        "kernels": {"cuda": [[k, 1.0, 1] for k in kc], "portable": [[k, 1.0, 1] for k in kp], "pt_compiled": [[k, 1.0, 1] for k in kt]},
    }


G3 = ladder.BY_ID["tm_out_D128"]          # a three-rung group
G2 = ladder.BY_ID["swa_attn"]             # CUDA vs PyTorch only


@pytest.mark.parametrize(("cuda", "portable", "pytorch", "expect"), [
    (1.0, 2.0, 3.0, "OK"),
    (2.5, 2.0, 3.0, "VIOLATION(cuda_vs_portable)"),
    (1.0, 3.5, 3.0, "VIOLATION(portable_vs_pytorch)"),
    (4.0, 3.5, 3.0, "VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)"),
    (2.0, 2.0, 3.0, "VIOLATION(cuda_vs_portable)"),            # a tie is not "faster"
    (None, None, 3.0, "n.a."),
    (1.0, None, 3.0, "n.a."),                                  # neither pair can be formed without the portable rung
])
def test_verdict_is_the_invariant(cuda, portable, pytorch, expect) -> None:
    assert ladder.verdict(_row(cuda, portable, pytorch), G3)["verdict"] == expect


def test_a_group_without_a_portable_rung_is_judged_cuda_against_pytorch() -> None:
    assert ladder.verdict(_row(1.0, None, 3.0), G2)["verdict"] == "OK"
    assert ladder.verdict(_row(4.0, None, 3.0), G2)["verdict"] == "VIOLATION(cuda_vs_pytorch)"


def test_a_cuda_row_that_ran_the_portable_kernels_is_not_a_cuda_measurement() -> None:
    row = _row(2.5, 2.0, 3.0, kc=("x",), kp=("x",))
    out = ladder.verdict(row, G3)
    assert "VIOLATION(cuda_vs_portable)" not in out["verdict"] and "no distinct CUDA path" in out["note"]


def test_a_portable_row_that_ran_the_pytorch_kernels_is_not_a_portable_measurement() -> None:
    row = _row(1.0, 2.0, 3.0, kp=("t",), kt=("t",))
    assert "portable rung launched the PyTorch kernels" in ladder.verdict(row, G3)["note"]


def test_the_gate_fails_on_an_unlisted_violation_and_passes_when_it_is_listed(tmp_path) -> None:
    row = {**_row(2.5, 2.0, 3.0), "group": "tm_out_D128", "op": "TriMul", "mode": "inference", "dtype": "bf16", "L": 384, "shape": "D128",
           "weight": 3, "pt_compiled_ms": 3.0, "pt_eager_graph_ms": None, "pt_eager_ms": None, "pt_best": "pt_compiled", "noise": 0.0,
           "verdict": "VIOLATION(cuda_vs_portable)", "note": "", "dev": "H100", "torch": "2.13", "target": "x"}
    path = tmp_path / "ladder.csv"
    ladder.write_csv([row], path, {})
    assert ladder.cmd_gate(ladder.build_parser().parse_args(["gate", "--csv", str(path)])) == 1
    allow = tmp_path / "allow.txt"
    allow.write_text("tm_out_D128:inference:384:cuda_vs_portable\n")
    assert ladder.cmd_gate(ladder.build_parser().parse_args(["gate", "--csv", str(path), "--allow", str(allow)])) == 0
    assert ladder.cmd_gate(ladder.build_parser().parse_args(["gate", "--csv", str(path), "--allow", str(allow), "--strict"])) == 1


def test_the_cli_exposes_the_ladder() -> None:
    from miniworld_engine import cli

    args = cli.build_parser().parse_args(["ladder", "plan", "--tier", "smoke"])
    assert args.ladder_args == ["plan", "--tier", "smoke"]
