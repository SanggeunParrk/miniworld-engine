"""Paired training regimes and stochastic replay must not silently change the workload."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from benchmarks.runners.bench_policy import graph_regimes
from benchmarks.runners.measurement import check_stochastic_graph_replay

ROOT = Path(__file__).resolve().parents[2]


def test_auto_training_runs_both_regimes_even_if_first_fails(monkeypatch):
    script = ROOT / "benchmarks/runners/bench.py"
    tree = ast.parse(script.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node.decorator_list = []
    node.args.args[0].annotation = None
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(script), "target=triangle_attention", "level=module",
                                     "mode=training", "compile=true", "cudagraph=auto"])
    namespace: dict[str, Any] = {"sys": sys, "Path": Path, "__file__": str(script), "__name__": __name__}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), namespace)
    with pytest.raises(SystemExit) as caught:
        namespace["main"](SimpleNamespace(level="module", mode="training", metric="time", cudagraph="auto"))
    assert caught.value.code == 1
    assert [cmd[-1] for cmd in calls] == ["cudagraph=disabled", "cudagraph=manual"]
    assert all("compile=true" in cmd and "cudagraph=auto" not in cmd for cmd in calls)


def test_explicit_regimes_memory_and_kernel_training_remain_single_runs():
    for regime in ("disabled", "manual", "graphed"):
        assert graph_regimes("module", "training", "time", regime) == (regime,)
    assert graph_regimes("module", "training", "memory", "auto") == ("auto",)
    assert graph_regimes("kernel", "training", "time", "auto") == ("auto",)
    assert graph_regimes("module", "inference", "time", "auto") == ("auto",)


def stochastic_steps(*, frozen=False, corrupt_gradient=False):
    captured: dict[str, Any] = {"result": torch.empty(256), "gradients": (torch.empty(256),)}

    def step():
        mask = (torch.rand(256) > 0.25).float() / 0.75
        return {"result": mask * 2, "gradients": (mask,)}

    fixed = step()

    def replay():
        value = fixed if frozen else step()
        captured["result"].copy_(value["result"])
        captured["gradients"][0].copy_(value["gradients"][0] + int(corrupt_gradient))

    return step, replay, captured


def test_stochastic_replay_checks_outputs_gradients_and_preserves_rng():
    step, replay, captured = stochastic_steps()
    state = torch.random.get_rng_state().clone()
    result = check_stochastic_graph_replay(step, replay, captured)
    assert result["dropout_replay_changes_output"]
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("fault", ["frozen", "corrupt_gradient"])
def test_bad_stochastic_replay_is_rejected(fault):
    step, replay, captured = stochastic_steps(**{fault: True})
    with pytest.raises(RuntimeError, match="timed execution differs"):
        check_stochastic_graph_replay(step, replay, captured)


def test_replay_reseeding_every_call_is_rejected_after_seeded_checks():
    step, replay, captured = stochastic_steps()

    def frozen_offset_replay():
        torch.manual_seed(torch.initial_seed())
        replay()

    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="did not advance"):
        check_stochastic_graph_replay(step, frozen_offset_replay, captured)
    assert torch.equal(state, torch.random.get_rng_state())
