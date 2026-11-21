"""The prebuilder's plan (``python -m miniworld_engine.kernels._prebuild --list``) on a CPU node: it plans without a GPU and without
building, reports no problem, and two runs plan the same artifacts (names and content keys). ``--check`` (the plan against the code and
requirements_kernel.csv) passes.

The --list / --check runs are separate processes (as the CLI is used); ``test_plan_is_reentrant`` plans twice in this one: a plan
clears the loader caches that kept its stand-ins when it ends (``_nvcc.plan_session``), so the second plan records the same."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SRC = Path(__file__).resolve().parents[2] / "src"
ENTRY = re.compile(r"^\[prebuild\]\s+(have|todo) (\S+)\s+(\S+)\s+\[([0-9a-f]{8})\]\s+(\S+)$")


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "miniworld_engine.kernels._prebuild", "--arch", "sm90a", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def _plan(out: str) -> list[tuple]:
    return [m.group(2, 3, 4, 5) for line in out.splitlines() if (m := ENTRY.match(line))]


@pytest.fixture(scope="module")
def runs():
    return _run("--list"), _run("--check")


def test_list_plans_without_problems(runs):
    listed, _ = runs
    assert listed.returncode == 0, listed.stdout[-2000:] + listed.stderr[-2000:]
    assert "problem:" not in listed.stdout
    head = next(line for line in listed.stdout.splitlines() if "plan for sm90" in line)
    m = re.search(r"plan for sm90: (\d+) artifacts", head)
    assert m is not None, head
    n = int(m.group(1))
    assert n > 100
    assert len(_plan(listed.stdout)) == n


def test_plan_is_stable(runs):
    listed, checked = runs
    assert _plan(listed.stdout) == _plan(checked.stdout)


def test_check_runs(runs):
    """``--check`` is a per-arch inventory (an sm90 plan leaves the sm80 / sm100 call sites and sources unreached, and its name-token
    match of requirements_kernel.csv rows is a heuristic), so its exit status is a finding count, not a verdict: it must complete its
    report and find no recipe problem."""
    _, checked = runs
    assert "requirements_kernel.csv:" in checked.stdout, checked.stdout[-2000:] + checked.stderr[-2000:]
    assert "problem:" not in checked.stdout


def test_plan_is_reentrant():
    """Two plans in one process record the same artifacts: the ``cast`` family's loader is an ``lru_cache``, which kept the first
    plan's stand-in and made the second plan record nothing before ``plan_session`` cleared it."""
    from miniworld_engine.kernels import _nvcc, _prebuild

    _nvcc.ensure_cuda_home()  # as the CLI does before planning
    t = _prebuild.target("sm90a")
    first, problems = _prebuild.plan(t, ["cast"])
    second, problems2 = _prebuild.plan(t, ["cast"])
    assert not problems and not problems2, problems + problems2
    assert first, "the cast family plans nothing on sm90a"
    assert [s.key for s in first] == [s.key for s in second]


def _list_names(arch: str) -> set[str]:
    env = {**os.environ, "PYTHONPATH": f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, "-m", "miniworld_engine.kernels._prebuild", "--arch", arch, "--list"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return {name for _kind, name, _key, _fam in _plan(done.stdout)}


def test_a100_plan_ships_fused_sm80_transition():
    """``transition_fused_sm80`` (the A100 one-kernel D 128 / n 4 Transition, rows from FUSED_MIN_ROWS) is reached by bf16 parameters, which the requirements
    carry as master rows: it was missing from the 3.0.0 .. 3.0.5 kernels wheels (a ``KernelBuildError`` on an installed A100). The RTX 5090 does not serve it (shared memory)."""
    assert "transition_fused_sm80" in _list_names("sm80")
    assert "transition_fused_sm80" not in _list_names("sm120a")


def test_h100_plan_ships_ieee_f32_sm90():
    """The H100 fp32 Transition with TF32 off runs the IEEE fp32 chain (``kernels/ieee_f32`` built for sm_90a; every float32 requirements row reaches it): the
    sm90a plan carries ``ieee_f32_sm90`` so the wheel ships it (no first-call compile), and the other cards' plans keep their own builds."""
    assert "ieee_f32_sm90" in _list_names("sm90a")
    assert "ieee_f32_sm90" not in _list_names("sm80")
