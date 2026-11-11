"""The prebuilder's plan (``python -m miniworld_engine.kernels._prebuild --list``) on a CPU node: it plans without a GPU and without
building, reports no problem, and two runs plan the same artifacts (names and content keys). ``--check`` (the plan against the code and
requirements_kernel.csv) passes.

Each run is its own process: planning is not re-entrant within one process (the loaders' ``lru_cache`` keeps the planning stand-ins, so
a second plan in the same process records nothing for them)."""

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
    return subprocess.run([sys.executable, "-m", "miniworld_engine.kernels._prebuild", "--arch", "sm90a", *args], capture_output=True,
                          text=True, env=env, timeout=300, check=False)


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
