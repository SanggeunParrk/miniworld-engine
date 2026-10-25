"""The performance ladder as a test: CUDA < portable (Triton) < PyTorch, on one H100 (`benchmarks/runners/ladder.py`).

Not part of any default run: it takes minutes (every cell is compiled and CUDA-graph timed three times) and it asserts a performance property, so it
only runs when asked for -- `MINIWORLD_LADDER=1 pytest -m ladder tests/gpu/sm90/test_performance_ladder_gpu.py` (through Slurm, like any GPU job).
`MINIWORLD_LADDER_GROUPS` (a regex over the group ids of `ladder.GROUPS`, default: the smoke tier) picks the cells.

The test fails on a violation that is not listed in `docs/records/data/performance_ladder_known.txt` (one `group:mode:L:pair` per line, the debt
of the last full run): a NEW violation, or a known one that got fixed and must be taken off the list (`MINIWORLD_LADDER_STRICT=1` ignores the list).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from paths import ROOT

pytestmark = [
    pytest.mark.ladder,
    pytest.mark.skipif(os.environ.get("MINIWORLD_LADDER") != "1", reason="set MINIWORLD_LADDER=1 to run the performance ladder"),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="the ladder is measured on an H100"),
]

KNOWN = ROOT / "docs" / "records" / "data" / "performance_ladder_known.txt"


def test_the_ladder_holds(tmp_path):
    sys.path.insert(0, str(ROOT))
    from benchmarks.runners import ladder

    smoke = "|".join(g.id for g in ladder.GROUPS if g.tier == "smoke")
    pattern = os.environ.get("MINIWORLD_LADDER_GROUPS", f"^({smoke})$")
    args = ladder.build_parser().parse_args(["run", "--group", pattern, "--repeats", "3", "--out", str(tmp_path)])
    assert ladder.cmd_run(args) == 0
    table, meta = ladder.load(Path(tmp_path))
    assert table, "the ladder produced no rows"
    ladder.write_csv(table, tmp_path / "ladder.csv", meta)
    (tmp_path / "ladder.md").write_text(ladder.render_md(table, meta))
    gate = ["gate", "--csv", str(tmp_path / "ladder.csv")]
    if KNOWN.is_file():
        gate += ["--allow", str(KNOWN)]
    if os.environ.get("MINIWORLD_LADDER_STRICT") == "1":
        gate += ["--strict"]
    assert ladder.cmd_gate(ladder.build_parser().parse_args(gate)) == 0, (tmp_path / "ladder.md").read_text()[:4000]
