"""No launcher may bypass its own autotuner, and every pinned config must be on its ladder.

`Autotuner.fn[grid](...)` launches the wrapped JIT function directly. It does not prune, does not
bench, does not record -- the tuning round never happens. Two launchers did this, and the cost was
not the one the comments anticipated:

  * `_bwd_x` forced the covering tile at N <= 1024, and `_dgrad_condln` forced a large-M config at
    M >= 8192. Both were argued from "the driver tunes at 512 rows while production runs 36864".
  * `76daae51` made the driver tune at `max(L, 8192)` rows, which removed that premise -- and made
    the driver's own M cross BOTH thresholds, so every build unit of both kernels took the bypass.
  * The result: every unit came out EMPTY (10 s, 0 ops, "nothing captured"), neither op had a
    usable cache entry on any card, and A5000-measured constants with no arch condition shipped
    everywhere. A build reported success the whole time.
  * Neither pinned config was even reachable by tuning -- `_bwd_x` pinned `num_warps=8` against a
    ladder of `1 2 4`, `_dgrad_condln` pinned `BLOCK_K_NC=128` against `16 32 64`.

The second half is the part a rule can state: a config good enough to hard-code is a config that
belongs in the search space, where the sweep can find it, price it, and record it per shape.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "src" / "miniworld_engine" / "kernels"
GRID = ROOT / "src" / "miniworld_engine" / "autotune" / "configs" / "grid"

#: A launcher may still reach `.fn` for something that is not a launch (introspection, a warmup
#: that deliberately compiles one config). Nothing needs that today; add a name here WITH the
#: measurement that justifies it, not to make this test pass.
ALLOWED: frozenset[str] = frozenset()


def _sources() -> list[Path]:
    return sorted(p for p in KERNELS.rglob("*.py") if "notes" not in p.parts)


def test_no_launcher_calls_fn_and_skips_the_tuner() -> None:
    hits = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text())):
            # `<something>.fn[...]` used as a call target: `kernel.fn[grid](...)`.
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
                continue
            sub = node.func.value
            if isinstance(sub, ast.Attribute) and sub.attr == "fn":
                name = getattr(sub.value, "id", "?")
                if name in ALLOWED:
                    continue
                rel = path.relative_to(ROOT)
                hits.append(f"{rel}:{node.lineno}: {name}.fn[...] launches past the autotuner")
    assert not hits, (
        "a launcher bypasses its own autotuner -- the tuning round never runs, the op records "
        "nothing, and every build of it comes out EMPTY with no cache entry on any card:\n  "
        + "\n  ".join(hits))


def _ladders(csv_path: Path) -> dict[str, list[int]]:
    out = {}
    for line in csv_path.read_text().splitlines()[1:]:
        if "," not in line:
            continue
        axis, values = line.split(",", 1)
        out[axis.strip()] = [int(v) for v in values.split() if v.strip().isdigit()]
    return out


#: Constants a launcher passes to a kernel that IS autotuned, as (op, axis, value). Kept as data
#: rather than derived: a launcher may legitimately compute a value (`BLOCK_K = next_pow2(N)`),
#: and only a literal can be checked against a ladder.
_PIN = re.compile(r"(BLOCK_[A-Z0-9_]+|num_warps|num_stages)\s*=\s*(\d+)")


@pytest.mark.parametrize("csv_path", sorted(GRID.glob("*.csv")), ids=lambda p: p.stem)
def test_every_ladder_value_is_a_number_the_axis_can_take(csv_path: Path) -> None:
    """A ladder with no rungs cannot be searched, and an empty axis is how a grid silently
    collapses to triton's substitute config."""
    ladders = _ladders(csv_path)
    assert ladders, f"{csv_path.name} declares no axes"
    empty = [a for a, v in ladders.items() if not v]
    assert not empty, f"{csv_path.name}: axes with no values: {empty}"


def test_the_two_configs_the_bypasses_pinned_are_on_their_ladders() -> None:
    """The specific regression. Both were outside their own search space, which is why removing
    the bypass without widening the ladder would have handed production a config nobody chose."""
    warps = _ladders(GRID / "adaln_bwd_pre_dx_triton.csv")["num_warps"]
    assert 8 in warps, (
        "adaln_bwd_pre_dx's launcher used to force num_warps=8; with it off the ladder the sweep "
        "cannot reproduce the config that was believed to win")
    nc = _ladders(GRID / "adaln_bwd_dx_dlnw_triton.csv")["BLOCK_K_NC"]
    assert 128 in nc, (
        "adaln_bwd_dx_dlnw's launcher used to force BLOCK_K_NC=128, which its ladder did not offer")
