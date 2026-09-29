"""A recovered `entry_grids` stamp must be a space the entry could actually have been measured with.

`space_backfill` reads each cache's history and stamps every entry with the grid the build that
wrote it was holding. `_first_seen` pins a key to the `config_space_hash` of the commit the key
FIRST appeared in -- and `store_ranked_configs` used to RESET a file when the grid moved and
re-measure the same keys against the new one. A reset does not rename a key, so `setdefault` never
saw the re-measurement and the entry was stamped with a grid nothing had been measured against.

Measured on the committed corpus after that backfill ran: 225 entries across 11 A100 caches whose
own stored winners were NOT in the space they were stamped with -- 140 of them not yet invalidated
by a `build_rev` bump, together claiming 46,560 configs as already searched. That is the exact
failure the per-entry record exists to prevent, produced by the code that writes it.

The entry's own winners are the check. A cache entry stores the configs that WON for that shape; if
one of them is outside the stamped space, the file is contradicting itself and the stamp is wrong
however plausible its provenance looked. Dropping a stamp costs a re-measurement. Keeping a wrong
one costs a winner nobody timed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache as C
from miniworld_engine.autotune import space_backfill as SB

DATA = Path(__file__).resolve().parents[2] / "src/miniworld_engine/autotune/data"


def _spaces(data: dict) -> dict:
    """entry key -> the set of config signatures it is stamped with."""
    refs, grids = data.get("entry_grids") or {}, data.get("grids") or {}
    out = {}
    for key, hs in refs.items():
        hs = [hs] if isinstance(hs, str) else list(hs or [])
        space: set = set()
        for h in hs:
            space |= set(grids.get(h) or [])
        if space:
            out[key] = space
    return out


def test_no_committed_entry_is_stamped_with_a_space_its_winners_are_outside_of() -> None:
    """The corpus-wide invariant. This is a property of the shipped data, not of a fixture: if a
    future backfill or merge reintroduces the bug, this fails on the committed files."""
    bad, stamped = [], 0
    for f in sorted(DATA.glob("*/*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        spaces = _spaces(data)
        for key, ranked in (data.get("entries") or {}).items():
            space = spaces.get(key)
            if not space or not ranked:
                continue
            stamped += 1
            outside = [c for c in ranked if repr(C._sig_from_dict(c)) not in space]
            if outside:
                bad.append(f"{f.parent.name} [{f.stem}] {key}: "
                           f"{len(outside)} of {len(ranked)} winners outside the stamped space")
    if not stamped:
        pytest.skip("no entry carries a recovered space yet")
    assert not bad, (
        f"{len(bad)} entrie(s) are stamped with a space they cannot have been measured with -- a "
        f"rebuild will skip configs nothing ever timed:\n  " + "\n  ".join(bad[:10]))


def test_the_backfill_refuses_a_stamp_the_entry_contradicts() -> None:
    ranked = [{"kwargs": {"BLOCK_M": 64}, "num_warps": 4, "num_stages": 2, "ms": 1.0}]
    inside = repr(C._sig_from_dict(ranked[0]))
    assert SB._contains_its_own_winners([inside], ranked)
    assert SB._contains_its_own_winners([inside, "something-else"], ranked)
    assert not SB._contains_its_own_winners(["something-else"], ranked), (
        "a space missing the entry's own winner was accepted")
    assert SB._contains_its_own_winners([], []), "an entry with no winners has nothing to contradict"


def test_the_repair_drops_exactly_the_contradicted_stamps(tmp_path, monkeypatch) -> None:
    """`backfill` skips a file whose keys are all stamped, so a bad stamp is invisible to it
    forever. `repair` is what can still reach one."""
    good = {"kwargs": {"BLOCK_M": 64}, "num_warps": 4, "num_stages": 2, "ms": 1.0}
    other = {"kwargs": {"BLOCK_M": 32}, "num_warps": 1, "num_stages": 1, "ms": 2.0}
    g_sig, o_sig = repr(C._sig_from_dict(good)), repr(C._sig_from_dict(other))
    d = tmp_path / "op_probe"
    d.mkdir()
    (d / "GPU.json").write_text(json.dumps({
        "entries": {"bf16|keep": [good], "bf16|drop": [good]},
        "grids": {"h_ok": [g_sig, o_sig], "h_bad": [o_sig]},
        "entry_grids": {"bf16|keep": ["h_ok"], "bf16|drop": ["h_bad"]},
    }))
    monkeypatch.setattr(SB, "_DATA", tmp_path)

    rows = SB.repair(apply=True)
    assert rows == [("op_probe", "GPU", 1)]
    after = json.loads((d / "GPU.json").read_text())
    assert set(after["entry_grids"]) == {"bf16|keep"}
    assert set(after["grids"]) == {"h_ok"}, "a grid nothing points at any more was kept"


def test_a_file_whose_every_stamp_is_wrong_loses_the_fields_entirely(tmp_path, monkeypatch) -> None:
    """No `entry_grids` means 'measure it all', which is the correct state for a file that can
    prove nothing -- an empty dict would read the same but leave a field asserting otherwise."""
    good = {"kwargs": {"BLOCK_M": 64}, "num_warps": 4, "num_stages": 2, "ms": 1.0}
    d = tmp_path / "op_probe"
    d.mkdir()
    (d / "GPU.json").write_text(json.dumps({
        "entries": {"bf16|a": [good]},
        "grids": {"h": ["not-the-winner"]},
        "entry_grids": {"bf16|a": ["h"]},
    }))
    monkeypatch.setattr(SB, "_DATA", tmp_path)
    SB.repair(apply=True)
    after = json.loads((d / "GPU.json").read_text())
    assert "entry_grids" not in after
    assert "grids" not in after
