"""Correct dimensions and reporting without narrowing schedules from sampled GPU winners."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniworld_engine.autotune import builder, configs, width_evidence
from miniworld_engine.autotune.cache import _sig, _sig_from_dict
from miniworld_engine.autotune.module_registry import transition_driver_shapes
from miniworld_engine.viz import sweep_page

ROOT = Path(__file__).resolve().parents[2]
REPORT = json.loads((ROOT / "docs/reports/autotune-space-20260916.json").read_text())


@pytest.mark.parametrize("record", REPORT["kernels"], ids=lambda r: r["kernel"])
def test_original_candidate_space_and_uniqueness(record):
    space = configs._read(sweep_page.GRID / f"{record['kernel']}.csv")
    assert len(space) == len({_sig(c) for c in space}) == record["after"]
    assert len(space) == record["before"]
    axes, parsed = sweep_page._config_inventory(record["kernel"])
    assert axes
    assert len(parsed) == len(space)


def test_b2b_prune_covers_real_widths_and_keeps_cached_candidates():
    from miniworld_engine.kernels.transition.triton.fused import _prefer_covering_b2b

    op = "transition_fwd_b2b_triton"
    space = configs._read(sweep_page.GRID / f"{op}.csv")
    live = {_sig(c) for c in space}
    for width in (32, 64, 128, 256):
        chosen = _prefer_covering_b2b(space, {"K": width})
        assert len(chosen) == 560
        assert all(c.kwargs["BLOCK_K_D"] == c.kwargs["BLOCK_K_ND"] == width for c in chosen)
        assert all(c.kwargs["GROUP_M"] == 1 for c in chosen)
    for path in (sweep_page.DATA / op).glob("*.json"):
        for ranked in json.loads(path.read_text())["entries"].values():
            assert all(_sig_from_dict(c) in live for c in ranked)


def test_partial_width_evidence_keeps_unknown_and_collapses_known_groups():
    evidence = {"op": {"64": ["a"], "128": ["a"], "256": ["b"], "512": ["b"]}}
    assert width_evidence.distinct_widths("op", (16, 64, 128, 256, 512, 768), evidence) == (16, 128, 512, 768)
    assert width_evidence.distinct_widths("missing", (64, 128), evidence) == (64, 128)


def test_b2b_dimensions_are_exact_pairs_not_other_modules_widths():
    rows = transition_driver_shapes("transition_fwd_b2b_triton")
    assert {width for _, _, width, _ in rows} == {64, 128}
    assert {(side, width, ratio) for side, _, width, ratio in rows} == {
        ("pair", 64, 2), ("pair", 128, 2), ("pair", 128, 4),
        ("atom", 128, 2), ("msa", 64, 4), ("msa", 128, 4)}
    units = builder.op_units({"transition_fwd_b2b_triton"})
    assert len(units) == len(rows)
    assert all(u.width <= 128 and u.heads in {2, 4} for u in units)
    assert len({u.stem for u in units}) == len(units)


def test_conditional_dimension_pairs_do_not_overwrite_each_other():
    units = builder.op_units({"adaln_fwd_gate_triton"})
    pairs = {(u.width, u.heads) for u in units if u.side == "token"}
    assert {(768, 384), (768, 768)} <= pairs


def test_materialized_inventory_does_not_reexpand_axis_marginals(tmp_path, monkeypatch):
    # Explicit config lists remain supported, but GPU-specific winners do not
    # justify replacing the universal search grid with a smaller list.
    (tmp_path / "probe.csv").write_text(
        "BLOCK_M1,BLOCK_N,num_warps,num_stages\n32,64,4,2\n64,128,8,3\n")
    with monkeypatch.context() as patch:
        patch.setattr(sweep_page, "GRID", tmp_path)
        axes, space = sweep_page._config_inventory("probe")
    import math

    assert len(space) == 2
    assert math.prod(len(v) for v in axes.values()) == 16


def test_inventory_does_not_add_an_unreached_b2b_driver():
    from miniworld_engine.autotune import derive

    rows, _ = sweep_page.collect()
    required = {r["kernel"] for r in derive.kernel_rows("sm86")}
    assert {r["kernel"] for r in rows} == required
    assert len(sweep_page._config_inventory("transition_fwd_b2b_triton")[1]) == 27944


def test_b2b_inventory_uses_recorded_dimensions_for_existing_prune(monkeypatch):
    from miniworld_engine.autotune import derive, plan
    from miniworld_engine.autotune.shape_key import pack

    monkeypatch.setattr(plan, "load", lambda *a: {"complete": True})
    monkeypatch.setattr(sweep_page, "_coverage", dict)
    monkeypatch.setattr(derive, "units", lambda *a, **kw: [])
    monkeypatch.setattr(sweep_page, "_derived_rows", lambda *a: [{
        "kernel": "transition_fwd_b2b_triton", "streams": "token_pair",
        "bucket": f"HAS_LN=1,shape_key={pack(128, K=64, ND=128)}",
        "shapes": '{"token_pair": [128]}', "lengths": "128",
    }])
    rows, _ = sweep_page.collect()
    row, = rows
    assert row["grid"] == 27944
    assert row["cost"] == 560
    assert row["sides"][0]["dimensions"] == ["K=64, ND=128"]
    assert "verified keys" in row["cost_kind"]


@pytest.mark.parametrize("record", REPORT["kernels"], ids=lambda r: r["kernel"])
def test_original_grid_and_all_legal_cached_candidates_are_preserved(record, tmp_path):
    original = tmp_path / "original.csv"
    original.write_text(record["original_grid_csv"])
    before = {_sig(c) for c in configs._read(original)}
    current = configs._read(sweep_page.GRID / f"{record['kernel']}.csv")
    after = {_sig(c) for c in current}
    assert after == before
    for path in (sweep_page.DATA / record["kernel"]).glob("*.json"):
        data = json.loads(path.read_text())
        ranked_groups = [*data["entries"].values(), *(
            workload["entries"] for workloads in data.get("measurements", {}).values()
            for workload in workloads.values())]
        for ranked in ranked_groups:
            assert {_sig_from_dict(c) for c in ranked} & before <= after
    if record["kernel"] != "transition_fwd_b2b_triton":
        assert {signature[0] for signature in before} == {signature[0] for signature in after}


@pytest.mark.parametrize(("width", "ratio", "side"), [(64, 2, "pair"), (128, 4, "msa"), (128, 2, "atom"), (384, 2, "token")])
def test_driver_really_launches_the_requested_dimensions(width, ratio, side, monkeypatch):
    import importlib

    import torch

    from miniworld_engine.kernels import drivers
    from miniworld_engine.kernels.transition.triton import fused

    monkeypatch.setattr(drivers, "DRIVER_WIDTH", width)
    monkeypatch.setattr(drivers, "DRIVER_HEADS", ratio)
    monkeypatch.setattr(drivers, "DRIVER_LENGTH", 128)
    monkeypatch.setenv("MINIWORLD_DRIVER_SIDE", side)
    module = importlib.import_module("miniworld_engine.kernels.drivers.transition")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "rows2d", lambda m, n: torch.empty(m, n, device="meta"))
    monkeypatch.setattr(module, "vec", lambda n: torch.empty(n, device="meta"))
    seen = []
    def launch(x, _g, _b, wa, wb, ws, *_args, **kwargs):
        seen.append((tuple(x.shape), tuple(wa.shape), tuple(wb.shape), tuple(ws.shape)))
    name = "transition_b2b" if width <= 128 else "transition_b2b_ktiled"
    monkeypatch.setattr(fused, name, launch)
    getattr(module, "transition_fwd_b2b_triton" if width <= 128 else "transition_fwd_b2b_ktiled_triton")()
    rows = 128 * 128 if side == "pair" else 8 * 128 if side == "msa" else 128
    assert seen
    assert all(shapes == ((rows, width), (width * ratio, width), (width * ratio, width), (width, width * ratio)) for shapes in seen)
