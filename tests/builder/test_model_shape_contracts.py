"""Checkpoint census is independent of the registry being pruned."""
import json
from pathlib import Path

import pytest
import torch

from miniworld_engine.autotune import builder, checkpoint_cases, module_registry
from miniworld_engine.viz import sweep_page

ROOT = Path(__file__).resolve().parents[2]
CENSUS = json.loads((ROOT / "docs/checkpoint-shapes-20260915.json").read_text())["groups"]


def test_all_affine_checkpoint_norm_dimensions_survive():
    # Model paths decide stream membership; this independent constructor census
    # additionally prevents removing any actual affine width/bias combination.
    want = set()
    for group in CENSUS:
        if group["class"] in {"LayerNorm", "OpenFoldLayerNorm"} and "weight" in group["parameters"]:
            want.add((group["parameters"]["weight"][0], int("bias" in group["parameters"])))
    have = {(r.dims["d_norm"], r.dims["has_bias"]) for r in module_registry.module_rows() if r.module == "layernorm_native"}
    assert want <= have


def test_projected_checkpoint_token_attention_pairs_survive():
    want = set()
    for group in CENSUS:
        if group["class"] == "AttentionPairBias" and not group["attrs"]["cross_attention_mode"]:
            want.add((group["attrs"]["n_heads"], group["parameters"]["attention.linear_q.weight"][0]))
    have = {(r.dims["n_head"], r.dims["d_hidden"]) for r in module_registry.module_rows() if r.module == "projected_attention" and r.stream == "token_single"}
    assert want <= have


def test_transition_census_pairs_are_preserved():
    want = set()
    for group in CENSUS:
        if group["class"] == "Transition":
            a = group["attrs"]
            want.add((a.get("c_in", a.get("d_hidden")), a.get("n", a.get("num_intermediate_factor"))))
    have = {(r.dims["d_hidden"], r.dims.get("n", 4)) for r in module_registry.module_rows() if r.module == "transition"}
    assert want <= have


def test_semantic_streams_and_lengths():
    for row in module_registry.module_rows():
        assert row.lengths == module_registry.STREAM_LADDERS[row.stream]
        if row.module == "layernorm_native" and row.dims["d_norm"] == 16:
            assert row.stream == "atom_pair"
        if row.module == "layernorm_native" and row.dims["d_norm"] == 256:
            assert row.stream in {"token_pair", "noise"}
    assert module_registry.STREAM_LADDERS["token_single"] == (128, 256, 384, 512, 640, 768)
    assert module_registry.STREAM_LADDERS["atom_single"] == tuple(range(1024, 8193, 1024))


@pytest.mark.parametrize(("stream", "length", "width", "batch", "shape"), [
    ("atom_pair", 4096, 16, 1, (1, 128, 32, 128, 16)),
    ("noise", 1, 256, 48, (48, 1, 256)),
    ("token_pair", 384, 384, 1, (1, 384, 384, 384)),
    ("msa_token", 384, 128, 1, (1, 8, 384, 128)),
])
def test_stream_inputs_match_model_layout(monkeypatch, stream, length, width, batch, shape):
    def meta_randn(*size: int, device=None, dtype=None):
        return torch.empty(size, device="meta", dtype=dtype)

    monkeypatch.setattr(torch, "randn", meta_randn)
    x, = checkpoint_cases._inputs("layernorm_native", batch, length, {"d_norm": width}, torch.bfloat16, stream)
    assert tuple(x.shape) == shape


def test_case_names_and_bench_mappings_remain_valid():
    from miniworld_engine import cli
    names = {case.name for case in builder.cases()}
    assert names == set(builder.CASE_NAMES)
    assert all(set(cases) <= names for cases in cli.KERNEL_TARGETS.values())


def test_stale_html_plan_cannot_reintroduce_cartesian_shapes(monkeypatch):
    from miniworld_engine.autotune import plan
    def stale(*args):
        raise ValueError("stale derivation")
    monkeypatch.setattr(plan, "load", stale)
    with pytest.raises(ValueError, match="stale derivation"):
        sweep_page.collect()


def test_same_dimensions_keep_distinct_augmentation_rows(monkeypatch):
    from miniworld_engine.autotune import derive, plan

    monkeypatch.setattr(builder, "device_sm", lambda: "sm86")
    case = next(c for c in builder.cases() if c.name == "layernorm_native")
    units = builder.units([case])
    labels = [plan.label(u, case) for u in units]
    assert len(labels) == len(set(labels))
    assert set(labels) == {u.label for u in derive.units(list(case.rows), arch="sm86")}
    assert len({u.stem for u in units}) == len(units)
