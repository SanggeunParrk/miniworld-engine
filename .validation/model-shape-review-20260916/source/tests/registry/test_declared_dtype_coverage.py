"""The work list must cover every DECLARED dtype, and the audit must check that it did.

registry.csv declares the precisions each kernel has to be tuned for -- token kernels bf16, atom
and both bf16|fp32. `op_units` emitted bfloat16 for everything, so the fp32 half of 66 kernels was
never driven; and `check_cache_coverage` counted (op, bucket) with no dtype axis, so it reported
527/527 and missing_pairs=0 over a cache holding one of the two declared precisions. Two halves of
the same blind spot: the builder did not do the work and the check could not see it missing.
"""
from __future__ import annotations

import collections

import pytest
from paths import registry_rows

ALIAS = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}


def _declared():
    out = {}
    for r in registry_rows():
        if r["backend"] == "triton" and (r["driver"] or "").strip():
            out[r["kernel"]] = {ALIAS.get(x, x) for x in r["dtypes"].split("|") if x}
    return out


@pytest.fixture
def units():
    from miniworld_engine.autotune.builder import op_units
    return op_units()


def test_the_work_list_drives_every_declared_dtype(units):
    got = collections.defaultdict(set)
    for u in units:
        got[u.op].add(u.dtype)
    missing = {op: d - got[op] for op, d in _declared().items() if op in got and d - got[op]}
    assert not missing, f"declared dtypes never driven: {dict(list(missing.items())[:6])}"


def test_fp32_is_actually_in_the_work_list(units):
    """The concrete regression: 527 units, every one bfloat16."""
    by = collections.Counter(u.dtype for u in units)
    assert by.get("float32", 0) > 0, f"no fp32 units at all: {dict(by)}"


def test_a_token_kernel_is_not_driven_in_fp32(units):
    """Declared coverage cuts both ways -- driving fp32 where only bf16 is declared tunes a
    precision the model never asks for, at the price of a bucket it does."""
    decl = _declared()
    for u in units:
        if u.op in decl:
            assert u.dtype in decl[u.op], f"{u.op} driven as {u.dtype}, declared {decl[u.op]}"


def test_coverage_counts_the_dtype_axis(tmp_path, monkeypatch):
    """A cache holding only bf16 must NOT audit clean when fp32 is declared."""
    import json

    from miniworld_engine.autotune import cache
    from miniworld_engine.build import audit

    root = tmp_path / "data"
    (root / "op_x").mkdir(parents=True)
    (root / "op_x" / "g.json").write_text(json.dumps({
        "gpu": "g", "op": "op_x", "entries": {"bfloat16|shape_key=128": [{"kwargs": {}}]}}))
    monkeypatch.setattr(cache, "_CACHE_ROOT", root)

    # The real OpUnit, not a stub: the coverage check reads `u.bucket` (a both-level unit's key
    # is its row count, not its length), and a stub that only carries `length` would have kept
    # passing while the check it stands in for stopped working.
    import miniworld_engine.autotune.builder as B

    monkeypatch.setattr(audit, "_declared_dtypes", lambda: {"op_x": {"bfloat16", "float32"}})
    monkeypatch.setattr(B, "op_units",
                        lambda *a, **k: [B.OpUnit("op_x", 128, "bfloat16"),
                                         B.OpUnit("op_x", 128, "float32")])
    rep = audit.Report()
    audit.check_cache_coverage(rep, gpu="g")
    assert rep.stats["missing_pairs"] == 1, rep.findings
    assert "float32" in rep.findings[0].detail
