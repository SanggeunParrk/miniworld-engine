"""A kernel with no ``shape_key`` in its autotune key must not be failed for having one bucket.

``check_key_spread`` FAILs an op whose cache holds a single shape bucket, on the reasoning that
every other shape then falls back to the full grid in production. That is right for a kernel whose
best config moves with shape -- and wrong for one whose key carries no ``shape_key`` at all, e.g.
a weight-only kernel whose N and K are its whole shape; the builder already drives such a kernel
at one length only. The audit once disagreed with the builder and reported the build broken while
it was right.

Both sides now answer through ``builder._keys_on_shape``, and these tests are what keeps them from
drifting apart again. No registered kernel is weight-only today (the last one,
``transition_fold_triton``, was removed in v2.2.0), so the no-shape_key side is exercised on a
synthetic kernel source and a patched resolver.
"""
from __future__ import annotations

from miniworld_engine.autotune import builder
from miniworld_engine.build import audit

_WEIGHT_ONLY = '''
import triton

@triton.autotune(configs=[], key=['N', 'K'])
@triton.jit
def _weight_only_kernel(w_ptr, N, K):
    pass
'''


def test_a_kernel_without_shape_key_does_not_key_on_shape(tmp_path):
    src = tmp_path / "weight_only.py"
    src.write_text(_WEIGHT_ONLY)
    assert builder._keys_on_shape(src, "_weight_only_kernel") is False


def test_a_shape_keyed_kernel_still_reports_true():
    # A registered kernel that DOES carry shape_key -- if this ever returns False the resolver is
    # broken, not the kernel, and the test above would pass for the wrong reason.
    assert audit._keys_on_shape_key("transition_expand_swiglu_triton") is True


def test_an_unknown_op_is_assumed_shape_keyed():
    """The strict reading: an op the registry does not name keeps the FAIL."""
    assert audit._keys_on_shape_key("not_a_kernel_triton") is True


def test_one_bucket_is_OK_for_a_kernel_with_no_shape_key(tmp_path, monkeypatch):
    """End to end through the check, not just the helper."""
    import json

    op = "weight_only_triton"
    real = audit._keys_on_shape_key
    monkeypatch.setattr(audit, "_keys_on_shape_key", lambda name: name != op and real(name))
    shard = tmp_path / "shards"
    shard.mkdir()
    (shard / "unit.json").write_text(json.dumps({
        op: {"entries": {"bfloat16+float32|K=128,N=512": {}}},
    }))
    rep = audit.Report()
    audit.check_key_spread(rep, [shard])
    rows = [f for f in rep.of("spread") if f.subject == op]
    assert rows, "the check reported nothing for the op"
    assert all(f.level != audit.FAIL for f in rows), rows


def test_one_bucket_still_fails_a_shape_keyed_kernel(tmp_path):
    import json

    shard = tmp_path / "shards"
    shard.mkdir()
    (shard / "unit.json").write_text(json.dumps({
        "transition_expand_swiglu_triton": {"entries": {"bfloat16|K=128,N=512": {}}},
    }))
    rep = audit.Report()
    audit.check_key_spread(rep, [shard])
    rows = [f for f in rep.of("spread") if f.subject == "transition_expand_swiglu_triton"]
    assert any(f.level == audit.FAIL for f in rows), rows
