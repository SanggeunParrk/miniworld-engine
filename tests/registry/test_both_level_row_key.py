"""A `level=both` kernel keys on its ROW COUNT, because length collides across its two sides.

A pair activation (B, L, L, D) at L=1024 and an atom activation (B, A, D) at A=1024 both have
`shape[-2] == 1024`, so `both_key(length)` put them in one bucket while the first launches
1,048,576 rows and the second launches 1,024. It is visible in the shipped A6000 cache for
transition_layernorm_expand_swiglu_triton:

    shape_key=384    0.6103 ms     pair  L=384,  M = 147,456
    shape_key=1024   0.0215 ms     atom  A=1024, M =   1,024

The bucket grows and the time falls 28x. The module bench then ran the pair side at L=1024
against a config tuned at 1,024 rows and measured 9.50 ms where the same card in the same
configuration had measured 5.50 ms a month earlier.
"""
from __future__ import annotations

import collections

import pytest
from paths import ROOT, registry_rows

from miniworld_engine.autotune import cache
from miniworld_engine.autotune.builder import op_units
from miniworld_engine.autotune.shape_key import (
    ATOM_SHAPES,
    BOTH_ROWS,
    TOKEN_SHAPES,
    both_key,
    rows_of,
)


def test_a_pair_and_an_atom_of_the_same_length_are_different_buckets():
    """The regression, stated directly."""
    pair = both_key(rows_of((1, 1024, 1024, 128)))
    atom = both_key(rows_of((1, 1024, 128)))
    assert pair != atom, "a 1,048,576-row launch and a 1,024-row launch share a bucket"
    assert atom == 1024
    assert pair == max(BOTH_ROWS)


@pytest.mark.parametrize("length", TOKEN_SHAPES)
def test_a_pair_launch_buckets_on_its_row_count(length):
    assert both_key(rows_of((1, length, length, 128))) == length * length


@pytest.mark.parametrize("length", ATOM_SHAPES)
def test_an_atom_launch_buckets_on_its_row_count(length):
    assert both_key(rows_of((1, length, 128))) == length


def test_rows_of_refuses_a_flattened_shape():
    with pytest.raises(ValueError, match="already flattened"):
        rows_of((262144, 128))


def test_the_bucket_set_is_exactly_what_the_work_list_drives():
    """Every BOTH_ROWS bucket has a unit somewhere, and every unit lands on one.

    This used to take a single op and demand its buckets BE the whole set, which held only while
    every `level=both` row was driven from the same two sides. They are not: the row now says which
    streams it runs on, because `level=both` was never a claim about that. AlphaFold-3's Transition
    is applied to the pair representation, to the single representation at token granularity
    (`pairformer.transition_single`) and to the MSA stack, and never to atoms -- so those fourteen
    kernels were built at six atom lengths they never see and at none of the token shapes they do.

    So the invariant splits in two: no bucket in the set is unreachable (the union covers it), and
    no unit lands outside the set (nothing floors into a neighbour's bucket).
    """
    rows = [r for r in registry_rows()
            if r["level"] == "both" and r["backend"] == "triton" and (r["driver"] or "").strip()]
    assert rows, "no level=both triton kernels in the registry"
    union, outside = set(), []
    for r in rows:
        want = {x for x in (r.get("sides") or "pair|atom").split("|") if x}
        assert want <= {"pair", "atom", "token"}, f"{r['kernel']}: unknown side in {want}"
        seen = set()
        for u in op_units(only={r["kernel"]}):
            assert u.side in want, (
                f"{u.stem}: driven from {u.side!r}, which the row does not claim ({sorted(want)})")
            seen.add(u.side)
            union.add(u.bucket)
            if u.bucket not in set(BOTH_ROWS):
                outside.append(f"{u.stem} -> bucket {u.bucket}")
        # A kernel that does not key on shape_key has ONE bucket, so the builder drives it once
        # and the sides it claims describe where it is USED, not how many units it needs.
        # transition_fold is the only one: it reads the weights (Wa, Wb, gamma, beta) and never
        # touches the activation.
        from miniworld_engine.autotune.builder import _keys_on_shape

        keyed = _keys_on_shape(ROOT / "src" / r["file"], r["symbol"])
        if keyed:
            assert seen == want, (
                f"{r['kernel']}: claims {sorted(want)}, driven from {sorted(seen)}")
        else:
            assert len(seen) == 1, (
                f"{r['kernel']} does not key on shape_key, so it has ONE bucket and should be "
                f"driven once; got {sorted(seen)}")
            assert seen <= want, (
                f"{r['kernel']} is driven from {sorted(seen)}, which the row does not claim "
                f"({sorted(want)})")
    assert not outside, "units landing outside BOTH_ROWS:\n  " + "\n  ".join(outside)
    assert union == set(BOTH_ROWS), (
        f"buckets no unit reaches: {sorted(set(BOTH_ROWS) - union)}; "
        f"driven but not in BOTH_ROWS: {sorted(union - set(BOTH_ROWS))}")


def test_every_both_level_family_names_all_three_streams_it_runs_on():
    """`sides` has to be traced from the model, not defaulted, and the default was wrong for all.

    Traced in the model (team-gm):

      transition           pair (pairformer.py:116), single at token granularity
                           (pairformer.py:123, `transition_single = Transition(d_single)`), MSA
                           (msa_module.py:106). No atom use -- the atom blocks build
                           ConditionedTransition, a separate family.
      layernorm            all of them. `attention_pair_bias.py:49  ln_single =
      layernorm_linear     nn.LayerNorm(d_single)` and `outer_product.py:33` are the token stream;
                           `ln_pair` / `ln_msa` the others; AdaLN normalises atoms in the DiT.
      gated_projection     pair through triangle_multiplication, and token+atom through
                           conditioned_transition, which imports `_sigmul_fwd` / `_sigmul_bwd`
                           from it (`conditioned_transition/triton/training.py:28`).

    So every one of these runs on the token stream and not one was built there. This test names the
    expectation per family so the next kernel added to one inherits a traced answer rather than a
    default.
    """
    expect = {"transition": {"pair", "token"},
              # rmsnorm: pair through triangle_attention's qk-norm
              # (kernels/triangle_attention/whole_op.py) and atom through the SWA atom block's
              # `_qk_norm` (modules/swa_atom_attention/module.py). No token stream: the token-DiT
              # user was the adaLN modulate, now the rmsnorm_adamod family.
              "rmsnorm": {"pair", "atom"},
              # rmsnorm_adamod: the two DiT streams and no more. adaLN conditions the diffusion
              # blocks -- atom_dit and token_dit -- and the pair stream has no adaptive
              # normalization at all, so claiming `pair` here would be a guess and not a trace.
              "rmsnorm_adamod": {"token", "atom"},
              "layernorm": {"pair", "token", "atom"},
              "layernorm_linear": {"pair", "token", "atom"},
              "gated_projection": {"pair", "token", "atom"}}
    bad = []
    for r in registry_rows():
        if r["level"] != "both":
            continue
        want = expect.get(r["family"])
        got = {x for x in (r.get("sides") or "").split("|") if x}
        if want is None:
            bad.append(f"{r['kernel']}: family {r['family']!r} is level=both and this test has no "
                       f"traced answer for it -- trace it in the model and add one")
        elif got != want:
            bad.append(f"{r['kernel']}: sides={sorted(got)}, traced {sorted(want)}")
    assert not bad, "\n  ".join(["a level=both row disagrees with the model:", *bad])


def test_a_transition_kernel_is_not_built_on_atoms():
    """The finding that produced the `sides` column, pinned so it cannot quietly come back.

    `Transition` in the model is constructed in pairformer (pair and single), msa_module (msa and
    pair), template (pair) and the mini_ variants. There is no atom-level use: the atom blocks use
    ConditionedTransition, which is a different family with its own rows. A `transition` kernel
    driven at ATOM_SHAPES is six units per precision spent on a stream it never sees.
    """
    bad = []
    for r in registry_rows():
        if r["family"] != "transition" or r["level"] != "both":
            continue
        if "atom" in (r.get("sides") or "pair|atom").split("|"):
            bad.append(r["kernel"])
    assert not bad, ("transition kernels claiming an atom side, which the model never gives them:"
                     "\n  " + "\n  ".join(bad))


def test_a_units_side_reaches_the_driver():
    units = [u for u in op_units() if u.side]
    assert units
    assert units[0].env()["MINIWORLD_DRIVER_SIDE"] in ("pair", "atom", "token")
    assert "--side" in units[0].cmd_args()


def test_the_two_sides_get_different_shards():
    """Same op, same length, two sides -- they must not overwrite one another.

    This used to name pair and atom at L=256, which stopped being a pair of sides when the atom
    ladder moved to 1024 and up. The sides that now share a length are pair and token: a pair L is
    a token count too (the activation is (B, L, L, D)), so both walk TOKEN_SHAPES and every length
    collides unless the stem separates them.
    """
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for u in op_units():
        if u.side:
            by[(u.op, u.dtype, u.length)][u.side].add(u.stem)
    shared = {k: v for k, v in by.items() if len(v) > 1}
    assert shared, "no op is driven from two sides at the same length; this test has no subject"
    for (op, dtype, length), sides in sorted(shared.items()):
        stems = [s for group in sides.values() for s in group]
        assert len(stems) == len(set(stems)), (
            f"{op} [{dtype}] at length {length}: sides {sorted(sides)} share a shard stem, so one "
            f"overwrites the other -- {stems}")


def test_an_entry_from_the_old_scheme_is_not_served_to_a_both_level_kernel():
    assert cache._scheme_stale("transition_expand_swiglu_triton", None) is True
    assert cache._scheme_stale("transition_expand_swiglu_triton", cache.KEY_SCHEME) is False


def test_a_token_level_kernel_keeps_its_entries():
    """Scheme 2 re-based only both-level keys; invalidating the rest would discard a good build."""
    token = next(r["kernel"] for r in registry_rows() if r["level"] == "token")
    assert cache._scheme_stale(token, None) is False


def test_an_unknown_op_is_treated_as_stale():
    assert cache._scheme_stale("not_a_kernel_triton", None) is True


def test_every_write_stamps_the_key_scheme(tmp_path, monkeypatch):
    """A file has to say which scheme its keys are in, not leave it to be reconstructed."""
    import json

    import triton

    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    cfg = triton.Config({"BLOCK_M1": 64}, num_warps=4, num_stages=2)
    fp = cache.store_ranked_configs("op_probe", "gpu0", "bfloat16", "shape_key=256",
                                    [(cfg, 1.0)], "hash0", op_id="op0", env_id="env0")
    assert json.loads(fp.read_text())["key_scheme"] == cache.KEY_SCHEME

    # ... and again on an APPEND to a file that is not being reset.
    fp = cache.store_ranked_configs("op_probe", "gpu0", "bfloat16", "shape_key=512",
                                    [(cfg, 1.0)], "hash0", op_id="op0", env_id="env0")
    data = json.loads(fp.read_text())
    assert data["key_scheme"] == cache.KEY_SCHEME
    assert len(data["entries"]) == 2, "the second write reset the file instead of appending"


def test_the_coverage_check_compares_buckets_not_lengths():
    """`want` used `u.length`; a both-level unit's key is its row count, so every one would miss."""
    units = [u for u in op_units() if u.side == "pair"]
    assert units
    u = units[0]
    assert u.bucket == both_key(u.length * u.length)
    assert u.bucket != u.length, "a pair unit's bucket must not be its length"


def test_a_token_or_atom_unit_buckets_on_its_length():
    units = [u for u in op_units() if not u.side]
    assert units
    assert all(u.bucket == u.length for u in units)


def test_a_shard_records_the_scheme_its_buckets_are_in(tmp_path, monkeypatch):
    """Without this a merge after a bump folds old and new keys into one file."""
    import json

    from miniworld_engine.autotune import capture

    monkeypatch.setattr(capture, "_CAPTURE", {})
    p = tmp_path / "unit.json"
    capture.dump_shard(str(p))
    assert json.loads(p.read_text())["_key_scheme"] == cache.KEY_SCHEME


def test_the_merge_skips_a_shard_from_an_older_scheme(tmp_path, monkeypatch):
    """An old pair measurement and a new atom one both write `shape_key=256`.

    Merging both means the file's 256 bucket is whichever shard the merge reached last -- a
    65,536-row measurement served to a 256-row launch, or the reverse, by file order.
    """
    import json

    from miniworld_engine.autotune import capture

    op = "transition_expand_swiglu_triton"                     # level=both
    entry = [{"kwargs": {"BLOCK_M1": 64}, "num_warps": 4, "num_stages": 2, "ms": 1.0}]
    old = tmp_path / "old.json"
    old.write_text(json.dumps({op: {"grid": entry, "entries": {"bfloat16|shape_key=256": entry},
                                    "op_id": "x"}}))            # no _key_scheme = scheme 1
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"_key_scheme": cache.KEY_SCHEME,
                               op: {"grid": entry, "entries": {"bfloat16|shape_key=256": entry},
                                    "op_id": "x"}}))

    written = []
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path / "data")
    monkeypatch.setattr(capture, "store_ranked_configs",
                        lambda *a, **k: written.append(a) or (tmp_path / "f.json"))
    capture.merge_shards([str(old), str(new)], gpu="g")
    assert any("old.json" in s for s, _ in capture._MERGE_SKIPPED), capture._MERGE_SKIPPED
    assert not any("new.json" in s for s, _ in capture._MERGE_SKIPPED)


def test_the_merge_keeps_an_old_shard_for_an_op_the_bump_did_not_touch():
    """Scheme 2 re-based both-level keys only; a token op's shard is still good."""
    from miniworld_engine.autotune import cache as c

    token = next(r["kernel"] for r in registry_rows() if r["level"] == "token")
    assert c._scheme_stale(token, None) is False
