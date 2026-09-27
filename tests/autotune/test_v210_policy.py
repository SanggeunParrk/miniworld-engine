"""Policy changes must not erase inference workloads or validated global winners."""
import dataclasses

import pytest

from miniworld_engine.autotune import configs, derive, policy
from miniworld_engine.autotune.cache import _sig, as_cfg_dict
from miniworld_engine.autotune.module_registry import module_rows


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    monkeypatch.delenv("MINIWORLD_BUILD_SCOPE", raising=False)
    monkeypatch.delenv("MINIWORLD_BUILD_MODE", raising=False)


def test_inference_ladders_survive_and_token_training_is_cropped():
    for row in module_rows():
        units = derive.units([row])
        if "eval" in row.modes:
            assert {u.length for u in units if u.mode == "eval"} == set(row.lengths)
        if "train" in row.modes:
            expected = set(row.lengths)
            if row.stream in policy.TOKEN_STREAMS:
                expected &= {384, 768}
            assert {u.length for u in units if u.mode == "train"} == expected
        assert all(u.option is None or u.option[0] == "p_drop" for u in units)
        if "miniworld" in row.impls:
            assert all(u.impl == "miniworld" for u in units)


def test_global_search_and_alternative_dispatch_are_independent(monkeypatch):
    row = dataclasses.replace(module_rows()[0], options=(("ln_bwd_path", "atomic"),),
                              impls=("miniworld", "triton"), modes=("eval", "train"))
    production = derive.units([row])
    monkeypatch.setenv("MINIWORLD_BUILD_SCOPE", "all")
    alternatives = derive.units([row])
    assert len(alternatives) > len(production)
    assert any(u.impl == "triton" for u in alternatives)
    assert any(u.option == ("ln_bwd_path", "atomic") for u in alternatives)
    monkeypatch.setenv("MINIWORLD_BUILD_MODE", "eval")
    assert all(u.mode == "eval" for u in derive.units([row]))


def test_every_default_is_small_nonempty_and_in_the_global_domain():
    defaults = configs.CONFIG_ROOT / "default"
    global_dir = configs.CONFIG_ROOT / "grid"
    assert {p.name for p in defaults.glob("*.csv")} == {p.name for p in global_dir.glob("*.csv")}
    for path in defaults.glob("*.csv"):
        candidates = configs._read(path)
        assert 1 <= len(candidates) <= 32, path
        valid = configs.validated_global_configs(path.stem, [as_cfg_dict(c) for c in candidates])
        assert {_sig(c) for c in valid} == {_sig(c) for c in candidates}, path


def test_a_measured_global_winner_outside_default_is_still_eligible():
    op = "rmsnorm_fwd_triton"
    small = {_sig(c) for c in configs._read(configs.CONFIG_ROOT / "default" / (op + ".csv"))}
    full = configs._read(configs.CONFIG_ROOT / "grid" / (op + ".csv"))
    wider = next(c for c in full if _sig(c) not in small)
    restored = configs.validated_global_configs(op, [as_cfg_dict(wider)])
    assert [_sig(c) for c in restored] == [_sig(wider)]
    invalid = as_cfg_dict(wider)
    invalid["kwargs"] = {**invalid["kwargs"], "BLOCK_K": 999999}
    assert not configs.validated_global_configs(op, [invalid])


def test_cli_default_and_explicit_global_modes():
    from miniworld_engine.cli import build_parser
    parse = build_parser().parse_args
    args = parse(["build", "all"])
    assert args.config_type == "default"
    assert not args.include_alternatives
    args = parse(["build", "all", "grid", "--include-alternatives", "--mode", "eval"])
    assert args.config_type == "grid" and args.include_alternatives and args.mode == "eval"


def test_per_op_modes_preserve_atom_and_mpnn(monkeypatch):
    from miniworld_engine.autotune.builder import OpUnit
    edge = OpUnit(op="mpnn_edge_tail_bwd_layernorm_saveact_triton", length=1024,
                  dtype="bfloat16", side="edge", width=128)
    atom = OpUnit(op="layernorm_bwd_split_triton", length=4096,
                  dtype="bfloat16", side="atom", width=128)
    pair = OpUnit(op="layernorm_bwd_split_triton", length=256,
                  dtype="bfloat16", side="pair", width=128)
    inference = [OpUnit(op="trimul_fwd_sm90_cuda", length=n, dtype="bfloat16", width=128)
                 for n in (128, 256, 384, 512, 640, 768)]
    units = [edge, atom, pair, dataclasses.replace(pair, length=384), *inference]
    monkeypatch.setenv("MINIWORLD_BUILD_MODE", "train")
    selected = policy.filter_op_units(units)
    assert edge in selected and atom in selected
    assert pair not in selected and dataclasses.replace(pair, length=384) in selected
    assert not any(u.op == "trimul_fwd_sm90_cuda" for u in selected)
    monkeypatch.setenv("MINIWORLD_BUILD_MODE", "eval")
    assert policy.filter_op_units(units) == inference


def test_global_cache_winner_is_rechecked_by_resource_pruning(monkeypatch):
    from types import SimpleNamespace
    from miniworld_engine import settings
    from miniworld_engine.autotune import cache
    op = "rmsnorm_fwd_triton"
    small = configs._read(configs.CONFIG_ROOT / "default" / (op + ".csv"))
    signatures = {_sig(c) for c in small}
    winner = next(c for c in configs._read(configs.CONFIG_ROOT / "grid" / (op + ".csv"))
                  if _sig(c) not in signatures)
    entry = [as_cfg_dict(winner)]
    data = {"op_identity": "current-source", "env_identity": cache.env_identity(),
            "key_scheme": cache.KEY_SCHEME, "entries": {"bfloat16|test": entry}}
    monkeypatch.setattr(configs, "using_default_space", lambda: True)
    monkeypatch.setattr(configs, "op_of", lambda _: op)
    monkeypatch.setattr(cache, "gpu_key", lambda: "cpu-probe")
    monkeypatch.setattr(cache, "dtype_of_args", lambda _: "bfloat16")
    monkeypatch.setattr(cache, "bucket_of_autotuner", lambda *a: "test")
    monkeypatch.setattr(cache, "_load", lambda *a: data)
    monkeypatch.setattr(cache, "op_identity", lambda _: "current-source")
    monkeypatch.setattr(cache, "implementation_identity", lambda _: "current-source")
    monkeypatch.setattr(cache, "_stored_rev", lambda _: cache.build_rev(op))
    monkeypatch.setattr(cache, "runtime_candidates", lambda *a: entry)
    monkeypatch.setattr(cache, "_miss", lambda *a: None)
    autotuner = SimpleNamespace(configs=small)
    old = settings.configure(run_autotune=False)
    try:
        kept = cache._cached_subset(autotuner, small, {}, {}, resource_prune=lambda c, *a, **kw: c)
        assert [_sig(c) for c in kept] == [_sig(winner)]
        assert cache._cached_subset(autotuner, small, {}, {}, resource_prune=lambda *a, **kw: []) is None
        data["op_identity"] = "obsolete-source"
        assert cache._cached_subset(autotuner, small, {}, {}) is None
    finally:
        settings.configure(**vars(old))
