"""There must be a config set without setting an environment variable, and it must be `grid`.

`MINIWORLD_CONFIG_DIR` used to be the only way a config set was ever selected. Unset, `_DIR`
stayed None, every op registered an empty list, triton substituted its own `Config({})`, and the
first launch of every triton kernel died with

    TypeError: dynamic_func() missing 2 required positional arguments: 'BLOCK_M1' and 'BLOCK_K'

which names neither the op nor the cause. Every sbatch script and every bench entry point in this
repo exports the variable, so the failure only showed up when one of them did not -- a bench run
with no `MINIWORLD_CONFIG_DIR`, where the `miniworld` row came back `status=failed` with that
message while `pytorch` next to it was fine.

The second test is the one that says WHICH set the default has to be: the cache reader intersects
a shipped entry against the live config list, so a default narrower than the space the cache was
built over resolves every entry to nothing and re-tunes on every call. That failure is silent --
correct numbers, no warning, just slow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache, configs

DATA = Path(cache.__file__).parent / "data"


def test_a_config_set_is_selected_without_the_environment_variable():
    d = configs.default_config_dir()
    assert d is not None, "no packaged or repo-root configs/grid; every triton kernel would fail"
    assert d.is_dir()


def test_the_default_is_grid():
    d = configs.default_config_dir()
    assert d is not None
    assert d.name == "grid"


def test_ops_get_configs_with_no_environment_variable(monkeypatch):
    monkeypatch.delenv("MINIWORLD_CONFIG_DIR", raising=False)
    assert len(configs.configs_for("transition_fold_triton")) > 1


@pytest.mark.parametrize("op", ["transition_fold_triton", "layernorm_fwd_mmajor_triton"])
def test_a_shipped_cache_entry_still_exists_in_the_default_config_set(op):
    """The intersection the reader performs must be non-empty, or the cache buys nothing."""
    files = sorted((DATA / op).glob("*.json")) if (DATA / op).is_dir() else []
    if not files:
        pytest.skip(f"no shipped cache for {op}")
    live = {cache._sig(c) for c in configs.configs_for(op)}
    assert live, f"{op} has no configs under the default set"
    for f in files:
        entries = json.loads(f.read_text()).get("entries", {})
        for bucket, ranked in entries.items():
            hit = [c for c in ranked if cache._sig_from_dict(c) in live]
            assert hit, f"{f.name} {bucket}: none of its {len(ranked)} configs is in the grid"
