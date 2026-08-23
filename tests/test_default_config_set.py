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


def test_the_default_ships_inside_the_package():
    """A wheel contains `src/miniworld_engine/**`, and nothing else.

    `configs/` at the repo root is not package data, so a pip install used to have no config set
    at all -- `default_config_dir()` would fall through to the RuntimeWarning and every triton
    kernel would fail at launch. The default set is packaged; the A-B sets stay at the root.
    """
    packaged = Path(configs.__file__).parent / "configs" / "grid"
    assert packaged.is_dir(), "the default set is not inside the package; a wheel would ship none"
    assert configs.default_config_dir() == packaged, "the packaged copy is not the one chosen"
    assert len(list(packaged.glob("*.csv"))) > 80


def test_the_two_copies_of_grid_are_identical():
    """While the repo root still has one -- every sweep launcher points at it -- it must match.

    Two copies of a search space that drift apart produce a cache tuned over one and read
    against the other, and the reader's intersection silently empties. The root copy goes away
    once no running job depends on it.
    """
    packaged = Path(configs.__file__).parent / "configs" / "grid"
    root = Path(configs.__file__).resolve().parents[3] / "configs" / "grid"
    if not root.is_dir():
        pytest.skip("the repo-root copy is gone, which is the intended end state")
    names = {p.name for p in packaged.glob("*.csv")}
    assert names == {p.name for p in root.glob("*.csv")}, "the two copies list different ops"
    differing = [n for n in sorted(names)
                 if (packaged / n).read_bytes() != (root / n).read_bytes()]
    assert not differing, f"{len(differing)} file(s) differ: {differing[:5]}"


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


def test_every_runtime_data_extension_is_declared_as_package_data():
    """A data file the wheel does not carry is a file that only works from a checkout.

    `[tool.setuptools.package-data]` is a list of globs. Anything under the package that the
    RUNTIME reads -- the cache JSONs, the config CSVs, the registry, the JIT-compiled CUDA
    sources, the build matrix -- has to match one of them, or it works in an editable install and
    vanishes on `pip install`. That is how the config set came to be missing from the wheel.
    """
    # `importorskip`, not `import tomllib`: stdlib TOML is 3.11+ and this package's floor is
    # 3.10, so a plain import is a type error at the declared version even though every
    # environment that runs the suite is newer.
    tomllib = pytest.importorskip("tomllib")

    root = Path(configs.__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as fh:
        globs = tomllib.load(fh)["tool"]["setuptools"]["package-data"]["miniworld_engine"]
    allowed = {g.rsplit(".", 1)[-1] for g in globs}

    pkg = Path(configs.__file__).resolve().parents[1]
    runtime = [pkg / "autotune" / "data", pkg / "autotune" / "configs",
               pkg / "build", pkg / "kernels" / "registry.csv"]
    # Scaffolding and prose, not data the runtime reads.
    skip = {".py", ".pyc"}
    skip_names = {".gitkeep", "README.md"}
    missed = []
    for target in runtime:
        files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
        missed += [p.relative_to(pkg) for p in files
                   if p.name not in skip_names and p.suffix not in skip
                   and p.suffix.lstrip(".") not in allowed]
    assert not missed, f"{len(missed)} runtime file(s) no glob would ship: {missed[:5]}"
