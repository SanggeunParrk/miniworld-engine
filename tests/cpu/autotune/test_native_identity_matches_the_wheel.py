"""The native caches are keyed by `native.source_identity()`, a hash of the kernel sources ON DISK. A wheel ships fewer
files than the source tree (pyproject `[tool.setuptools.packages.find] exclude`: optimization `notes`, build-time `gen`
scripts), so the identity must leave out exactly what the wheel leaves out -- else every install computes another
identity, rejects every shipped native cache ("native cache source changed") and runs default configs whose kernels
the prebuilt bundle does not carry (2.4.0's first empty-cache verification: PrebuiltMissing in FoldForge and pdall)."""
from __future__ import annotations

from pathlib import Path

import tomllib

from miniworld_engine.autotune import native

ROOT = Path(native.__file__).resolve().parents[2].parent


def test_every_directory_the_wheel_excludes_is_left_out_of_the_identity() -> None:
    find = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]
    excluded = {pattern.split(".")[-1] for pattern in find["exclude"] if not pattern.endswith(".*")}
    # `autotune.plans` is outside kernels/ and never hashed
    kernel_dirs = excluded - {"plans"}
    source = Path(native.__file__).read_text()
    for name in kernel_dirs:
        assert f'"{name}"' in source.split("def source_identity", 1)[1].split("def ", 1)[0], name


def test_a_native_ops_graph_holds_only_files_the_wheel_ships_and_is_per_op() -> None:
    """``native.op_identity`` hashes the same files in a wheel as in the checkout: no `notes`, no generator `gen`; and an unrelated
    kernel family is not part of an op's graph (the point of replacing the one global hash)."""
    from miniworld_engine.autotune import identity

    for op in native.BUILD_OPS:
        files = identity.native_files(op)
        assert files
        assert not [f for f in files if {"notes", "gen", "plans"} & set(f.split("/"))], op
        assert not [f for f in files if f.startswith("autotune/data/")], op
        assert "kernels/triangle_attention/triton/main.py" not in files, op
    assert len({native.op_identity(op) for op in native.BUILD_OPS}) == len(native.BUILD_OPS)


def test_the_global_source_identity_is_only_a_salt() -> None:
    """Nothing is judged stale by ``source_identity()``: the cache scan, the coverage check and the chooser use ``op_identity``."""
    import inspect

    from miniworld_engine.autotune import cache_status, derive

    for fn in (cache_status._current_op_identity, derive.coverage, native.choose_config):
        assert "source_identity" not in inspect.getsource(fn).replace("native_source_identity", ""), fn.__name__
