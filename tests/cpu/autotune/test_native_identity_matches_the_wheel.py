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
