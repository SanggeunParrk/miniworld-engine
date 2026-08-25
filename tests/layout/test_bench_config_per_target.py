"""Every bench target owns its config, and the config names its own directory.

``@hydra.main``'s ``config_path`` is a decorator argument, so it was a constant:
``benchmarks/modules/triangle_multiplication/configs``. Every run loaded that file -- a kernel
bench, an atom bench, all of them -- and then applied command-line overrides. The other 25
targets' ``configs/bench.yaml`` were files nothing read, and they did not agree with what ran:
``augmented_attention_atom`` declares ``min_seq_len: 128 / max_seq_len: 384`` and was swept at
384-1024, token-scale lengths on an atom-scale op. Its own committed tables show 128/256/384, so
the tables and the runner disagreed too.

``_target_config_path()`` now reads ``level=``/``target=`` off argv before hydra starts and
returns that target's directory. This file is the static half of the guarantee: that the
directory always exists, and that what is in it says the same thing its path does.

bench.py cannot be imported without a GPU, so its tables are read out of the source with ``ast``,
the same way tests/compile/test_compiled_flag_is_what_ran.py does.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BENCH = REPO / "benchmarks" / "runners" / "bench.py"
SOURCE = BENCH.read_text()
TREE = ast.parse(SOURCE)


def _targets(table: str) -> list[str]:
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == table for t in node.targets)):
            return [str(k.value) for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError(f"{table} not found in {BENCH}")


CONFIGS = [("kernel", t) for t in _targets("KERNEL_TARGETS")] + \
          [("module", t) for t in _targets("MODULE_TARGETS")]


def _path(level: str, target: str) -> Path:
    return REPO / "benchmarks" / f"{level}s" / target / "configs" / "bench.yaml"


@pytest.mark.parametrize(("level", "target"), CONFIGS, ids=lambda v: v)
def test_the_target_has_a_config_that_names_its_own_directory(level: str, target: str) -> None:
    """A missing config is now a hard error at startup rather than a fall back to someone else's
    ladders, so the file has to be there; and its declared target/level have to match the path it
    was found at, or reading the file tells you about a different run than the one it configures.
    """
    path = _path(level, target)
    assert path.is_file(), f"missing {path}"
    conf = yaml.safe_load(path.read_text())
    assert conf["target"] == target, f"{path} declares target={conf['target']!r}"
    assert conf["level"] == level, f"{path} declares level={conf['level']!r}"


def test_every_config_carries_the_same_keys() -> None:
    """Hydra composes one file, with no defaults list: a key absent from a target's config is a
    key that does not exist for that run, and an override of it fails with "not in struct" rather
    than falling back to anything."""
    base = set(yaml.safe_load(_path("module", "triangle_multiplication").read_text()))
    for level, target in CONFIGS:
        keys = set(yaml.safe_load(_path(level, target).read_text()))
        assert keys == base, {
            "config": str(_path(level, target).relative_to(REPO)),
            "missing": sorted(base - keys), "extra": sorted(keys - base),
        }


def test_the_config_directory_is_computed_not_hard_coded() -> None:
    """The regression this file exists for: one target's directory wired in as a constant."""
    decorator = next(
        node for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in node.decorator_list if isinstance(node, ast.Call))
    path_arg = next(k for k in decorator.keywords if k.arg == "config_path")
    assert not isinstance(path_arg.value, ast.Constant), (
        "config_path is a literal again; it must be _target_config_path(), or every target loads "
        "whichever one is written here")
    assert isinstance(path_arg.value, ast.Call)
    assert isinstance(path_arg.value.func, ast.Name)
    assert path_arg.value.func.id == "_target_config_path"


def test_no_target_directory_is_left_without_a_config() -> None:
    """A directory under benchmarks/{kernels,modules}/ with no config is a target that cannot be
    run; `.gitkeep` used to stand in for the file and hid exactly that."""
    for level in ("kernel", "module"):
        for target_dir in sorted((REPO / "benchmarks" / f"{level}s").iterdir()):
            if not target_dir.is_dir():
                continue
            assert (target_dir / "configs" / "bench.yaml").is_file(), \
                f"{target_dir.relative_to(REPO)} has no configs/bench.yaml"
            assert not (target_dir / "configs" / ".gitkeep").exists(), \
                f"{target_dir.relative_to(REPO)}/configs/.gitkeep is left over; the dir is not empty"
