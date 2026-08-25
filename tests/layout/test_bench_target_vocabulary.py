"""One vocabulary across the four name spaces a bench run crosses.

A single measurement threads through four tables that used to be written by hand, independently,
in different spellings:

    bench.py   KERNEL_TARGETS / MODULE_TARGETS   what can be measured
    cli.py     KERNEL_TARGETS / MODULE_TARGETS   what `bench_kernel` / `bench_module` accept,
                                                 and which build case fills each one's cache
    builder.py CASE_NAMES                        what `build` can drive
    the tree    benchmarks/<level>s/<target>/    where the results land

They drifted, and every drift was silent. `bench_kernel triangle_attention` was rejected as an
unknown target because the kernel-level table spelled it `tri_attn`. `bench_module all` ran eight
of the nine module targets because the "all" group read a table that was missing one.
`augmented_attention_token` and `_atom` shared one directory named after neither of them, so the
coverage report looked them up, found nothing, and reported their kernels as never launched.

bench.py cannot be imported without a GPU (it raises at import), so the two tables in it are read
out of its source with `ast`, the same way tests/compile/test_compiled_flag_is_what_ran.py does.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from miniworld_engine import cli
from miniworld_engine.autotune.builder import CASE_NAMES

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BENCH = REPO / "benchmarks" / "runners" / "bench.py"
TREE = ast.parse(BENCH.read_text())


def _target_table(name: str) -> dict[str, str]:
    """`{target: bench function name}` for one of bench.py's two tables."""
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        return {str(k.value): v.id for k, v in zip(node.value.keys, node.value.values, strict=False)
                if isinstance(k, ast.Constant) and isinstance(v, ast.Name)}
    raise AssertionError(f"{name} not found in {BENCH}")


BENCH_KERNEL = _target_table("KERNEL_TARGETS")
BENCH_MODULE = _target_table("MODULE_TARGETS")
LEVELS = {"kernel": (BENCH_KERNEL, cli.KERNEL_TARGETS),
          "module": (BENCH_MODULE, cli.MODULE_TARGETS)}


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_the_cli_offers_exactly_what_bench_py_can_run(level: str) -> None:
    """A CLI target with no bench function cannot run; a bench function the CLI does not list
    cannot be reached through the front door, which is how `adaln_bwd` stayed unreachable."""
    in_bench, in_cli = LEVELS[level]
    assert set(in_cli) == set(in_bench), {
        "cli only": sorted(set(in_cli) - set(in_bench)),
        "bench.py only": sorted(set(in_bench) - set(in_cli)),
    }


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_every_target_owns_exactly_one_results_directory(level: str) -> None:
    """`benchmarks/<level>s/<target>/` is derived, never guessed -- so it has to be real, and no
    two targets may share one. Two targets in one directory is what silently dropped
    augmented_attention from the coverage report."""
    in_bench, _ = LEVELS[level]
    root = REPO / "benchmarks" / f"{level}s"
    for target in in_bench:
        assert (root / target).is_dir(), f"missing {root / target}"
    listed = {p.name for p in root.iterdir() if p.is_dir()}
    assert listed == set(in_bench), {
        "no target owns these directories": sorted(listed - set(in_bench)),
        "target has no directory": sorted(set(in_bench) - listed),
    }


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_the_bench_function_of_a_target_is_named_after_it(level: str) -> None:
    """bench.py asserts this at import, which no CPU test reaches. Assert it here too."""
    in_bench, _ = LEVELS[level]
    for target, fn in in_bench.items():
        assert fn == f"bench_{level}_{target}", f"{target}: {fn}"


def test_no_bench_target_abbreviates_a_name_the_engine_spells_out() -> None:
    """The rule that produced the current names, written as a check.

    Every kernel target names a kernel FAMILY from kernels/registry.csv, except four that bench a
    fused op SHAPE implemented by several families and are named after the shape. Anything else --
    `tri_attn` for `triangle_attention`, `ln_mask` for `fused_ln_mask` -- fails here.
    """
    import csv

    with (REPO / "src" / "miniworld_engine" / "kernels" / "registry.csv").open() as fh:
        families = {row["family"] for row in csv.DictReader(fh)}
    #: Targets that bench a fused GEMM/epilogue shape rather than one family. They are named after
    #: the shape because more than one family implements it, so no family name would be right.
    shape_named = {"dual_gemm_epilogue", "dual_gemm_epilogue_bwd", "gemm_epilogue",
                   "gemm_epilogue_bwd", "gemm_gate", "gemm_gate_bwd", "transition_b2b",
                   "transition_b2b_bwd", "conditioned_transition_tail"}
    for target in BENCH_KERNEL:
        if target in shape_named:
            continue
        stem = target.removesuffix("_bwd")
        assert stem in families, (
            f"kernel target {target!r} names neither a registry family nor a declared op shape; "
            f"families: {sorted(families)}")


def test_every_target_builds_from_a_real_case() -> None:
    """The CLI's target -> build-case mapping is the one place the two name spaces meet. A case
    name that no longer exists makes the pre-bench build fill nothing, and the bench then measures
    an untuned kernel."""
    for level, table in (("kernel", cli.KERNEL_TARGETS),
                         ("module", {k: v.cases for k, v in cli.MODULE_TARGETS.items()})):
        for target, cases in table.items():
            assert cases, f"{level} target {target!r} has no build case"
            unknown = [c for c in cases if c not in CASE_NAMES]
            assert not unknown, f"{level} target {target!r} -> unknown case(s) {unknown}"


def test_every_group_is_made_of_module_targets_and_all_means_all() -> None:
    """`bench_module all` has to mean every module target. It meant eight of nine while "all"
    read a table that `triangle_multiplication_bidirectional` had never been added to."""
    assert set(cli.GROUPS["all"]) == set(cli.MODULE_TARGETS)
    for group, members in cli.GROUPS.items():
        unknown = [m for m in members if m not in cli.MODULE_TARGETS]
        assert not unknown, f"group {group!r} names non-targets {unknown}"


def test_the_shape_ladder_and_the_dispatch_pins_name_real_targets() -> None:
    """SHAPES and PINS are keyed by module target. A stale key is a pin that never applies and a
    ladder that never overrides -- both invisible, both wrong."""
    for key in cli.SHAPES:
        if key == "default":
            continue
        assert key in cli.MODULE_TARGETS, f"SHAPES key {key!r} is not a module target"
    for switch, (_values, targets, _modes) in cli.PINS.items():
        unknown = [t for t in targets if t not in cli.MODULE_TARGETS]
        assert not unknown, f"PINS[{switch!r}] names non-targets {unknown}"
