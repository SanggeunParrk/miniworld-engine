"""Every `IMPL_MIN_ARCH` key must name a target and an implementation that exist.

The table is what stops `bench_kernel all` offering an H100 kernel to an A100: 99 of the 307 failed
rows in one A100 sweep were a card being asked for an implementation it cannot run, answered with
`AssertionError: SM90 (H100) only`, `NotImplementedError: Gemm Sm80 is not implemented yet` or
`OpError: expects arch to be sm_90a`.

It has to be DECLARED rather than derived, which is why it needs a test at all. Both derivations
were tried and neither resolves: an implementation's `path=` string names a function or a package
as often as a file (22 of 34 match `registry.csv`'s `file`), and its imports go through the flat
`miniworld_engine.kernels` re-export (16 of 32). A declaration drifts, and this drift is silent --
a key naming a renamed target simply stops gating, and the rows come back among the real failures.

Read with `ast`: bench.py raises at import without a GPU, as the other bench tests here note.

The arch VALUES are not checked. They are what each card actually answered; no CPU test can
confirm an sm90 claim.
"""
from __future__ import annotations

import ast

from paths import ROOT

BENCH = ROOT / "benchmarks" / "runners" / "bench.py"
TREE = ast.parse(BENCH.read_text())


def _dict_literal(name: str) -> ast.Dict:
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Dict)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node.value
    raise AssertionError(f"{name} not found in {BENCH}")


def _target_names(name: str) -> set[str]:
    return {k.value for k in _dict_literal(name).keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _gated() -> list[tuple[str, str, str]]:
    """(target, implementation, arch) for every IMPL_MIN_ARCH entry."""
    out = []
    d = _dict_literal("IMPL_MIN_ARCH")
    for k, v in zip(d.keys, d.values, strict=True):
        assert isinstance(k, ast.Tuple), "an IMPL_MIN_ARCH key is a (target, implementation) tuple"
        assert len(k.elts) == 2, "an IMPL_MIN_ARCH key has exactly two parts"
        out.append((k.elts[0].value, k.elts[1].value, v.value))
    return out


def _impls_of(target: str) -> set[str]:
    """The `implementation == "..."` names in that target's bench function, as `target_impls` reads
    them. A module-level target has no such chain and returns an empty set, which is not checked."""
    for node in ast.walk(TREE):
        # Exact names, not endswith: `bench_kernel_dual_gemm_epilogue` ends with
        # "gemm_epilogue" and answered for it, which is how this helper first read the wrong
        # function's implementations.
        if not (isinstance(node, ast.FunctionDef)
                and node.name in (f"bench_kernel_{target}", f"bench_module_{target}")):
            continue
        names = set()
        for n in ast.walk(node):
            if (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
                    and n.left.id == "implementation"):
                for cmp_ in n.comparators:
                    elts = ([cmp_] if isinstance(cmp_, ast.Constant)
                            else list(getattr(cmp_, "elts", [])))
                    names |= {e.value for e in elts
                              if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        return names
    return set()


def test_every_key_names_a_target_that_exists() -> None:
    known = _target_names("KERNEL_TARGETS") | _target_names("MODULE_TARGETS")
    bad = sorted({t for t, _, _ in _gated() if t not in known})
    assert not bad, (
        f"IMPL_MIN_ARCH names target(s) that do not exist: {bad}. A key that matches nothing gates "
        f"nothing, and the rows it was added for come back as bench failures.")


def test_every_key_names_an_implementation_that_target_defines() -> None:
    bad = []
    for target, impl, arch in _gated():
        impls = _impls_of(target)
        if impls and impl not in impls:
            bad.append(f"{target}: {impl!r} (needs {arch}) is not one of {sorted(impls)}")
    assert not bad, "\n  ".join(
        ["IMPL_MIN_ARCH names an implementation the target does not define:", *bad])
