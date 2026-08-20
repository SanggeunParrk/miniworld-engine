"""registry.csv is the declaration of what kernels exist. These tests make it stay one.

The rule this enforces: a new kernel is not finished until it has a row here, with `kind` and
`level` filled in BY HAND. Both halves matter.

* Without the row, the kernel is invisible: `autotune.run_all`, the coverage report and every
  accuracy sweep iterate the registry, so an undeclared kernel is not "untested", it is unknown.
  The repo used to discover kernels by walking the AST for `configs_for(...)`; that inferred the
  answer and got it wrong in both directions, which is why the declaration exists.
* Without `kind`/`level`, the row is present but says nothing about what the kernel is or where the
  model uses it. `level` in particular CANNOT be derived -- it is a property of the architecture,
  not of the kernel's text -- so a machine cannot fill it in later.

The classification is hand-entered, and `test_kind_matches_the_source` checks it against what the
source actually does rather than replacing it. Hand-declared, machine-verified: a disagreement is a
question for a human (did the kernel change, or was the row wrong?), not something to overwrite.
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REG = SRC / "miniworld_engine/kernels/registry.csv"
KERNELS = SRC / "miniworld_engine/kernels"

KINDS = {"gemm", "reduce", "elem"}
LEVELS = {"atom", "token", "both"}


def _rows() -> list[dict]:
    with REG.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_column_is_filled() -> None:
    """`check` may be empty (a kernel that cannot run on any available card has no reference);
    every other column must have a value, and `driver` must too -- a kernel nothing can launch is
    a hole, and an empty cell hides it."""
    rows = _rows()
    required = ("kernel", "backend", "family", "kind", "level", "file", "symbol", "driver")
    missing = [(r["kernel"] or "<no name>", c) for r in rows for c in required
               if not (r.get(c) or "").strip()]
    assert not missing, f"empty cells in registry.csv: {missing}"


def test_kind_and_level_use_the_declared_vocabulary() -> None:
    bad = [(r["kernel"], r["kind"], r["level"]) for r in _rows()
           if r["kind"] not in KINDS or r["level"] not in LEVELS]
    assert not bad, (f"kind must be one of {sorted(KINDS)} and level one of {sorted(LEVELS)}; "
                     f"offending rows: {bad}")


def test_level_is_consistent_within_a_family() -> None:
    """Where a family is used is a property of the family, so two rows of one family disagreeing is
    a typo, not a distinction."""
    seen: dict[str, str] = {}
    bad = []
    for r in _rows():
        prev = seen.setdefault(r["family"], r["level"])
        if prev != r["level"]:
            bad.append((r["family"], prev, r["kernel"], r["level"]))
    assert not bad, f"one family with two levels: {bad}"


def test_names_and_files_resolve() -> None:
    rows = _rows()
    dupes = [k for k, n in __import__("collections").Counter(r["kernel"] for r in rows).items()
             if n > 1]
    assert not dupes, f"duplicate kernel names: {dupes}"
    missing = [(r["kernel"], r["file"]) for r in rows if not (SRC / r["file"]).is_file()]
    assert not missing, f"registry rows whose file does not exist: {missing}"


def test_every_autotuned_op_is_declared() -> None:
    """Every `configs_for("op")` in the tree must have a row. This is the direction that actually
    catches a forgotten kernel: the op string is what Triton keys the config set on, so a kernel
    with configs but no row runs in production and appears in no sweep."""
    declared = {r["kernel"] for r in _rows()}
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        if "autotune/data" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "configs_for"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found[node.args[0].value] = str(path.relative_to(SRC))
    undeclared = {op: where for op, where in found.items() if op not in declared}
    assert not undeclared, (
        "these ops ask for autotune configs but have no registry.csv row -- add one, with kind and "
        f"level filled in: {undeclared}")


def test_drivers_and_checkers_resolve() -> None:
    """`module:function` must name something that exists. Importing the modules needs a GPU, so
    this parses instead: the function has to be defined in the named module's source."""
    bad = []
    for r in _rows():
        for col in ("driver", "check"):
            spec = (r.get(col) or "").strip()
            if not spec:
                continue
            mod, _, fn = spec.partition(":")
            if not fn:
                bad.append((r["kernel"], col, spec, "not 'module:function'"))
                continue
            path = SRC / (mod.replace(".", "/") + ".py")
            if not path.is_file():
                bad.append((r["kernel"], col, spec, "module file not found"))
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError as exc:
                bad.append((r["kernel"], col, spec, f"module does not parse: {exc}"))
                continue
            names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            if fn not in names:
                bad.append((r["kernel"], col, spec, "function not defined at module level"))
    assert not bad, f"unresolvable driver/checker references: {bad}"


def test_kind_matches_the_source() -> None:
    """The hand-entered `kind` against what the source does, via tools/kernel-audit/classify.py.

    A mismatch is not automatically the row's fault -- a kernel that gained a `tl.dot` really did
    change kind -- so the failure names both sides and leaves the decision to a person.
    """
    tool = ROOT / "tools/kernel-audit/classify.py"
    if not tool.is_file():                       # tools/ is not shipped in a wheel
        return
    sys.path.insert(0, str(tool.parent))
    try:
        from classify import classify
    finally:
        sys.path.pop(0)
    bad = []
    for r in _rows():
        measured, signals, how = classify(SRC / r["file"], r["symbol"])
        if how != "closure":
            bad.append((r["kernel"], "symbol not resolved in its file", how))
        elif measured != r["kind"]:
            bad.append((r["kernel"], f"row says {r['kind']}, source says {measured}",
                        signals or "no gemm/reduce signal found"))
    assert not bad, "kind disagrees with the source:\n" + "\n".join(
        f"  {k}: {why} [{extra}]" for k, why, extra in bad)


def test_the_rule_is_written_down() -> None:
    """The rule has to be findable by someone adding a kernel, not only enforced after the fact."""
    spec = (ROOT / "docs/kernels/naming.md").read_text()
    assert "registry.csv" in spec, "docs/kernels/naming.md must document the registry requirement"
