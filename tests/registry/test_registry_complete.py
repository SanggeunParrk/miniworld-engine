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
import sys

from paths import ROOT, registry_rows

SRC = ROOT / "src"

KERNELS = SRC / "miniworld_engine/kernels"

KINDS = {"gemm", "reduce", "elem"}
LEVELS = {"atom", "token", "both"}


def _rows() -> list[dict]:
    return registry_rows()


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


def test_dtypes_stay_within_what_the_level_permits() -> None:
    """Which operand dtypes a kernel must be tuned for, declared per level.

        token          bf16              pair/token-level kernels run bf16 only
        atom, both     bf16 and/or fp32

    A SUBSET, not an equality. The column's whole point is that level says WHERE a kernel is used
    and dtypes says WHAT it must be tuned for, and those are different facts -- but this test used
    to demand `dtypes == {"atom": "bf16|fp32"}[level]`, which is the derivation the column exists
    to avoid, and it made the over-declaration unavoidable: 29 kernels that are fixed-precision by
    construction were forced to claim both.

    What that cost: the work list took the claim at face value and built both halves, both
    measured the same thing, and the second wrote its result under a dtype the kernel never saw.
    192 of 922 units were that duplicate, and `audit` reported 192 declared-but-absent pairs as
    holes against a cache that was complete.

    Narrowing is therefore allowed and has to be earned:
    tests/registry/test_declared_dtypes_match_reality.py checks each narrowed row against what the shipped
    cache actually recorded, so a row cannot claim less than its kernel runs either.
    """
    permitted = {"token": {"bf16"}, "atom": {"bf16", "fp32"}, "both": {"bf16", "fp32"}}
    bad = []
    for r in _rows():
        got = {x for x in (r["dtypes"] or "").split("|") if x}
        if not got or not got <= permitted[r["level"]]:
            bad.append((r["kernel"], r["level"], r["dtypes"]))
    assert not bad, f"dtypes must be a non-empty subset of {permitted}; offending rows: {bad}"


def test_dtypes_use_the_declared_vocabulary() -> None:
    allowed = {"bf16", "fp32", "fp16"}
    bad = [(r["kernel"], r["dtypes"]) for r in _rows()
           if not r["dtypes"] or set(r["dtypes"].split("|")) - allowed]
    assert not bad, f"dtypes must be '|'-joined members of {sorted(allowed)}; offending: {bad}"


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


def test_the_harness_is_one_module_per_family() -> None:
    """The driver/check harness is laid out BY the `family` column, one module per family.

    It used to be eleven flat modules grouped by nothing the repo declares -- `drivers_attn.py`
    served three families, `drivers_trans.py` two, and the suffixes (`attn`, `ln`, `trans`,
    `trimul`) abbreviated no name in `registry.csv`. So "which module does this kernel's driver
    go in" had no answer a machine could give, and every new kernel guessed.

    The `family` column already answers it, so the harness follows it: row -> module, with no
    grouping left to drift. A helper several families share lives in `drivers/__init__.py` or
    `checks/__init__.py`; a kernel's own driver and checker live in its family's module.
    """
    bad = []
    for r in _rows():
        for col, pkg in (("driver", "drivers"), ("check", "checks")):
            spec = (r.get(col) or "").strip()
            if not spec:
                continue
            mod = spec.partition(":")[0]
            want = f"miniworld_engine.kernels.{pkg}.{r['family']}"
            if mod != want:
                bad.append((r["kernel"], col, mod, f"family is {r['family']}, so expected {want}"))
    assert not bad, ("driver/check modules that do not match their row's family:\n  "
                     + "\n  ".join(f"{k} [{c}]: {m} -- {why}" for k, c, m, why in bad))


def test_every_family_has_a_harness_module() -> None:
    """The other direction: a family named in the registry must have both modules on disk, so a
    new family cannot be declared with nowhere for its drivers to go."""
    families = sorted({r["family"] for r in _rows()})
    missing = [f"{pkg}/{f}.py" for f in families for pkg in ("drivers", "checks")
               if not (KERNELS / pkg / f"{f}.py").is_file()]
    assert not missing, f"registry families with no harness module: {missing}"


def test_kind_matches_the_source() -> None:
    """The hand-entered `kind` against what the source does, via src/miniworld_engine/build/classify.py.

    A mismatch is not automatically the row's fault -- a kernel that gained a `tl.dot` really did
    change kind -- so the failure names both sides and leaves the decision to a person.
    """
    tool = ROOT / "src/miniworld_engine/build/classify.py"
    if not tool.is_file():                       # absent from a wheel-only install
        return
    sys.path.insert(0, str(tool.parent))
    try:
        from miniworld_engine.build.classify import classify
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


def test_launch_keywords_match_the_kernel_signature() -> None:
    """Every `kernel[grid](..., NAME=...)` keyword must be a parameter of that kernel.

    This exists because of a real regression. Renaming 90 key-only `GROUP_M`/`GROUP_N`/`seq_group`
    parameters to `shape_key` rewrote the kernel signatures and the `key=[...]` lists, but skipped
    files that define no kernel of their own -- so 14 launch sites in four files kept passing the old
    keyword, one of them on a production path (`layernorm_linear/autograd.py`). Triton swallows an
    unknown keyword and then dies on the missing required one, so the message names the missing
    parameter and not the stale keyword that caused it:

        TypeError: dynamic_func() missing 1 required positional argument: 'shape_key'

    Nothing checked that the two sides agreed, so the break reached a pushed commit. Now something
    does. This is static -- no GPU, no launch.

    Scope: only calls whose callee name is a kernel defined somewhere in the package, and only
    keyword arguments. A `**kwargs` forward is skipped rather than guessed at, and so is any name
    the calling file REBINDS -- `from .cute import layernorm_linear as _fwd` makes the local `_fwd`
    a different function from the package's `_fwd`, and resolving by bare name cannot tell them
    apart. Without that exclusion this reported three false positives.
    """
    params: dict[str, set[str]] = {}
    starstar: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            names = ({a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
                     | {a.arg for a in fn.args.posonlyargs})
            if fn.args.kwarg is not None:
                starstar.add(fn.name)
            # a name defined twice in the package: union, so we only flag a keyword no
            # definition accepts
            params.setdefault(fn.name, set()).update(names)

    bad = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        rebound = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                rebound |= {a.asname or a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.Assign):
                rebound |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.keywords:
                continue
            f = node.func
            if isinstance(f, ast.Subscript):          # kernel[grid](...) -- the Triton launch
                name = getattr(f.value, "id", None) or getattr(f.value, "attr", None)
            elif isinstance(f, ast.Name):
                name = f.id
            else:
                continue
            if name is None or name not in params or name in starstar or name in rebound:
                continue
            for kw in node.keywords:
                if kw.arg is None:                    # **kwargs forward
                    continue
                if kw.arg not in params[name]:
                    bad.append(f"{path.relative_to(SRC)}:{node.lineno} "
                               f"{name}(..., {kw.arg}=...) -- not a parameter of {name}")
    assert not bad, "launch keywords that the kernel does not accept:\n" + "\n".join(
        f"  {b}" for b in bad)


def test_no_launch_omits_a_required_kernel_parameter() -> None:
    """A launch must supply every parameter the kernel has no default for.

    The sibling test above checks the OTHER direction -- every keyword PASSED must exist in the
    signature -- which catches a stale keyword and can never catch an omitted one. That gap was
    real: the shape-key unification added ``shape_key`` to ``_ln_bwd_persistent`` and to its
    ``key=['N', 'shape_key']``, updated the two callers inside persistent.py, and missed the one
    in ``layernorm_linear/triton/mmajor_bwd.py``. Every launch down that branch raised

        TypeError: dynamic_func() missing 1 required positional argument: 'shape_key'

    and since it is the wide-N large-M branch, the trimul backward died at L >= 548 with
    d_hidden > 128. The build recorded it as a one-line "skip", so it read as an unsupported
    shape rather than a defect.

    The analysis lives in ``miniworld_engine.build.launch_bind`` and is shared rather than
    reimplemented here: it took three corrections to become sound (resolve imports instead of
    matching bare names, skip starred calls, do not count dict subscripts as launches), and two
    copies would have drifted apart at the first of them. That module also reports what it CANNOT
    decide -- 15 attention launches spread their strides with ``*x.stride()``, whose arity is not
    knowable statically -- and those are covered by actually launching them (``run_all`` over the
    drivers, ``bucket_count`` over the build matrix).
    """
    from miniworld_engine.build.launch_bind import audit

    findings, _skipped, checked = audit()[:3]
    # A floor on "the audit still resolves launches", not a count of them: 94 resolve today and the
    # number moves whenever a kernel is added or removed. It was 100, which the removal of five
    # superseded kernels tripped.
    assert checked > 60, f"only {checked} launches were resolved; the audit stopped resolving"
    assert not findings, "kernel launch(es) missing a required parameter:\n  " + "\n  ".join(
        f"{f}:{ln} {n} omits {miss}" for f, ln, n, miss in findings)


def test_no_constexpr_is_invisible_to_the_autotune_cache() -> None:
    """A constexpr that is neither a tuned tile axis nor in ``key=[...]`` cannot be told apart by
    the cache: two differently-compiled programs share one entry, so the config tuned for one code
    path is served to the other. Nothing fails -- Triton still specializes per constexpr value --
    which is why these lasted.

    Judgement is not automatable and is therefore RECORDED, in
    ``miniworld_engine/build/key_gaps_allowed.csv``, one row per (op, param) with the reason: ``eps``
    is a tolerance, ``D == K`` is a launch-site identity, ``stride_m == H*HEAD_DIM`` is a product
    of two already-keyed dims. Without that file the check reports the same ~19 findings forever
    and a NEW gap hides among them; with it, the signal is zero.

    Resolution is by the registry's (file, symbol) PAIR. Keying a ``name -> constexprs`` dict and
    unioning across files -- which an earlier throwaway version did -- merges the two different
    kernels both named ``_gate_mul_kernel`` and the four both named ``_attn_fwd``, and invents
    findings out of the union: it reported 31 ops, of which roughly a third were parameters the
    named kernel does not have, while missing two ops in ``modules/`` entirely.
    """
    from miniworld_engine.autotune.configs import config_set
    from miniworld_engine.build.key_gaps import audit

    findings, checked = audit(config_set("accuracy"))
    assert checked > 70, f"only {checked} kernels resolved; the audit stopped resolving"
    assert not findings, (
        "constexpr(s) invisible to the autotune cache -- add to the kernel's key=[...], or record "
        "the judgement in miniworld_engine/build/key_gaps_allowed.csv:\n  "
        + "\n  ".join(f"{op} ({f}) not-in-key={gap}" for f, op, _k, gap in findings))


def test_length_of_refuses_an_already_flattened_shape() -> None:
    """`length_of` returns shape[-2], which is L only BEFORE the activation is flattened.

    Given a 2-D (M, D) it used to return M and say nothing. For a pair kernel M = L*L, which
    both_key clamps to the top bucket at any L >= 91, so every sequence length shared one cached
    config -- and the value that came back was a perfectly plausible integer, which is why this
    survived across the whole kernel layer. Measured before the drivers were fixed: 38 of 91 drivers
    recorded a bucket that did not move at all between L=256 and L=512.

    Refusing is safe because nothing legitimate passes 2-D: over a real module run, 25 of 25 call
    sites pass >= 3-D, and over the driver suite every 2-D call was a bug.
    """
    import pytest

    from miniworld_engine.autotune.shape_key import length_of

    assert length_of((1, 384, 384, 128)) == 384       # pair
    assert length_of((1, 512, 128)) == 512            # token/atom

    with pytest.raises(ValueError, match="already flattened"):
        length_of((147456, 128))                      # the real defect: L=384 pair, flattened
    with pytest.raises(ValueError, match="already flattened"):
        length_of((4096, 128))
