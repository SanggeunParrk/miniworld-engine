"""Constexprs that are invisible to the autotune cache.

A `tl.constexpr` that is neither a tuned tile axis (supplied by the op's config CSV) nor in
`key=[...]` cannot be distinguished by the cache: two differently-compiled programs share one
entry, so the config tuned for one code path is served to the other. Triton still specializes per
constexpr value, so nothing fails -- the cache just cannot tell them apart.

Resolution is by the registry's (file, symbol) PAIR, never by the bare symbol. An earlier
throwaway version of this check keyed a `name -> constexprs` dict and unioned across files, which
silently merged the two different kernels both called `_gate_mul_kernel`
(`tm1/cute/launch.py` takes `BLOCK_E`; `trimul_inproj/triton/gate_elem.py` takes
`N, BLOCK_M1, BLOCK_K, SAVE_GATE, ADD_RESIDUAL, USE_DROPOUT`). Each then appeared to be missing
the other's parameters, producing two entirely fabricated findings and a third false conclusion --
that two "sibling" kernels disagreed about keying USE_DROPOUT, when one of them does not have it.
The same name-collision mistake, in the same repo, that `launch_bind.py` documents.

Reported, not decided: whether a visible constexpr BELONGS in the key is a judgement call. A
numeric tolerance (`eps`) never does; a value fully determined by another key entry does not
either. This tool finds candidates.
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import pathlib
from pathlib import Path

from miniworld_engine.autotune.configs import config_set

SRC = Path(__file__).resolve().parents[2]
REG = SRC / "miniworld_engine/kernels/registry.csv"
ALLOWED = Path(__file__).parent / "key_gaps_allowed.csv"
#: padding helpers derived from a keyed dim, never independent
IGNORE = {"HEAD_DIM_PAD", "N_PAD"}


def _kernel_ast(path: Path, symbol: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    want = symbol.split(".")[-1]
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == want:
            return fn
    return None


def _constexprs(fn: ast.FunctionDef) -> set[str]:
    a = fn.args
    return {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs
            if p.annotation and "constexpr" in ast.unparse(p.annotation)}


def _key_list(fn: ast.FunctionDef) -> list[str] | None:
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        if getattr(dec.func, "attr", getattr(dec.func, "id", "")) != "autotune":
            continue
        for kw in dec.keywords:
            if kw.arg == "key" and isinstance(kw.value, (ast.List, ast.Tuple)):
                return [e.value for e in kw.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def allowed() -> dict[tuple[str, str], str]:
    """(op, param) a human has judged NOT to belong in the key, with the reason.

    Without this the tool always reports the same ~19 findings and a NEW gap hides among them.
    Judging is not automatable -- `eps` is a tolerance, `D == K` is a launch-site identity, a
    stride equals a product of two keyed dims -- so the judgement is recorded rather than
    re-derived, and the file is the reviewable artefact.
    """
    if not ALLOWED.is_file():
        return {}
    return {(r["op"], r["param"]): r["reason"] for r in csv.DictReader(ALLOWED.open())}


def _axes_of(expr) -> set[str]:
    """The shape axes a ``shape_key=`` expression folds in.

    A launcher that can be handed a key from its caller writes a conditional --
    ``both_key(M, K=K) if shape_key is None else pack(shape_key, K=K)`` -- so the two branches are
    INTERSECTED: an axis counts only if BOTH paths fold it. A launcher that folds K on one path and
    not the other writes two different keys for one shape, which is the thing this check exists to
    catch, not to wave through.
    """
    if isinstance(expr, ast.Call):
        return {k.arg for k in expr.keywords if k.arg}
    if isinstance(expr, ast.IfExp):
        return _axes_of(expr.body) & _axes_of(expr.orelse)
    return set()


@functools.lru_cache(maxsize=1)
def _folded_into_shape_key() -> dict[tuple[str, str], set[str]]:
    """(file, symbol) -> the axes every launch of that kernel folds into ``shape_key``.

    ``shape_key.pack`` lets a launcher put a kernel's width axes INTO the shape key
    (``atom_key(L, H=H, HEAD_DIM=D)``) instead of listing them beside it in ``key=[...]``. They are
    then still keyed -- more finely, since the axis names are folded in too -- but this audit,
    which reads only the ``key=[...]`` list, would report them as invisible.

    So read the launches. For each ``<symbol>[grid](...)`` call, take the keywords of the
    ``shape_key=`` expression, and INTERSECT across every launch of that symbol: an axis only
    counts as folded if EVERY launcher folds it. One site that forgets it makes the whole kernel
    report the gap, which is the answer that matches what the cache does -- that site writes a key
    the others never read.
    """
    out: dict[tuple[str, str], set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        # `from ...persistent import _ln_bwd_persistent as _ln_bwd_persistent_jit` launches the
        # SAME kernel under another name. Without this the aliased site is invisible: it is not
        # intersected, so a site that forgets to fold an axis reports nothing and quietly writes a
        # key the other launchers never read -- the exact failure this check exists to prevent.
        alias, origin = {}, {}
        here = path.relative_to(SRC).parent
        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom):
                continue
            base = here
            for _ in range(max(0, n.level - 1)):
                base = base.parent
            mod = (n.module or "").replace(".", "/")
            src_file = f"{base.as_posix()}/{mod}.py" if n.level else f"{mod}.py".replace(
                "miniworld_engine/", "")
            for a in n.names:
                if a.asname:
                    alias[a.asname] = a.name
                origin[a.asname or a.name] = src_file
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # `<symbol>[grid](...)`: a Subscript whose value is the kernel name
            if not (isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name)):
                continue
            sk = next((k.value for k in node.keywords if k.arg == "shape_key"), None)
            folded = _axes_of(sk)
            local = f.value.id
            key = (str(path.relative_to(SRC)), alias.get(local, local),
                   origin.get(local, str(path.relative_to(SRC))))
            prev = out.get(key)
            out[key] = folded if prev is None else (prev & folded)
    return out


def _folds_for(kernel_file: str, symbol: str) -> set[str]:
    """The axes EVERY launch of this kernel folds into ``shape_key``.

    A launch site counts for this kernel when the launching file either IS the kernel's file, or
    IMPORTS that symbol from it. Resolving by symbol alone unions the four different kernels named
    ``_attn_fwd`` and the two named ``_gate_mul_kernel`` -- the mistake this module's audit
    docstring already records for a ``name -> constexprs`` dict. Resolving by DIRECTORY is not
    enough either: ``layernorm_linear/triton/mmajor_bwd.py`` launches ``layernorm``'s
    ``_ln_bwd_persistent``, so a directory rule drops the one site most likely to be forgotten.
    """
    def _norm(path: str) -> str:
        # registry rows, scanned paths and resolved imports disagree about the prefix
        # (`miniworld_engine/kernels/...` vs `kernels/...`). Compare from `kernels/` on.
        parts = pathlib.PurePosixPath(path).parts
        return "/".join(parts[parts.index("kernels"):]) if "kernels" in parts else path

    want = _norm(kernel_file)
    sets = [v for (f, sym, src), v in _folded_into_shape_key().items()
            if sym == symbol and want in (_norm(src), _norm(f))]
    return set.intersection(*sets) if sets else set()


def audit(config_dir: Path) -> tuple[list, int]:
    rows = list(csv.DictReader(REG.open()))
    findings, checked = [], 0
    for r in rows:
        if r["backend"] != "triton":
            continue
        fn = _kernel_ast(SRC / r["file"], r["symbol"])
        if fn is None:
            continue
        keys = _key_list(fn)
        if keys is None:
            continue
        spec = config_dir / f"{r['kernel']}.csv"
        axes: set[str] = set()
        if spec.is_file():
            head = list(csv.DictReader(spec.open(newline="")))
            if head and set(head[0]) >= {"axis", "values"}:
                axes = {h["axis"] for h in head}          # grid-spec form
            elif head:
                axes = set(head[0])                        # materialised form
        checked += 1
        ok = allowed()
        folded = _folds_for(r["file"], r["symbol"])
        gap = sorted(p for p in _constexprs(fn) - set(keys) - axes - folded - IGNORE
                     if (r["kernel"], p) not in ok)
        if gap:
            findings.append((r["file"], r["kernel"], sorted(keys), gap))
    return findings, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(config_set("accuracy")))
    args = ap.parse_args()
    findings, checked = audit(Path(args.config_dir))
    print(f"triton kernels checked: {checked}   "
          f"judged-and-recorded exclusions: {len(allowed())}   "
          f"UNEXPLAINED invisible constexpr: {len(findings)}")
    for f, op, keys, gap in findings:
        print(f"\n{f}\n    {op}\n      key={keys}\n      not-in-key={gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
