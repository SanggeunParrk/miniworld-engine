"""The portable family is pure Triton: no file of a ``kernels/*/triton/`` tree (nor a shared Triton helper) reaches for hand-CUDA.

Which of the two families runs is decided by the CALLER (module / interface routing, ``kernels/_pair_route.py``, ``modules/dispatch.py``,
``kernels/_ops_routed.py``), never from inside a Triton file. The ladder (hand-CUDA < portable Triton < PyTorch) measures the portable rung with
``settings.force_portable_kernels=True`` on H100, so a CUDA call hidden in a Triton file contaminates that rung. This test reads the sources
(AST, no GPU) and fails on, in any file of a ``kernels/*/triton/`` tree or a shared Triton helper:

* an import of a module with a ``cuda`` path component (``kernels.<family>.cuda...``, ``from ..cuda import``, ``from <pkg> import cuda``),
* an import of ``miniworld_engine.integrations`` (the hand-CUDA stages),
* an import of the extension builders (``_nvcc``, ``prebuilt``, ``torch.utils.cpp_extension``),
* a call of ``load_extension`` / ``load_inline`` / ``_ext(...)`` or a ``torch.ops.miniworld*`` / ``torch.ops.mwln*`` access.

It also pins the card-specific branches of those files: a ``get_device_name`` call or a card name in code must be allow-listed with its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
KERNELS = REPO / "src" / "miniworld_engine" / "kernels"
PKG = "miniworld_engine.kernels"

#: Shared Triton helpers outside a ``triton/`` tree.
SHARED_TRITON_HELPERS = ("_tma.py", "_compile.py", "_tiles.py", "_dot_precision.py")

#: EXPLICIT exceptions to the import rules: (path under kernels/, imported module) -> why it is not a hand-CUDA call.
ALLOW_IMPORTS = {
    ("adaln/triton/training.py", "miniworld_engine.integrations.h100_master"):
        "H100 master-weight pack: one multi-tensor PyTorch bf16 copy (torch._foreach_copy_), no CUDA kernel; card-specific branch, reported in docs/records/pure_triton_trees_20261027.md",
    ("conditioned_transition/triton/training.py", "miniworld_engine.integrations.h100_master"):
        "same H100 master-weight pack (a PyTorch cast), no CUDA kernel",
}

#: EXPLICIT card-specific code (a ``get_device_name`` call, a card name in code) in a Triton file: path -> why it stays.
ALLOW_CARD_SPECIFIC = {
    "layernorm_linear/triton/fused.py":
        "A5000/A6000 (sm_86) workaround: one autotune schedule (K=128, N=520, BLOCK_M1=64, BLOCK_N=128, 1 warp, 1 stage) makes tl.dot write outside "
        "shared memory there (reproduced with compute-sanitizer); pruned from the search on those two cards only",
}

_BUILDERS = {"_nvcc", "prebuilt", "cpp_extension"}
_BUILD_CALLS = {"load_extension", "load_inline", "_ext"}
_CARDS = ("A5000", "A6000", "H100", "B200", "A100")


def _allowed(rel: str, what: str) -> bool:
    return any(rel == r and (what == m or what.startswith(m + ".")) for r, m in ALLOW_IMPORTS)


def _files() -> list[Path]:
    files = [p for p in KERNELS.glob("*/triton/**/*.py")]
    files += [KERNELS / name for name in SHARED_TRITON_HELPERS if (KERNELS / name).is_file()]
    return sorted(files)


def _resolve(path: Path, node: ast.ImportFrom) -> str:
    """The absolute dotted module of a ``from ... import`` (relative levels resolved against the file's package)."""
    if not node.level:
        return node.module or ""
    package = ["miniworld_engine", *path.relative_to(REPO / "src" / "miniworld_engine").with_suffix("").parts[:-1]]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _violations(path: Path, source: str) -> list[tuple[str, str]]:
    """``(kind, what)`` for every hand-CUDA reach in ``source`` (a Triton file at ``path``)."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(path, node)
            modules = [base, *(f"{base}.{a.name}" for a in node.names)]
        for m in modules:
            parts = m.split(".")
            if parts[0] == "miniworld_engine" and "cuda" in parts[2:]:
                found.append(("imports a cuda module", m))
            elif m == "miniworld_engine.integrations" or m.startswith("miniworld_engine.integrations."):
                found.append(("imports an integration", m))
            elif _BUILDERS & set(parts):
                found.append(("imports an extension builder", m))
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in _BUILD_CALLS:
                found.append(("builds / loads an extension", name))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            chain = ast.unparse(node)
            if chain.startswith("torch.ops.") and chain.split(".")[2].startswith(("miniworld", "mwln")):
                found.append(("calls a hand-CUDA torch op", chain))
    return found


def _card_specific(source: str) -> list[str]:
    """Card-name checks in CODE (not docstrings or comments): ``get_device_name`` calls and string literals naming a card."""
    tree = ast.parse(source)
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(getattr(n.body[0], "value", None), ast.Constant)
    }
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "get_device_name":
            out.append("get_device_name")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings and any(c in node.value for c in _CARDS):
            out.append(node.value)
    return out


def test_triton_trees_do_not_reach_for_cuda():
    files = _files()
    assert len(files) > 100, "the Triton trees were not found"
    bad = []
    for path in files:
        rel = path.relative_to(KERNELS).as_posix()
        for kind, what in _violations(path, path.read_text()):
            if not _allowed(rel, what):
                bad.append(f"{rel}: {kind}: {what}")
    assert not bad, "a Triton file must not call hand-CUDA (the caller picks the family):\n  " + "\n  ".join(bad)


def test_allow_listed_exceptions_are_still_needed():
    """An allow-list entry that no longer matches anything is dead weight that would hide a future violation."""
    seen = set()
    for path in _files():
        rel = path.relative_to(KERNELS).as_posix()
        seen |= {(rel, what) for _, what in _violations(path, path.read_text())}
    stale = [k for k in ALLOW_IMPORTS if not any(k[0] == r and (w == k[1] or w.startswith(k[1] + ".")) for r, w in seen)]
    assert not stale, f"stale ALLOW_IMPORTS entries: {stale}"
    assert all(reason.strip() for reason in ALLOW_IMPORTS.values())


def test_card_specific_code_in_triton_trees_is_listed():
    found = {}
    for path in _files():
        rel = path.relative_to(KERNELS).as_posix()
        hits = _card_specific(path.read_text())
        if hits:
            found[rel] = hits
    unexpected = {rel: hits for rel, hits in found.items() if rel not in ALLOW_CARD_SPECIFIC}
    assert not unexpected, f"card-specific code in a portable Triton file (list it with its reason, or remove it): {unexpected}"
    stale = [rel for rel in ALLOW_CARD_SPECIFIC if rel not in found]
    assert not stale, f"stale ALLOW_CARD_SPECIFIC entries: {stale}"


def test_the_detector_sees_what_it_is_meant_to_see():
    """The scanner itself: each forbidden shape is found in a synthetic Triton file, a clean one is not."""
    path = KERNELS / "layernorm" / "triton" / "synthetic.py"
    cases = {
        "from ..cuda import layer_norm_bwd_cuda": "imports a cuda module",
        "from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd": "imports a cuda module",
        "from miniworld_engine.kernels.augmented_attention import cuda as sm90": "imports a cuda module",
        "from miniworld_engine.integrations.token_dit import update": "imports an integration",
        "from miniworld_engine.kernels._nvcc import load_extension": "imports an extension builder",
        "x = load_extension(name='a')": "builds / loads an extension",
        "y = m._ext('attn')": "builds / loads an extension",
        "torch.ops.miniworld.something(x)": "calls a hand-CUDA torch op",
    }
    for src, kind in cases.items():
        assert kind in {k for k, _ in _violations(path, src)}, src
    assert not _violations(path, "import triton\nfrom miniworld_engine.kernels.layernorm.triton.main import triton_layernorm\ntorch.ops.aten.foo(x)\n")
    assert _card_specific("def f(x):\n    return torch.cuda.get_device_name(x)\n") == ["get_device_name"]
    assert _card_specific('def f():\n    """Measured on an A5000."""\n    # an H100 comment\n    return 1\n') == []
