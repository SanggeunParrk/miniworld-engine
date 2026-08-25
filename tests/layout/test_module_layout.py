"""Every model-level op has the same shape on disk.

``modules/__init__.py`` has always opened by stating the rule -- "Each op is its own folder
(``modules/<op>/``)" -- and eight of the eleven ops followed it. ``attention_pair_bias``,
``msa_pair_weighted_averaging`` and ``swa_atom_attention`` were flat files, so the sentence
described the majority rather than the layout, and the next op had two neighbours to copy.

The shape, and what each part is for:

``module.py``     the connecting nn.Module. Every op folder has one; it is what the folder is for.
``reference.py``  the torch definition, where the op has one distinct from the module itself.
``dispatch.py``   a CHOICE among implementations, when the op makes one of its own.
anything else     an additional implementation of the same op (``bidirectional.py``,
                  ``baseline_dtv1.py``, ``graph_runner.py``). Free-form: an op with two forms has
                  two files, and nothing is gained by demanding a fixed name for the second.

Four modules sit flat under ``modules/`` and are NOT ops -- they are the shared infrastructure the
ops are built from. They are listed in :data:`SHARED` so a fifth cannot appear without a decision,
and so a new flat *op* file fails here instead of quietly becoming the fourth exception.

One import style, absolute, is NOT checked here -- ruff's TID252 bans relative imports across
the whole package, so a second check in this file would only be a place for the two to disagree.
It matters for this rule though: ``from .ops import sigmoid_gate`` kept working from
``modules/attention_pair_bias.py`` and resolved to nothing from
``modules/attention_pair_bias/module.py``, so the flat-to-folder move is exactly when a relative
import breaks.

The kernel-side twin of this file is ``tests/layout/test_kernel_layout.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULES = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "miniworld_engine" / "modules"

#: The flat modules that are infrastructure, not ops. Anything else flat under `modules/` is an op
#: that skipped the folder rule.
SHARED = {
    "__init__.py",
    "dispatch.py",      # public ImplementationType -> internal KernelBackend
    "exceptions.py",    # the public implementation enum and its error
    "primitives.py",    # layer classes the ops compose (Linear, LayerNorm, Dropout, MPLinear)
    "functional.py",    # free functions the ops compose (sigmoid_gate, swish_gate)
}


def _ops() -> list[Path]:
    return sorted(p for p in MODULES.iterdir() if p.is_dir() and p.name != "__pycache__")


@pytest.mark.parametrize("op", _ops(), ids=lambda p: p.name)
def test_an_op_is_a_package_with_a_module(op: Path) -> None:
    """The folder rule, both halves of it.

    ``__init__.py`` because a namespace package works by PEP 420 accident and hides a typo in the
    directory name; ``module.py`` because that is the file the rule promises is there -- every
    importer, and every reader looking for the nn.Module, goes to the same place.
    """
    assert (op / "__init__.py").is_file(), f"{op.name}/ is not a package"
    assert (op / "module.py").is_file(), f"{op.name}/ has no module.py"


def test_nothing_flat_under_modules_except_the_declared_infrastructure() -> None:
    """A new op added as a flat file fails here, naming the fix."""
    flat = {p.name for p in MODULES.iterdir() if p.is_file() and p.suffix == ".py"}
    extra = sorted(flat - SHARED)
    assert not extra, (
        f"{extra} sit flat under modules/. An op is a folder: move it to "
        f"modules/<op>/module.py with an __init__.py that re-exports its class. If it is shared "
        f"infrastructure and not an op, add it to SHARED in this file with a comment saying what "
        f"it is.")
    gone = sorted(SHARED - flat - {"__init__.py"})
    assert not gone, f"SHARED names {gone}, which no longer exist"


def test_every_exported_name_resolves() -> None:
    """``__all__`` is the package's contract; a name in it that does not import is a broken one."""
    import miniworld_engine.modules as modules

    missing = [n for n in modules.__all__ if not hasattr(modules, n)]
    assert not missing, missing


def test_the_exports_and_the_imports_are_both_sorted() -> None:
    """Unsorted, the two lists drift into different orders and a name goes missing from one of
    them without any diff looking wrong -- which is how ``AttentionPairBias`` came to sit between
    ``Linear`` and ``MSAPairWeightedAveraging`` under its old name."""
    import miniworld_engine.modules as modules

    assert list(modules.__all__) == sorted(modules.__all__)
    tree = ast.parse((MODULES / "__init__.py").read_text())
    froms = [n.module for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and n.module and n.level == 0]
    assert froms == sorted(froms), froms
