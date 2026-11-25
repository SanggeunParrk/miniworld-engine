"""Compatibility alias: ``miniworld-engine`` was renamed ``foldjet-engine`` in 1.0.0.

``import miniworld_engine`` and ``import miniworld_engine.<sub>`` return the SAME module objects as ``foldjet_engine`` /
``foldjet_engine.<sub>`` (one module per name, no second copy of any state), with one ``FutureWarning`` per process. New code
imports ``foldjet_engine``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import foldjet_engine as _new

_OLD, _NEW = "miniworld_engine", "foldjet_engine"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``miniworld_engine.x.y`` -> the module ``foldjet_engine.x.y`` (imported once, shared under both names)."""

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD) :])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
warnings.warn(
    "miniworld_engine was renamed foldjet_engine (foldjet-engine 1.0.0); import foldjet_engine instead",
    FutureWarning,
    stacklevel=2,
)
sys.modules[_OLD] = _new
