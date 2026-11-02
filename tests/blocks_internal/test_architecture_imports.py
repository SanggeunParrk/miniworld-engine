"""The internal blocks package imports on its own (the names the coverage driver and InputFeatureEmbedder build on)."""

import importlib
import pkgutil

import miniworld_engine._blocks as architecture


def test_every_module_imports():
    names = [
        m.name
        for m in pkgutil.walk_packages(
            architecture.__path__, architecture.__name__ + "."
        )
    ]
    assert names
    for name in names:
        importlib.import_module(name)


def test_public_names():
    from miniworld_engine._blocks.blocks import (
        DiffusionTransformer,
        Pairformer,
        SWAAtomTransformer,
    )
    from miniworld_engine._blocks.exceptions import ImplementationType
    from miniworld_engine._blocks.typing import typecheck

    assert callable(typecheck)
    assert {
        DiffusionTransformer,
        Pairformer,
        SWAAtomTransformer,
        ImplementationType,
    }
