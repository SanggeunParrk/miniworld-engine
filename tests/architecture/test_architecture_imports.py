"""The architecture package imports on its own and keeps the names MiniWorld builds on."""

import importlib
import pkgutil

import miniworld_engine.architecture as architecture


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
    from miniworld_engine.architecture.blocks import (
        DiffusionTransformer,
        Pairformer,
        SWAAtomTransformer,
    )
    from miniworld_engine.architecture.diffusion.edm import EDMScheduler
    from miniworld_engine.architecture.exceptions import ImplementationType
    from miniworld_engine.architecture.typing import typecheck

    assert callable(typecheck)
    assert {
        DiffusionTransformer,
        Pairformer,
        SWAAtomTransformer,
        EDMScheduler,
        ImplementationType,
    }
