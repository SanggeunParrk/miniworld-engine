"""Fixtures shared by the card-routing tests (tests/cpu/compile)."""
from __future__ import annotations

import dataclasses

import pytest

from miniworld_engine import settings
from miniworld_engine.modules import dispatch


@pytest.fixture(autouse=True)
def clean_routing_state():
    previous = settings.current()
    settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False, b200_portable_kernels=False, capture=False)
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()
    yield
    settings.configure(**dataclasses.asdict(previous))
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()


@pytest.fixture(autouse=True)
def clean_routing_unrecorded(monkeypatch):
    previous = settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False)
    yield
    settings.configure(**vars(previous))
    dispatch._unrecorded_warned.clear()
