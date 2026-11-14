"""Override ``miniworld_engine.settings`` fields for one test (what ``monkeypatch.setenv`` did for the old switches)."""

import dataclasses
import typing

from miniworld_engine import settings


def _coerce(name: str, value):
    hint = {f.name: f.type for f in dataclasses.fields(settings.Settings)}[name]
    hint = hint if isinstance(hint, str) else getattr(hint, "__name__", str(hint))
    if value is None:
        field = next(f for f in dataclasses.fields(settings.Settings) if f.name == name)
        if field.default is not dataclasses.MISSING:
            return field.default
        factory = field.default_factory
        return factory() if callable(factory) else None
    if hint.startswith("frozenset") and not isinstance(value, str):
        return frozenset(value)
    if not isinstance(value, str):
        return value
    if hint.startswith("frozenset[int"):
        return frozenset(int(p) for p in value.replace(",", " ").split())
    if hint.startswith("bool"):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if hint.startswith("int"):
        return int(value) if value.strip() else None
    return value


def override_setting(monkeypatch, name: str, value) -> None:
    """Set ``settings.<name>`` for the rest of the test. ``value`` None restores the default; strings are converted by field type."""
    monkeypatch.setattr(
        settings, "_ACTIVE", settings.current()
    )  # restored by monkeypatch at teardown
    settings.configure(**{name: _coerce(name, value)})





import contextlib

import pytest
import torch


@contextlib.contextmanager
def forced_triton():
    """Run a block with the internal A/B switch ``settings.force_portable_kernels`` on (what ``implementation=TRITON`` used to say for one module)."""
    previous = settings.configure(force_portable_kernels=True)
    try:
        yield
    finally:
        settings.configure(**vars(previous))

__all__ = ["forced_triton", "override_setting", "typing"]


@pytest.fixture(autouse=True)
def policy_blackwell_portable_off():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.fixture(autouse=True)
def policy_b200_portable_off():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.fixture(autouse=True)
def policy_portable_off():
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.fixture(autouse=True)
def policy_h100_portable_off():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))
