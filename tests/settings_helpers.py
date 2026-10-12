"""Override ``miniworld_engine.settings`` fields for one test (what ``monkeypatch.setenv`` did for the old switches)."""

import dataclasses
import typing

from miniworld_engine import settings


def _coerce(name: str, value):
    hint = {f.name: f.type for f in dataclasses.fields(settings.Settings)}[name]
    hint = hint if isinstance(hint, str) else getattr(hint, "__name__", str(hint))
    if value is None:
        field = next(f for f in dataclasses.fields(settings.Settings) if f.name == name)
        return (
            field.default
            if field.default is not dataclasses.MISSING
            else field.default_factory()
        )
    if not isinstance(value, str):
        return value
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


__all__ = ["override_setting", "typing"]
