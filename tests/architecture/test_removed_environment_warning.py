"""A removed environment switch is reported, not ignored silently."""

import pytest

from miniworld_engine import settings


def test_a_removed_variable_is_reported_with_its_replacement(monkeypatch):
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_TRAIN", "0")
    with pytest.warns(UserWarning, match="bias_only_dit_train"):
        settings._warn_about_removed_environment()


def test_a_deleted_switch_is_reported_as_removed(monkeypatch):
    monkeypatch.setenv("MINIWORLD_TRIATTN_Q_FWD", "0")
    with pytest.warns(UserWarning, match="MINIWORLD_TRIATTN_Q_FWD \\(removed\\)"):
        settings._warn_about_removed_environment()


def test_nothing_is_reported_without_them(monkeypatch, recwarn):
    for name in settings._REMOVED_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    settings._warn_about_removed_environment()
    assert not [w for w in recwarn if "no longer reads" in str(w.message)]
