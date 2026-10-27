"""Fixtures of the card-routing tests that several modules in this directory share."""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import records


@pytest.fixture
def manifests(tmp_path):
    with records.use_manifest_dir(tmp_path):
        yield tmp_path
