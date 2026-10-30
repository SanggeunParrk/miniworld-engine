"""``integrations.trimul_h100._legacy_native_faster`` (inference): which single-direction cells keep the packaged tmn K1 / K3
instead of the B200-style port (measured 2026-10-30, docs/records/trimul_h100_bigl_20261030.md)."""

from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.integrations import trimul_h100

ONE_DIRECTION = SimpleNamespace(outgoing=True)


def _legacy(d, n):
    return trimul_h100._legacy_native_faster(ONE_DIRECTION, torch.empty(1, n, n, d, device="meta"), True)


@pytest.mark.parametrize("n", [512, 768, 1024, 1152, 1536, 2048, 2560])
def test_d256_keeps_tmn_except_l384(n):
    assert _legacy(256, n)
    assert not _legacy(256, 384)


@pytest.mark.parametrize("n", [384, 512, 1024, 1152, 1536, 2048, 2560])
def test_d384_takes_the_port_at_every_length(n):
    """The port is 17 % faster than tmn at L1536-2560 (it already served L <= 1024); L % 128 != 0 falls back to tmn upstream."""
    assert not _legacy(384, n)
