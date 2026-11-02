"""``kernels.integrations.trimul_h100._legacy_native_faster`` (inference): single-direction D384 never keeps the packaged tmn K1 / K3
and D256 only below L256; the B200-style port serves the rest (records trimul_h100_bigl_20261030.md, trimul_h100_d256_20261031.md)."""

from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.kernels.integrations import trimul_h100

ONE_DIRECTION = SimpleNamespace(outgoing=True)


def _legacy(d, n):
    return trimul_h100._legacy_native_faster(ONE_DIRECTION, torch.empty(1, n, n, d, device="meta"), True)


@pytest.mark.parametrize("d", (256, 384))
@pytest.mark.parametrize("n", [256, 384, 512, 1024, 1152, 1536, 2048, 2560])
def test_one_direction_wide_takes_the_port(d, n):
    assert not _legacy(d, n)


@pytest.mark.parametrize("n", [128, 200])
def test_d256_below_l256_keeps_the_packaged_path(n):
    assert _legacy(256, n)
    assert not _legacy(384, n)


def test_lt_table_keys_are_well_formed():
    """``_h100_bmm.LT_CONFIGS``: (L, a transposed, planes per launch) -> (tile id, custom option, cluster shape id)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm

    for (n, transposed, planes), cfg in _h100_bmm.LT_CONFIGS.items():
        assert n % 64 == 0 and isinstance(transposed, bool) and planes > 0
        assert len(cfg) == 3 and all(isinstance(v, int) for v in cfg)
