"""CPU-side checks of the B200 e4m3-plane TriMul route (``b200_fp8``): import without a GPU, the scale arithmetic, the in-place refresh / rescale contract."""

import math
import os

import pytest
import torch

from foldjet_engine.kernels.trimul_inproj.cuda import b200_fp8 as F8


def test_off_by_default():
    if "MINIWORLD_TRIMUL_FP8" in os.environ or "MINIWORLD_FP8" in os.environ:
        pytest.skip("an FP8 switch is set in the environment")
    assert F8.POLICY is None


def test_pow2_scale_floor_and_zero():
    s = F8.pow2_scale(torch.tensor([0.0, 448.0, 449.0, 1e-30, 4.0, 1e-9]))
    assert s[0] == 1.0 and s[1] == 1.0 and s[2] == 2.0
    assert s[3] == F8.SCALE_FLOOR and s[5] == F8.SCALE_FLOOR  # never below 2^-30: a ~0 amax must not give a scale that saturates a later input
    assert s[4] == 2.0**-6 and 4.0 / s[4] <= 448.0 < 4.0 / (s[4] / 2)


@pytest.mark.parametrize("ng", [2, 4])
def test_derive_refresh_rescale_in_place(ng):
    chunks = 2 * ng
    s = torch.tensor([2.0**-3, 2.0**-4, 2.0**-5, 2.0**-6][:ng])
    st = {"s": s, "ng": ng, "chunks": chunks, "qinv": torch.empty(chunks), "csc": torch.empty(2)}
    F8._derive(st)
    qinv, csc = st["qinv"], st["csc"]
    assert torch.equal(qinv, (1.0 / s).repeat_interleave(2))
    assert csc[0] == (s[0] * s[2] if ng == 4 else s[0] * s[1]) and csc[1] == (s[1] * s[3] if ng == 4 else s[0] * s[1])
    F8._CACHES.clear()
    F8._CACHES[1] = st
    s0 = s.clone()  # (st["s"] is s: rescaled in place)
    f = F8.rescale_all(900.0)  # 2^ceil(log2(900 / 448 * 2)) = 8
    assert f == 2.0 ** math.ceil(math.log2(900 / 448 * 2))
    assert st["qinv"] is qinv and st["csc"] is csc  # in place: a captured graph keeps reading these tensors
    assert torch.equal(qinv, (1.0 / (s0 * f)).repeat_interleave(2))
    F8._CACHES.clear()


def test_min_l_table_and_counters():
    assert F8.MIN_L[(64, 1)] == 512 and F8.MIN_L_DEFAULT == 256
    assert set(F8.GUARD_STATS) == {"calls", "retries", "fallbacks", "capture_fallbacks", "calibrations"}
