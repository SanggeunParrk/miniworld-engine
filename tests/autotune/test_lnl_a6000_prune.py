"""Keep the proven A5000/A6000 shared-memory fault out of tuning without widening its scope."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import triton

from miniworld_engine.kernels.layernorm_linear.triton.fused import _prefer_covering_lnl


def _config(*, bk=128, bm=64, bn=128, warps=1, stages=1):
    return triton.Config({"BLOCK_K": bk, "BLOCK_M1": bm, "BLOCK_N": bn},
                         num_warps=warps, num_stages=stages)


def _args(**overrides):
    args = {"K": 128, "N": 520, "M": 147456,
            "x_ptr": SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cuda:0"))}
    args.update(overrides)
    return args


@pytest.fixture(autouse=True, params=["NVIDIA RTX A5000", "NVIDIA RTX A6000"])
def ampere_card(monkeypatch, request):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: request.param)


@pytest.mark.parametrize("rows", [128, 147456])
def test_only_faulting_schedule_is_removed(rows):
    bad = _config()
    good = [_config(stages=2), _config(warps=2), _config(warps=4), _config(warps=8),
            _config(bn=256), _config(bm=32)]
    assert _prefer_covering_lnl([bad, *good], _args(M=rows)) == good


@pytest.mark.parametrize("change", ["n512", "n528", "k64", "fp32", "fp16", "cpu",
                                    "sm80", "sm89", "sm90", "unverified_sm86", "missing_x"])
def test_other_inputs_and_devices_keep_the_candidate(change, monkeypatch):
    args = _args()
    if change.startswith("n"):
        args["N"] = int(change[1:])
    elif change == "k64":
        args["K"] = 64
    elif change in ("fp32", "fp16"):
        args["x_ptr"].dtype = torch.float32 if change == "fp32" else torch.float16
    elif change == "cpu":
        args["x_ptr"].device = torch.device("cpu")
    elif change.startswith("sm"):
        cap = (int(change[2]), int(change[3]))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: cap)
    elif change == "unverified_sm86":
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 3090")
    else:
        args.pop("x_ptr")
    candidate = _config()
    assert _prefer_covering_lnl([candidate], args) == [candidate]


def test_covering_and_k_tiled_selection_remains_intact():
    small, covering, wide = _config(bk=64), _config(warps=2), _config(bk=256)
    assert _prefer_covering_lnl([small, covering, wide], _args()) == [covering]
    # The issue concerns BLOCK_K=128 only; a larger sole covering tile stays selectable.
    assert _prefer_covering_lnl([wide], _args()) == [wide]
    assert _prefer_covering_lnl([small, covering, wide], _args(K=512)) == [small, covering, wide]
