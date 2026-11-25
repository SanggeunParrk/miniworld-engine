"""Extreme-logit behaviour of the FP8-PV 32 / 16-channel TriAttn core (``triattn_h16_sm90.cu`` -DTA_F8PV=1).

P is built as an e4m3 byte directly: ``byte = round(8 x + 56)`` with ``x = logit - m + 8``.  Byte 127 is the e4m3 NaN and 128+ is a negative number, so x must
stay <= 8.75.  The row offset ``m`` is rounded to fp16 and only rises when a logit exceeds it by TA_TAU = 0.5, and the fp16 ``logit * scale + 8 - m`` rounds
too: with logits in the hundreds / thousands x overshoots 8.875.  The core clamps x (like the 64-channel core).  This test drives large |q.k|, huge
biases (up to the fp16 limit), -inf / -65504 masks and all-masked rows, and checks finite outputs and a bounded error against an fp64 attention.
"""

from __future__ import annotations

import math
import zlib

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


pytestmark.append(pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)"))


def _reference(q, k, v, sbias, scale):
    """fp64 attention of the same fp16 operands; rows without a finite logit give 0 (the core's empty-sum convention)."""
    out = torch.empty_like(q, dtype=torch.float64)
    for n0 in range(0, q.shape[1], 16):
        lg = scale * (torch.einsum("bnjhd,bnkhd->bnhjk", q[:, n0 : n0 + 16].double(), k[:, n0 : n0 + 16].double()) + sbias.double()[:, None])
        p = torch.softmax(lg, -1)
        p = torch.nan_to_num(p, nan=0.0)
        out[:, n0 : n0 + 16] = torch.einsum("bnhjk,bnkhd->bnjhd", p, v[:, n0 : n0 + 16].double())
    return out


def _case(name, length, heads, dh, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    shape = (1, length, length, heads, dh)
    scale = dh**-0.5
    keep = None
    v = (3.0 * r(*shape)).half()
    qs = ks = 1.5
    bias = r(1, heads, length, length)
    if name.startswith("qk"):  # logit std = s^2 (fp16 q.k accumulator: keep the raw max below 65504)
        qs = ks = math.sqrt(float(name[2:]))
    elif name.startswith("bias") and name[4:].isdigit():  # huge additive bias (pair bias; the core scales q.k + bias)
        bias = (bias * float(name[4:])).clamp(-60000.0, 60000.0)  # (no fp16 inf in the input: q.k + bias must stay inside the fp16 accumulator)
    elif name == "bias_limit":  # +-65504, the fp16 limit
        bias = (bias.sign() * 65504.0).clamp(-65504.0, 65504.0)
        bias[..., :3] = 0.0
    elif name == "mask_big_neg":
        bias = bias * 300.0
        bias[..., length // 2 :] = -65504.0
    elif name == "mask_inf":
        bias = bias * 800.0
        bias[..., length - 40 :] = float("-inf")
    elif name == "mask_allrows":
        bias = bias * 800.0
        bias[:, :, 5:9, :] = float("-inf")  # rows without any finite logit
        bias[:, :, 20:23, : length // 2] = float("-inf")
    elif name == "keymask_tile":
        bias = bias * 800.0
        keep = torch.ones(1, length, dtype=torch.bool, device="cuda")
        keep[:, 128:256] = False  # a whole key tile masked
        keep[:, length - 40 :] = False
        bias = bias.masked_fill(~keep[:, None, None, :], float("-inf"))
    elif name == "spike":  # one dominant key per row, ~1000s above the rest, the others carry big noise
        bias = bias * 100.0
        idx = torch.randint(0, length, (1, heads, length), device="cuda", generator=g)
        bias.scatter_(3, idx[..., None], 3000.0)
    q, k = (qs * r(*shape)).half(), (ks * r(*shape)).half()
    return q, k, v, bias.half().contiguous(), keep, scale


NAMES = ["qk50", "qk300", "qk1000", "qk1500", "bias300", "bias3000", "bias20000", "bias_limit", "mask_big_neg", "mask_inf", "mask_allrows", "keymask_tile", "spike"]


@pytest.mark.parametrize(("heads", "dh", "length"), [(4, 32, 256), (4, 16, 256), (8, 32, 512)])
@pytest.mark.parametrize("name", NAMES)
def test_f8pv_core_extreme_logits(monkeypatch, name, heads, dh, length):
    from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    q, k, v, sbias, keep, scale = _case(name, length, heads, dh, seed=zlib.crc32(f"{name}{heads}{dh}".encode()) % 1000)
    monkeypatch.setattr(hs, "FP8_PV", True)
    got = hs.core_h16().triattn_fwd_h16(q, k, v, sbias, scale, False, key_mask=keep)[0].float()
    monkeypatch.setattr(hs, "FP8_PV", False)
    fp16 = hs.core_h16().triattn_fwd_h16(q, k, v, sbias, scale, False, key_mask=keep)[0].float()
    ref = _reference(q, k, v, sbias, scale)
    ok = torch.isfinite(fp16).flatten(3).all(-1)[..., None, None] & torch.ones_like(got, dtype=torch.bool)  # rows the fp16 core itself keeps finite
    bad = (~torch.isfinite(got)) & ok
    assert not bad.any(), f"{name}: {int(bad.sum())} non-finite outputs of {got.numel()} (fp16 core finite)"
    sel = ok & torch.isfinite(ref.float())
    if not sel.any():  # (the default core overflows everywhere: only the finite check above applies)
        return
    d8 = (got.double() - ref)[sel]
    d16 = (fp16.double() - ref)[sel]
    n = ref[sel].norm().clamp_min(1e-9)
    e8, e16 = (d8.norm() / n).item(), (d16.norm() / n).item()
    print(f"[extreme] {name:13s} H{heads} dh{dh} L{length}: err_f8 {e8:.4f}  err_fp16 {e16:.4f}  max|f8-ref| {d8.abs().max().item():.3f}")
    if name != "bias_limit":  # (+-65504 biases overflow the fp16 accumulator of the default core too: finite output is all that is asked)
        assert e8 < max(0.10, 1.5 * e16), (name, e8, e16)  # (bias20000: the fp16 sums themselves lose the keys, the default core is as far off)
