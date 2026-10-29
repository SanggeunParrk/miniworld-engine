"""B200 triangle-attention core: the forward's overflow guard, driven directly.

The core's softmax offset is the row's first finite 32-key maximum, raised only when a 128-key tile's exponentials sum past 2^64;
the tile is then recomputed from q / k (its P in TMEM may already hold inf) under the tile maximum, and l / O are rescaled. A row
whose first keys are all masked starts from the provisional offset 0 and takes a tile's maximum once its sum underflows there.
These cases put the logits where each of those branches runs: spikes far above the offset in the first tile (no O yet) and in later
tiles (O rescaled, repeatedly; inf under the old offset), different keys per query (lanes of one warp disagree), a masked prefix
followed by logits far below zero with fully masked tiles in between, and raises where the keys summed before the raise still weigh
~10 % of the output (a wrong l / O rescale would show). Reference: float64 softmax on the same bf16 inputs.
"""

import math

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100) required",
    ),
]

H, D = 2, 32


def _reference(q, k, v, bias, scale):
    """(out [B, N, S, H, D], lse base 2 [B, N, H, S]) in float64."""
    qf, kf, vf = (t.double().permute(0, 1, 3, 2, 4) for t in (q, k, v))       # [B, N, H, S, D]
    x = scale * qf @ kf.transpose(-1, -2) + bias.double()[:, None]             # [B, N, H, S, S]
    out = torch.softmax(x, -1) @ vf
    return out.permute(0, 1, 3, 2, 4), torch.logsumexp(x, -1) / math.log(2.0)


def _inputs(case, n, s, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(1, n, s, H, D, device="cuda", generator=g).bfloat16() for _ in range(3))
    bias = torch.randn(1, H, s, s, device="cuda", generator=g) * 2
    qi = torch.arange(s, device="cuda")
    if case == "escalating":
        # each query: a spike in tile 0 (chunk 2), then larger ones in every later tile, at a key that depends on the query
        for t in range(s // 128):
            key = t * 128 + (70 if t == 0 else 8 + 37 * (t % 3)) + (qi % 8)
            bias[0, :, qi, key] = 60.0 * (t + 1)                               # nats: 86 log2 units more per tile
    elif case == "some_lanes":
        # only every third query sees a spike, in the middle tile's first chunk (S / P chunk 0 already overwritten)
        sel = qi[qi % 3 == 0]
        bias[0, :, sel, 128 + 5] = 200.0                                       # 289 log2 units: inf under the old offset
    elif case == "masked_prefix_underflow":
        # first tile masked, the rest far below zero (offset 0 would underflow them all), one later tile masked again
        bias = bias - 300.0
        bias[..., :128] = -math.inf
        if s >= 512:
            bias[..., 256:384] = -math.inf
    elif case == "raise_tile0":
        # keys 32.. of tile 0 at +41 nats (59 log2 units over the first 32 keys' maximum: the tile sums past 2^64 and is redone
        # under its maximum), tile 1 on at +42: both tiles weigh in the output
        bias = bias * 0.25
        bias[..., 32:128] += 41.0
        bias[..., 128:] += 42.0
    elif case == "raise_tile1":
        # tile 0's keys 32.. at +38 nats (55 log2 units over the offset: kept), tile 1 at +40 (sums past 2^64: offset raised, O and
        # l rescaled), later tiles at +39: tile 0 still weighs ~10 % in the output, so a wrong O rescale shows
        bias = bias * 0.25
        bias[..., 32:128] += 38.0
        bias[..., 128:256] += 40.0
        bias[..., 256:] += 39.0
    elif case == "masked_prefix_then_spike":
        bias[..., :128] = -math.inf
        bias[0, :, qi, s - 1 - (qi % 16)] = 150.0
    return q, k, v, bias.bfloat16()


CASES = ["escalating", "some_lanes", "masked_prefix_underflow", "masked_prefix_then_spike", "raise_tile0", "raise_tile1"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(("n", "s"), [(3, 384), (5, 512)])                    # 2 and 4 pair rows a task (5: a padding row)
def test_overflow_guard(case, n, s):
    from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as T

    q, k, v, bias = _inputs(case, n, s, seed=CASES.index(case) * 1000 + s)
    scale = D**-0.5
    out, lse = T._core_forward(q, k, v, bias, scale, True)
    torch.cuda.synchronize()
    want, lse_want = _reference(q, k, v, bias, scale)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    err = float((out.double() - want).norm() / want.norm())
    assert err < 1e-2, err                                                     # bf16 P and output rounding: ~3e-3
    lerr = float(((lse.double() - lse_want).abs() / (1 + lse_want.abs())).max())
    assert lerr < 1e-4, lerr
