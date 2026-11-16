"""H100 fp16 inference core on heads of 16 channels (the compact path of AF3's d_pair 64): the exponential pairs split between the MUFU and the
FMA polynomial (``TA_POLY16``) -- deterministic run to run, and inside the engine's accuracy standard against an fp64 reference
(``max|x - gold| / max|gold|`` <= the judge's fixed 8e-3 for the master dtype), batch of two with a key mask, logits of a wide spread."""

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


@pytest.mark.parametrize("spread", [1.0, 2.2])
def test_dh16_core_deterministic_and_accurate(spread: float) -> None:
    _needs_h100()
    torch.manual_seed(1)
    B, L, H, dh = 2, 256, 4, 16
    scale = 32**-0.5
    q, k, v = (torch.randn(B, L, L, H, dh, device="cuda") * (spread if i < 2 else 1.0) for i in range(3))
    q, k, v = q.half(), k.half(), v.half()
    bias = torch.randn(B, H, L, L, device="cuda") * 2 * spread
    mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
    mask[1, 200:] = False
    bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
    sb = (bias / scale).half().contiguous()
    core = hs._core_h16().triattn_fwd_h16
    o1 = core(q, k, v, sb, scale, False, **hs.key_args(mask, L))[0]
    o2 = core(q, k, v, sb, scale, False, **hs.key_args(mask, L))[0]
    assert torch.equal(o1, o2)
    for b, n in [(0, 3), (1, 100), (1, 255)]:
        qq, kk, vv = (t[b, n].double().permute(1, 0, 2) for t in (q, k, v))
        s = scale * (qq @ kk.transpose(1, 2) + sb[b].double())
        gold = torch.softmax(s, -1) @ vv  # [H, L, dh]
        err = (o1[b, n].double().permute(1, 0, 2) - gold).abs().max() / gold.abs().max()
        assert err < 8e-3, (b, n, float(err))
