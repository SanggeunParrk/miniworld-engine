"""The token DiT's sm_100a attention kernels at every head layout they are built for (16 x 48, 24 x 32, 12 x 64, 16 x 64) against
fp64: the training forward / backward (A samples sharing one pair bias) and the gated inference core."""

import math

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]


@pytest.fixture(autouse=True)
def blackwell():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _ref(q, k, v, bias, mask=None):
    """softmax(q k^T / sqrt(dh) + bias) v per sample and head in fp64; q, k, v [A, L, H, dh], bias [H, L, L] -> [A, L, H, dh]."""
    dh = q.shape[-1]
    s = torch.einsum("alhd,amhd->ahlm", q.double(), k.double()) / math.sqrt(dh) + bias.double()
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return torch.einsum("ahlm,amhd->alhd", s.softmax(-1), v.double()), s


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("L", [128, 384])
def test_training_core(L, layout):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    H, dh = layout
    A, W = 2, H * dh
    g = torch.Generator(device="cuda").manual_seed(L + H)
    q, k, v = (torch.randn(A, L, H, dh, device="cuda", generator=g).bfloat16() for _ in range(3))
    bias = torch.randn(H, L, L, device="cuda", generator=g).bfloat16()
    q2, k2, v2 = (t.reshape(A * L, W).contiguous() for t in (q, k, v))
    O, LSE = sm100.forward(q2, k2, v2, bias.contiguous(), A, L, H, dh)
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    bd = bias.double().requires_grad_()
    want, s = _ref(qd, kd, vd, bd)
    assert _rel(O.view(A, L, H, dh), want) < 1e-2
    assert (LSE.double() - torch.logsumexp(s, -1) / math.log(2)).abs().max() < 2e-2
    dO = torch.randn(A, L, H, dh, device="cuda", generator=g).bfloat16()
    Dd = (dO.float() * O.view(A, L, H, dh)).sum(-1).permute(0, 2, 1).contiguous()          # [A, H, L]
    do2 = dO.reshape(A * L, W).contiguous()
    DQ, DK, DV, DB = sm100.backward(q2, k2, v2, do2, bias.contiguous(), LSE, Dd, A, L, H, dh)
    want.backward(dO.double())
    for got, ref in ((DQ, qd.grad), (DK, kd.grad), (DV, vd.grad)):
        assert _rel(got.view(A, L, H, dh), ref) < 1.5e-2
    assert _rel(DB, bd.grad) < 1.5e-2


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("L", [128, 200, 384])
def test_inference_core(L, layout):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    H, dh = layout
    S, W, nb = 3, H * dh, 2
    c = math.log2(math.e)
    g = torch.Generator(device="cuda").manual_seed(L * H)
    q, k, v, gt = (torch.randn(S, L, H, dh, device="cuda", generator=g) for _ in range(4))
    bias = torch.randn(nb * H, L, L, device="cuda", generator=g)
    mask = torch.rand(L, device="cuda", generator=g) > 0.2
    mask[0] = True
    qkvg = torch.cat([(q * (c / math.sqrt(dh))).reshape(S * L, W), k.reshape(S * L, W), v.reshape(S * L, W), gt.reshape(S * L, W)], 1)
    qkvg = qkvg.bfloat16().contiguous()
    b2 = (bias * c).masked_fill(~mask, float("-inf")).bfloat16().contiguous()
    core = sm100.GatedInferenceCore(torch.cuda.current_device(), torch.bfloat16, H, dh)
    core(qkvg, b2, 1, S)
    torch.cuda.synchronize()
    qb, kb, vb, gb = (t.bfloat16() for t in (q, k, v, gt))
    o, _ = _ref(qb, kb, vb, bias[H:].bfloat16(), mask)
    want = torch.sigmoid(gb.double()) * o
    assert _rel(qkvg[:, :W].view(S, L, H, dh), want) < 1.5e-2
