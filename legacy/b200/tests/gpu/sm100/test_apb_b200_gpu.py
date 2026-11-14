"""AttentionPairBias B200 kernels (kernels/augmented_attention/cuda/apb, kernels/augmented_attention/cuda/sm100 apb_*) against
fp64 references."""

import math

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def blackwell():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("heads", [8, 12, 16, 24])
@pytest.mark.parametrize("L", [128, 256, 384, 768])
def test_pair_bias_forward_and_backward(L, heads):
    from miniworld_engine.kernels.augmented_attention.cuda import apb

    g = torch.Generator(device="cuda").manual_seed(L)
    pair = (torch.randn(L, L, 128, device="cuda", generator=g) * 1.5 + 0.3).to(torch.bfloat16)
    wf = (torch.randn(heads, 128, device="cuda", generator=g) * 128 ** -0.5).to(torch.bfloat16)
    mask = torch.rand(L, device="cuda", generator=g) > 0.2
    got = apb.pair_bias(pair.view(L * L, 128), wf, mask, L, 1e-5, -1e30).float()
    xh = torch.nn.functional.layer_norm(pair.double(), (128,), eps=1e-5)
    want = torch.einsum("ijc,hc->hij", xh, wf.double())
    assert (got[:, :, ~mask] == torch.tensor(-1e30).bfloat16().float()).all()
    assert _rel(got[:, :, mask], want[:, :, mask]) < 1e-2
    # backward: dbias -> dpair, dWf, per-head sums
    dbias = torch.randn(heads, L, L, device="cuda", generator=g)
    dz, dwf, hs = apb.pair_bias_bwd(pair.view(L * L, 128), dbias, wf, L, 1e-5)
    p = pair.double().requires_grad_()
    w = wf.double().requires_grad_()
    torch.einsum("ijc,hc->hij", torch.nn.functional.layer_norm(p, (128,), eps=1e-5), w).backward(dbias.double())
    assert _rel(dz.view(L, L, 128), p.grad) < 1e-2
    assert _rel(dwf, w.grad) < 1e-2
    assert _rel(hs, dbias.double().sum((1, 2))) < 1e-4


GEOMETRY = {(8, 384): (48, 48), (12, 384): (32, 32), (16, 384): (24, 32), (24, 384): (16, 16), (16, 512): (32, 32)}
SHAPES = list(GEOMETRY)          # heads -> (real head dim, head width in memory: 16 x 24 zero-padded to 32)


def _core_inputs(L, seed, shape, mask_frac=0.2):
    """q, k, v, g [L, heads, dh] (real channels), bias [heads, L, L], key mask [L]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    heads, dh = shape[0], GEOMETRY[shape][0]
    q, k, v, gt = (torch.randn(L, heads, dh, device="cuda", generator=g) for _ in range(4))
    bias = torch.randn(heads, L, L, device="cuda", generator=g)
    mask = torch.rand(L, device="cuda", generator=g) > mask_frac
    mask[0] = True
    return q, k, v, gt, bias, mask


def _pad(t, shape):
    """[L, heads, dh] -> [L, heads * dhp] with zero pad channels."""
    heads, (dh, dhp) = shape[0], GEOMETRY[shape]
    return torch.nn.functional.pad(t, (0, dhp - dh)).reshape(t.shape[0], heads * dhp)


def _real(t, shape):
    """[L, heads * dhp] -> ([L, heads, dh] real channels, the pad channels)."""
    heads, (dh, dhp) = shape[0], GEOMETRY[shape]
    t = t.view(t.shape[0], heads, dhp)
    return t[..., :dh], t[..., dh:]


def _ref_attn(q, k, v, bias, mask):
    """softmax(q k^T / sqrt(dh) + bias) v per head in fp64, masked keys dropped; q, k, v [L, H, dh] -> [L, H, dh]."""
    dh = q.shape[-1]
    qh, kh, vh = (t.double().transpose(0, 1) for t in (q, k, v))
    s = qh @ kh.transpose(1, 2) / math.sqrt(dh) + bias.double()
    s = s.masked_fill(~mask, float("-inf"))
    return (s.softmax(-1) @ vh).transpose(0, 1), s


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("L", [128, 200, 256, 384, 640, 768])
def test_apb_inference_core(L, shape):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, gt, bias, mask = _core_inputs(L, L, shape)
    c = math.log2(math.e)
    heads, d, dh = shape[0], shape[1], GEOMETRY[shape][0]
    qkvg = torch.cat([_pad(q * (c / math.sqrt(dh)), shape), _pad(k, shape), _pad(v, shape), _pad(gt, shape)], 1)
    qkvg = qkvg.to(torch.bfloat16).contiguous()
    b2 = (bias * c).masked_fill(~mask, -1e30).to(torch.bfloat16).contiguous()
    core = sm100.ApbInferenceCore(torch.cuda.current_device(), heads, d)
    core(qkvg, b2)
    torch.cuda.synchronize()
    qb, kb, vb, gb = (t.to(torch.bfloat16) for t in (q, k, v, gt))
    o, _ = _ref_attn(qb, kb, vb, bias.to(torch.bfloat16), mask)
    want = torch.sigmoid(gb.double()) * o
    got, pad = _real(qkvg[:, :sm100.apb_width(heads, d)], shape)
    assert _rel(got, want) < 1.5e-2
    assert (pad == 0).all()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("L", [128, 256, 384, 768])
def test_apb_training_core(L, shape):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    heads, d = shape
    q, k, v, _, bias, mask = _core_inputs(L, L + 1, shape)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q, k, v))
    qp, kp, vp = (_pad(t, shape).contiguous() for t in (qb, kb, vb))
    bb = bias.masked_fill(~mask, -1e30).to(torch.bfloat16).contiguous()
    O, LSE = sm100.apb_forward(qp, kp, vp, bb, L, heads, d)
    qd, kd, vd = (t.double().requires_grad_() for t in (qb, kb, vb))
    bd = bias.to(torch.bfloat16).double().requires_grad_()
    want, s = _ref_attn(qd, kd, vd, bd, mask)
    o_real, o_pad = _real(O, shape)
    assert _rel(o_real, want) < 1e-2
    assert (o_pad == 0).all()
    lse2 = torch.logsumexp(s, -1) / math.log(2)
    assert (LSE.double() - lse2).abs().max() < 2e-2
    dO = torch.randn(L, heads, GEOMETRY[shape][0], device="cuda")
    dob = dO.to(torch.bfloat16)
    dobp = _pad(dob, shape).contiguous()
    Dd = (dob.float() * o_real).sum(-1).t().contiguous()
    DQP, DK, DV, DB = sm100.apb_backward(qp, kp, vp, dobp, bb, LSE, Dd, L, heads, d)
    assert DQP.shape == (L // 128, L, sm100.apb_width(heads, d))      # key-chunk partials, summed by the caller
    DQ = DQP.sum(0)
    want.backward(dob.double())
    for got, ref in ((DQ, qd.grad), (DK, kd.grad), (DV, vd.grad)):
        g_real, g_pad = _real(got, shape)
        assert _rel(g_real, ref) < 1.5e-2
        assert (g_pad == 0).all()
    assert _rel(DB[:, :, mask], bd.grad[:, :, mask]) < 1.5e-2
