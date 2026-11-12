"""FoldForge 5I28 confidence head (APB B > 1): ``AttentionPairBias`` on ``cuda`` serves a batch of B rows (the confidence PairformerBlock's
single attention batched over the diffusion samples) on H100 and A100, inference and training, as one step per batch row inside each op (the A100
fp32 composed path: its pair backward is one tensor-core launch for the batch, which computes each row as the one-row launch does).

The batched call equals the B one-row calls: outputs and the input gradients bitwise (each row runs the one-row launches), the weight
gradients as their sum over the rows (fp32 masters: summed in fp32 in row order, as autograd accumulates B calls -- bitwise; bf16 parameters:
summed in fp32 before one rounding, where B calls round each sample's gradient and accumulate in bf16 -- close).
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.attention import AttentionPairBias

pytestmark = [pytest.mark.gpu]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _built(d_pair: int, gate_bias: float, dtype: str) -> typing.Any:
    m = AttentionPairBias(d_single=384, d_pair=d_pair, n_head=16, gate_bias=gate_bias).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    built: typing.Any = C.cast(m, dtype)
    built.set_backend("cuda")
    return built


def _inputs(m, samples: int, length: int, dtype: str, grad: bool = False):
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=act, device="cuda", generator=g)
    # a different key mask per row (the rows must not share one)
    inp["mask"] = torch.arange(length, device="cuda")[None] < (length - 3 - 5 * torch.arange(samples, device="cuda"))[:, None]
    if grad:
        inp["single"].requires_grad_(True)
        inp["pair"].requires_grad_(True)
    return inp


def _autocast(dtype: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master")


CASES = [
    (128, 0.0, "master"),
    (128, 0.0, "bfloat16"),
    (256, 1.0, "master"),
    (384, 0.0, "master"),
    (512, 0.0, "float32"),
    (256, 1.0, "float32"),
]


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize(("d_pair", "gate_bias", "dtype"), CASES)
def test_batched_inference_equals_rows(d_pair: int, gate_bias: float, dtype: str) -> None:
    m = _built(d_pair, gate_bias, dtype).eval()
    inp = _inputs(m, 3, 128, dtype)
    with torch.no_grad(), _autocast(dtype):
        assert m.unsupported(inp["single"], inp["pair"], inp["mask"]) is None
        y = m(inp["single"], inp["pair"], inp["mask"])
        rows = [m(inp["single"][b : b + 1], inp["pair"][b : b + 1], inp["mask"][b : b + 1]) for b in range(3)]
    torch.testing.assert_close(y, torch.cat(rows), rtol=0, atol=0)


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("dtype", ["master", "float32"])
def test_batched_extra_pair_bias_equals_rows(dtype: str) -> None:
    m = _built(384, 0.0, dtype).eval()
    inp = _inputs(m, 3, 128, dtype)
    act = inp["single"].dtype
    extra = torch.randn(3, 16, 128, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2)).to(act)
    with torch.no_grad(), _autocast(dtype):
        assert m.unsupported(inp["single"], inp["pair"], inp["mask"], extra) is None
        y = m(inp["single"], inp["pair"], inp["mask"], extra)
        rows = [m(inp["single"][b : b + 1], inp["pair"][b : b + 1], inp["mask"][b : b + 1], extra[b : b + 1]) for b in range(3)]
    torch.testing.assert_close(y, torch.cat(rows), rtol=0, atol=0)


def _grads(m, inp, dtype: str, rows: bool):
    """(y, d single, d pair, {name: d param}) of sum(y * w) for a fixed w, batched or as B one-row calls."""
    m.zero_grad(set_to_none=True)
    single, pair = (inp[k].detach().clone().requires_grad_() for k in ("single", "pair"))
    mask = inp["mask"]
    w = torch.randn(single.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    with _autocast(dtype):
        if rows:
            ys, dss, dps = [], [], []
            for b in range(single.shape[0]):
                s_b, p_b = single[b : b + 1].detach().requires_grad_(), pair[b : b + 1].detach().requires_grad_()
                y_b = m(s_b, p_b, mask[b : b + 1])
                (y_b.float() * w[b : b + 1]).sum().backward()
                ys.append(y_b.detach())
                dss.append(s_b.grad)
                dps.append(p_b.grad)
            y, ds, dp = torch.cat(ys), torch.cat(dss), torch.cat(dps)
        else:
            assert m.unsupported(single, pair, mask) is None
            y = m(single, pair, mask)
            (y.float() * w).sum().backward()
            y, ds, dp = y.detach(), single.grad, pair.grad
    return y, ds, dp, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize(("d_pair", "gate_bias", "dtype"), CASES)
def test_batched_training_equals_rows(d_pair: int, gate_bias: float, dtype: str) -> None:
    if dtype == "bfloat16" and torch.cuda.get_device_capability() == (8, 0) and d_pair != 128:
        pytest.skip("A100: bf16 parameters with a wide pair are the composed path's (covered by master)")
    m = _built(d_pair, gate_bias, dtype).train()
    inp = _inputs(m, 3, 128, dtype)
    y, ds, dp, gw = _grads(m, inp, dtype, rows=False)
    y1, ds1, dp1, gw1 = _grads(m, inp, dtype, rows=True)
    torch.testing.assert_close(y, y1, rtol=0, atol=0)
    torch.testing.assert_close(ds, ds1, rtol=0, atol=0)
    torch.testing.assert_close(dp, dp1, rtol=0, atol=0)
    assert gw.keys() == gw1.keys() and len(gw) == 10
    for name in gw:
        a, b = gw[name].float(), gw1[name].float()
        if dtype == "bfloat16":  # the batch sums in fp32 before one rounding; B calls accumulate bf16-rounded gradients
            assert float((a - b).abs().max()) <= 2e-2 * float(b.abs().max()) + 1e-6, name
        else:  # equal up to the A100 fused backward's atomic dWf / LN sums (run to run ~2e-7 of the largest entry)
            assert float((a - b).abs().max()) <= 2e-6 * float(b.abs().max()) + 1e-9, name


def _pair_bias_backward_fp64(z, wcm, db, mask, eps):
    """The pair-bias backward (``sm80.pair_bias_generic_backward``'s math) in float64: (dz, dW' per batch row [B, H, C])."""
    z64 = z.double()
    mean = z64.mean(-1, keepdim=True)
    rstd = (z64.var(-1, unbiased=False, keepdim=True) + eps).rsqrt()
    xh = (z64 - mean) * rstd
    g = db.double().permute(0, 2, 3, 1) * mask[:, None, :, None].double()  # [B, L, L, H]; a masked key passes no gradient
    dxh = g @ wcm.double().t()
    dz = rstd * (dxh - dxh.mean(-1, keepdim=True) - xh * (dxh * xh).mean(-1, keepdim=True))
    return dz, torch.einsum("bijh,bijc->bhc", g, xh)


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)), reason="A100 (sm_80)")
@pytest.mark.parametrize(("channels", "heads"), [(512, 16), (384, 8), (256, 16), (128, 24)])
def test_tc_pair_backward(channels: int, heads: int) -> None:
    """The A100 composed fp32 path's pair backward (``pair_bias_generic_backward_tc``, one launch for the batch): dz at ~fp32 accuracy (3xTF32),
    dW' at TF32-GEMM accuracy; each batch row's result equals the one-row launch's bitwise, and a replay is bit-identical."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm80

    g = torch.Generator(device="cuda").manual_seed(4)
    batch, length = 3, 128
    z = torch.randn(batch, length, length, channels, device="cuda", generator=g) * 1.3 + 0.7
    wcm = torch.randn(channels, heads, device="cuda", generator=g) * 0.1
    db = torch.randn(batch, heads, length, length, device="cuda", generator=g) * 1e-2
    mask = torch.arange(length, device="cuda")[None] < (length - 3 - 5 * torch.arange(batch, device="cuda"))[:, None]
    dz, dw = sm80.pair_bias_generic_backward_tc(z, wcm, db, mask, length)
    rz, rw = _pair_bias_backward_fp64(z, wcm, db, mask, 1e-5)
    assert float((dz.double() - rz).abs().max()) <= 2e-6 * float(rz.abs().max())
    assert float((dw.double() - rw).abs().max()) <= 2e-3 * float(rw.abs().max())
    for b in range(batch):
        rows = sm80.pair_bias_generic_backward_tc(z[b : b + 1], wcm, db[b : b + 1], mask[b : b + 1], length)
        assert torch.equal(rows[0][0], dz[b]) and torch.equal(rows[1][0], dw[b])
    again = sm80.pair_bias_generic_backward_tc(z, wcm, db, mask, length)
    assert torch.equal(again[0], dz) and torch.equal(again[1], dw)


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)), reason="H100 (sm_90)")
@pytest.mark.parametrize(("d_pair", "heads", "extra"), [(128, 8, False), (128, 12, True), (256, 16, False)])
def test_h100_batched_chunks_equal_rows(monkeypatch, d_pair: int, heads: int, extra: bool) -> None:
    """H100 bf16 inference runs the batch rows together (one launch per stage), in chunks of rows bounded by ``BATCH_BIAS_BYTES``: with
    the bound forcing chunks of 2 rows out of 5 (and the last of 1), every row still equals its one-row call bitwise."""
    from miniworld_engine.kernels.integrations import attention_pair_bias_b200 as native

    m = AttentionPairBias(d_single=384, d_pair=d_pair, n_head=heads).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    built: typing.Any = C.cast(m, "master")
    built.set_backend("cuda").eval()
    inp = _inputs(built, 5, 128, "master")
    planes = None
    if extra:
        planes = torch.randn(5, heads, 128, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2)).to(torch.bfloat16)
    monkeypatch.setattr(native, "BATCH_BIAS_BYTES", 2 * 2 * heads * 128 * 128)
    assert native._batch_chunk(5, heads, 128) == 2
    with torch.no_grad(), _autocast("master"):
        y = built(inp["single"], inp["pair"], inp["mask"], planes)
        rows = [
            built(inp["single"][b : b + 1], inp["pair"][b : b + 1], inp["mask"][b : b + 1], None if planes is None else planes[b : b + 1])
            for b in range(5)
        ]
    torch.testing.assert_close(y, torch.cat(rows), rtol=0, atol=0)
