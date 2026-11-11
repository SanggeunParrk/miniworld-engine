"""FoldForge 5I28 confidence head (APB B > 1): ``AttentionPairBias`` on ``cuda`` serves a batch of B rows (the confidence PairformerBlock's
single attention batched over the diffusion samples) on H100 and A100, inference and training, as one step per batch row inside each op.

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
    m = C.cast(m, dtype)
    m.set_backend("cuda")
    return m


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


CASES = [(128, 0.0, "master"), (128, 0.0, "bfloat16"), (256, 1.0, "master"), (384, 0.0, "master"), (512, 0.0, "float32")]


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
        else:
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6, msg=name)
