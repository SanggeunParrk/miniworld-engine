"""MSAPairWeightedAveraging training (H100, bf16), 3.0.1:

* ``pwa_wide_aux.cu``: the dropout mask of the wide widths' backward (``drop_bwd_w``) and the BIG build's dv permute-copy (``perm_dv_w``)
  are bit-identical to the PyTorch ops they replace;
* a wide-width training step is bit-identical with ``pwa_wide.FUSED_BWD_AUX`` on and off (seeded dropout);
* the native builds serve every shape the refusal admits (``pwa_train.NATIVE_ANY_SHAPE``): deterministic (two calls give the same bytes: no
  float atomics in dgamma / dbeta), within the engine tolerance of the fp32 PyTorch module, and the pair side beyond the six qualified shapes
  stays the Triton kernel (``NATIVE_PAIR_ANY_SHAPE`` off): ln_pair.weight / to_bias.weight equal the 3.0.0 routing's.
"""

import pytest
import torch

from miniworld_engine.kernels.integrations import pwa_train as PT
from miniworld_engine.kernels.integrations import pwa_wide as PW
from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]

_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("s", "n", "d", "p"), [(3, 5, 64, 0.25), (70, 208, 128, 0.15), (130, 96, 256, 0.25), (1, 1, 64, 0.5)])
def test_drop_bwd_w_bit_identical(s: int, n: int, d: int, p: float) -> None:
    torch.manual_seed(0)
    dres = torch.randn(s, n, d, device="cuda").bfloat16()
    dres[0, 0, :4] = torch.tensor([1e38, -0.0, 1e-30, -3e38], device="cuda").bfloat16()  # specials pass through the same arithmetic
    keep = (torch.rand(n, d, device="cuda", dtype=torch.bfloat16) > p).to(torch.bfloat16)
    scale = 1.0 / (1.0 - p)
    ref = (dres * keep[None] * scale).contiguous()
    got = PW._aux().drop_bwd_w(dres, keep, scale)
    assert got.dtype == torch.bfloat16 and got.shape == ref.shape
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("h", "n", "s", "c"), [(8, 16, 5, 32), (8, 24, 70, 32), (8, 8, 9, 16), (8, 3, 4, 8)])
def test_perm_dv_w_bit_identical(h: int, n: int, s: int, c: int) -> None:
    torch.manual_seed(0)
    hc = h * c
    dvh = torch.randn(h, n, s * c, device="cuda").bfloat16()
    ref = torch.zeros(s, n, 2 * hc, device="cuda", dtype=torch.bfloat16)
    ref[..., hc:].view(s, n, h, c).copy_(dvh.view(h, n, s, c).permute(2, 1, 0, 3))
    got = torch.zeros_like(ref)
    PW._aux().perm_dv_w(dvh, got, hc)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def _module(d_msa: int, d_pair: int, c: int, p: float):
    gen = torch.Generator(device="cuda").manual_seed(0)
    m = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=c, p_drop=p, master_weights=True, backend="cuda").cuda().train()
    C.randomize(m, gen)
    return m


def _step(m, s: int, n: int, d_msa: int, d_pair: int, seed: int = 1234):
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(1, s, n, d_msa, device="cuda", generator=g).bfloat16().requires_grad_()
    z = torch.randn(1, n, n, d_pair, device="cuda", generator=g).bfloat16().requires_grad_()
    gy = torch.randn(1, s, n, d_msa, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    params = [q for q in m.parameters() if q.requires_grad]
    torch.manual_seed(seed)
    y = m(x, z, mask)
    return [y.detach(), *torch.autograd.grad(y, [x, z, *params], gy)]


@pytest.fixture
def switches():
    saved = (PT.NATIVE_ANY_SHAPE, PT.NATIVE_PAIR_ANY_SHAPE, PW.FUSED_BWD_AUX)
    yield
    PT.NATIVE_ANY_SHAPE, PT.NATIVE_PAIR_ANY_SHAPE, PW.FUSED_BWD_AUX = saved


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("d_msa", "d_pair", "c", "n", "s"), [(64, 128, 8, 128, 70), (128, 256, 8, 160, 33), (256, 512, 32, 128, 66)])
def test_wide_step_fused_aux_bit_identical(switches, d_msa: int, d_pair: int, c: int, n: int, s: int) -> None:
    m = _module(d_msa, d_pair, c, 0.25)
    PW.FUSED_BWD_AUX = False
    ref = _step(m, s, n, d_msa, d_pair)
    PW.FUSED_BWD_AUX = True
    got = _step(m, s, n, d_msa, d_pair)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a.float(), b.float())


@pytest.mark.skipif(not _H100, reason="H100")
def test_native_routing_predicate(switches) -> None:
    PT.NATIVE_ANY_SHAPE = True
    assert PT._training_default_shape(1024, 512) and PT._training_default_shape(384, 768) and PT._training_default_shape(512, 128)
    assert not PT._training_default_shape(1024, 192) and not PT._training_default_shape(1025, 256) and not PT._training_default_shape(1024, 1152)
    PT.NATIVE_ANY_SHAPE = False
    assert PT._training_default_shape(1024, 384) and not PT._training_default_shape(1024, 512)
    PT.NATIVE_PAIR_ANY_SHAPE = False
    assert PT._native_pair_shape(2048, 768) and not PT._native_pair_shape(384, 768)


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("n", "s"), [(256, 64), (512, 66)])
def test_native_any_shape_deterministic_and_close(switches, n: int, s: int) -> None:
    m = _module(64, 128, 32, 0.0)
    PT.NATIVE_ANY_SHAPE = False
    old = _step(m, s, n, 64, 128)  # the 3.0.0 routing (plain builds)
    PT.NATIVE_ANY_SHAPE = True
    a = _step(m, s, n, 64, 128)
    b = _step(m, s, n, 64, 128)
    for x, y in zip(a, b, strict=True):  # no float atomics: the same bytes on every call
        assert torch.equal(x.float(), y.float())
    names = ["out", "msa", "pair", *[q for q, p in m.named_parameters() if p.requires_grad]]
    for name, x, y in zip(names, a, old, strict=True):
        den = float(y.float().abs().max()) or 1.0
        assert float((x.float() - y.float()).abs().max()) / den < 2e-2, name
    # the pair side beyond the six qualified shapes is the plain (Triton) one: its parameter gradients are the 3.0.0 routing's
    for name, x, y in zip(names, a, old, strict=True):
        if name in ("ln_pair.weight", "to_bias.weight"):
            den = float(y.float().abs().max()) or 1.0
            assert float((x.float() - y.float()).abs().max()) / den < 2e-3, name
