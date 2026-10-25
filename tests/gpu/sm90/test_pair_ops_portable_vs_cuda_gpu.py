"""On H100 the portable kernels of the three pair ops (`kernels/*/triton`, selected by the internal A/B switch) against the CUDA kernels the card runs by default: the same bins
(bit-identical codes), the same forward values, the same gradients (the table / LayerNorm / gamma / beta gradients), and the same routing (an H100 call is CUDA unless the switch is on)."""
import pytest
import torch
from settings_helpers import forced_triton as portable_switch   # the internal A/B switch: the one place the tests name it

from miniworld_engine.kernels import _pair_route as R
from miniworld_engine.kernels.pair_embed_adaln import pair_embed_adaln
from miniworld_engine.kernels.pair_embed_geom import pair_embed_ca, pair_embed_geom, pair_embed_geom_train
from miniworld_engine.kernels.token_pair_init import token_pair_init

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"),
]


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _geom_inputs(B, n, seed):
    from miniworld_engine.kernels.drivers.pair_embed_geom import geom_inputs

    torch.manual_seed(seed)
    tbl, bb, ca, cb, mask, idx, chains, dlim, alim = geom_inputs(B, n)
    return bb, ca.bool(), cb.bool(), mask.bool(), idx, chains


def test_h100_routing_is_cuda_unless_the_switch_is_on():
    dev = torch.device("cuda")
    assert R.route("pair_embed_geom", dev) == R.CUDA
    assert R.route("token_pair_init", dev, cuda_caps=((9, 0), (10, 0))) == R.CUDA
    with portable_switch():
        assert R.route("pair_embed_geom", dev) == R.PORTABLE


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (1, 257), (4, 150)])
def test_geom_codes_and_values_are_those_of_the_cuda_kernels(B, n):
    from miniworld_engine.kernels.pair_embed_geom import triton as pt
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    from miniworld_engine.kernels.drivers.pair_embed_geom import ca_inputs, geom_inputs

    torch.manual_seed(B * n)
    args = geom_inputs(B, n)
    for eps in (None, 1e-5):
        o1, c1 = sm90.forward(*args, with_codes=True, ln_eps=eps)
        o2, c2 = pt.portable_forward(*args, with_codes=True, ln_eps=eps)
        assert torch.equal(c1, c2), "the bins (codes) must be bit-identical"
        if eps is None:
            assert torch.equal(o1, o2)
        else:      # the LayerNorm: the statistics differ in the last fp32 bit, so a bf16 output may differ by one rounding step
            assert float(((o1.float() - o2.float()).abs() > 0.02 + 0.02 * o1.float().abs()).float().mean()) == 0
    dout = torch.randn(B, n, n, 256, device="cuda").bfloat16()
    assert torch.equal(sm90.backward(dout, c1, 0), pt.portable_backward(dout, c1, 0))
    assert _rel(pt.portable_ln_backward(args[0], c1, dout, 0, 1e-5), sm90.ln_backward(args[0], c1, dout, 0, 1e-5)) < 1e-4
    cargs = ca_inputs(B, n)
    for eps in (None, 1e-5):
        a1, k1 = sm90.forward_ca(*cargs, with_codes=True, ln_eps=eps)
        a2, k2 = pt.portable_forward_ca(*cargs, with_codes=True, ln_eps=eps)
        assert torch.equal(k1[..., 0], k2[..., 0])
        assert float(((a1.float() - a2.float()).abs() > 0.02 + 0.02 * a1.float().abs()).float().mean()) == 0
    assert _rel(pt.portable_backward(dout, k1, 1), sm90.backward(dout, k1, 1)) < 1e-6


@pytest.mark.parametrize("ln_eps", [None, 1e-5])
def test_geom_train_and_ca_gradients_match_the_cuda_path(ln_eps):
    B, n = 3, 130
    bb, ca, cb, mask, idx, chains = _geom_inputs(B, n, 5)
    g = torch.Generator(device="cuda").manual_seed(1)
    w = torch.randn(256, 317, device="cuda", generator=g) * 0.3
    wc = torch.randn(256, 157, device="cuda", generator=g) * 0.3
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    cax = bb[:, :, 1] * mask[..., None]
    res = {}
    for name, ctx in (("cuda", None), ("portable", portable_switch)):
        outs = []
        for build, wt in ((lambda w_: pair_embed_geom_train(w_, bb, ca, cb, mask, idx, chains, ln_eps=ln_eps), w),
                          (lambda w_: pair_embed_ca(w_, cax, mask, ln_eps=ln_eps), wc)):
            wr = wt.clone().requires_grad_()
            if ctx is None:
                out = build(wr)
                out.backward(dout)
            else:
                with ctx():
                    out = build(wr)
                    out.backward(dout)
            outs += [out.detach(), wr.grad]
        res[name] = outs
    for a, e in zip(res["portable"], res["cuda"], strict=True):
        assert _rel(a.float(), e.float()) < 2e-3


def test_geom_inference_entry_point_matches():
    bb, ca, cb, mask, idx, chains = _geom_inputs(2, 90, 9)
    w = torch.randn(256, 317, device="cuda") * 0.3
    ref = pair_embed_geom(w, bb, ca, cb, mask, idx, chains)
    with portable_switch():
        got = pair_embed_geom(w, bb, ca, cb, mask, idx, chains)
    assert torch.equal(ref, got)


def _adaln_case(B, n, with_sc, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    params = [(r(256, 217) * 0.3).requires_grad_(), (1 + 0.1 * r(256)).requires_grad_(), (0.1 * r(256)).requires_grad_(),
              torch.sigmoid(r(B, 256)).requires_grad_(), (0.2 * r(B, 256)).requires_grad_()]
    xt, sc = r(B, n, 3) * 0.8, (r(B, n, 3) * 0.8 if with_sc else None)
    idx = torch.randint(1, 4, (B, n), device="cuda", generator=g).cumsum(1)
    lengths = torch.randint(max(2, n // 2), n + 1, (B,), device="cuda", generator=g)
    mask = torch.arange(n, device="cuda")[None] < lengths[:, None]
    return params, xt, sc, idx, mask, r(B, n, n, 256)


@pytest.mark.parametrize(("B", "n", "with_sc"), [(2, 64, True), (3, 100, True), (2, 130, False), (1, 257, True)])
def test_adaln_matches_the_cuda_path(B, n, with_sc):
    params, xt, sc, idx, mask, wout = _adaln_case(B, n, with_sc)
    wout = wout * (mask[:, :, None] & mask[:, None, :])[..., None]
    res = {}
    for name, ctx in (("cuda", None), ("portable", portable_switch)):
        for p in params:
            p.grad = None
        if ctx is None:
            out = pair_embed_adaln(*params, xt, sc, idx, mask)
            (out.float() * wout).sum().backward()
        else:
            with ctx():
                out = pair_embed_adaln(*params, xt, sc, idx, mask)
                (out.float() * wout).sum().backward()
        res[name] = [out.detach().float(), *(p.grad.clone() for p in params)]
    assert float(((res["portable"][0] - res["cuda"][0]).abs() > 0.02 + 0.02 * res["cuda"][0].abs()).float().mean()) == 0
    for name, a, e in zip(("out", "dw", "dln_w", "dln_b", "dgamma", "dbeta"), res["portable"], res["cuda"], strict=True):
        assert _rel(a, e) < 1e-3, (name, _rel(a, e))


def _tpi_case(b, l, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [torch.randint(0, hi, (b, l), generator=g).cuda() for hi in (3, 90, 60, 2, 6)]
    bond = (torch.rand(b, l, l, generator=g) < 0.05).cuda()
    leaves = [(torch.randn(b, l, 128, generator=g).cuda()).requires_grad_(), (torch.randn(b, l, 128, generator=g).cuda()).requires_grad_(),
              (torch.randn(128, 139, generator=g) * 0.3).cuda().requires_grad_(), (torch.randn(128, 2, generator=g) * 0.3).cuda().requires_grad_()]
    return leaves, ids, bond


@pytest.mark.parametrize(("b", "l"), [(1, 384), (2, 130), (3, 64), (1, 7)])
def test_token_pair_init_matches_the_cuda_path(b, l):
    leaves, ids, bond = _tpi_case(b, l)
    dz = torch.randn(b, l, l, 128, device="cuda")
    res = {}
    for name, ctx in (("cuda", None), ("portable", portable_switch)):
        if ctx is None:
            z = token_pair_init(*leaves, *ids, bond)
            grads = torch.autograd.grad(z, leaves, dz)
        else:
            with ctx():
                z = token_pair_init(*leaves, *ids, bond)
                grads = torch.autograd.grad(z, leaves, dz)
        res[name] = [z.detach(), *grads]
    assert torch.equal(res["portable"][0], res["cuda"][0])
    for name, a, e in zip(("dleft", "dright", "dw_rel", "dw_bond"), res["portable"][1:], res["cuda"][1:], strict=True):
        assert _rel(a, e) < 1e-5, (name, _rel(a, e))
