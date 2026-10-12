"""fp32 master parameters over bf16 activations on the B200 training paths (AMP's ``bf16-mixed``): the kernels cast the parameters
to bf16 outside autograd, so every parameter gets an fp32 gradient that was never rounded to bf16, and it matches the gradient
the bf16-parameter path gives (same bf16 kernels, the same math up to the parameters' own bf16 rounding)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
                                reason="B200 (sm_100a) paths")

BF, F32 = torch.bfloat16, torch.float32


def _randomize(m):
    """Nonzero, well-scaled parameters (zero-initialised projections would zero most gradients)."""
    g = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif "weight" in n:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return m


def _grads(make, inputs, call, pdt):
    torch.manual_seed(0)
    m = _randomize(make()).cuda().to(pdt).train()
    params = [p for p in m.parameters() if p.requires_grad]
    y = call(m, [x.clone().requires_grad_() for x in inputs])
    dy = torch.randn(y.shape, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda").to(y.dtype) * 0.1
    return [n for n, p in m.named_parameters() if p.requires_grad], torch.autograd.grad(y, params, dy, allow_unused=True)


def _check(make, inputs, call, tol=3e-2):
    names, g32 = _grads(make, inputs, call, F32)
    _, g16 = _grads(make, inputs, call, BF)
    for n, a, b in zip(names, g32, g16, strict=True):
        if a is None:
            assert b is None, n
            continue
        assert a.dtype == F32, f"{n}: {a.dtype}"
        if a.numel() > 64 and a.any():
            assert not torch.equal(a, a.to(BF).float()), f"{n}: the fp32 gradient was rounded to bf16"
        ref = b.float()
        scale = ref.norm()
        if n.endswith("ln_pair.bias"):     # its true gradient is 0 (the softmax cancels a per-head constant): rounding noise only
            continue
        if scale > 0:
            assert float((a - ref).norm() / scale) < tol, f"{n}: fp32-master gradient vs bf16-parameter gradient"


def _r(*shape):
    return (torch.randn(*shape, device="cuda") * 0.5).to(BF)


def test_trimul_bidirectional_and_outgoing():
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    x = _r(1, 256, 256, 128)
    _check(lambda: BidirectionalTriangleMultiplication(128, p_drop=0.0, implementation=IT.MINIWORLD), [x], lambda m, v: m(v[0]))
    _check(lambda: TriangleMultiplication(128, d_hidden=128, outgoing=True, p_drop=0.0, implementation=IT.MINIWORLD), [x],
           lambda m, v: m(v[0]))


@pytest.mark.parametrize("d", [64, 128])
def test_transition_sm100(d):
    from miniworld_engine.kernels.transition.cuda import fused_sm100a, fused_wide_sm100a

    class T(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_w, self.ln_b = torch.nn.Parameter(torch.ones(d)), torch.nn.Parameter(torch.zeros(d))
            self.wa, self.wb = torch.nn.Parameter(torch.empty(4 * d, d)), torch.nn.Parameter(torch.empty(4 * d, d))
            self.ws = torch.nn.Parameter(torch.empty(d, 4 * d))

        def forward(self, x):
            k = fused_sm100a if d == 128 else fused_wide_sm100a
            assert k.available(x, self.wa, self.ws), "the sm_100a Transition must serve fp32 and bf16 weights alike"
            entry = fused_sm100a.transition_fused_sm100a if d == 128 else fused_wide_sm100a.transition_wide_sm100a
            return entry(x, self.ln_w, self.ln_b, self.wa, self.wb, self.ws, 1e-5)

    _check(T, [_r(256 * 128, d)], lambda m, v: m(v[0]))


def test_attention_pair_bias():
    from miniworld_engine.modules import AttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    _check(lambda: AttentionPairBias(384, 128, 16, implementation=IT.MINIWORLD), [_r(1, 256, 384), _r(1, 256, 256, 128)],
           lambda m, v: m(v[0], v[1], None))


def test_outer_product_mean():
    from miniworld_engine.modules import OuterProductMean
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    _check(lambda: OuterProductMean(64, 128, 32, implementation=IT.MINIWORLD), [_r(1, 256, 256, 64)], lambda m, v: m(v[0], None))


def test_local_atom_dit():
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.local_dit import LocalDiTBlock

    _check(lambda: LocalDiTBlock(128, 128, 16, 4, n=2, cross_attention=True, implementation=IT.MINIWORLD),
           [_r(8, 1, 1024, 128), _r(8, 1, 1024, 128), _r(1, 32, 32, 128, 16)], lambda m, v: m(v[0], v[1], v[2], None))


def test_swa_dit_block_and_modulation():
    from miniworld_engine.kernels.swa_dit.interface import (
        refusal,
        swa_dit_block,
        swa_dit_hoist_modulation,
    )

    A, B, S, C = 4, 1, 1024, 128
    ang = torch.rand(B * S, C // 8, device="cuda") * 6
    cos, sin = ang.cos(), ang.sin()
    seq = torch.full((A * B,), S, device="cuda", dtype=torch.int32)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ws = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(*s)) for s in
                                              ((6 * C, C), (3 * C, C), (C, C), (C, C), (512, C), (C, 256))])

        def forward(self, q, c):
            assert refusal(q, cos, sin, seq, *self.ws[1:], n_head=4, half_window=64, cond=c, wmod=self.ws[0]) is None
            return swa_dit_block(q, swa_dit_hoist_modulation(c, self.ws[0]), cos, sin, seq, *self.ws[1:], B, 64)

    _check(Block, [_r(A * B, S, C), _r(B, S, C)], lambda m, v: m(v[0], v[1]))


def test_token_pair_init_bf16_streams_fp32_weights():
    """A bf16-autocast embedder hands bf16 left / right and fp32 master weights: served, computed in fp32 on the bf16 values,
    the stream gradients bf16 and the weight gradients fp32 and unrounded -- the same numbers as the all-fp32 call on those values."""
    from miniworld_engine.kernels.token_pair_init import refusal, token_pair_init

    g = torch.Generator().manual_seed(0)
    b, l, p = 1, 256, 128
    n_rel = 2 * (2 * 32 + 2) + (2 * 2 + 2) + 1
    ids = [torch.randint(0, hi, (b, l), generator=g).cuda() for hi in (3, 90, 60, 2, 6)]
    bond = (torch.rand(b, l, l, generator=g) < 0.05).cuda()
    left16, right16 = (torch.randn(b, l, p, generator=g).cuda().to(BF) for _ in range(2))
    w_rel, w_bond = (torch.randn(p, n_rel, generator=g) * 0.3).cuda(), (torch.randn(p, 2, generator=g) * 0.3).cuda()
    dz = torch.randn(b, l, l, p, generator=g).cuda()

    def run(left, right):
        leaves = [t.detach().clone().requires_grad_() for t in (left, right, w_rel, w_bond)]
        assert refusal(*leaves, bond) is None
        z = token_pair_init(*leaves, *ids, bond)
        return z, torch.autograd.grad(z, leaves, dz)

    z16, g16 = run(left16, right16)
    z32, g32 = run(left16.float(), right16.float())
    assert z16.dtype == F32
    assert torch.equal(z16, z32)
    assert g16[0].dtype == BF
    assert g16[1].dtype == BF
    for a, e in zip(g16[:2], g32[:2], strict=True):              # dright sums with atomics: its last fp32 bits (so a bf16
        assert float((a.float() - e).norm() / e.norm()) < 5e-3   # rounding now and then) differ from run to run
    for a, e in zip(g16[2:], g32[2:], strict=True):
        assert a.dtype == F32
        assert not torch.equal(a, a.to(BF).float())
        assert float((a - e).norm() / e.norm()) < 1e-5            # the class-bin sums use atomics: last-bit noise only


def test_pair_weighted_averaging():
    from miniworld_engine.integrations import pwa_train
    from miniworld_engine.modules import MSAPairWeightedAveraging
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    served = pwa_train.STATS["served"]
    _check(lambda: MSAPairWeightedAveraging(64, 128, n_head=8, d_hidden=32, p_drop=0.0, implementation=IT.MINIWORLD),
           [_r(1, 256, 256, 64), _r(1, 256, 256, 128)], lambda m, v: m(v[0], v[1], None))
    assert pwa_train.STATS["served"] >= served + 2, f"the sm_100a PWA training path was not taken: {pwa_train.STATS['refused']}"
