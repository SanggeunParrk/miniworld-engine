"""The sm_100 bf16 pair-bias attention core (kernels/augmented_attention/cuda/sm100) against an fp64 truth, and its
module wiring.

As for the sm_90 core: the kernels round their operands to bf16, so the bar is "no worse than the same math on the
bf16-rounded inputs", and the gate matters as much as the numbers (the tile shapes and the sample pairing are literals).
"""
import math

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
BLACKWELL = CUDA and torch.cuda.get_device_capability() == (10, 0)
needs_blackwell = pytest.mark.skipif(not BLACKWELL, reason="the kernels are sm_100a only")


def _inputs(A, L, seed=0, bias_scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(A, 1, L, 16, 48, device="cuda", generator=g) for _ in range(3))
    bias = torch.randn(1, L, L, 16, device="cuda", generator=g) * bias_scale
    return q, k, v, bias


def _truth(q, k, v, bias, do):
    """fp64 autograd: (O, dq, dk, dv, dbias) in the op's layouts."""
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, bias)]
    qd, kd, vd, bd = leaves
    s = torch.einsum("aihd,ajhd->ahij", qd[:, 0], kd[:, 0]) / math.sqrt(48) + bd[0].permute(2, 0, 1)[None]
    o = torch.einsum("ahij,ajhd->aihd", torch.softmax(s, -1), vd[:, 0])[:, None]
    o.backward(do.double())
    return (o.detach(), *(t.grad for t in leaves))


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_what_the_tiles_cannot_take():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, _, _, bias = _inputs(4, 128)
    assert sm100.supported(q, bias) is BLACKWELL
    assert sm100.supported(q, bias.permute(3, 0, 1, 2).contiguous(), bias_head_major=True) is BLACKWELL
    assert not sm100.supported(q, bias.permute(3, 0, 1, 2).contiguous())               # layout flag disagrees
    assert not sm100.supported(q[:3], bias)                                            # odd A: samples run in pairs
    assert not sm100.supported(q, bias, mask=torch.ones(4, 1, 128, device="cuda", dtype=torch.bool))  # no key mask
    q96, _, _, b96 = _inputs(4, 96)
    assert not sm100.supported(q96, b96)                                              # L not a multiple of 128
    assert not sm100.supported(q[..., :32], bias)                                     # head dim 32
    assert not sm100.supported(torch.randn(4, 2, 128, 16, 48, device="cuda"), bias)   # B == 2
    assert not sm100.supported(q.cpu(), bias.cpu())


@needs_blackwell
@pytest.mark.parametrize(("A", "L", "bias_scale", "head_major"), [
    (48, 384, 1.0, False),     # the token DiT's training shape
    (2, 768, 1.0, True),       # one sample pair, head-major bias (as the DiT hoists it)
    (6, 256, 4.0, False),      # wide logits
    (4, 128, 1.0, True),       # one query tile per head
])
def test_matches_fp64_at_the_bf16_input_floor(A, L, bias_scale, head_major):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, seed=A * 1000 + L, bias_scale=bias_scale)
    do = torch.randn(A, 1, L, 16, 48, device="cuda")
    truth = _truth(q, k, v, bias, do)
    r = lambda t: t.bfloat16().float()
    floor = _truth(r(q), r(k), r(v), r(bias), r(do))

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    b = (bias.permute(3, 0, 1, 2).contiguous() if head_major else bias.clone()).requires_grad_()
    assert sm100.available(leaves[0], b, bias_head_major=head_major)
    o = sm100.augmented_attention_bf16_sm100(leaves[0], leaves[1], leaves[2], b, bias_head_major=head_major)
    o.backward(do)
    db = b.grad.permute(1, 2, 3, 0) if head_major else b.grad
    got = (o.detach(), leaves[0].grad, leaves[1].grad, leaves[2].grad, db)
    for name, x, t, f in zip(("O", "dq", "dk", "dv", "dbias"), got, truth, floor, strict=False):
        e, ef = _rel(x, t), _rel(f, t)
        assert math.isfinite(e), f"{name}: not finite"
        assert e < 1.3 * ef + 1e-4, f"{name}: {e:.2e} against a floor of {ef:.2e}"


@needs_blackwell
def test_graph_capture_replays_the_same_forward():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(4, 256)
    with torch.no_grad():
        eager = sm100.augmented_attention_bf16_sm100(q, k, v, bias)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap = sm100.augmented_attention_bf16_sm100(q, k, v, bias)
        q.mul_(0.5)
        g.replay()
        assert torch.equal(cap, sm100.augmented_attention_bf16_sm100(q, k, v, bias))
        assert not torch.equal(cap, eager)


@needs_blackwell
def test_module_bf16_core_takes_the_kernels_and_keeps_the_triton_error():
    """AugmentedAttentionPairBias with compute_dtype=bf16 on a supported shape runs the sm_100 kernels; its error
    against the fp32 PyTorch module is no worse than the bf16 Triton core's (the switch is off in the second run)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.modules import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType

    torch.manual_seed(0)
    A, L = 4, 256
    ref = AugmentedAttentionPairBias(768, 384, 128, 16, use_qk_norm=True, implementation=ImplementationType.PYTORCH).cuda()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(std=p.shape[-1] ** -0.5 if p.dim() > 1 else 0.3).add_(1.0 if p.dim() == 1 else 0.0)
    eng = AugmentedAttentionPairBias(768, 384, 128, 16, use_qk_norm=True,
                                     implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    single = torch.randn(A, 1, L, 768, device="cuda")
    cond = torch.randn(A, 1, L, 384, device="cuda")
    pair = torch.randn(1, L, L, 128, device="cuda")
    w = torch.randn(A, 1, L, 768, device="cuda")

    def run(m, **kw):
        ins = [t.clone().requires_grad_() for t in (single, cond, pair)]
        out = m(*ins, None, **kw)
        (out * w).sum().backward()
        grads = [t.grad for t in ins]
        m.zero_grad(set_to_none=True)
        return [out.detach(), *grads]

    truth = run(ref)
    calls = []
    orig = sm100.augmented_attention_bf16_sm100
    sm100.augmented_attention_bf16_sm100 = lambda *a, **k: calls.append(1) or orig(*a, **k)  # ty: ignore[invalid-assignment] -- deliberate spy
    try:
        got = run(eng, compute_dtype=torch.bfloat16)
    finally:
        sm100.augmented_attention_bf16_sm100 = orig
    assert calls, "the bf16 core did not take the sm_100 kernels"
    prev = settings.configure(augmented_attention_bf16_sm100=False)
    try:
        tri = run(eng, compute_dtype=torch.bfloat16)
    finally:
        settings.configure(augmented_attention_bf16_sm100=prev.augmented_attention_bf16_sm100)
    for name, a, b, t in zip(("out", "dsingle", "dcond", "dpair"), got, tri, truth, strict=False):
        es, et = _rel(a, t), _rel(b, t)
        assert es < 1.5 * et + 1e-4, f"{name}: sm100 {es:.2e} vs triton bf16 {et:.2e}"
