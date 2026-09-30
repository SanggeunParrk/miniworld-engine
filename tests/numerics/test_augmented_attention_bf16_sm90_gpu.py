"""The sm_90 bf16 pair-bias attention core (kernels/augmented_attention/cuda) against an fp64 truth, and its
module wiring.

The kernels round their operands to bf16, so the bar is not "close to fp64" but "no worse than the same math on the
bf16-rounded inputs" -- a kernel bug shows up as an error well above that floor, a correct kernel sits on it. The
gate matters as much: the tile shapes are literals, so a wrongly accepted call is a wrong answer, not a slow one.
"""
import math

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
HOPPER = CUDA and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(not HOPPER, reason="the kernels are sm_90a only")
LOG2E = 1.0 / math.log(2.0)


def _inputs(A, L, seed=0, bias_scale=1.0, mask_frac=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(A, 1, L, 16, 48, device="cuda", generator=g) for _ in range(3))
    bias = torch.randn(1, L, L, 16, device="cuda", generator=g) * bias_scale
    mask = (torch.rand(A, 1, L, device="cuda", generator=g) >= mask_frac) if mask_frac else None
    return q, k, v, bias, mask


def _truth(q, k, v, bias, mask, do):
    """fp64 autograd: (O, dq, dk, dv, dbias) in the op's layouts."""
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, bias)]
    qd, kd, vd, bd = leaves
    s = torch.einsum("aihd,ajhd->ahij", qd[:, 0], kd[:, 0]) / math.sqrt(48) + bd[0].permute(2, 0, 1)[None]
    if mask is not None:
        s = s.masked_fill(~mask[:, 0][:, None, None, :], torch.finfo(s.dtype).min)
    o = torch.einsum("ahij,ajhd->aihd", torch.softmax(s, -1), vd[:, 0])[:, None]
    o.backward(do.double())
    return (o.detach(), *(t.grad for t in leaves))


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_what_the_tiles_cannot_take():
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

    q, _, _, bias, _ = _inputs(3, 128)
    assert cuda_sm90.supported(q, bias) is HOPPER
    assert cuda_sm90.supported(q, bias.permute(3, 0, 1, 2).contiguous(), bias_head_major=True) is HOPPER
    assert not cuda_sm90.supported(q, bias.permute(3, 0, 1, 2).contiguous())            # layout flag disagrees
    q96, _, _, b96, _ = _inputs(3, 96)
    assert not cuda_sm90.supported(q96, b96)                                           # L not a multiple of 128
    assert not cuda_sm90.supported(q[..., :32], bias)                                  # head dim 32
    assert not cuda_sm90.supported(torch.randn(3, 2, 128, 16, 48, device="cuda"), bias)  # B == 2
    assert not cuda_sm90.supported(q.cpu(), bias.cpu())


@needs_hopper
@pytest.mark.parametrize(("A", "L", "bias_scale", "mask_frac", "head_major"), [
    (6, 384, 1.0, 0.0, False),    # A % 3 == 0: dbias summed on chip (attn_dqb)
    (4, 384, 1.0, 0.0, True),     # A % 3 != 0: per-sample atomics (attn_dq + attn_dkv), head-major bias
    (3, 768, 1.0, 0.2, True),     # key mask, the 2-CTA-per-row forward at L768
    (6, 256, 4.0, 0.0, False),    # wide logits; L % 192 != 0 takes the 2-warpgroup dQ build on the fallback
])
def test_matches_fp64_at_the_bf16_input_floor(A, L, bias_scale, mask_frac, head_major):
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

    q, k, v, bias, mask = _inputs(A, L, seed=A * 1000 + L, bias_scale=bias_scale, mask_frac=mask_frac)
    do = torch.randn(A, 1, L, 16, 48, device="cuda")
    truth = _truth(q, k, v, bias, mask, do)
    # the same math on the operands as the kernels round them: the floor a correct bf16 kernel sits on
    r = lambda t: t.bfloat16().float()
    floor = _truth(r(q), r(k), r(v), (bias * LOG2E).bfloat16().double() / LOG2E, mask, r(do))

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    b = (bias.permute(3, 0, 1, 2).contiguous() if head_major else bias.clone()).requires_grad_()
    assert cuda_sm90.available(leaves[0], b, bias_head_major=head_major)
    o = cuda_sm90.augmented_attention_bf16_sm90(leaves[0], leaves[1], leaves[2], b, mask, bias_head_major=head_major)
    o.backward(do)
    db = b.grad.permute(1, 2, 3, 0) if head_major else b.grad
    got = (o.detach(), leaves[0].grad, leaves[1].grad, leaves[2].grad, db)
    for name, x, t, f in zip(("O", "dq", "dk", "dv", "dbias"), got, truth, floor, strict=False):
        e, ef = _rel(x, t), _rel(f, t)
        assert math.isfinite(e), f"{name}: not finite"
        assert e < 1.3 * ef + 1e-4, f"{name}: {e:.2e} against a floor of {ef:.2e}"


@needs_hopper
def test_no_grad_keeps_nothing():
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

    q, k, v, bias, _ = _inputs(6, 256)
    with torch.no_grad():
        cuda_sm90.augmented_attention_bf16_sm90(q, k, v, bias)          # builds, warms the allocator
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        o = cuda_sm90.augmented_attention_bf16_sm90(q, k, v, bias)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - base == o.numel() * o.element_size()


@needs_hopper
def test_module_bf16_core_takes_the_kernels_and_keeps_the_triton_error():
    """AugmentedAttentionPairBias with compute_dtype=bf16 on a supported shape runs the sm_90 kernels; its error against
    the fp32 PyTorch module is no worse than the bf16 Triton core's (the switch is off in the second run)."""
    from miniworld_engine import settings
    from miniworld_engine.modules import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType

    torch.manual_seed(0)
    A, L = 3, 256
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
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90
    orig = cuda_sm90.augmented_attention_bf16_sm90
    cuda_sm90.augmented_attention_bf16_sm90 = lambda *a, **k: calls.append(1) or orig(*a, **k)  # ty: ignore[invalid-assignment] -- deliberate spy
    try:
        sm90 = run(eng, compute_dtype=torch.bfloat16)
    finally:
        cuda_sm90.augmented_attention_bf16_sm90 = orig
    assert calls, "the bf16 core did not take the sm_90 kernels"
    prev = settings.configure(augmented_attention_bf16_sm90=False)
    try:
        tri = run(eng, compute_dtype=torch.bfloat16)
    finally:
        settings.configure(augmented_attention_bf16_sm90=prev.augmented_attention_bf16_sm90)
    for name, a, b, t in zip(("out", "dsingle", "dcond", "dpair"), sm90, tri, truth, strict=False):
        es, et = _rel(a, t), _rel(b, t)
        assert es < 1.5 * et + 1e-4, f"{name}: sm90 {es:.2e} vs triton bf16 {et:.2e}"


@needs_hopper
def test_checkpoint_keeping_attention_skips_the_recompute_and_changes_nothing():
    """Per-block checkpointing with checkpoint_context_keeping_attention(): the backward's recompute reuses O and the
    LSE instead of launching the forward kernel again, and every gradient is bitwise what plain checkpointing gives
    (the kernel is deterministic, so the kept O is the O a recompute would produce)."""
    from torch.utils.checkpoint import checkpoint

    from miniworld_engine.kernels.augmented_attention import cuda as sm90

    A, L, NB = 3, 256, 3
    torch.manual_seed(0)
    ws = [torch.randn(768, 768, device="cuda") * 768 ** -0.5 for _ in range(3 * NB)]
    x0 = torch.randn(A, 1, L, 768, device="cuda")
    bias = torch.randn(1, L, L, 16, device="cuda")

    def block(x, i):
        q, k, v = ((x @ ws[3 * i + j]).view(A, 1, L, 16, 48) for j in range(3))
        return x + sm90.augmented_attention_bf16_sm90(q, k, v, bias).reshape(A, 1, L, 768)

    def run(keep):
        x = x0.clone().requires_grad_()
        y = x
        for i in range(NB):
            kw = {"context_fn": sm90.checkpoint_context_keeping_attention} if keep else {}
            y = checkpoint(block, y, i, use_reentrant=False, **kw)
        before = sm90.FWD_LAUNCHES[0]
        y.square().mean().backward()
        return x.grad, sm90.FWD_LAUNCHES[0] - before

    g_plain, recomputes_plain = run(False)
    g_keep, recomputes_keep = run(True)
    assert recomputes_plain == NB
    assert recomputes_keep == 0
    assert torch.equal(g_plain, g_keep)
