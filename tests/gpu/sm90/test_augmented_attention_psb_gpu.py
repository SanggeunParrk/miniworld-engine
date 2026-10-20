"""The sm_90 bf16 pair-bias attention core with one bias PER SAMPLE (kernels/augmented_attention/cuda, ``PSB=1``) against
an fp64 truth. Same bar as ``test_augmented_attention_bf16_sm90_gpu.py``: no worse than the same math on the bf16-rounded
operands (the floor a correct kernel sits on)."""
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
    bias = torch.randn(16, A, L, L, device="cuda", generator=g) * bias_scale
    mask = (torch.rand(A, 1, L, device="cuda", generator=g) >= mask_frac) if mask_frac else None
    return q, k, v, bias, mask


def _truth(q, k, v, bias, mask, do):
    """fp64 autograd: (O, dq, dk, dv, dbias) in the op's layouts."""
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, bias)]
    qd, kd, vd, bd = leaves
    s = torch.einsum("aihd,ajhd->ahij", qd[:, 0], kd[:, 0]) / math.sqrt(48) + bd.permute(1, 0, 2, 3)
    if mask is not None:
        s = s.masked_fill(~mask[:, 0][:, None, None, :], torch.finfo(s.dtype).min)
    o = torch.einsum("ahij,ajhd->aihd", torch.softmax(s, -1), vd[:, 0])[:, None]
    o.backward(do.double())
    return (o.detach(), *(t.grad for t in leaves))


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


@needs_hopper
@pytest.mark.parametrize(("A", "L", "bias_scale", "mask_frac"), [
    (4, 384, 1.0, 0.0),
    (3, 256, 1.0, 0.2),     # key mask
    (6, 384, 4.0, 0.0),     # wide logits
    (5, 128, 1.0, 0.0),     # one query tile per head
])
def test_per_sample_bias_matches_fp64_at_the_bf16_input_floor(A, L, bias_scale, mask_frac):
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

    q, k, v, bias, mask = _inputs(A, L, seed=A * 1000 + L, bias_scale=bias_scale, mask_frac=mask_frac)
    do = torch.randn(A, 1, L, 16, 48, device="cuda")
    truth = _truth(q, k, v, bias, mask, do)
    r = lambda t: t.bfloat16().float()
    floor = _truth(r(q), r(k), r(v), (bias * LOG2E).bfloat16().double() / LOG2E, mask, r(do))

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    b = bias.clone().requires_grad_()
    o = cuda_sm90.augmented_attention_bf16_sm90_psb(*leaves, b, mask)
    o.backward(do)
    got = (o.detach(), leaves[0].grad, leaves[1].grad, leaves[2].grad, b.grad)
    for name, x, t, f in zip(("O", "dq", "dk", "dv", "dbias"), got, truth, floor, strict=False):
        e, ef = _rel(x, t), _rel(f, t)
        assert math.isfinite(e), f"{name}: not finite"
        assert e < 1.3 * ef + 1e-4, f"{name}: {e:.2e} against a floor of {ef:.2e}"


@needs_hopper
def test_the_samples_really_read_their_own_bias():
    """A per-sample bias must differ from the shared one when the biases differ: swapping two samples' biases changes the
    output of exactly those two samples."""
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

    A, L = 3, 256
    q, k, v, bias, _ = _inputs(A, L, seed=7)
    o = cuda_sm90.augmented_attention_bf16_sm90_psb(q, k, v, bias)
    swapped = bias[:, [1, 0, 2]].contiguous()
    o2 = cuda_sm90.augmented_attention_bf16_sm90_psb(q, k, v, swapped)
    assert _rel(o2[2], o[2]) < 1e-6                      # sample 2 kept its bias
    assert _rel(o2[0], o[0]) > 1e-2 and _rel(o2[1], o[1]) > 1e-2


@needs_hopper
def test_the_backward_is_bitwise_repeatable():
    from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

    A, L = 4, 256
    q, k, v, bias, _ = _inputs(A, L, seed=3)
    do = torch.randn(A, 1, L, 16, 48, device="cuda")
    grads = []
    for _ in range(3):
        leaves = [t.clone().requires_grad_() for t in (q, k, v, bias)]
        cuda_sm90.augmented_attention_bf16_sm90_psb(*leaves[:3], leaves[3]).backward(do)
        grads.append([t.grad for t in leaves])
    for other in grads[1:]:
        assert all(torch.equal(a, b) for a, b in zip(grads[0], other, strict=True))
