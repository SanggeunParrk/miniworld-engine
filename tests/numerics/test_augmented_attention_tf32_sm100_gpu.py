"""The fp32 (TF32 tensor core) training kernels of the augmented pair-bias attention core on B200
(kernels/augmented_attention/cuda/sm100: attn_fwd_tf32, attn_dkv_tf32, attn_dqb_tf32) against fp64."""

import math

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
H, DH = 16, 48


@pytest.fixture(autouse=True)
def blackwell():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _inputs(A, L, masked, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(A * L, H * DH, device="cuda", generator=g) for _ in range(3))
    bias = torch.randn(H, L, L, device="cuda", generator=g)
    if masked:
        keep = torch.rand(L, device="cuda", generator=g) > 0.2
        bias[:, :, ~keep] = float("-inf")
    return q, k, v, bias


def _reference(q, k, v, bias, A, L):
    qd, kd, vd = (t.double().view(A, L, H, DH) for t in (q, k, v))
    s = torch.einsum("alhd,ajhd->ahlj", qd, kd) / math.sqrt(DH) + bias.double()[None]
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("ahlj,ajhd->alhd", torch.softmax(s, -1), vd).reshape(A * L, H * DH)
    return o, lse * (1 / math.log(2.0)), s


@pytest.mark.parametrize(("A", "L"), [(2, 128), (4, 256), (2, 384)])
@pytest.mark.parametrize("masked", [False, True])
def test_forward_tf32_matches_fp64(A, L, masked):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, masked, A * 1000 + L + masked)
    O, LSE = sm100.forward_tf32(q, k, v, bias, A, L)
    torch.cuda.synchronize()
    o_ref, lse_ref, _ = _reference(q, k, v, bias, A, L)
    assert _rel(O, o_ref) < 2e-3, _rel(O, o_ref)
    # TF32 drops the operands' low mantissa bits: the logits (and LSE) shift by ~1e-3; the backward recomputes P from the same
    # operands, so the shift cancels there
    assert float((LSE.double() - lse_ref).abs().max()) < 5e-3


@pytest.mark.parametrize(("A", "L"), [(2, 128), (4, 256), (2, 384)])
@pytest.mark.parametrize("masked", [False, True])
def test_backward_tf32_matches_fp64(A, L, masked):
    """dQ, dK, dV and dbias (summed over the samples) of the fp32 kernels against fp64 autograd, from the kernels' own O / LSE."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, masked, A * 1000 + L + masked + 7)
    do = torch.randn(A * L, H * DH, device="cuda", generator=torch.Generator(device="cuda").manual_seed(L))
    O, LSE = sm100.forward_tf32(q, k, v, bias, A, L)
    Dd = (do * O).view(A, L, H, DH).sum(-1).permute(0, 2, 1).contiguous()
    DQ, DK, DV, DB = sm100.backward_tf32(q, k, v, do, bias, LSE, Dd, A, L)
    torch.cuda.synchronize()
    leaves = [t.double().requires_grad_() for t in (q, k, v)]
    bd = bias.double().requires_grad_()
    o, _, _ = _reference(leaves[0], leaves[1], leaves[2], bd, A, L)
    o.backward(do.double())
    for name, got, want in (("dq", DQ, leaves[0].grad), ("dk", DK, leaves[1].grad), ("dv", DV, leaves[2].grad), ("dbias", DB, bd.grad)):
        assert _rel(got, want) < 3e-3, f"{name}: {_rel(got, want):.2e}"
