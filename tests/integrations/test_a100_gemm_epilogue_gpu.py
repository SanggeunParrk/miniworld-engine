"""The A100 hand-CUDA GEMM-with-epilogue building blocks of the kernel-level benches (kernels/transition/cuda/gemm_epilogue_sm80.py) against fp32 PyTorch:
LayerNorm + Linear (forward and backward) and the gated dual-GEMM front of the triangle multiplication (forward and backward).  Errors are held to the bf16
PyTorch computation's own error."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("shape", [(1, 24, 24), (130,), (3, 77)])
def test_layernorm_linear_forward_and_backward(d, shape):
    from miniworld_engine.kernels.transition.cuda import gemm_epilogue_sm80 as ge

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(*shape, d, device="cuda", generator=g)
    dy = torch.randn(*shape, d, device="cuda", generator=g)
    gamma, beta = torch.rand(d, device="cuda", generator=g) + 0.5, torch.randn(d, device="cuda", generator=g) * 0.1
    w = torch.randn(d, d, device="cuda", generator=g) * d**-0.5
    assert ge.supported(x.bfloat16(), d)

    def grads(dtype, fn):
        leaves = [t.clone().to(dtype).requires_grad_() for t in (x, gamma, beta, w)]
        y = fn(*leaves)
        y.backward(dy.to(dtype))
        return [y.detach().float(), *[t.grad.float() for t in leaves]]

    def ref(x_, g_, b_, w_):
        return F.linear(F.layer_norm(x_, (d,), g_, b_, 1e-5), w_)

    want = grads(torch.float32, ref)
    base = grads(torch.bfloat16, ref)
    got = grads(torch.bfloat16, lambda x_, g_, b_, w_: ge.layernorm_linear_sm80(x_, g_, b_, w_, 1e-5))
    for name, a, b, r in zip(("y", "dx", "dgamma", "dbeta", "dW"), got, base, want, strict=True):
        assert _rel(a, r) <= 1.1 * _rel(b, r) + 1e-4, (name, _rel(a, r), _rel(b, r))


def test_layernorm_linear_gate_and_compile():
    from miniworld_engine.kernels.transition.cuda import gemm_epilogue_sm80 as ge

    x = torch.randn(1, 20, 20, 256, device="cuda", dtype=torch.bfloat16)
    assert not ge.supported(x.float(), 256)
    assert not ge.supported(torch.randn(10, 192, device="cuda", dtype=torch.bfloat16), 192)      # width without a build
    assert not ge.supported(x, 100)                                                              # N not a multiple of 128
    gamma, beta = (torch.rand(256, device="cuda", dtype=torch.bfloat16) + 0.5), torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.06
    torch._dynamo.reset()
    with torch.no_grad():
        want = ge.layernorm_linear_sm80(x, gamma, beta, w, 1e-5)
        got = torch.compile(lambda a: ge.layernorm_linear_sm80(a, gamma, beta, w, 1e-5), fullgraph=True)(x)
    assert torch.equal(got, want)
    torch._dynamo.reset()


@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("m", [130, 4096])
def test_dual_gemm_gate_forward_and_backward(d, m):
    from miniworld_engine.kernels.transition.cuda import gemm_epilogue_sm80 as ge

    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(m, d, device="cuda", generator=g)
    dl, dr = (torch.randn(m, d, device="cuda", generator=g) for _ in range(2))
    ws = [torch.randn(d, d, device="cuda", generator=g) * d**-0.5 for _ in range(4)]          # WL, WLg, WR, WRg: used as x @ W
    assert ge.supported(x.bfloat16(), d)

    def front(xx, wl, wlg, wr, wrg):
        return (xx @ wl) * torch.sigmoid(xx @ wlg), (xx @ wr) * torch.sigmoid(xx @ wrg)

    def grads(dtype, fn):
        leaves = [t.clone().to(dtype).requires_grad_() for t in (x, *ws)]
        left, right = fn(*leaves)
        torch.autograd.backward([left, right], [dl.to(dtype), dr.to(dtype)])
        return [left.detach().float(), right.detach().float(), *[t.grad.float() for t in leaves]]

    want = grads(torch.float32, front)
    base = grads(torch.bfloat16, front)
    got_fwd = ge.dual_gemm_gate_sm80(x.bfloat16(), *[w.bfloat16() for w in ws])
    for name, a, b, r in zip(("left", "right"), got_fwd, base[:2], want[:2], strict=True):
        assert _rel(a, r) <= 1.1 * _rel(b, r) + 1e-4, (name, _rel(a, r), _rel(b, r))
    dxn, dwl, dwlg, dwr, dwrg = ge.dual_gemm_gate_bwd_sm80(dl.bfloat16(), dr.bfloat16(), x.bfloat16(), *[w.bfloat16() for w in ws])
    for name, a, b, r in zip(("dx", "dWL", "dWLg", "dWR", "dWRg"), (dxn, dwl, dwlg, dwr, dwrg), base[2:], want[2:], strict=True):
        assert _rel(a, r) <= 1.1 * _rel(b, r) + 1e-4, (name, _rel(a, r), _rel(b, r))
