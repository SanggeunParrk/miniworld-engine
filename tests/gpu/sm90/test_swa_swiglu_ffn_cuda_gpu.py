"""The SWA modules' SwiGLUFFN on H100 runs the hand-CUDA ``kernels.swiglu_ffn`` (bf16, the registry's declared precision), not the PyTorch statements.

Both modules (``modules.swa_atom_transformer.SwiGLUFFN``, ``modules.swa_dit.SwiGLUFFN``) at the declared atom shape (d_model 128, hidden 256 -- registry_module.csv
``swiglu_ffn,atom_single``; 5 evaluation / 48 training augmentations of 1024 atoms). Asserted: the CUDA kernel IS called (the PyTorch statements would pass the
accuracy bar too, so a silent fallback is a failure), the output and the gradients are within 1.5x of the bf16 PyTorch module's error against an fp64 run of the
same weights (the repo's yardstick), and fp32 -- not a declared precision of the row -- keeps the PyTorch statements and equals the PyTorch module bit for bit.
"""

import copy

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]


def _modules(which):
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    if which == "atom":
        from miniworld_engine.modules.swa_atom_transformer import SwiGLUFFN
    else:
        from miniworld_engine.modules.swa_dit.module import SwiGLUFFN
    torch.manual_seed(7)
    mw = SwiGLUFFN(128, 2, implementation=IT.MINIWORLD).cuda()
    with torch.no_grad():
        for p in mw.parameters():
            p.normal_(0, p.shape[-1] ** -0.5)
    pt = SwiGLUFFN(128, 2, implementation=IT.PYTORCH).cuda()
    pt.load_state_dict(mw.state_dict())
    return mw, pt


def _rel(a, b):
    return float((a.double() - b).norm() / b.norm())


def _spy(monkeypatch):
    from miniworld_engine.kernels.conditioned_transition.cuda import swiglu_ffn as cuda

    calls = []
    orig = cuda.swiglu_ffn

    def spy(*a, **k):
        out = orig(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(cuda, "swiglu_ffn", spy)
    return calls


@pytest.mark.parametrize("which", ["atom", "dit"])
@pytest.mark.parametrize(("aug", "train"), [(5, False), (48, True)], ids=["eval-A5", "train-A48"])
def test_bf16_runs_the_cuda_kernel_and_holds_the_yardstick(monkeypatch, which, aug, train):
    mw, pt = _modules(which)
    ref = copy.deepcopy(pt).double()
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(aug, 1024, 128, device="cuda", generator=g)
    dy = torch.randn(aug, 1024, 128, device="cuda", generator=g)
    calls = _spy(monkeypatch)

    def run(m, dtype, autocast):
        xi = x.to(dtype).clone().requires_grad_(train)
        m.zero_grad()
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            y = m(xi)
        out = {"y": y.detach().double()}
        if train:
            y.backward(dy.to(y.dtype))
            out["dx"] = xi.grad.double()
            out["w_up"] = m.w_up.weight.grad.double()
            out["w_down"] = m.w_down.weight.grad.double()
        return out

    r = run(ref, torch.float64, False)
    e = run(mw, torch.bfloat16, True)
    assert calls and all(calls), "the hand-CUDA swiglu_ffn did not serve the call"
    b = run(pt, torch.bfloat16, True)
    for k in r:
        assert _rel(e[k], r[k]) <= 1.5 * _rel(b[k], r[k]) + 1e-6, (k, _rel(e[k], r[k]), _rel(b[k], r[k]))


@pytest.mark.parametrize("which", ["atom", "dit"])
def test_fp32_keeps_the_pytorch_statements(monkeypatch, which):
    mw, pt = _modules(which)
    x = torch.randn(5, 1024, 128, device="cuda")
    calls = _spy(monkeypatch)
    with torch.no_grad():
        assert torch.equal(mw(x), pt(x))
    assert not calls, "fp32 is not a declared precision of the swiglu_ffn rows; the CUDA kernel must not be called"
