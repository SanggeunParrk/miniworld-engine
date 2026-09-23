"""The fused OuterProductMean TRAINING path (integrations.opm_train) against this engine's own path. Needs an H100."""
import os

import pytest
import torch

from miniworld_engine.integrations import opm_train as ot
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0),
                       reason="the kernels are built for sm_90a"),
]

D_MSA, D_PAIR, D_HID, L, S = 64, 128, 32, 384, 256
DEV, DT = "cuda", torch.bfloat16
PARAMS = ("ln_msa.weight", "ln_msa.bias", "to_left.weight", "to_right.weight", "to_out.weight", "to_out.bias")


@pytest.fixture(autouse=True)
def explicit_baseline(monkeypatch):
    """A baseline call explicitly disables the now-default native path."""
    monkeypatch.setenv(ot.ENV, "0")
    monkeypatch.delenv("MINIWORLD_PWA_INFER", raising=False)


class opted_in:
    def __enter__(self):
        self.saved = os.environ.get(ot.ENV)
        os.environ[ot.ENV] = "1"

    def __exit__(self, *a):
        if self.saved is None:
            os.environ.pop(ot.ENV, None)
        else:
            os.environ[ot.ENV] = self.saved


def rel(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float())).item()


@pytest.fixture(scope="module")
def module():
    torch.manual_seed(0)
    m = OuterProductMean(D_MSA, D_PAIR, D_HID, implementation=ImplementationType.MINIWORLD).to(DEV).to(DT)
    with torch.no_grad():                      # gamma = 1 / zero-initialised projections hide kernel bugs: randomise everything
        m.ln_msa.weight.normal_(1.0, 0.3); m.ln_msa.bias.normal_(0.0, 0.3)
        m.to_left.weight.normal_(0, 0.15); m.to_right.weight.normal_(0, 0.15)
        m.to_out.weight.normal_(0, 0.05); m.to_out.bias.normal_(0, 0.2)
    return m.train()


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(1)
    return {
        "msa": (torch.randn(1, S, L, D_MSA, device=DEV, dtype=DT) * 0.5).contiguous(),
        "pair": (torch.randn(1, L, L, D_PAIR, device=DEV, dtype=DT) * 0.5).contiguous(),
        "gz": (torch.randn(1, L, L, D_PAIR, device=DEV, dtype=DT) * 0.1).contiguous(),
        "full": torch.ones(1, S, L, device=DEV, dtype=torch.bool),
        "ragged": ((torch.rand(1, S, 1, device=DEV) < 0.7) & (torch.rand(1, 1, L, device=DEV) < 0.85)),
    }


def _run(module, inputs, mask_key, fused, save_o=True):
    msa = inputs["msa"].clone().requires_grad_(True)
    pair = inputs["pair"].clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    if fused:
        os.environ["MINIWORLD_OPM_TRAIN_SAVE_O"] = "1" if save_o else "0"
        with opted_in():
            out = module(msa, inputs[mask_key], residual=pair)
    else:
        out = module(msa, inputs[mask_key], residual=pair)
    out.backward(inputs["gz"])
    grads = {n: p.grad.detach().clone() for n, p in module.named_parameters()}
    return out.detach(), msa.grad.detach().clone(), pair.grad.detach().clone(), grads


@pytest.mark.parametrize("mask_key", ["full", "ragged"])
@pytest.mark.parametrize("save_o", [True, False])
def test_fused_training_path_matches_the_engines_own(module, inputs, mask_key, save_o):
    out_e, dmsa_e, dpair_e, g_e = _run(module, inputs, mask_key, fused=False)
    out_f, dmsa_f, dpair_f, g_f = _run(module, inputs, mask_key, fused=True, save_o=save_o)
    res = inputs["pair"]
    assert rel(out_f.float() - res.float(), out_e.float() - res.float()) < 1e-2      # the OPM output itself, not the residual
    assert rel(dpair_f, dpair_e) == 0.0                                              # the residual's gradient passes straight through
    assert rel(dmsa_f, dmsa_e) < 2e-2
    for n in PARAMS:
        assert rel(g_f[n], g_e[n]) < 2e-2, n


def test_refusals(inputs):
    kw = {"interchain": False, "normalize_before_proj": True}
    assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, **kw) is not None           # not opted in
    with opted_in():
        assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, **kw) is None
        assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, interchain=True, normalize_before_proj=True) is not None
        assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, interchain=False, normalize_before_proj=False) is not None
        assert ot.refusal(inputs["msa"][:, :200], D_MSA, D_HID, D_PAIR, **kw) is not None   # S % 256


def test_default_auto_path_compiles_forward_and_backward(module, inputs, monkeypatch):
    from miniworld_engine import settings
    monkeypatch.delenv(ot.ENV)
    monkeypatch.setenv("MINIWORLD_OPM_TRAIN_SAVE_O", "1")
    assert settings.current().engine_backend == "auto"
    assert ot.wanted(module.implementation)
    x = inputs["msa"].transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_(True)
    args = (x, *module.parameters())
    y = module(x, inputs["ragged"])
    expected = torch.autograd.grad(y, args, inputs["gz"])
    from torch._dynamo.testing import CompileCounterWithBackend
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(module, backend=counter, fullgraph=True)
    z = compiled(x, inputs["ragged"])
    actual = torch.autograd.grad(z, args, inputs["gz"])
    for _ in range(12):
        compiled(x, inputs["ragged"])
    assert counter.frame_count == 1
    torch.testing.assert_close(z, y, rtol=0, atol=0)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x, inputs["ragged"]), y, rtol=0, atol=0)


@pytest.mark.parametrize("length", [384, 768])
def test_residual_epilogue_rounding_and_identity_gradient(length, monkeypatch):
    monkeypatch.setenv(ot.ENV, "1")
    torch.manual_seed(71)
    m = OuterProductMean(64, 128, 32, implementation="miniworld").cuda().bfloat16()
    with torch.no_grad():
        m.to_out.weight.normal_(std=.03)
    x = torch.randn(1, 256, length, 64, device=DEV, dtype=DT, requires_grad=True)
    mask = torch.rand(1, 256, length, device=DEV) > .2
    # Noncontiguous residual must keep both its value and its gradient mapping.
    residual = torch.randn(1, length, length, 128, device=DEV, dtype=DT).transpose(1, 2).requires_grad_()
    with torch.no_grad():
        expected = m(x, mask) + residual
        actual = m(x, mask, residual=residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    leaves = (x, residual, *m.parameters())
    baseline = m(x, mask) + residual
    expected_grads = torch.autograd.grad(baseline, leaves, grad)
    compiled = torch.compile(m, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    actual = compiled(x, mask, residual=residual)
    actual_grads = torch.autograd.grad(actual, leaves, grad)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads[1], grad, rtol=0, atol=0)
