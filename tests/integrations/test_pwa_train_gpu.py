"""The fused MSAPairWeightedAveraging TRAINING path (integrations.pwa_train) against this engine's own path.

Needs an H100; skipped everywhere else. The reference is the SAME module with the opt-in variable unset.
"""
import os

import pytest
import torch

from miniworld_engine.integrations import pwa_train as pt
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0),
                       reason="the kernels are built for sm_90a"),
]

D_MSA, D_PAIR, D_HID, HEADS, L, S = 64, 128, 32, 8, 384, 256
DEV, DT = "cuda", torch.bfloat16
PARAMS = ("ln_msa.weight", "ln_msa.bias", "to_value.weight", "to_gate.weight", "ln_pair.weight", "ln_pair.bias",
          "to_bias.weight", "to_out.weight")


class opted_in:
    def __enter__(self):
        self.saved = os.environ.get(pt.ENV)
        os.environ[pt.ENV] = "1"

    def __exit__(self, *a):
        if self.saved is None:
            os.environ.pop(pt.ENV, None)
        else:
            os.environ[pt.ENV] = self.saved


def rel(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float())).item()


@pytest.fixture(scope="module")
def module():
    torch.manual_seed(0)
    m = MSAPairWeightedAveraging(D_MSA, D_PAIR, HEADS, D_HID, implementation=ImplementationType.MINIWORLD).to(DEV).to(DT)
    with torch.no_grad():                      # zero / unit initialisations hide kernel bugs: randomise everything
        for n, p in m.named_parameters():
            p.normal_(0, 0.3) if "ln_" in n and n.endswith("bias") else p.normal_(1.0 if "ln_" in n else 0.0, 0.3 if "ln_" in n else 0.15)
    return m.train()


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(1)
    return {
        "msa": (torch.randn(1, S, L, D_MSA, device=DEV, dtype=DT) * 0.5).contiguous(),
        "pair": (torch.randn(1, L, L, D_PAIR, device=DEV, dtype=DT) * 0.5).contiguous(),
        "gz": (torch.randn(1, S, L, D_MSA, device=DEV, dtype=DT) * 0.1).contiguous(),
        "full": torch.ones(1, L, device=DEV, dtype=torch.bool),
        "ragged": (torch.rand(1, L, device=DEV) < 0.85),
    }


def _run(module, inputs, mask_key, fused):
    msa = inputs["msa"].clone().requires_grad_(True)
    pair = inputs["pair"].clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    if fused:
        with opted_in():
            out = module(msa, pair, inputs[mask_key])
    else:
        out = module(msa, pair, inputs[mask_key])
    out.backward(inputs["gz"])
    grads = {n: p.grad.detach().clone() for n, p in module.named_parameters()}
    return out.detach(), msa.grad.detach().clone(), pair.grad.detach().clone(), grads


@pytest.mark.parametrize("mask_key", ["full", "ragged"])
def test_fused_training_path_matches_the_engines_own(module, inputs, mask_key):
    out_e, dmsa_e, dpair_e, g_e = _run(module, inputs, mask_key, fused=False)
    out_f, dmsa_f, dpair_f, g_f = _run(module, inputs, mask_key, fused=True)
    assert rel(out_f, out_e) < 5e-3
    assert rel(dmsa_f, dmsa_e) < 5e-3                    # the residual dominates; also check the update part below
    gz = inputs["gz"]
    assert rel(dmsa_f.float() - gz.float(), dmsa_e.float() - gz.float()) < 2e-2
    assert rel(dpair_f, dpair_e) < 1e-2
    for n in PARAMS:
        if n == "ln_pair.bias":                          # a softmax gradient sums to zero over j: dbeta is rounding noise on both sides
            assert g_f[n].float().norm() < 0.05 * g_e["ln_pair.weight"].float().norm()
            continue
        assert rel(g_f[n], g_e[n]) < 1e-2, n


def test_refusals(module, inputs):
    with opted_in():
        assert pt.refusal(inputs["msa"], inputs["pair"], D_MSA, D_PAIR, HEADS, D_HID) is None
        assert pt.refusal(inputs["msa"][:, :, :320], inputs["pair"], D_MSA, D_PAIR, HEADS, D_HID) is not None
    assert pt.refusal(inputs["msa"], inputs["pair"], D_MSA, D_PAIR, HEADS, D_HID) is not None   # not opted in


def test_fused_dropout_is_the_modules_row_dropout(module, inputs):
    """With the module's drop_msa live, the fused path draws its own keep-mask; check it against the engine's own path
    driven by the same mask: the update is masked and scaled, the residual is not."""
    p = 0.15
    torch.manual_seed(7)
    module.drop_msa.p_drop = p
    try:
        msa = inputs["msa"].clone().requires_grad_(True); pair = inputs["pair"].clone().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        with opted_in():
            out_f = module(msa, pair, inputs["full"])
        out_f.backward(inputs["gz"])
        dmsa_f = msa.grad.clone(); dpair_f = pair.grad.clone(); g_f = {n: p_.grad.clone() for n, p_ in module.named_parameters()}
        assert pt.STATS["served"] > 0
        # recover the keep-mask the path used from its own output: update = out - msa is 0 exactly where the mask is 0
        module.drop_msa.p_drop = 0.0
        with torch.no_grad(), opted_in():
            out_nodrop = module(inputs["msa"], inputs["pair"], inputs["full"])
        upd_f = (out_f.detach().float() - inputs["msa"].float())
        keep = (upd_f.abs().sum(1, keepdim=True) > 0)                                      # [1, 1, L, D]
        assert 0.75 < keep.float().mean().item() < 0.95
        # the engine's own path with dropout off, fed the masked/scaled output gradient, gives the same update-side gradients
        msa_e = inputs["msa"].clone().requires_grad_(True); pair_e = inputs["pair"].clone().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        out_e = module(msa_e, pair_e, inputs["full"])
        gz_m = (inputs["gz"] * keep.to(DT) / (1 - p)).to(DT)
        out_e.backward(gz_m)
        assert rel(dpair_f, pair_e.grad) < 1e-2
        for n in PARAMS:
            if n == "ln_pair.bias":
                continue
            assert rel(g_f[n], module.get_parameter(n).grad) < 1e-2, n
        # dmsa: ours = residual grad (gz) + update grad; the engine's = gz_m + update grad
        assert rel(dmsa_f.float() - inputs["gz"].float(), msa_e.grad.float() - gz_m.float()) < 2e-2
        # forward: the kept, scaled update matches the dropout-free update
        upd_n = (out_nodrop.float() - inputs["msa"].float()) * keep.float() / (1 - p)
        assert rel(upd_f, upd_n) < 1e-2
    finally:
        module.drop_msa.p_drop = 0.0


def test_inference_takes_the_same_kernels(module, inputs):
    module.eval()
    try:
        with torch.no_grad():
            out_e = module(inputs["msa"], inputs["pair"], inputs["ragged"])
            served = pt.STATS["served"]
            with opted_in():
                out_f = module(inputs["msa"], inputs["pair"], inputs["ragged"])
        assert pt.STATS["served"] == served + 1
        assert rel(out_f, out_e) < 5e-3
        assert rel(out_f.float() - inputs["msa"].float(), out_e.float() - inputs["msa"].float()) < 2e-2   # the update itself
    finally:
        module.train()
