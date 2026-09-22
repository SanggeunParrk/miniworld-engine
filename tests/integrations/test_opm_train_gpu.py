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
    kw = dict(interchain=False, normalize_before_proj=True)
    assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, **kw) is not None           # not opted in
    with opted_in():
        assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, **kw) is None
        assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, interchain=True, normalize_before_proj=True) is not None
        assert ot.refusal(inputs["msa"], D_MSA, D_HID, D_PAIR, interchain=False, normalize_before_proj=False) is not None
        assert ot.refusal(inputs["msa"][:, :200], D_MSA, D_HID, D_PAIR, **kw) is not None   # S % 256
