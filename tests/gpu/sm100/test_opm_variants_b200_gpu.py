"""The OuterProductMean options of FoldForge's families on B200 inference: Protenix-v2 / OpenDDE widths (d_msa 128, d_pair 256 / 384),
either normalisation order, RoseTTAFold3's biased projections, Chai-1's 8 groups with the product LayerNorm, a batch of two MSA stacks.
The widths of ``opm_train.SM100_D_MSA`` / ``SM100_D_PAIR`` (the biases included) run the sm_100a kernels (``integrations.opm_train``),
Chai-1 its sm_100a passes (``kernels.outer_product_mean.chai``); a batch of MSA stacks runs the same kernels one stack at a time (since
2026-10-29; it ran the composed CUDA / cuBLAS path before).  Against the fp64 statements: the engine's bf16 error held to 1.5x the bf16
statements' own, and the call must not run the PyTorch statements."""

import copy

import pytest
import torch

from miniworld_engine.integrations import opm_train
from miniworld_engine.kernels.outer_product_mean import chai, composed
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100a) required",
    ),
]

# name: (d_msa, d_pair, d_hidden, options, the engine path that must serve it)
VARIANTS = {
    "protenix2_after": (128, 256, 32, {"normalize_before_proj": False}, "opm_train"),
    "protenix2_before": (128, 256, 32, {}, "opm_train"),
    "opendde_after": (128, 384, 32, {"normalize_before_proj": False}, "opm_train"),
    "rf3_before": (64, 128, 32, {"projection_bias": True}, "opm_train"),
    "rf3_after": (64, 128, 32, {"projection_bias": True, "normalize_before_proj": False}, "opm_train"),
    "rf3_wide_after": (128, 256, 32, {"projection_bias": True, "normalize_before_proj": False}, "opm_train"),
    "chai1": (64, 256, 64, {"groups": 8}, "chai"),
}


def build(d_msa, d_pair, d_hidden, options):
    torch.manual_seed(0)
    m = OuterProductMean(d_msa, d_pair, d_hidden, implementation=ImplementationType.MINIWORLD, **options).cuda().eval()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize(("b", "s", "n", "residual"), [(1, 512, 128, True), (1, 1024, 256, False), (2, 256, 192, False), (1, 200, 100, True)])
def test_variant_matches_fp64_statements(name, b, s, n, residual, monkeypatch):
    d_msa, d_pair, d_hidden, options, served_by = VARIANTS[name]
    calls = []
    if served_by == "opm_train":
        real = opm_train.outer_product_mean
        monkeypatch.setattr(opm_train, "outer_product_mean", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    elif served_by == "chai":
        real = chai.forward
        monkeypatch.setattr(chai, "forward", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    else:
        real = getattr(composed, served_by)
        monkeypatch.setattr(composed, served_by, lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    m = build(d_msa, d_pair, d_hidden, options)
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(b, s, n, d_msa, device="cuda", generator=g)
    mask = torch.rand(b, s, n, device="cuda", generator=g) > 0.2
    mask[:, :, 3] = False                     # a token without a valid row: count clamped to 1
    pair = torch.randn(b, n, n, d_pair, device="cuda", generator=g) if residual else None
    gold, ref16, eng = copy.deepcopy(m).double(), copy.deepcopy(m).bfloat16(), copy.deepcopy(m).bfloat16()
    with torch.no_grad():
        want = gold._variant_reference(msa.double(), mask, None, None if pair is None else pair.double())
        base = ref16._variant_reference(msa.bfloat16(), mask, None, None if pair is None else pair.bfloat16())
        got = eng(msa.bfloat16(), mask, residual=None if pair is None else pair.bfloat16())

    def rel(x):
        return ((x.double() - want).norm() / want.norm()).item()

    assert calls, f"{name}: not served by {served_by}"
    assert torch.isfinite(got).all()
    assert rel(got) <= max(1.5 * rel(base), 1e-2), f"{name}: engine {rel(got):.2e} vs bf16 statements {rel(base):.2e}"
