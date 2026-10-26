"""The widened H100 OuterProductMean path (integrations.opm_train): d_msa 128 with d_pair 256 / 384, ragged lengths and MSA depths,
and the ESMFold2 order (divide after the projection). Reference: the module's own PyTorch statements in fp32.

Needs an H100 (sm_90a); skipped elsewhere.
"""

import pytest
import torch

from miniworld_engine.integrations import opm_train as ot
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0),
        reason="the wide kernels are sm_90a",
    ),
]

DEV, DT = "cuda", torch.bfloat16


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def build(d_msa, d_pair, impl, dtype, after):
    torch.manual_seed(0)
    m = OuterProductMean(
        d_msa, d_pair, 32, normalize_before_proj=not after, implementation=impl
    ).to(DEV)
    with torch.no_grad():  # gamma = 1 / zero-initialised projections hide kernel bugs: randomise everything
        m.ln_msa.weight.normal_(1.0, 0.3)
        m.ln_msa.bias.normal_(0.0, 0.3)
        m.to_left.weight.normal_(0, 0.15)
        m.to_right.weight.normal_(0, 0.15)
        m.to_out.weight.normal_(0, 0.05)
        m.to_out.bias.normal_(0, 0.2)
    return m.to(dtype)


def run(module, msa, mask, pair, gz):
    msa = msa.clone().requires_grad_(True)
    pair = pair.clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    out = module(msa, mask, residual=pair)
    out.backward(gz)
    return (
        out.detach(),
        msa.grad.clone(),
        pair.grad.clone(),
        {n: p.grad.clone() for n, p in module.named_parameters()},
    )


@pytest.mark.parametrize(("d_msa", "d_pair"), [(128, 256), (128, 384), (64, 128)])
@pytest.mark.parametrize(
    ("length", "depth"), [(128, 256), (100, 300)]
)  # the second pair: ragged length and MSA depth (padded)
@pytest.mark.parametrize("after", [False, True])
def test_matches_fp32_reference_and_is_deterministic(
    d_msa, d_pair, length, depth, after
):
    torch.manual_seed(1)
    msa = (torch.randn(1, depth, length, d_msa, device=DEV) * 0.5).to(DT)
    pair = (torch.randn(1, length, length, d_pair, device=DEV) * 0.5).to(DT)
    gz = (torch.randn(1, length, length, d_pair, device=DEV) * 0.1).to(DT)
    mask = (torch.rand(1, depth, 1, device=DEV) < 0.7) & (
        torch.rand(1, 1, length, device=DEV) < 0.85
    )
    ref = build(d_msa, d_pair, ImplementationType.PYTORCH, torch.float32, after)
    r_out, r_dm, _r_dp, r_g = run(ref, msa.float(), mask, pair.float(), gz.float())
    fused = build(d_msa, d_pair, ImplementationType.MINIWORLD, DT, after)
    ot.STATS["served"] = 0
    f_out, f_dm, f_dp, f_g = run(fused, msa, mask, pair, gz)
    assert ot.STATS["served"] == 1, ot.STATS["refused"]
    assert cos(f_out.float() - pair.float(), r_out - pair.float()) > 0.9999
    assert cos(f_dm, r_dm) > 0.9999
    assert torch.equal(f_dp, gz)  # the residual's gradient passes straight through
    for n, g in f_g.items():
        assert cos(g, r_g[n]) > 0.9999, n
    f_out2, f_dm2, _, f_g2 = run(fused, msa, mask, pair, gz)  # bitwise deterministic
    assert torch.equal(f_out, f_out2)
    assert torch.equal(f_dm, f_dm2)
    assert all(torch.equal(f_g[n], f_g2[n]) for n in f_g)
    with torch.no_grad():
        inf = fused(msa, mask, residual=pair)
        # Inference of the MiniWorld width is the single fused kernel (`opm_fused_infer`): the same bf16 products as the training forward
        # (cuBLAS + the epilogue kernel), summed in another order, so equal to bf16 noise but not bitwise; every other width shares the
        # training forward exactly.
        if (d_msa, d_pair) == (64, 128):
            assert cos(inf.float() - pair.float(), f_out.float() - pair.float()) > 0.99999
            assert torch.equal(fused(msa, mask, residual=pair), inf)
        else:
            assert torch.equal(inf, f_out)


def test_refusals():
    msa = torch.zeros(1, 256, 128, 128, device=DEV, dtype=DT)
    kw = {"interchain": False, "normalize_before_proj": True}
    assert ot.refusal(msa, 128, 32, 256, **kw) is None
    assert ot.refusal(msa, 128, 32, 384, **kw) is None
    assert (
        ot.refusal(msa, 128, 32, 128, **kw) is not None
    )  # (128, 128) is not a registry shape
    assert ot.refusal(msa, 128, 16, 256, **kw) is not None  # d_hidden
    assert (
        ot.refusal(msa, 128, 32, 256, interchain=True, normalize_before_proj=True)
        is not None
    )
    assert (
        ot.refusal(msa[:, :, :100], 128, 32, 256, **kw) is None
    )  # ragged length: padded by the wrapper
    assert ot.refusal(msa[:, :200], 128, 32, 256, **kw) is None  # ragged depth
    assert (
        ot.refusal(msa, 128, 32, 256, interchain=False, normalize_before_proj=False)
        is None
    )  # ESMFold2 order
