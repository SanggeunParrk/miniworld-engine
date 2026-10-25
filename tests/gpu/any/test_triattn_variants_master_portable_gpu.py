"""The TriangleAttention variants (gate / out-projection biases, untransposed output, dual out-projection) with fp32 MASTER parameters under
bf16 activations -- the mixed-precision layout -- on the portable core: the module casts the projections to the activation dtype inside
(an autograd cast), so every gradient comes back in the master dtype (fp32). Against the module's fp64 statements; the bf16 engine error is
held to the PyTorch statements run under bf16 autocast (the same layout), x 1.5, floor 2e-3. bf16 parameters (the previous contract) are
checked to be unchanged by test_triattn_variants_triton_gpu.py.
"""
from __future__ import annotations

import copy
import warnings

import pytest
import torch
from portable_helpers import portable_on, rel

from miniworld_engine.modules.dispatch import KernelRefusedWarning, UnrecordedCardWarning
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_attention import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF = "cuda", torch.bfloat16


@pytest.fixture(autouse=True)
def portable():
    with portable_on():
        yield


def _module(d_pair, d_hidden, n_head, starting, transposed_bias, **options):
    torch.manual_seed(0)
    m = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, transposed_bias=transposed_bias,
                          implementation=I.MINIWORLD, p_drop=0.0, **options).to(DEV)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name == "ln_pair.weight" else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


CASES = [
    (64, 256, 4, True, False, {"gate_bias": True, "out_bias": True}),            # RoseTTAFold3 template
    (128, 128, 4, True, True, {"gate_bias": True, "out_bias": True}),
    (128, 256, 8, True, True, {"gate_bias": True, "out_bias": True, "dual_output": True}),
    (256, 256, 4, False, False, {"untransposed_output": True}),                  # Chai-1 trunk
    (256, 256, 4, True, False, {"dual_output": True}),                            # Chai-1 confidence head
    (64, 128, 4, False, False, {"untransposed_output": True}),
]


@pytest.mark.parametrize("case", range(len(CASES)), ids=[str(c[:5]) + "".join(sorted(c[5])) for c in CASES])
@pytest.mark.parametrize(("b", "n", "real"), [(1, 160, 160), (2, 192, 150)])
def test_fp32_master_parameters_run_the_portable_core(monkeypatch, case, b, n, real):
    d_pair, d_hidden, n_head, starting, transposed_bias, options = CASES[case]
    m = _module(d_pair, d_hidden, n_head, starting, transposed_bias, **options)   # parameters stay fp32
    assert all(p.dtype == torch.float32 for p in m.parameters())
    g = torch.Generator(device=DEV).manual_seed(1)
    z = torch.randn(b, n, n, d_pair, device=DEV, generator=g)
    mask = (torch.arange(n, device=DEV) < real).expand(b, n).contiguous()
    valid = (mask[:, :, None] & mask[:, None, :])[..., None]
    dy = torch.randn(b, n, n, d_pair, device=DEV, generator=g) * valid
    gold64 = copy.deepcopy(m).double().train(False)

    def run(module, dtype, reference, autocast=False):
        zz = z.to(dtype).requires_grad_()
        with torch.autocast("cuda", dtype=BF, enabled=autocast):
            out = zz + module._biased_reference(zz, mask) if reference else module(zz, mask)
        params = list(module.parameters())
        return out, torch.autograd.grad(out, [zz, *params], dy.to(dtype))

    out_gold, g_gold = run(gold64, torch.float64, True)
    out_16, g_16 = run(m, BF, True, autocast=True)                       # the statements in the same layout: fp32 masters, bf16 stream
    monkeypatch.setattr(TriangleAttention, "_biased_reference", lambda *a, **k: pytest.fail("the PyTorch statements ran"))
    with warnings.catch_warnings():
        for category in (UnrecordedCardWarning, KernelRefusedWarning):
            warnings.simplefilter("error", category)
        out, g_eng = run(m, BF, False)
    rows = (slice(None), slice(0, real), slice(0, real))
    assert torch.isfinite(out[rows]).all()
    assert rel(out[rows], out_gold[rows]) <= max(1.5 * rel(out_16[rows], out_gold[rows]), 2e-3), \
        (CASES[case], rel(out[rows], out_gold[rows]), rel(out_16[rows], out_gold[rows]))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_16, strict=True)):
        assert rel(a, r) <= max(1.5 * rel(c, r), 2e-3), (CASES[case], i, rel(a, r), rel(c, r))
        if i:
            assert a.dtype == torch.float32, "parameter gradients stay in the master dtype"


def test_mixed_parameter_dtypes_and_non_bf16_activations_still_refuse():
    from miniworld_engine.kernels.triangle_attention.triton.variants import refusal

    z = torch.zeros(1, 32, 32, 64, device=DEV, dtype=BF)
    w32, w16 = torch.zeros(8, 64, device=DEV), torch.zeros(8, 64, device=DEV, dtype=BF)
    assert refusal(z, (w32, w16, None), 4, 128) is None
    assert refusal(z.float(), (w32,), 4, 128) is not None
    assert refusal(z, (w32.half(),), 4, 128) is not None
