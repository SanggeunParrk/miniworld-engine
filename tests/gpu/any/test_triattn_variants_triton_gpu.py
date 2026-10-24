"""The TriangleAttention variants (RoseTTAFold3's gate / out-projection biases, Chai-1's untransposed output and dual out-projection,
AlphaFold 3's untransposed ending-node bias) on the portable Triton core, forward and gradients, against the module's fp64 torch
statements (``_biased_reference``): the bf16 engine error is held to the bf16 statements' own (x 1.5, floor 2e-3).

Run with ``force_triton_kernels=True`` so the H100 CUDA kernels step aside; the PyTorch statements are made to fail if they run.
Padded keys (mask) and padded query rows are in every case; the cotangent is zero outside the real rows.
"""
from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_attention import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF = "cuda", torch.bfloat16


@pytest.fixture(autouse=True)
def triton_policy():
    previous = settings.current()
    settings.configure(force_triton_kernels=True)
    yield
    settings.configure(**vars(previous))


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _module(d_pair, d_hidden, n_head, starting, transposed_bias, **options):
    torch.manual_seed(0)
    m = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, transposed_bias=transposed_bias,
                          implementation=I.MINIWORLD, p_drop=0.0, **options).to(DEV)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name == "ln_pair.weight" else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


# (d_pair, d_hidden, n_head, starting, transposed_bias, options)
CASES = [
    (256, 256, 4, False, False, {"untransposed_output": True}),                  # Chai-1 trunk, ending, AF3 bias orientation
    (256, 256, 4, True, False, {"dual_output": True}),                            # Chai-1 confidence head
    (256, 256, 4, False, False, {"dual_output": True}),
    (128, 128, 4, False, False, {"untransposed_output": True}),
    (64, 128, 4, False, False, {"untransposed_output": True}),                   # Chai-1 template, head 32
    (64, 256, 4, True, False, {"gate_bias": True, "out_bias": True}),            # RoseTTAFold3 template, head 64
    (64, 256, 4, False, False, {"gate_bias": True, "out_bias": True}),
    (128, 128, 4, True, True, {"gate_bias": True, "out_bias": True}),            # head 32
    (128, 64, 4, False, True, {"gate_bias": True, "out_bias": True}),            # head 16
    (128, 256, 8, True, True, {"gate_bias": True, "out_bias": True, "dual_output": True}),    # 8 heads of 32
    (256, 256, 16, False, False, {"gate_bias": True, "out_bias": True}),         # 16 heads of 16
    (384, 384, 12, True, False, {"gate_bias": True, "dual_output": True}),       # width 384, 12 heads of 32
]


@pytest.mark.parametrize("case", range(len(CASES)), ids=[str(c[:5]) + "".join(sorted(c[5])) for c in CASES])
@pytest.mark.parametrize(("b", "n", "real"), [(1, 160, 160), (2, 192, 150)])
def test_variants_on_the_triton_core(monkeypatch, case, b, n, real):
    d_pair, d_hidden, n_head, starting, transposed_bias, options = CASES[case]
    m = _module(d_pair, d_hidden, n_head, starting, transposed_bias, **options)
    g = torch.Generator(device=DEV).manual_seed(1)
    z = torch.randn(b, n, n, d_pair, device=DEV, generator=g)
    mask = (torch.arange(n, device=DEV) < real).expand(b, n).contiguous()
    valid = (mask[:, :, None] & mask[:, None, :])[..., None]
    dy = torch.randn(b, n, n, d_pair, device=DEV, generator=g) * valid
    gold64 = copy.deepcopy(m).double().train(False)
    ref16 = copy.deepcopy(m).to(BF)
    eng = m.to(BF)

    def run(module, dtype, reference):
        zz = z.to(dtype).requires_grad_()
        out = zz + module._biased_reference(zz, mask) if reference else module(zz, mask)       # the module returns pair + update
        params = list(module.parameters())
        return out, torch.autograd.grad(out, [zz, *params], dy.to(dtype))

    out_gold, g_gold = run(gold64, torch.float64, True)
    out_16, g_16 = run(ref16, BF, True)
    monkeypatch.setattr(TriangleAttention, "_biased_reference", lambda *a, **k: pytest.fail("the PyTorch statements ran"))
    out, g_eng = run(eng, BF, False)
    rows = (slice(None), slice(0, real), slice(0, real))
    assert torch.isfinite(out[rows]).all()
    assert _rel(out[rows], out_gold[rows]) <= max(1.5 * _rel(out_16[rows], out_gold[rows]), 2e-3), \
        (CASES[case], _rel(out[rows], out_gold[rows]), _rel(out_16[rows], out_gold[rows]))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_16, strict=True)):
        assert _rel(a, r) <= max(1.5 * _rel(c, r), 2e-3), (CASES[case], i, _rel(a, r), _rel(c, r))


@pytest.mark.parametrize("case", [5, 1])
def test_variants_in_one_compiled_graph(case):
    """Forward + backward of a variant compile as one graph (no graph break at the Triton core) and agree with eager."""
    from torch._dynamo.testing import CompileCounterWithBackend

    d_pair, d_hidden, n_head, starting, transposed_bias, options = CASES[case]
    m = _module(d_pair, d_hidden, n_head, starting, transposed_bias, **options).to(BF)
    z = torch.randn(1, 160, 160, d_pair, device=DEV, dtype=BF, requires_grad=True)
    mask = torch.rand(1, 160, device=DEV) > 0.1
    eager = m(z, mask)
    eager_g = torch.autograd.grad(eager.float().sum(), [z, *m.parameters()])
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(lambda x: m(x, mask), backend=counter, fullgraph=True)
    for _ in range(2):
        out = compiled(z)
    got_g = torch.autograd.grad(out.float().sum(), [z, *m.parameters()])
    assert counter.frame_count == 1
    for a, b in zip((out, *got_g), (eager, *eager_g), strict=True):
        assert _rel(a, b) < 2e-2
