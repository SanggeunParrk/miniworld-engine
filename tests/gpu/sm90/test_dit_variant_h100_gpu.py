"""The Boltz-2 / RoseTTAFold3 / Chai-1 token DiT variants (``modules.dit_variant``) on the H100 inference path against the fp64 PyTorch
reference: the engine's bf16 error is held to the bf16 reference's own error against fp64."""

import copy

import pytest
import torch

from miniworld_engine.kernels.dit_variants import interface
from miniworld_engine.modules.dit_variant import (
    BOLTZ2,
    BOLTZ2_ATOM,
    CHAI1,
    CHAI1_ATOM,
    ROSETTAFOLD3,
    ROSETTAFOLD3_ATOM,
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

MODELS = {"boltz2": BOLTZ2, "rf3": ROSETTAFOLD3, "chai1": CHAI1}


def _stack(options, n_blocks=3):
    torch.manual_seed(0)
    stack = DiTVariantStack(n_blocks, **options).cuda()
    with torch.no_grad():
        for name, p in stack.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.3)
    return stack


def _inputs(options, a, n, real, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(a, n, 768, device="cuda", generator=g)
    s = torch.randn(n, options["d_cond"], device="cuda", generator=g)
    z = torch.randn(n, n, options["d_pair"], device="cuda", generator=g)
    mask = torch.arange(n, device="cuda") < real
    return x, s, z, mask


def _rel(a, b, rows):
    a, b = a[:, rows].double(), b[:, rows].double()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize(("a", "n", "real"), [(5, 256, 256), (2, 384, 300), (1, 128, 100)])
def test_variant_matches_reference(model, a, n, real):
    options = MODELS[model]
    stack = _stack(options)
    x, s, z, mask = _inputs(options, a, n, real)
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [t.to(torch.bfloat16) for t in (x, s, z)]
    with torch.no_grad():
        assert interface.serves(eng, bf[0], bf[1], bf[2], mask)
        gold = ref64(x.double(), s.double(), z.double(), mask)
        base = ref16(*bf, mask)
        got = eng(*bf, mask)
    assert got.shape == gold.shape
    assert torch.isfinite(got).all()
    rows = slice(0, real)
    err, err16 = _rel(got, gold, rows), _rel(base, gold, rows)
    assert err <= max(1.5 * err16, 1e-2), f"{model}: engine {err:.2e} vs bf16 reference {err16:.2e}"


def test_variant_reuses_pair_bias_and_pack():
    stack = _stack(CHAI1).to(torch.bfloat16)
    x, s, z, mask = (t.to(torch.bfloat16) if t.is_floating_point() else t for t in _inputs(CHAI1, 2, 256, 256))
    with torch.no_grad():
        first = stack(x, s, z, mask)
        second = stack(x, s, z, mask)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


ATOM_MODELS = {"boltz2": BOLTZ2_ATOM, "rf3": ROSETTAFOLD3_ATOM, "chai1": CHAI1_ATOM}


def _atom_stack(options, key_window):
    torch.manual_seed(0)
    stack = AtomDiTVariantStack(3, key_window=key_window, **options).cuda()
    with torch.no_grad():
        for name, p in stack.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.3)
    return stack


# (A, N, real atoms): full; ragged; under 128 real atoms; N an odd number of windows (the engine pads one)
@pytest.mark.parametrize("model", list(ATOM_MODELS))
@pytest.mark.parametrize("key_window", ["slide", "pad"])
@pytest.mark.parametrize(("a", "n", "real"), [(5, 1024, 1024), (2, 768, 700), (1, 256, 100), (3, 544, 544)])
def test_atom_variant_matches_reference(model, key_window, a, n, real):
    stack = _atom_stack(ATOM_MODELS[model], key_window)
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(a, n, 128, device="cuda", generator=g)
    c = torch.randn(n, 128, device="cuda", generator=g)
    z = torch.randn(n // 32, 32, 128, 16, device="cuda", generator=g)
    mask = torch.arange(n, device="cuda") < real
    uid = (torch.arange(n, device="cuda") // 37) if model == "chai1" else None
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [t.to(torch.bfloat16) for t in (x, c, z)]
    with torch.no_grad():
        assert interface.atom_serves(eng, *bf, mask, uid)
        gold = ref64(x.double(), c.double(), z.double(), mask, uid)
        base = ref16(*bf, mask, uid)
        got = eng(*bf, mask, uid)
    assert got.shape == gold.shape
    assert torch.isfinite(got[:, :real]).all()
    rows = slice(0, real)
    err, err16 = _rel(got, gold, rows), _rel(base, gold, rows)
    assert err <= max(1.5 * err16, 1e-2), f"{model} {key_window}: engine {err:.2e} vs bf16 reference {err16:.2e}"
