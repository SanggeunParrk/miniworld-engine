"""R17 (FoldForge): ``PoolAtoms`` trains -- the kernels' backward (cuda: H100 sm_90a / A100 sm_80; triton) equals the gradient of the reference
statement in float64 up to bf16 rounding, is bitwise reproducible (no atomics, although random atom indices repeat), and a mask that requires
grad is refused."""

import pytest
import torch

from miniworld_engine.modules import PoolAtoms
from miniworld_engine.modules.functional import pool_atoms

pytestmark = [pytest.mark.gpu]


def _cuda_card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _grad(module: PoolAtoms, inp: dict, gout: torch.Tensor) -> torch.Tensor:
    x = inp["x"].detach().clone().requires_grad_(True)
    out = module(x, inp["idx"], inp["gather_mask"], inp["atom_mask"])
    out.backward(gout)
    assert x.grad is not None
    return x.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize(("length", "samples"), [(1024, 1), (2048, 48), (1000, 5)])
def test_pool_atoms_backward(backend: str, length: int, samples: int) -> None:
    if backend == "cuda" and not _cuda_card():
        pytest.skip("the cuda backend is the H100 / A100 build")
    g = torch.Generator(device="cuda").manual_seed(0)
    m = PoolAtoms(backend=backend)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    t = inp["idx"].shape[0]
    gout = torch.randn((samples, t, inp["x"].shape[-1]), device="cuda", generator=g).to(torch.bfloat16)
    assert m.unsupported(inp["x"].requires_grad_(True), inp["idx"], inp["gather_mask"], inp["atom_mask"]) is None
    inp["x"] = inp["x"].detach()

    got = _grad(m, inp, gout)
    again = _grad(m, inp, gout)
    assert torch.equal(got, again), "the backward must be bitwise reproducible"

    x64 = inp["x"].double().requires_grad_(True)
    pool_atoms(x64, inp["idx"], inp["gather_mask"], inp["atom_mask"]).backward(gout.double())
    gold = x64.grad
    assert gold is not None
    err = (got.double() - gold).abs().max() / gold.abs().max()
    assert err < 8e-3, f"{backend}: relative error {err:.3e}"
    # the untouched rows (no gathered slot) and the relu's zero side: zeros, up to an fp32 cancellation of two equal slot terms (fma rounding, ~1e-9)
    assert float(got[gold == 0].abs().max()) <= 1e-6 * float(gold.abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_pool_atoms_refuses_mask_grad() -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    m = PoolAtoms(backend="triton")
    inp = m.make_inputs(length=1024, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    am = inp["atom_mask"].float().requires_grad_(True)
    assert m.unsupported(inp["x"], inp["idx"], inp["gather_mask"], am) is not None
