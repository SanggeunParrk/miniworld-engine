"""R2 (FoldForge): ``OuterProductMean(count_offset=1e-3)`` (AlphaFold 3's pair divisor ``n + 1e-3``) on the kernels -- cuda (H100 sm_90a, A100 sm_80)
and triton serve it, forward and backward, and match the float64 reference as closely as the ``pytorch`` backend does (padded tokens included: their
pairs divide by 1e-3, not by 1)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import OuterProductMean

pytestmark = [pytest.mark.gpu]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _run(module, msa, mask, gout, train: bool):
    msa = msa.detach().clone().requires_grad_(train)
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=msa.dtype != torch.float64):
        out = module(msa, mask)
    tensors = {"out": out.detach().double()}
    if train:
        module.zero_grad(set_to_none=True)
        out.backward(gout.to(out.dtype))
        assert msa.grad is not None
        tensors["grad:msa"] = msa.grad.double()
        tensors.update({f"grad:{n}": p.grad.double() for n, p in module.named_parameters() if p.grad is not None})
    return tensors


@pytest.mark.skipif(not _card(), reason="H100 / A100")
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize(("d_msa", "d_pair", "length"), [(64, 128, 128), (64, 128, 256), (128, 256, 192)])
def test_count_offset_on_kernels(backend: str, train: bool, d_msa: int, d_pair: int, length: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = OuterProductMean(d_msa, d_pair, 32, normalize_before_proj=False, count_offset=1e-3).cuda()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=1024, dtype=torch.bfloat16, device="cuda", generator=gen)
    msa, mask = inp["msa"], inp["mask"]
    assert not bool(mask.any(1).all()), "the inputs must hold padded tokens (pair count 0)"
    gout = torch.randn((1, length, length, d_pair), device="cuda", generator=gen)

    gold_module = copy.deepcopy(base).double()
    gold = _run(gold_module, msa.double(), mask, gout.double(), train)
    ref = _run(copy.deepcopy(base), msa, mask, gout, train)
    kern = copy.deepcopy(base)
    kern.set_backend(backend)
    assert kern.unsupported(msa.requires_grad_(train), mask) is None
    got = _run(kern, msa, mask, gout, train)

    for name, g in gold.items():
        scale = g.abs().max().clamp_min(1e-30)
        err_ref = float((ref[name] - g).abs().max() / scale)
        err = float((got[name] - g).abs().max() / scale)
        assert err <= max(err_ref, 8e-3), f"{backend} {name}: err {err:.3e}, pytorch {err_ref:.3e}"
