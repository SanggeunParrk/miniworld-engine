"""The atom DiT ``cuda`` backend past 8192 atoms (FoldForge's 5I28 benchmark: 12288 / 16384 atoms) matches the ``pytorch`` reference.

Regression (2026-11-12): on H100 the split conditioning kernel (sm90_atom ``cond90_split``) ran its CTA's units 8 .. 15 on LN(c) when the
launcher split the 16 weight units in two (grid.y 2 at 132 .. 263 row tiles = 8448 .. 16832 rows), so the out gates (so, st) of every
non-identity class were sigmoid(LN(c) W + b): output error ~0.08 against the reference (``cond_fwd`` now never splits in two). 8448 is the
first such row count, 12288 the 5I28 one."""

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import atom_dit as ad

pytestmark = [pytest.mark.gpu]

_CLASSES = ("AtomDiTOpenFold3", "AtomDiTProtenix", "AtomDiTBoltz2", "AtomDiTRoseTTAFold3", "AtomDiTChai1")


def _err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.double() - ref.double()).abs().max() / ref.double().abs().max()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("length", [8448, 12288])
@pytest.mark.parametrize("cls", _CLASSES)
def test_cuda_matches_pytorch_past_8192_atoms(cls: str, length: int) -> None:
    m = getattr(ad, cls)(key_window="slide").cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m = typing.cast("ad._LocalAtomDiT", m.train(True))
    inputs = m.make_inputs(
        length=length, samples=1, depth=None, dtype=torch.float32, device=torch.device("cuda"), generator=torch.Generator("cuda").manual_seed(0)
    )
    m.set_backend("cuda")
    why = m.unsupported(**inputs)
    if why is not None:
        pytest.skip(f"cuda refuses: {why}")
    dy = torch.randn(inputs["single"].shape, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    out = {}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for b in ("pytorch", "cuda"):
            m.set_backend(b)
            x = inputs["single"].clone().requires_grad_(True)
            c = inputs["cond"].clone().requires_grad_(True)
            y = m(x, c, inputs["pair"], inputs["mask"])
            (y * dy).sum().backward()
            assert x.grad is not None and c.grad is not None
            out[b] = (y.detach(), x.grad, c.grad)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    (y_ref, dx_ref, dc_ref), (y, dx, dc) = out["pytorch"], out["cuda"]
    # TF32 kernels against the fp32 reference: ~5e-4 (out) / ~2e-3 (gradients); the bug gave ~8e-2 on the output
    assert _err(y, y_ref) < 3e-3
    assert _err(dx, dx_ref) < 1e-2
    assert _err(dc, dc_ref) < 1e-2
