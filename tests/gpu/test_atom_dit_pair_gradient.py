"""R16 (FoldForge): the atom-DiT block sends a gradient into its atom-pair (pair-bias) input on every backend that serves training, the
same as the ``pytorch`` reference's (H100: kernels/integrations/local_dit_h100; A100: the sm_80 block; triton).

FoldForge's Protenix v1 parameters that get no gradient (``single_to_pair_cond_row`` / ``_col``, ``embed_pair_offsets``,
``embed_pair_distances`` of the input embedder and the diffusion encoder, without the ``_1`` suffix) are not upstream of the engine block:
its ``AtomCrossAttEncoder.prepare`` discards the pair that ``_per_atom_conditioning`` builds from them (``token_atoms_single_cond, _ =``),
as AF3's release does; they get None on the ``pytorch`` backend too."""

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTProtenix

pytestmark = [pytest.mark.gpu]


def _backends() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [b for b in ("cuda", "triton") if b in AtomDiTProtenix.backends]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("key_window", ["slide", "pad"])
@pytest.mark.parametrize("dtype", ["float32", "master"])
def test_pair_gradient_matches_pytorch(backend: str, key_window: str, dtype: str) -> None:
    m = AtomDiTProtenix(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, key_window=key_window).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m = typing.cast("AtomDiTProtenix", C.cast(m, dtype).train(True))
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    n, samples, real = 1024, 2, 610
    single = torch.randn(samples, 1, n, 128, device="cuda", generator=g).to(act)
    cond = torch.randn(samples, 1, n, 128, device="cuda", generator=g).to(act)
    pair = torch.randn(1, n // 32, 32, 128, 16, device="cuda", generator=g).to(act)
    mask = (torch.arange(n, device="cuda") < real)[None]
    dy = torch.randn(samples, 1, n, 128, device="cuda", generator=g)
    m.set_backend(backend)
    why = m.unsupported(single, cond, pair, mask)
    if why is not None:
        pytest.skip(f"{backend}: {why}")
    grads = {}
    for b in ("pytorch", backend):
        m.set_backend(b)
        z = pair.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master"):
            y = m(single, cond, z, mask)
        (y.float() * dy).sum().backward()
        assert z.grad is not None, b
        grads[b] = z.grad.float()
    ref, got = grads["pytorch"], grads[backend]
    assert ref.abs().max() > 0
    err = ((got - ref).abs().max() / ref.abs().max()).item()
    assert err < (5e-3 if dtype == "float32" else 3e-2), err
