"""R16 (FoldForge): the atom-DiT block sends a gradient into its atom-pair (pair-bias) input on every backend that serves training, the
same as the ``pytorch`` reference's (H100: kernels/integrations/local_dit_h100; A100: the sm_80 block / kernels/dit_variants; triton).

FoldForge's Protenix v1 parameters that get no gradient (``single_to_pair_cond_row`` / ``_col``, ``embed_pair_offsets``,
``embed_pair_distances`` of the input embedder and the diffusion encoder, without the ``_1`` suffix) are not upstream of the engine block:
its ``AtomCrossAttEncoder.prepare`` discards the pair that ``_per_atom_conditioning`` builds from them (``token_atoms_single_cond, _ =``),
as AF3's release does; they get None on the ``pytorch`` backend too.

FoldForge's pattern: 610 real atoms of 1024, the output masked to the real atoms (``_mask_atoms``), so the upstream gradient is zero on
the padding rows. ``test_padding_windows_past_the_last_real_one`` records where ``cuda`` "slide" differs from the reference: the query
windows wholly past the last real atom (padding queries only)."""

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


def _grads(backend: str, key_window: str, dtype: str, *, masked_dy: bool, real: int = 610):
    """{backend: (output, d pair, d single)} for ``pytorch`` and ``backend`` (None: ``backend`` refuses the call)."""
    m = AtomDiTProtenix(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, key_window=key_window).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m = typing.cast("AtomDiTProtenix", C.cast(m, dtype).train(True))
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    n, samples = 1024, 2
    single = torch.randn(samples, 1, n, 128, device="cuda", generator=g).to(act)
    cond = torch.randn(samples, 1, n, 128, device="cuda", generator=g).to(act)
    pair = torch.randn(1, n // 32, 32, 128, 16, device="cuda", generator=g).to(act)
    mask = (torch.arange(n, device="cuda") < real)[None]
    dy = torch.randn(samples, 1, n, 128, device="cuda", generator=g)
    if masked_dy:
        dy = dy * mask[None, :, :, None]
    m.set_backend(backend)
    if m.unsupported(single, cond, pair, mask) is not None:
        return None
    out = {}
    for b in ("pytorch", backend):
        m.set_backend(b)
        z = pair.clone().requires_grad_(True)
        x = single.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master"):
            y = m(x, cond, z, mask)
        (y.float() * dy).sum().backward()
        assert z.grad is not None and x.grad is not None, b
        out[b] = (y.detach().float(), z.grad.float(), x.grad.float())
    return out


_KNOWN = "known: cuda 'slide' padding query windows past the last real window keep the clipped key window"


def _err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("key_window", ["slide", "pad"])
@pytest.mark.parametrize("dtype", ["float32", "master"])
def test_pair_gradient_matches_pytorch(backend: str, key_window: str, dtype: str) -> None:
    res = _grads(backend, key_window, dtype, masked_dy=True)
    if res is None:
        pytest.skip(f"{backend} refuses the call")
    (_, dz_ref, _), (_, dz, _) = res["pytorch"], res[backend]
    assert dz_ref.abs().max() > 0
    err = _err(dz, dz_ref)
    assert err < (5e-3 if dtype == "float32" else 3e-2), err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize(
    "backend", [pytest.param(b, marks=pytest.mark.xfail(strict=True, reason=_KNOWN)) if b == "cuda" else b for b in _backends()]
)
def test_padding_windows_past_the_last_real_one(backend: str) -> None:
    """ "slide": a query window wholly past the last real atom (padding queries only) reads, in the reference, the slid key window
    ``[R - 128, R)`` (``functional.slide_starts``). ``cuda`` (H100 local_dit_h100, A100 kernels/dit_variants) recomputes only the five edge
    windows chosen on the device (0, 1 and the last three real ones), so those padding windows keep the clipped window: their output rows,
    and with a nonzero upstream gradient on them the pair gradient of those windows and their single gradient, differ from the reference
    (the real rows and every gradient under a masked output agree: ``test_pair_gradient_matches_pytorch``)."""
    res = _grads(backend, "slide", "float32", masked_dy=False)
    if res is None:
        pytest.skip(f"{backend} refuses the call")
    (y_ref, dz_ref, dx_ref), (y, dz, dx) = res["pytorch"], res[backend]
    assert _err(y, y_ref) < 5e-3
    assert _err(dz, dz_ref) < 5e-3
    assert _err(dx, dx_ref) < 5e-3
