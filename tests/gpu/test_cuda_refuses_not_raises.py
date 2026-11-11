"""Calls from FoldForge's 5I28 run (2026-11-11, engine 2.5.1, H100) where a ``cuda`` backend claimed a call and its kernel then raised,
killing the fold instead of letting the caller fall back:

- Chai-1's token DiT (n <= 4 copies, 128-512 tokens): bf16 ``single`` with the fp32 conditioning / pair; the H100 gate claimed it and
  ``conditioned_transition.cuda.token_inference`` raised ``ValueError('Hopper token CUDA candidate requires ...')``. The gate now asks for the
  launcher's own conditions (the pair in ``single``'s dtype, [B, L, L, w]), and the Chai-1 classes round the master layout's fp32
  activations to bf16 at the entry (FoldForge R7), so the call is served.
- OpenDDE's triangle attention at 2048 tokens (16 copies): one sample's working set over the inference cap runs in blocks of pair rows
  (``triattn_h100_sm90._generic_rows``), whose q | k | v | g GEMM an enclosing bf16 autocast turned bf16; the fp16 core raised
  ``RuntimeError('q / k / v: [B, N, S, H, 32 | 16] fp16 rows at one common stride')``. The row-block path now runs with autocast off, as
  the one-launch path does. The test forces the row blocks at a small length.
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.primitives import BackendUnsupported
from miniworld_engine.modules.token_dit import TokenDiTChai1, TokenDiTProtenix
from miniworld_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def _master(module: torch.nn.Module, seed: int = 0) -> typing.Any:
    module = module.cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(seed))
    return C.cast(module, "master").eval()


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("length", [128, 256, 512])
def test_chai1_token_dit_mixed_dtypes_served(length: int) -> None:
    m = _master(TokenDiTChai1(d_single=768, d_cond=384, d_pair=256, n_head=16, n=2))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(
        length=length, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_fp32_cond"
    )
    assert (inp["single"].dtype, inp["cond"].dtype, inp["pair"].dtype) == (torch.bfloat16, torch.float32, torch.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # the master layout (fp32 weights, bf16 compute)
        m.set_backend("cuda")
        assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"]) is None
        y = m(**inp)
        m.set_backend("pytorch")
        ref = m(**inp)
    assert y.dtype == torch.bfloat16 and y.shape == ref.shape
    assert _rel(y, ref) < 2e-2


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_token_dit_gate_refuses_what_the_launcher_rejects() -> None:
    """A block without the entry rounding (TokenDiTProtenix) refuses a pair in another dtype than ``single`` or of another shape than
    [B, L, L, w], with a reason, instead of claiming it and raising in the launcher."""
    m = _master(TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=128, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    m.set_backend("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"]) is None  # served as it is
        for pair in (inp["pair"].float(), inp["pair"][0]):  # fp32 pair next to a bf16 single; [L, L, w] (no batch dimension)
            why = m._supports_cuda(inp["single"], inp["cond"], pair, inp["mask"])
            assert isinstance(why, str) and why
            with pytest.raises(BackendUnsupported):
                m(inp["single"], inp["cond"], pair, inp["mask"])


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) triangle attention")
@pytest.mark.parametrize(("d_pair", "n_head"), [(384, 12), (64, 2)])  # OpenDDE's two triangle attentions
@pytest.mark.parametrize("starting", [True, False])
def test_triattn_row_blocks_under_autocast(monkeypatch: pytest.MonkeyPatch, d_pair: int, n_head: int, starting: bool) -> None:
    from miniworld_engine.kernels.integrations import triattn_h100_sm90 as tri

    m = _master(TriangleAttention(d_pair=d_pair, n_head=n_head, starting=starting, model="opendde"))
    m.set_backend("cuda")
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=256, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    with torch.no_grad():
        one = m(**inp)  # one launch
        calls = []
        rows = tri._generic_rows

        def spy(*args):
            calls.append(args[-1])
            return rows(*args)

        monkeypatch.setattr(tri, "_rows_blocks", lambda kind, pair, n_head: 4)  # as a sample over the inference cap
        monkeypatch.setattr(tri, "_generic_rows", spy)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(**inp)
        m.set_backend("pytorch")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = m(**inp)
    assert calls == [4]
    assert y.dtype == torch.bfloat16 and y.shape == one.shape
    assert _rel(y, one) < 1e-2
    assert _rel(y, ref) < 2e-2
