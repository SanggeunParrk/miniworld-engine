"""Boltz-2's up-gated conditioned transition (``UpGatedConditionedTransition``: ``hidden = silu(W_a h) (W_b h) (W_u h)``) on the RTX 5090's
sm_80-family ``cuda`` path at inference (``conditioned_transition_sm80.inference_up``: AdaLN rows, one ``[a | b | u]`` GEMM, ``silu(a) b u`` in one
row pass, cached weight casts).

Before (TokenDiTBoltz2 cuda:5090_sm120 2), the token DiT ran this transition as the module's own statements: every call cast the fp32 master
weights to bf16 (seven casts a block) and ran six split-K cuBLAS GEMMs -- FoldJet Boltz-2 5I28 n = 1 spent ~0.55 s of its 1.42 s there.
A gradient, and the A100, keep the module's statements (``serves`` refuses them).
"""

from __future__ import annotations

import typing

import pytest
import torch

from foldjet_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80
from foldjet_engine.measure import config as C
from foldjet_engine.modules.token_dit import TokenDiTBoltz2

pytestmark = [pytest.mark.gpu]

_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _block(seed: int = 0) -> typing.Any:
    m = TokenDiTBoltz2(d_single=768, d_cond=768, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(seed))
    return m.eval()


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a.detach().double() - ref.detach()).abs().max() / ref.detach().abs().max().clamp_min(1e-30))


@pytest.mark.skipif(not _SM120, reason="RTX 5090 (sm_120) cuda path")
def test_serves_inference_only() -> None:
    m = _block()
    x = torch.randn(5, 1, 128, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(1, 1, 128, 768, device="cuda", dtype=torch.bfloat16).expand(5, 1, 128, 768)
    assert ct_sm80.serves(m.transition, x, c, torch.bfloat16, False) is True
    assert ct_sm80.serves(m.transition, x, c, torch.float32, False) is True
    assert ct_sm80.serves(m.transition, x, c, torch.bfloat16, True) is False  # training keeps the module's statements


@pytest.mark.skipif(not _SM120, reason="RTX 5090 (sm_120) cuda path")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
def test_transition_vs_fp64(dtype: torch.dtype, shared: bool) -> None:
    """The transition alone (update with the residual): within the pytorch error of the fp64 reference (bf16: fp32 master weights)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    m = _block(3)
    m64 = _block(3).double()
    m64.load_state_dict(m.state_dict())
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(5, 1, 256, 768, device="cuda", generator=g)
    c = torch.randn(1 if shared else 5, 1, 256, 768, device="cuda", generator=g).expand(5, 1, 256, 768)
    with torch.no_grad():
        ref = m64.transition(x.double(), c.double())
        xk, ck = x.to(dtype), c.to(dtype)
        yk = ct_sm80.update(m.transition, xk, ck, True, dtype, False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
            ypt = m.transition(xk, ck)
    assert yk.shape == x.shape and yk.dtype == dtype
    assert _rel(yk, ref) <= max(_rel(ypt, ref), 8e-3 if dtype is torch.bfloat16 else 1e-5)


@pytest.mark.skipif(not _SM120, reason="RTX 5090 (sm_120) cuda path")
@pytest.mark.parametrize("length", [128, 256])
def test_token_dit_master_cuda_serves_and_matches(length: int) -> None:
    """The block (bf16 activations, fp32 master weights under bf16 autocast, the conditioning shared by 5 samples as FoldJet passes it):
    the cuda route runs the transition on ``inference_up`` and stays within the pytorch error of the fp64 reference."""
    m = _block()
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single")
    inp["cond"] = inp["cond"][:1].expand_as(inp["cond"])
    m64 = _block()
    m64.load_state_dict(m.state_dict())
    m64 = m64.double()
    inp64 = {k: (v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in inp.items()}
    calls = []
    real = ct_sm80.inference_up

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    ct_sm80.inference_up = spy  # pyright: ignore[reportAttributeAccessIssue]
    try:
        with torch.no_grad():
            ref = m64(**inp64)
            m.set_backend("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"]) is None
                y = m(**inp)
            m.set_backend("pytorch")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yp = m(**inp)
    finally:
        ct_sm80.inference_up = real  # pyright: ignore[reportAttributeAccessIssue]
    assert calls, "the up-gate transition did not run on conditioned_transition_sm80.inference_up"
    assert _rel(y, ref) <= max(_rel(yp, ref), 8e-3)
