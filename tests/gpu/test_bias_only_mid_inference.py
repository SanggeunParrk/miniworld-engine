"""Bias-only token DiT inference at 1024 < L <= 3072 (FoldForge AF3 n = 9 .. 24: 1152 .. 3072 tokens), bf16 / master / fp32 parameters, on H100, A100
and RTX 5090.  3.0.5 raised ``RuntimeError: softmax_t: L a multiple of 32, at most 1024`` on the H100 for bf16 / master inference there (the fp32-logit hoist
had moved to ``softmax_t``'s P-only form, which holds 1024 keys): ``softmax_hoist`` takes ``softmax_rows`` past 1024.  A100 / RTX 5090 never called
``softmax_t`` in inference (``softmax_rows``), and are pinned here too.  The cuda backend must serve the lengths (no CUDA hole), run both the pair path
and (H100) the hoisted-bias path, and match a row-chunked float64 reference as closely as the ``pytorch`` backend does (the engine accuracy standard)."""

# torch's Tensor.pow / mean overloads are not resolved by the checker in the error-measure helpers:
# pyright: reportCallIssue=false
from __future__ import annotations

import copy
import gc

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly, pair_bias_all

pytestmark = [pytest.mark.gpu]

_CARD = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
requires_cuda_card = pytest.mark.skipif(_CARD not in ((9, 0), (8, 0), (12, 0)), reason="H100 / A100 / RTX 5090")
S = 2
DTYPES = ["master", "bfloat16", "float32"]


def _block(dtype: str):
    m = TokenDiTMiniWorldBiasOnly(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    out: torch.nn.Module = C.cast(m, dtype)
    return out.eval().requires_grad_(False)


def _inputs(m, length: int, dtype: str, pad: int):
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    inp = m.make_inputs(
        length=length, samples=S, depth=None, dtype=act, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
    if pad:
        mask[:, length - pad :] = False
    inp["mask"] = mask
    return inp


@torch.no_grad()
def _gold_f64(m, inp: dict, rows: int = 64) -> torch.Tensor:
    m64 = copy.deepcopy(m).set_backend("pytorch").double()
    at, tr = m64.attention, m64.transition
    x, c, pair, mask = inp["single"].double(), inp["cond"].double(), inp["pair"], inp["mask"]
    h = at.n_head
    s, res = at.ada_ln_in.forward_with_stream(x, c)
    value, gate = at.to_value(s), at.to_gate(s)
    a, _, length, _ = x.shape
    value = value.view(a, length, h, -1)
    out = torch.empty(a, length, h, value.shape[-1], dtype=torch.float64, device=x.device)
    for i0 in range(0, length, rows):
        z = pair[0, i0 : i0 + rows].double()
        ln = F.layer_norm(z, (z.shape[-1],), None, None, at.ln_pair.eps) * at.ln_pair.weight
        lg = F.linear(ln, at.to_bias.weight).masked_fill(~mask[0][None, :, None], float("-inf"))
        out[:, i0 : i0 + rows] = torch.einsum("ijh,ajhd->aihd", torch.softmax(lg, dim=1), value)
    o = at.to_out(torch.sigmoid(gate) * out.flatten(-2).view(a, 1, length, -1))
    return tr(res + torch.sigmoid(at.to_scale(c)) * o, c)


def _errs(out: torch.Tensor, gold: torch.Tensor) -> tuple[float, float]:
    d = out.double() - gold
    return float(d.pow(2).mean().sqrt() / gold.pow(2).mean().sqrt()), float(d.abs().max() / gold.abs().max())


def _check(dtype: str, length: int, pad: int, hoist: bool) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    m = _block(dtype)
    inp = _inputs(m, length, dtype, pad)
    kern = copy.deepcopy(m).set_backend("cuda")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master")
    with torch.no_grad():
        gold = _gold_f64(m, inp)
        with ctx:
            call = dict(inp)
            if hoist:  # the hoisted pair bias (pair_bias_all): the block takes the fp32 logits, P is made per call
                call.update(pair=None, bias=pair_bias_all([kern], inp["pair"])[0])
            assert kern.unsupported(**call) is None, kern.unsupported(**call)
            rms, mx = _errs(kern(**call), gold)
            try:
                rms_ref, mx_ref = _errs(copy.deepcopy(m).set_backend("pytorch")(**inp), gold)
            except torch.OutOfMemoryError:
                rms_ref = mx_ref = 0.0
            finally:
                torch.cuda.empty_cache()
    assert rms <= max(1.5 * rms_ref, 4e-3) and mx <= max(1.5 * mx_ref, 3e-2), (
        f"cuda rms {rms:.3e} max {mx:.3e}; pytorch rms {rms_ref:.3e} max {mx_ref:.3e}"
    )


@requires_cuda_card
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("length", "pad"), [(1152, 0), (1536, 100), (2048, 0), (3072, 200)])
def test_pair_inference_past_1024(dtype: str, length: int, pad: int) -> None:
    _check(dtype, length, pad, hoist=False)


@pytest.mark.skipif(_CARD != (9, 0), reason="H100 (the hoisted-bias path is the Hopper block's)")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("length", "pad"), [(1152, 0), (2048, 100), (3072, 0)])
def test_hoisted_bias_inference_past_1024(dtype: str, length: int, pad: int) -> None:
    _check(dtype, length, pad, hoist=True)


@pytest.mark.skipif(_CARD != (9, 0), reason="H100")
@pytest.mark.parametrize("length", [1024, 1152, 2048, 3072])
def test_softmax_hoist_every_length(length: int) -> None:
    """``softmax_hoist`` = the fp32 softmax of fp32 logits (huge, key spread ~500), P rounded to bf16, at the lengths either side of softmax_t's 1024."""
    from miniworld_engine.kernels.bias_only_dit import cuda as K

    g = torch.Generator(device="cuda").manual_seed(3)
    lg = 4000.0 + 120.0 * torch.randn(length, length, device="cuda", generator=g)  # [R, L] with R a multiple of L (softmax_t's [H L, L])
    mask = torch.rand(length, device="cuda", generator=g) > 0.2
    ref = torch.softmax(lg.double().masked_fill(~mask[None], float("-inf")), dim=-1)
    p = torch.empty(length, length, device="cuda", dtype=torch.bfloat16)
    K.softmax_hoist(lg, p, mask)
    assert ((p.double() - ref).abs() <= 2.0**-8 * ref + 1e-9).all()


@pytest.mark.skipif(_CARD not in ((8, 0), (12, 0)), reason="A100 / RTX 5090 (the H100 past 3072 keys: test_bias_only_long_inference_sm90)")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("length", "pad"), [(3200, 0), (4096, 200)])
def test_pair_inference_past_3072_sm80(dtype: str, length: int, pad: int) -> None:
    """AF3 n >= 25 (3200+ tokens): 3.0.5's row softmax held 3072 keys (``softmax_rows: L ... at most 3072``); the two-pass rows take them."""
    _check(dtype, length, pad, hoist=False)
