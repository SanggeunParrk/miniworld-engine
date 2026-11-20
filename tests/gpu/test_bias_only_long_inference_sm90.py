"""H100 bias-only token DiT inference past L 3072 (FoldForge AF3 n >= 25: 3200+ tokens): the softmax rows keep a row of up to 3072 keys in registers and
run two passes over longer ones (``bias_only_dit_rows.cu`` softmax_rows_long_kernel, ``bias_only_dit_f32_rows.cu`` softmax_rows_long_k), the PV cores
(pvx / pvx32) take L up to 8192, and a call past 3072 must fit the device memory (a refusal, never an out-of-memory).  The cuda backend must serve these
lengths (no CUDA hole), match a row-chunked float64 reference as closely as the ``pytorch`` backend does (the engine accuracy standard, not
bit-identity), and keep the pair-bias logits and the softmax fp32 (user rule 2026-11-19: bf16 logits break the peptide bonds)."""

# torch's Tensor.pow / mean overloads are not resolved by the checker in the error-measure helpers:
# pyright: reportCallIssue=false
from __future__ import annotations

import copy
import gc

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly

pytestmark = [pytest.mark.gpu]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


requires_h100 = pytest.mark.skipif(_card() != (9, 0), reason="H100 (sm_90)")


@pytest.fixture(autouse=True)
def _free_cuda_between_cases():
    """The long cases hold tens of GiB (pair, float64 gold, two module copies): free it before and after each case so one process runs them all."""
    gc.collect()
    torch.cuda.empty_cache()
    yield
    from miniworld_engine.kernels.integrations import bias_only_dit

    bias_only_dit._RUNNERS.clear()  # a runner keeps the last hoisted pair-bias P and its (pair, mask) key tensors alive: tens of GiB at these lengths
    gc.collect()
    torch.cuda.empty_cache()


S = 2
DTYPES = ["master", "float32"]


def _block(dtype: str, d_pair: int = 128, d_cond: int = 384):
    m = TokenDiTMiniWorldBiasOnly(d_single=768, d_cond=d_cond, d_pair=d_pair, n_head=16, n=2).cuda().float()
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


def _ctx(dtype: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master")


@torch.no_grad()
def gold_f64(m, inp: dict, rows: int = 64) -> torch.Tensor:
    """The block's math in float64 with the pair side (LayerNorm, bias projection, softmax) taken ``rows`` query rows at a time."""
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
        lg = F.linear(ln, at.to_bias.weight).masked_fill(~mask[0][None, :, None], float("-inf"))  # [r, L, h]
        out[:, i0 : i0 + rows] = torch.einsum("ijh,ajhd->aihd", torch.softmax(lg, dim=1), value)
    o = at.to_out(torch.sigmoid(gate) * out.flatten(-2).view(a, 1, length, -1))
    return tr(res + torch.sigmoid(at.to_scale(c)) * o, c)


def _errs(out: torch.Tensor, gold: torch.Tensor) -> tuple[float, float]:
    d = out.double() - gold
    return float(d.pow(2).mean().sqrt() / gold.pow(2).mean().sqrt()), float(d.abs().max() / gold.abs().max())


def _check(dtype: str, length: int, pad: int, d_pair: int = 128, d_cond: int = 384) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    m = _block(dtype, d_pair, d_cond)
    inp = _inputs(m, length, dtype, pad)
    kern = copy.deepcopy(m).set_backend("cuda")
    with torch.no_grad():
        assert kern.unsupported(**inp) is None, kern.unsupported(**inp)
        gold = gold_f64(m, inp)
        with _ctx(dtype):
            y = kern(**inp)
            rms, mx = _errs(y, gold)
            del y
            try:  # the pytorch backend materialises fp32 logits and a fp32 LayerNorm of the pair: not always next to the pair at the long lengths
                ref = copy.deepcopy(m).set_backend("pytorch")
                rms_ref, mx_ref = _errs(ref(**inp), gold)
            except torch.OutOfMemoryError:
                rms_ref = mx_ref = 0.0
            finally:
                torch.cuda.empty_cache()
    assert rms <= max(1.5 * rms_ref, 4e-3) and mx <= max(1.5 * mx_ref, 3e-2), (
        f"cuda rms {rms:.3e} max {mx:.3e}; pytorch rms {rms_ref:.3e} max {mx_ref:.3e}"
    )


@requires_h100
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("length", "pad"), [(3200, 0), (3328, 200), (3584, 0)])
def test_inference_past_3072(dtype: str, length: int, pad: int) -> None:
    _check(dtype, length, pad)


@requires_h100
@pytest.mark.parametrize("dtype", DTYPES)
def test_inference_wide_pair_and_cond_3328(dtype: str) -> None:
    _check(dtype, 3328, 128, d_pair=256, d_cond=768)


@pytest.mark.slow
@requires_h100
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("length", "pad"), [(4096, 0), (4096, 512), (6144, 256)])
def test_inference_4096_and_6144(dtype: str, length: int, pad: int) -> None:
    _check(dtype, length, pad)


@pytest.mark.slow
@requires_h100
def test_inference_8192_or_clean_refusal() -> None:
    """The longest length: served where the device memory holds it, else a refusal naming the memory (never an out-of-memory)."""
    m = _block("master")
    length = 8192
    act = torch.bfloat16
    z = torch.empty(1, length, length, 128, device="cuda", dtype=act)
    inp = {
        "single": torch.randn(S, 1, length, 768, device="cuda", dtype=act),
        "cond": torch.randn(S, 1, length, 384, device="cuda", dtype=act),
        "pair": z.normal_(),
        "mask": torch.ones(1, length, dtype=torch.bool, device="cuda"),
    }
    kern = m.set_backend("cuda")
    with torch.no_grad(), _ctx("master"):
        why = kern.unsupported(**inp)
        assert why is None, why
        y = kern(**inp)
    assert torch.isfinite(y).all()


@requires_h100
def test_refuses_past_the_bound_cleanly() -> None:
    m = _block("master").set_backend("cuda")
    length = 8320  # a multiple of 128 past 8192
    inp = {
        "single": torch.empty(1, 1, length, 768, device="cuda", dtype=torch.bfloat16),
        "cond": torch.empty(1, 1, length, 384, device="cuda", dtype=torch.bfloat16),
        "pair": torch.empty(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16).expand(1, length, length, 128),
        "mask": torch.ones(1, length, dtype=torch.bool, device="cuda"),
    }
    with torch.no_grad(), _ctx("master"):
        why = m.unsupported(**inp)
    assert why is not None and "8192" in why


@requires_h100
def test_refuses_what_cannot_fit_in_memory() -> None:
    """A pair that cannot sit beside P and the step (48 samples) in the device memory is refused with the memory as the reason (not an out-of-memory)."""
    m = _block("float32", d_pair=256).set_backend("cuda")
    length, samples = 8192, 48
    inp = {
        "single": torch.empty(samples, 1, length, 768, device="cuda", dtype=torch.float32),
        "cond": torch.empty(samples, 1, length, 384, device="cuda", dtype=torch.float32),
        "pair": torch.empty(1, 1, 1, 256, device="cuda", dtype=torch.float32).expand(
            1, length, length, 256
        ),  # 64 GiB logical + the 48 samples' step buffers
        "mask": torch.ones(1, length, dtype=torch.bool, device="cuda"),
    }
    with torch.no_grad():
        why = m.unsupported(**inp)
    assert why is not None and "GiB" in why


@requires_h100
@pytest.mark.parametrize("length", [3200, 4096, 6144, 8192])
def test_softmax_rows_two_pass(length: int) -> None:
    """The two-pass rows past the register softmax: bf16 P of fp32 logits (huge, key spread ~500) and fp32 P, masked keys, a fully masked row,
    against the float64 softmax (the fp32 logits are NOT rounded: a bf16 logit would move P by O(1))."""
    from miniworld_engine.kernels.bias_only_dit import cuda as K
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32

    g = torch.Generator(device="cuda").manual_seed(3)
    rows = 64
    lg = 4000.0 + 120.0 * torch.randn(rows, length, device="cuda", generator=g)
    mask = torch.rand(length, device="cuda", generator=g) > 0.2
    mask[-37:] = False
    lg[5] = torch.randn(length, device="cuda", generator=g)  # a row of ordinary logits
    ref = torch.softmax(lg.double().masked_fill(~mask[None], float("-inf")), dim=-1)
    p = torch.empty(rows, length, device="cuda", dtype=torch.bfloat16)
    K.softmax_rows(lg, p, mask)
    assert ((p.double() - ref).abs() <= 2.0**-8 * ref + 1e-9).all()
    p32 = torch.empty_like(lg)
    tf32.rows32().softmax_rows_cuda(lg, p32, mask)
    assert (p32.double() - ref).abs().max().item() <= 1e-6 * ref.max().item() * 4 + 1e-9
    assert (p32[:, ~mask] == 0).all() and (p[:, ~mask] == 0).all()
    none = torch.zeros(length, dtype=torch.bool, device="cuda")  # every key masked: a uniform row, as the references' finite fill
    q = torch.empty(rows, length, device="cuda", dtype=torch.bfloat16)
    K.softmax_rows(lg, q, none)
    assert torch.allclose(q.float(), torch.full_like(q.float(), 1.0 / length), rtol=2.0**-7)
