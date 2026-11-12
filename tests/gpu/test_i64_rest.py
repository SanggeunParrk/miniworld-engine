"""64-bit address arithmetic, the remaining Triton kernels (I64 audit, part 2): the LayerNorm statistics kernel and the gated output
projection read a channel-major operand at ``channel * M`` (WIDE constexpr past 2**31 elements); the bias-only pair-bias LayerNorm
writes / reads its head-major logits at ``head * R`` (cast to int64). Each test runs past the bound and checks the high offsets against a
float32 reference on those rows only.
"""

import pytest
import torch

from miniworld_engine.kernels._wide import needs_wide

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
EPS = 1e-5


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


requires_card = pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")


def _close(out: torch.Tensor, ref: torch.Tensor, what: str, rel: float = 2e-2) -> None:
    err = float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
    assert err <= rel, f"{what}: max error {err:.3e} of the reference's max"


def _rows(m: int) -> torch.Tensor:
    return torch.cat([torch.arange(64), torch.arange(m - 2048, m)]).cuda()


@requires_card
def test_layernorm_stats_mmajor_past_2_31() -> None:
    """``stats_triton`` on an m-major (M, 512) view of 2.18e9 elements: (K - 1) * M passes 2**31."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    k, m = 512, 2**22 + 2**16
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(k, m, device="cuda", dtype=torch.bfloat16, generator=gen).t()
    assert needs_wide(x) and (k - 1) * m >= 2**31
    rstd, c1 = stats_triton(x, EPS)
    rows = _rows(m)
    xr = x[rows].float()
    mean = xr.mean(-1)
    ref_rstd = torch.rsqrt(xr.var(-1, unbiased=False) + EPS)
    _close(rstd[rows], ref_rstd, "rstd", 1e-3)
    _close(c1[rows], mean * ref_rstd, "c1", 1e-2)


@requires_card
def test_gate_out_channel_major_past_2_31() -> None:
    """The gated output projection (bias-only attention ``_fwd`` and the SWA wrapper on the same body) reading channel-major gate /
    out_r operands of (M, 128) with 127 * M >= 2**31."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd
    from miniworld_engine.kernels.gated_projection.triton.swa import _forward as swa_forward

    dh, n = 128, 64
    m = 2**31 // (dh - 1) + 2**16
    gen = torch.Generator(device="cuda").manual_seed(1)
    gate = torch.randn(dh, m, device="cuda", dtype=torch.bfloat16, generator=gen).t()
    outr = torch.randn(dh, m, device="cuda", dtype=torch.bfloat16, generator=gen).t()
    wo = (0.1 * torch.randn(n, dh, device="cuda", generator=gen)).to(torch.bfloat16)
    assert needs_wide(gate) and gate.stride() == (1, m)
    rows = _rows(m)
    ref = (torch.sigmoid(gate[rows].float()) * outr[rows].float()) @ wo.float().t()
    with torch.no_grad():
        out = _fwd(gate, outr, wo)
        _close(out[rows], ref, "gate_out")
        del out
        out = swa_forward(gate, outr, wo, 0)
        _close(out[rows], ref, "swa gate_out")


@requires_card
def test_bias_only_pair_bias_head_major_past_2_31() -> None:
    """``pair_bias_logits`` with 64 heads over R = 34.6M pair rows (C 16): the head-major logits [64, R] have 2.2e9 elements, so
    ``head * R`` passes 2**31 in the forward store and the backward load."""
    from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import _bwd, _fwd

    c, n = 16, 64
    r = 2**25 + 2**20
    assert n * r >= 2**31
    gen = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(r, c, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = (0.2 * torch.randn(n, c, device="cuda", generator=gen)).to(torch.bfloat16)
    out, mean, rstd = _fwd(x, w, EPS)
    rows = _rows(r)
    xr = x[rows].float().requires_grad_(True)
    xh = (xr - xr.mean(-1, keepdim=True)) * torch.rsqrt(xr.var(-1, unbiased=False, keepdim=True) + EPS)
    ref = xh @ w.float().t()  # [rows, N]
    _close(out[:, rows].t(), ref.detach(), "pair-bias logits")
    del out
    dl = torch.randn(n, r, device="cuda", generator=gen).to(torch.bfloat16)
    dx, _ = _bwd(x, w, dl, mean, rstd)
    ref.backward(dl[:, rows].t().float())
    assert xr.grad is not None
    _close(dx[rows], xr.grad, "pair-bias dx", 3e-2)
