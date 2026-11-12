"""64-bit address arithmetic in the Triton kernels (I64 audit, 2026-11-12): every kernel below formed an int32 product that wraps past
2**31 elements (a channel index times the row count M on an m-major view, ``(i * N + j) * d`` over a pair), or launched B * H * L
programs on grid axis y / z (CUDA caps those at 65535). Each test runs the kernel at a size past the bound and checks the rows /
columns at the high offsets against a float32 reference computed on those rows only (the full reference would not fit next to the
kernel's tensors).
"""

import pytest
import torch

from miniworld_engine.kernels._wide import needs_wide

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# An m-major (M, K) view: the channel index c times M wraps once c * M >= 2**31, so (K - 1) * M must pass it (K 512: c >= 505 wraps).
K = 512
M = 2**22 + 2**16
EPS = 1e-5


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


requires_card = pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")


def _rows() -> torch.Tensor:
    return torch.cat([torch.arange(64), torch.arange(M - 2048, M)]).cuda()


def _close(out: torch.Tensor, ref: torch.Tensor, what: str, rel: float = 2e-2) -> None:
    err = float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
    assert err <= rel, f"{what}: max error {err:.3e} of the reference's max"


def _mmajor(gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(K, M, device="cuda", dtype=torch.bfloat16, generator=gen).t()  # (M, K) strides (1, M)
    assert x.stride() == (1, M) and needs_wide(x) and (K - 1) * M >= 2**31
    return x


def _ln_ref(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor):
    x = x.float()
    mean = x.mean(-1, keepdim=True)
    rstd = torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + EPS)
    return (x - mean) * rstd * g + b, mean.squeeze(-1), rstd.squeeze(-1)


@requires_card
def test_layernorm_linear_mmajor_past_2_31() -> None:
    """te_style ``_ln_materialize`` (fwd), the three m-major LN backward kernels (te_style atomic, mmajor_bwd persistent, layernorm
    persistent) and the recompute kernels, on an m-major (M, 512) view of 2.18e9 elements."""
    from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd, recompute, te_style

    gen = torch.Generator(device="cuda").manual_seed(0)
    x = _mmajor(gen)
    g = 1 + 0.1 * torch.randn(K, device="cuda", generator=gen)
    b = 0.1 * torch.randn(K, device="cuda", generator=gen)
    rows = _rows()
    ref, ref_mean, ref_rstd = _ln_ref(x[rows], g, b)

    xn, mean, rstd = te_style._ln_materialize(x, g, b, EPS)
    _close(xn[rows], ref, "materialize xn")
    _close(mean[rows], ref_mean, "materialize mean", 1e-3)
    _close(rstd[rows], ref_rstd, "materialize rstd", 1e-3)
    del xn
    _close(recompute._recompute_xnormed(x, g, b, mean, rstd)[rows], ref, "recompute xnormed")
    _close(recompute._recompute_xhat(x, mean, rstd)[rows], (ref - b) / g, "recompute xhat")

    dxn = torch.randn(K, M, device="cuda", dtype=torch.bfloat16, generator=gen).t()
    xr = x[rows].float().requires_grad_(True)
    torch.autograd.backward(_ln_ref(xr, g, b)[0], dxn[rows].float())
    assert xr.grad is not None
    for name, fn in (
        ("atomic", mmajor_bwd._ln_bwd_atomic),
        ("persistent", mmajor_bwd._ln_bwd_persistent_new),
        ("canonical", mmajor_bwd._ln_bwd_persistent_canonical),
    ):
        dx, _, _ = fn(dxn, x, g, mean, rstd, list(x.stride()))
        assert dx.stride() == x.stride()
        _close(dx[rows], xr.grad, f"{name} dx")
        del dx


@requires_card
def test_layernorm_transpose_past_2_31() -> None:
    """``_ln_transpose_dbn_bnd``: a channel-major (D, B, N) input normalised over D into (B, N, D)."""
    from miniworld_engine.kernels.layernorm.triton.transpose import _ln_transpose_dbn_bnd

    gen = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(K, 1, M, device="cuda", dtype=torch.bfloat16, generator=gen)
    g = 1 + 0.1 * torch.randn(K, device="cuda", generator=gen)
    b = 0.1 * torch.randn(K, device="cuda", generator=gen)
    rows = _rows()
    y = _ln_transpose_dbn_bnd(x, g, b, EPS)
    _close(y[0, rows], _ln_ref(x[:, 0, rows].t(), g, b)[0], "transpose LN")


@requires_card
def test_pwa_train_pair_past_2_31() -> None:
    """pwa_train ``pair_fwd`` / ``pair_bwd`` at N 4160, d_pair 128 (z: 2.2e9 elements): the last rows' softmax weights and dz."""
    from miniworld_engine.kernels.integrations.pwa_train import pair_bwd, pair_fwd

    n, dz, h = 4160, 128, 8
    gen = torch.Generator(device="cuda").manual_seed(2)
    z = torch.randn(n, n, dz, device="cuda", dtype=torch.bfloat16, generator=gen)
    assert needs_wide(z)
    mask = (torch.rand(n, n, device="cuda", generator=gen) > 0.1).to(torch.bfloat16)
    ln_w = 1 + 0.1 * torch.randn(dz, device="cuda", generator=gen)
    wb = 0.1 * torch.randn(h, dz, device="cuda", generator=gen)
    w = pair_fwd(z, mask, ln_w, EPS, wb, h)
    dw = torch.randn(h, n, n, device="cuda", generator=gen)
    dz_out, _, _ = pair_bwd(z, w, dw, ln_w, EPS, wb)
    for i in (0, n - 2, n - 1):
        zi = z[i].float().requires_grad_(True)
        xc = zi - zi.mean(-1, keepdim=True)
        zn = xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + EPS) * ln_w
        logits = (zn @ wb.t()).masked_fill(mask[i, :, None] < 0.5, float("-inf"))  # [N, H]
        wi = torch.softmax(logits, 0).t()  # [H, N]
        _close(w[:, i], wi.detach(), f"w row {i}")
        (wi * dw[:, i]).sum().backward()
        assert zi.grad is not None
        _close(dz_out[i], zi.grad, f"dz row {i}", 5e-2)


def _attn_ref(q, k, v, bias):
    """softmax(q k^T / sqrt(D) + bias) v per (b, h, row): q/k/v [B, H, L, L, D], bias [B, H, L, L] shared by every row."""
    s = torch.einsum("bhimd,bhind->bhimn", q, k) * q.shape[-1] ** -0.5 + bias[:, :, None]
    return torch.einsum("bhimn,bhind->bhimd", torch.softmax(s, -1), v)


# B * H * L = 66560 rows > 65535 (the old backward grid's axis y / z)
B, H, L, D = 130, 16, 32, 32


def _attn_inputs(seed: int, n: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(B, H, L, L, D, device="cuda", generator=gen).to(torch.bfloat16) for _ in range(n)] + [
        torch.randn(B, H, L, L, device="cuda", generator=gen).to(torch.bfloat16)
    ]


def _check_attn(fn, ins, ref_fn, what):
    assert B * H * L > 65535
    gen = torch.Generator(device="cuda").manual_seed(9)
    dy = torch.randn(B, H, L, L, D, device="cuda", generator=gen).to(torch.bfloat16)
    leaves = [t.clone().requires_grad_(True) for t in ins]
    out = fn(*leaves)
    out.backward(dy)
    gold = [t.float().requires_grad_(True) for t in ins]
    ref = ref_fn(*gold)
    ref.backward(dy.float())
    for s in (slice(0, 2), slice(B - 2, B)):  # the first and the last samples (the highest row ids)
        _close(out[s], ref[s].detach(), f"{what} out")
        for i, (a, r) in enumerate(zip(leaves, gold, strict=True)):
            assert a.grad is not None and r.grad is not None
            _close(a.grad[s], r.grad[s], f"{what} grad {i}", 3e-2)


@requires_card
def test_triangle_attention_backward_grid_past_65535_rows() -> None:
    from miniworld_engine.kernels.triangle_attention.triton.main import triton_triangle_attention_pair_bias

    _check_attn(triton_triangle_attention_pair_bias, _attn_inputs(3, 3), _attn_ref, "triattn")


@requires_card
def test_triangle_attention_atomic_backward_grid_past_65535_rows() -> None:
    from miniworld_engine.kernels.triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction

    _check_attn(TritonTriangleAttentionPairBiasFunction.apply, _attn_inputs(4, 3), _attn_ref, "triattn atomic")


@requires_card
def test_bias_only_attention_backward_grid_past_65535_rows() -> None:
    from miniworld_engine.kernels.bias_only_attention.triton.main import triton_bias_only_attention

    def ref(v, bias):
        return torch.einsum("bhimn,bhind->bhimd", torch.softmax(bias[:, :, None].expand(-1, -1, L, -1, -1), -1), v)

    _check_attn(triton_bias_only_attention, _attn_inputs(5, 1), ref, "bias-only")
