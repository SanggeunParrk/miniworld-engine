"""64-bit addressing of the OuterProductMean and MSAPairWeightedAveraging Triton kernels past 2**31 elements (I64 audit, 2026-11-12).

Both families already form every element offset in int64 (program ids cast at the origin, ``.to(tl.int64)`` before each product with
L / L CH / S C); these tests run them at sizes where the big operand passes 2**31 elements and check the outputs / gradients against a
float32 reference, so the claim is measured, not read off the source:

- OPM: the outer product ``O`` [L CH, L CH] (ungrouped) / [G, L K, L K] (grouped, Chai-1) and its gradient ``dO`` are read / written by
  the epilogue kernels at offsets up to (L CH)**2; the grouped LayerNorm'd products ``y`` [L L, P] too.
- PWA: the pair ``z`` [L, L, d_pair] is read (forward softmax) and its gradient written (backward) at offsets up to L**2 d_pair.

The forward is checked on the rows at the highest offsets; the backward against a reference computed in row chunks (the full fp32 one
would not fit next to the kernel's tensors) or, for PWA, exactly on the sliced rows (the pair row i only reaches output row i).
"""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

EPS = 1e-5


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


requires_card = pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")


def _close(out: torch.Tensor, ref: torch.Tensor, what: str, rel: float = 3e-2) -> None:
    err = float((out.detach().float() - ref.detach()).abs().max() / ref.detach().abs().max().clamp_min(1e-6))
    assert err <= rel, f"{what}: max error {err:.3e} of the reference's max"


def _ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, eps: float) -> torch.Tensor:
    x = x.float()
    xc = x - x.mean(-1, keepdim=True)
    y = xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps) * w
    return y if b is None else y + b


class _Rand:
    """Seeded tensors on the bf16 grid (the kernels round weights to bf16; the reference then sees the same values in fp32)."""

    def __init__(self, seed: int) -> None:
        self.g = torch.Generator(device="cuda").manual_seed(seed)

    def __call__(self, *shape: int, scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
        return (shift + scale * torch.randn(*shape, device="cuda", generator=self.g)).to(torch.bfloat16).float()


def _opm_params(rnd: _Rand, cm: int, ch: int, p: int, cz: int) -> list[torch.Tensor]:
    return [
        rnd(cm, scale=0.1, shift=1.0),  # ln weight
        rnd(cm, scale=0.1),  # ln bias
        rnd(ch, cm, scale=cm**-0.5),  # W_left
        rnd(ch, cm, scale=cm**-0.5),  # W_right
        rnd(cz, p, scale=p**-0.5),  # W_out
        rnd(cz, scale=0.1),  # b_out
    ]


@requires_card
def test_opm_triton_past_2_31() -> None:
    """Ungrouped OPM (normalize_before_proj, no mask) at L 1472, d_hidden 32: O / dO are [47104, 47104] = 2.2e9 elements."""
    from miniworld_engine.kernels.outer_product_mean.triton.main import triton_outer_product_mean

    S, L, CM, CH, CZ = 4, 1472, 64, 32, 128
    assert (L * CH) ** 2 > 2**31
    rnd = _Rand(0)
    msa = rnd(1, S, L, CM).to(torch.bfloat16)
    params = _opm_params(rnd, CM, CH, CH * CH, CZ)
    lnw, lnb, wl, wr, wo, bo = params
    x = msa.clone().requires_grad_(True)
    leaves = [p.clone().requires_grad_(True) for p in params]
    out = triton_outer_product_mean(x, None, *leaves, eps=EPS, normalize_before_proj=True)
    dz = rnd(1, L, L, CZ).to(torch.bfloat16)
    out.backward(dz)
    assert x.grad is not None and leaves[4].grad is not None

    xf = msa.float().requires_grad_(True)
    y = _ln(xf[0], lnw, lnb, EPS)
    a, b = y @ wl.t(), y @ wr.t()  # [S, L, CH]
    ad, bd = a.detach(), b.detach()
    n = float(S)  # no mask: every pair count is S
    for i in (0, L - 2, L - 1):  # the rows at the highest O offsets
        oi = torch.einsum("sd,sje->jde", ad[:, i], bd).reshape(L, CH * CH) / n
        _close(out[0, i], oi @ wo.t() + bo, f"out row {i}")
    dzn = dz[0].float() / n
    da, db, dwo = torch.zeros_like(ad), torch.zeros_like(bd), torch.zeros_like(wo)
    for i0 in range(0, L, 64):
        ic = slice(i0, min(i0 + 64, L))
        d_o = (dzn[ic] @ wo).view(-1, L, CH, CH)  # dO for rows ic: [ci, L, d, e]
        da[:, ic] = torch.einsum("sje,ijde->sid", bd, d_o)
        db += torch.einsum("sid,ijde->sje", ad[:, ic], d_o)
        oc = torch.einsum("sid,sje->ijde", ad[:, ic], bd).reshape(-1, CH * CH)
        dwo += dzn[ic].reshape(-1, CZ).t() @ oc
    torch.autograd.backward([a, b], [da, db])
    assert xf.grad is not None
    _close(x.grad[0], xf.grad[0], "dmsa", 5e-2)
    _close(leaves[4].grad, dwo, "dW_out")


@requires_card
def test_opm_triton_grouped_past_2_31() -> None:
    """Grouped OPM (Chai-1: groups 8, d_hidden 64, products LayerNorm'd) at L 2056: O / dO are [8, 16448, 16448] = 2.16e9 elements and
    the products y [L L, 512] 2.16e9."""
    from miniworld_engine.kernels.outer_product_mean.triton.main import triton_outer_product_mean

    S, L, CM, CH, G, CZ = 4, 2056, 64, 64, 8, 256
    K = CH // G
    P = G * K * K
    eps_p = 0.1
    assert G * (L * K) ** 2 > 2**31 and L * L * P > 2**31
    rnd = _Rand(1)
    msa = rnd(1, S, L, CM).to(torch.bfloat16)
    params = _opm_params(rnd, CM, CH, P, CZ)
    lnw, lnb, wl, wr, wo, bo = params
    gam, bet = rnd(P, scale=0.1, shift=1.0), rnd(P, scale=0.1)
    x = msa.clone().requires_grad_(True)
    leaves = [p.clone().requires_grad_(True) for p in params]
    out = triton_outer_product_mean(
        x, None, *leaves, eps=EPS, groups=G, product_norm_weight=gam, product_norm_bias=bet, product_norm_eps=eps_p
    )
    dz = rnd(1, L, L, CZ).to(torch.bfloat16)
    out.backward(dz)
    assert x.grad is not None and leaves[4].grad is not None

    xf = msa.float().requires_grad_(True)
    y = _ln(xf[0], lnw, lnb, EPS)
    a, b = y @ wl.t(), y @ wr.t()  # [S, L, CH], channel = g K + k
    ad, bd = a.detach().view(S, L, G, K), b.detach().view(S, L, G, K)

    def products(ai: torch.Tensor) -> torch.Tensor:  # ai [S, ci, G, K] -> LayerNorm'd products [ci, L, P]
        p = torch.einsum("sigk,sjgl->ijgkl", ai, bl.view(S, L, G, K)).reshape(ai.shape[1], L, P)
        return _ln(p, gam, bet, eps_p)

    bl = bd.reshape(S, L, CH)
    for i in (0, L - 2, L - 1):
        _close(out[0, i], products(ad[:, i : i + 1])[0] @ wo.t() + bo, f"out row {i}")
    da, db, dwo = torch.zeros(S, L, CH, device="cuda"), torch.zeros(S, L, CH, device="cuda"), torch.zeros_like(wo)
    for i0 in range(0, L, 32):
        ic = slice(i0, min(i0 + 32, L))
        ai = ad[:, ic].clone().requires_grad_(True)
        bl = bd.reshape(S, L, CH).clone().requires_grad_(True)
        yc = products(ai)
        dzc = dz[0, ic].float()
        gai, gbl = torch.autograd.grad(yc, [ai, bl], dzc @ wo)
        da[:, ic] = gai.reshape(S, -1, CH)
        db += gbl
        dwo += dzc.reshape(-1, CZ).t() @ yc.detach().reshape(-1, P)
    torch.autograd.backward([a, b], [da, db])
    assert xf.grad is not None
    _close(x.grad[0], xf.grad[0], "dmsa", 5e-2)
    _close(leaves[4].grad, dwo, "dW_out")


@requires_card
def test_pwa_triton_past_2_31() -> None:
    """PWA (row 48's widths: d_msa 64, d_pair 128, 8 heads x 8) at L 4160: the pair is [4160, 4160, 128] = 2.2e9 elements. The output rows
    and the pair-gradient rows at the highest offsets against a float32 autograd reference on those rows (pair row i reaches only row i)."""
    from miniworld_engine.kernels.pair_weighted_averaging.triton.main import triton_pair_weighted_averaging

    S, L, D, DZ, NH, C = 4, 4160, 64, 128, 8, 8
    HC = NH * C
    assert L * L * DZ > 2**31
    rnd = _Rand(2)
    msa = rnd(1, S, L, D).to(torch.bfloat16)
    pair = rnd(1, L, L, DZ).to(torch.bfloat16).requires_grad_(True)
    lnm_w, lnm_b = rnd(D, scale=0.1, shift=1.0), rnd(D, scale=0.1)
    wv, wg = rnd(HC, D, scale=D**-0.5), rnd(HC, D, scale=D**-0.5)
    lnz_w, wb = rnd(DZ, scale=0.1, shift=1.0), rnd(NH, DZ, scale=DZ**-0.5)
    wo = rnd(D, HC, scale=HC**-0.5)
    out = triton_pair_weighted_averaging(msa, pair, None, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo, eps_msa=EPS, eps_pair=EPS)
    dres = rnd(1, S, L, D).to(torch.bfloat16)
    out.backward(dres)
    assert pair.grad is not None

    y = _ln(msa[0], lnm_w, lnm_b, EPS)  # [S, L, D]
    v = (y @ wv.t()).view(S, L, NH, C)
    for i in (0, L - 2, L - 1):
        zi = pair[0, i].detach().float().requires_grad_(True)
        wi = torch.softmax(_ln(zi, lnz_w, None, EPS) @ wb.t(), dim=0)  # [L, NH], over the keys j
        oi = torch.einsum("jh,sjhc->shc", wi, v)
        gi = torch.sigmoid((y[:, i] @ wg.t()).view(S, NH, C))
        out_i = msa[0, :, i].float() + (gi * oi).reshape(S, HC) @ wo.t()
        _close(out[0, :, i], out_i, f"out row {i}")
        (out_i * dres[0, :, i].float()).sum().backward()
        assert zi.grad is not None
        _close(pair.grad[0, i], zi.grad, f"dpair row {i}", 5e-2)
