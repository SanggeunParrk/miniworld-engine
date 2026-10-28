"""Hand-CUDA bidirectional TriMul on H100 for fp32 activations on TF32 tensor cores (D = 128, hidden = 128).

``y = x + ds * sigmoid(LN_in(x) Wg^T) * (LN_out(contractions) Wo^T)`` with the front (LN_in, the gated left / right projections) and the back
(LN_out, output projection, gate, residual) as wgmma-TF32 kernels around two cuBLAS TF32 batched contractions (the planes are
``[channel][token]``: the contraction needs no copy).  All sources in ``h100_f32_sources/``; ``tmf32.cuh`` (forward) and ``tmf32_bwd.cuh``
(backward) describe the kernel structure.  Every matmul operand is rounded to nearest TF32 and accumulated in fp32, as cuBLAS does under
``allow_tf32``; the whole path is deterministic (no atomics; the per-block weight-gradient partials are summed in a fixed order).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent / "h100_f32_sources"
_inc = Path(__file__).parents[2] / "transition" / "cuda" / "anthropic_v5"  # tmn_ptx.cuh (the one copy, shared with the bf16 TriMul)
_REVISION = "v1"
D = 128
CH = 256          # channels per side (2 x hidden)


@functools.lru_cache(maxsize=2)
def _ext(bwd: bool = False):
    ensure_cuda_home()
    return load_extension(
        name=f"trimul_f32_h100_{_REVISION}" + ("_bwd" if bwd else ""),
        sources=[str(_dir / "bind.cu")],
        extra_cuda_cflags=[
            *host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{_inc}", f"-I{_dir}", "--expt-relaxed-constexpr",
            *(["-DTMF_BWD"] if bwd else []),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _rm(w: torch.Tensor) -> torch.Tensor:
    """Row-major view of a weight (the D128 module stores its front matrices column-major)."""
    return w if w.is_contiguous() else w.contiguous()


def _contract(planes: torch.Tensor, L: int) -> torch.Tensor:
    """Outgoing (channels 0..127) and incoming (128..255) contractions of the left / right planes [512, L*L] -> tri [256, L*L]."""
    left = planes[:CH].view(CH, L, L)
    right = planes[CH:].view(CH, L, L)
    tri = torch.empty(CH, L, L, device=planes.device, dtype=planes.dtype)
    torch.bmm(left[:D], right[:D].transpose(1, 2), out=tri[:D])
    torch.bmm(left[D:].transpose(1, 2), right[D:], out=tri[D:])
    return tri.view(CH, L * L)


def _contract_bwd(dtri: torch.Tensor, planes: torch.Tensor, L: int) -> torch.Tensor:
    """Gradient of the planes [512, L*L] given d_tri [256, L*L] (the four batched products of the two contractions)."""
    left = planes[:CH].view(CH, L, L)
    right = planes[CH:].view(CH, L, L)
    g = dtri.view(CH, L, L)
    dpl = torch.empty(2 * CH, L, L, device=planes.device, dtype=planes.dtype)
    dl, dr = dpl[:CH], dpl[CH:]
    torch.bmm(g[:D], right[:D], out=dl[:D])
    torch.bmm(g[:D].transpose(1, 2), left[:D], out=dr[:D])
    torch.bmm(right[D:], g[D:].transpose(1, 2), out=dl[D:])
    torch.bmm(left[D:], g[D:], out=dr[D:])
    return dpl.view(2 * CH, L * L)


# ------------------------------------------------------------------------------------------------------------------------------ inference
def _infer_fake(x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask, ds):
    """y: the input's shape and dtype."""
    return torch.empty_like(x)


@opaque(fake=_infer_fake, name="trimul_f32_h100_infer")
def infer(x: torch.Tensor, w_lg: torch.Tensor, w_l: torch.Tensor, w_rg: torch.Tensor, w_r: torch.Tensor, w_o: torch.Tensor,
          w_g: torch.Tensor, ln_in_w: torch.Tensor, ln_in_b: torch.Tensor, ln_out_w: torch.Tensor, ln_out_b: torch.Tensor,
          eps_in: float, eps_out: float, token_mask: torch.Tensor | None, ds: torch.Tensor | None) -> torch.Tensor:
    """x [1, L, L, 128] fp32 -> y = x + ds * trimul(x) [1, L, L, 128]; token_mask [L] bool or None; ds [L, 128] row-drop scale or None."""
    e = _ext(False)
    L = x.shape[1]
    xf = x.reshape(L * L, D)
    wpk, wo, wg = e.pack_weights(_rm(w_lg), _rm(w_l), _rm(w_rg), _rm(w_r), _rm(w_o), _rm(w_g))
    xn, _, _ = e.ln_in(xf, ln_in_w, ln_in_b, eps_in, False)
    planes, _ = e.front(xn, wpk, token_mask, L, False)
    tri = _contract(planes, L)
    del planes
    on, _, _ = e.ln_out_t(tri, ln_out_w, ln_out_b, eps_out, False)
    del tri
    y, _, _ = e.back(on, xn, wo, wg, xf, ds, L, False)
    return y.view(1, L, L, D)


# ------------------------------------------------------------------------------------------------------------------------------ training
def _train_fwd_fake(x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask, ds):
    """The eleven tensors of the training forward: y [1, L, L, 128], x_n, two LN_in statistics, planes [512, T], tri [256, T], on [T, 256], two LN_out statistics, gate logit and projection [T, 128] (T = L * L)."""
    L = x.shape[1]
    T = L * L
    f = lambda *s: x.new_empty(s)  # noqa: E731
    return (f(1, L, L, D), f(T, D), f(T), f(T), f(2 * CH, T), f(CH, T), f(T, CH), f(T), f(T), f(T, D), f(T, D))


@opaque(fake=_train_fwd_fake, name="trimul_f32_h100_train_fwd")
def train_fwd(x: torch.Tensor, w_lg: torch.Tensor, w_l: torch.Tensor, w_rg: torch.Tensor, w_r: torch.Tensor, w_o: torch.Tensor,
              w_g: torch.Tensor, ln_in_w: torch.Tensor, ln_in_b: torch.Tensor, ln_out_w: torch.Tensor, ln_out_b: torch.Tensor,
              eps_in: float, eps_out: float, token_mask: torch.Tensor | None, ds: torch.Tensor | None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                         torch.Tensor, torch.Tensor, torch.Tensor]:
    """(y, xn, mean_in, rstd_in, planes, tri, on, mean_out, rstd_out, gate logit, projection): everything the backward reads."""
    e = _ext(True)
    L = x.shape[1]
    xf = x.reshape(L * L, D)
    wpk, wo, wg = e.pack_weights(_rm(w_lg), _rm(w_l), _rm(w_rg), _rm(w_r), _rm(w_o), _rm(w_g))
    xn, mean_in, rstd_in = e.ln_in(xf, ln_in_w, ln_in_b, eps_in, True)
    planes, _ = e.front(xn, wpk, token_mask, L, False)
    tri = _contract(planes, L)
    on, mean_out, rstd_out = e.ln_out_t(tri, ln_out_w, ln_out_b, eps_out, True)
    y, gs, ps = e.back(on, xn, wo, wg, xf, ds, L, True)
    return y.view(1, L, L, D), xn, mean_in, rstd_in, planes, tri, on, mean_out, rstd_out, gs, ps


def _train_bwd_fake(dy, x, xn, mean_in, rstd_in, planes, tri, on, mean_out, rstd_out, gs, ps, w_lg, w_l, w_rg, w_r, w_o, w_g,
                    ln_in_w, ln_out_w, ln_out_b, token_mask, ds):
    """dx [1, L, L, 128], the four front weight gradients stacked [1024, 128], dW_o [128, 256], dW_g [128, 128], the LN_in and LN_out gradients."""
    L = x.shape[1]
    f = lambda *s: x.new_empty(s)  # noqa: E731
    return (f(1, L, L, D), f(4 * CH, D), f(D, CH), f(D, D), f(D), f(D), f(CH), f(CH))


@opaque(fake=_train_bwd_fake, name="trimul_f32_h100_train_bwd")
def train_bwd(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, mean_in: torch.Tensor, rstd_in: torch.Tensor, planes: torch.Tensor,
              tri: torch.Tensor, on: torch.Tensor, mean_out: torch.Tensor, rstd_out: torch.Tensor, gs: torch.Tensor,
              ps: torch.Tensor, w_lg: torch.Tensor, w_l: torch.Tensor, w_rg: torch.Tensor, w_r: torch.Tensor, w_o: torch.Tensor,
              w_g: torch.Tensor, ln_in_w: torch.Tensor, ln_out_w: torch.Tensor, ln_out_b: torch.Tensor,
              token_mask: torch.Tensor | None, ds: torch.Tensor | None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dW [1024, 128] = [dW_lg; dW_l; dW_rg; dW_r], dW_o, dW_g, dgamma_in, dbeta_in, dgamma_out, dbeta_out)."""
    e = _ext(True)
    L = x.shape[1]
    T = L * L
    xf = x.reshape(T, D)
    dyf = dy.reshape(T, D)
    if not dyf.is_contiguous():
        dyf = dyf.contiguous()
    woT, uv = e.wo_prep(_rm(w_o), ln_out_w, ln_out_b)
    dproj, dgl, s12 = e.gate_bwd(dyf, gs, ps, ds, uv, L)
    d_wg = torch.mm(dgl.t(), xn)                                  # [128, 128]
    d_wo = torch.mm(dproj.t(), on)                                # [128, 256]
    dtri, pdg, pdb = e.lnout_bwd(dproj, woT, tri, mean_out, rstd_out, s12, ln_out_w)
    del dproj, s12
    dpl = _contract_bwd(dtri, planes, L)
    del dtri
    wpk, _, _ = e.pack_weights(_rm(w_lg), _rm(w_l), _rm(w_rg), _rm(w_r), _rm(w_o), _rm(w_g))
    dpre = e.front_bwd(xn, wpk, token_mask, dpl, L).view(4 * CH, T)
    del dpl
    d_w = torch.mm(dpre, xn)                                      # [1024, 128] = [dW_lg; dW_l; dW_rg; dW_r]
    wcat = torch.cat([_rm(w_lg), _rm(w_l), _rm(w_rg), _rm(w_r)], 0)
    dxn = torch.mm(dpre.t(), wcat)
    del dpre
    dxn.addmm_(dgl, _rm(w_g))
    dx, pdgi, pdbi = e.ln_in_bwd(dxn, xf, mean_in, rstd_in, dyf, ln_in_w)
    return (dx.view(1, L, L, D), d_w, d_wo, d_wg, pdgi.sum(0), pdbi.sum(0), pdg.sum(0), pdb.sum(0))


class TriMulF32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask, ds):
        out = train_fwd(x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask, ds)
        y, saved = out[0], out[1:]
        ctx.save_for_backward(x, *saved, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_out_w, ln_out_b)
        ctx.token_mask, ctx.ds = token_mask, ds
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, xn, mean_in, rstd_in, planes, tri, on, mean_out, rstd_out, gs, ps,
         w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_out_w, ln_out_b) = ctx.saved_tensors
        g = train_bwd(dy, x, xn, mean_in, rstd_in, planes, tri, on, mean_out, rstd_out, gs, ps, w_lg, w_l, w_rg, w_r, w_o, w_g,
                      ln_in_w, ln_out_w, ln_out_b, ctx.token_mask, ctx.ds)
        dx, d_w, d_o, d_g, d_inw, d_inb, d_outw, d_outb = g
        return dx, d_w[:CH], d_w[CH:2 * CH], d_w[2 * CH:3 * CH], d_w[3 * CH:], d_o, d_g, d_inw, d_inb, d_outw, d_outb, None, None, None, None


def trimul(x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask=None, ds=None):
    """Bidirectional TriMul residual update for one sample: inference without autograd, the autograd function otherwise."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b)):
        return TriMulF32.apply(x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask, ds)
    return infer(x, w_lg, w_l, w_rg, w_r, w_o, w_g, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps_in, eps_out, token_mask, ds)
