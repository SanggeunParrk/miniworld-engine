"""H100 single-direction TriMul training at D=256/384 (hidden D, one contraction), L384/768.

The bidirectional wide sources (``h100_sources/wide_train``) compiled for hidden H = D:
  forward   weight pack -> ``front_pre_d384.cu`` at (C_Z, C_H) = (D, D): LN_in + gated projections,
            saves x_n and the channel-major gate/projection preactivations -> one cuBLAS contraction
            -> ``forward_norm_d384.cu`` (WIDTH=D/2, i.e. LN_out over H=D; saves norm, mean, rstd)
            -> projection and gate GEMMs -> ``output_dense.cu`` epilogue (gate, dropout, residual);
  backward  ``gate.cu`` (dProj, gate-logit gradient) -> dn GEMM -> ``output_ln_d{D}_l384.cu``
            (WIDTH=D/2) -> ``contract_gp_d384_l384.cu`` with GP_DIR (one direction; dl / dr are
            never materialised, the epilogue writes the gate/projection preactivation gradients)
            -> weight GEMMs -> dX GEMM -> ``input_ln_d{D}_l384.cu`` (input LN + residual,
            INPUT_NO_REDUCE: weight gradients come from the GEMMs).
B=1, BF16 pair and projection weights, FP32 LayerNorm affine, LN eps 1e-5.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_training as W

WIDTHS = (256, 384)
# front K1 tiles: (BI, BJ, ring slots, K chunks per slot, min blocks per SM)
_K1 = {256: (2, 64, 4, 4, 1), 384: (2, 64, 2, 6, 1)}


def supports(width, length) -> bool:
    return width in WIDTHS and length in (384, 768)


# --------------------------------------------------------------------------- forward
def _front(x, w1, ab, mask, gi, bi, xn, pre):
    n, D = x.shape[1], x.shape[-1]
    a, b, slots, sk, minb = _K1[D]
    defines = W._defines(MW_MINB=minb, MW_K1_STREAM=0, TMN_SIGMOID_TANH=1, TMN_WSKIP=1,
                         TMN_MASK_TEMPLATE=1, FRONT_CZ=D, FRONT_CH=D, FRONT_SLOTS=slots, FRONT_SK=sk)
    smem = W._k1_smem(D, _K1[D], a * b // 64 * 4096)
    k = W._kernel("front_pre_d384.cu", "mw_wide_front_pre_overlap", "forward", defines, smem)
    tj = (n + b - 1) // b
    tiles = ((n + a - 1) // a) * tj
    fields = [
        W._map(x, [64, b, a], [D, n, n], [D * 2, n * D * 2]),
        W._map(w1, [64, 64], [D, 4 * D], [D * 2]),
        W._map(ab, [64, 1, 32], [n, n, 2 * D], [n * 2, n * n * 2]),
        mask, gi, bi, ab, None, xn,
        W._map(pre, [64, 64], [pre.shape[1], pre.shape[0]], [pre.shape[1] * 2]),
        n, n, tj, tiles, 1, n, 1, 1e-5, n * D, D, 0, 0,
    ]
    W._launch(k, min(tiles, W._sms() * minb), 128 * (a * b // 64 + 1), smem,
              W._params(f"uni_front{D}", fields))


def _triangle(ab, tri, outgoing):
    D = tri.shape[0]
    left, right = ab[:D], ab[D:]
    if outgoing:
        torch.bmm(left, right.transpose(-1, -2), out=tri)
    else:
        torch.bmm(left.transpose(-1, -2), right, out=tri)


def _output_norm(tri, norm, go, bo, mu, rs):
    H, n = tri.shape[0], tri.shape[1]
    M = n * n
    defines = W._defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=H // 2, OUTPUT_ROWS=32, OUTPUT_THREADS=128)
    smem = 32 * H * 2 + 128 + H * 8
    args = ("forward_norm_d384.cu", "mw_wide_cached_forward_norm", "forward", defines)
    k = W._kernel(*args, smem)
    grid = W._resident_grid(*args, 128, smem)
    maps = [W._map(tri, [32, 64], [M, H], [M * 2], "64B"), W._map(norm, [64, 32], [H, M], [H * 2])]
    W._launch(k, grid, 128, smem, W._params(f"uni_forward_norm{H}", [*maps, go, bo, mu, rs, M]))


def saved_like(x):
    """ab, tri, x_n, norm, proj, gate, mean_out, rstd_out, pre, bf16 pair mask."""
    n, D = x.shape[1], x.shape[-1]
    M = n * n
    f32 = dict(dtype=torch.float32)
    return [x.new_empty((2 * D, n, n)), x.new_empty((D, n, n)), torch.empty_like(x),
            x.new_empty((M, D)), x.new_empty((M, D)), x.new_empty((M, D)),
            x.new_empty((M,), **f32), x.new_empty((M,), **f32), x.new_empty((4 * D, M)),
            x.new_empty((n, n))]


def _run_forward(leaves, mask, ds, outgoing, save):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, D = x.shape[1], x.shape[-1]
    if not supports(D, n):
        raise ValueError(f"single-direction wide TriMul training covers D256/384 at L384/768, not {(D, n)}")
    M = n * n
    saved = saved_like(x)
    ab, tri, xn, norm, proj, gate, mu, rs, pre, pmask = saved
    pmask.copy_(mask.reshape(n, n))
    w1 = x.new_empty((4 * D, D))
    W.pack_into(w1, wl, wlg, wr, wrg)
    _front(x, w1, ab, pmask, gi, bi, xn, pre)
    _triangle(ab, tri, outgoing)
    _output_norm(tri, norm, go, bo, mu, rs)
    torch.mm(norm, wp.t(), out=proj)
    torch.mm(xn.reshape(M, D), wg.t(), out=gate)
    y = torch.empty_like(x)
    defines = W._defines(MW_MINB=1, MW_K1_STREAM=0, TMN_SIGMOID_TANH=1, WIDTH=D, GROUPS=1, KCHUNK=1)
    k = W._kernel("output_dense.cu", "mw_wide_dense_output_epi", "forward", defines)
    W._launch(k, 1056, 256, 0, W._params(f"uni_epilogue{D}", [proj, gate, x, ds.reshape(n, D), y, M * D, n * D]))
    return [y, *saved] if save else y


def _forward_fake(leaves, mask, ds, outgoing):
    """y like x, then the saved set (``saved_like``)."""
    x = leaves[0]
    return [torch.empty_like(x), *saved_like(x)]


@opaque(fake=_forward_fake, name="trimul_h100_uni_wide_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool) -> list[torch.Tensor]:
    """Training forward; returns ``[y, *saved]`` (all freshly allocated)."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, outgoing, True)


def _forward_nograd_fake(leaves, mask, ds, outgoing):
    """y like x."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_uni_wide_dropout_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool) -> torch.Tensor:
    """Training-mode (dropout) forward without autograd."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, outgoing, False)


# --------------------------------------------------------------------------- backward
def _gate(proj, gate, dy, ds, dp, prefix):
    """dp = dy ds sigmoid(gate); the gate-logit gradient into the dX GEMM prefix."""
    M, D = proj.shape
    n = ds.shape[0]
    defines = W._defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=D, TMN_SIGMOID_TANH=1)
    smem = 49152 + 128
    k = W._kernel("gate.cu", "mw_prefix_gate_epi", "forward", defines, smem)
    fields = [W._rows(proj, D, M), W._rows(gate, D, M), W._rows(dy, D, M), W._rows(ds, D, n),
              W._rows(dp, D, M), W._map(prefix, [64, 64], [M, D], [M * 2]), M, n]
    W._launch(k, W._sms() * 4, 128, smem, W._params(f"uni_gate{D}", fields))


#: Output-LN backward per H: (source family, rows per tile, min blocks per SM). The "d256" source
#: single-buffers its tiles, the "d384" source double-buffers them.
OUT_LN = {256: ("d256", 32, 3), 384: ("d384", 16, 2)}


def _output_ln(tri, dt, dn, mu, rs, go, dgo, dbo):
    """Output-LN backward (dgo / dbo accumulate: zeroed by the caller)."""
    H, n = tri.shape[0], tri.shape[1]
    M = n * n
    family, rows, minblocks = OUT_LN[H]
    buffers = 2 if family == "d256" else 4
    defines_ = dict(MW_MINB=1, MW_K1_STREAM=0, WIDTH=H // 2, LN_ROWS=rows, LN_DN_TMA=1, LN_FENCE=0,
                    LN_MINBLOCKS=minblocks)
    sb = rows * H * 2
    stats = buffers * sb + 128 + 4 * H
    defines = W._defines(**defines_, LN_STATS_OFFSET=stats)
    smem = stats + rows * 8 * (buffers // 2)
    source = f"output_ln_{family}_l384.cu"
    swizzle = "64B" if rows == 32 else "32B"
    maps = [W._map(t, [rows, 64, H // 64], [M, 64, H // 64], [M * 2, M * 128], swizzle) for t in (tri, dt)]
    maps.append(W._map(dn, [64, rows, H // 64], [64, M, H // 64], [H * 2, 128]))
    k = W._kernel(source, "mw_independent_output_ln", "upstream", defines, smem)
    grid = W._resident_grid(source, "mw_independent_output_ln", "upstream", defines, 128, smem)
    W._launch(k, grid, 128, smem, W._params(f"uni_{source}", [*maps, dn, mu, rs, go, dgo, dbo, M]))


def _contract_gp(dt, ab, pre, mask, gp, outgoing):
    """dl / dr of one direction fused with the gate/projection preactivation gradients."""
    D, n = dt.shape[0], dt.shape[1]
    defines = W._defines(WIDTH=D, GP_DIR=1 if outgoing else 2)
    smem = 98304 + 128
    # L768: the fixed-length (fully unrolled K loop) kernel; L384: the staged-epilogue one (run-time N).
    source, entry = (("contract_gp_l768.cu", "mw_wide_fixed_length_gp") if n == 768
                     else ("contract_gp_d384_l384.cu", "mw_wide_staged_epilogue_gp"))
    k = W._kernel(source, entry, "upstream", defines, smem)
    left, right = ab[:D], ab[D:]

    def plane(t, channels):
        return W._map(t, [64, 64], [n, channels * n], [n * 2])
    # MODE 0/1 (outgoing): A = dt, B = right / left; MODE 2/3 (incoming): A = right / left, B = dt.
    a = (dt, dt, right, left)
    b = (right, left, dt, dt)
    maps = [plane(t, D) for t in (*a, *b)]
    tail = [plane(pre, 4 * D), plane(mask, 1), *[plane(g, D) for g in gp]]
    fields = [*maps, pre, mask, *gp, None, None, *tail, n]
    W._launch(k, 2 * D * (n // 128) ** 2, 256, smem, W._params(f"uni_contract_gp{D}", fields))


def _input_ln(x, dxn, dy, dx, gi, dgi, dbi):
    """Input-LN backward + identity residual (dgi / dbi accumulate: zeroed by the caller)."""
    n, D = x.shape[1], x.shape[-1]
    M = n * n
    defines = W._defines(WIDTH=D, INPUT_ROWS=16, INPUT_THREADS=128, INPUT_MINBLOCKS=4, INPUT_NO_REDUCE=1)
    smem = max(3 * 16 * D * 2 + 128, 2 * 4 * D * 4) + D * 4
    source = f"input_ln_d{D}_l384.cu"
    k = W._kernel(source, "mw_independent_input_ln", "upstream", defines, smem)
    maps = [W._map(t, [64, 16], [D, M], [D * 2]) for t in (x, dxn, dy, dx)]
    fields = [*maps, gi, dgi, dbi, None, None, None, None, None, *([None] if D == 384 else []), M]
    W._launch(k, W._sms() * {256: 6, 384: 5}[D], 128, smem, W._params(f"uni_{source}", fields))


def _run_backward(leaves, mask, ds, saved, dy, outgoing, master=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, D = x.shape[1], x.shape[-1]
    M = n * n
    ab, tri, xn, norm, proj, gate, mu, rs, pre, pmask = saved
    xn2 = xn.reshape(M, D)
    dy = dy.contiguous()
    f32 = dict(device=x.device, dtype=torch.float32)
    dp = x.new_empty((M, D))
    dxin = x.new_empty((5 * D, M))   # [gate-logit gradient; W_l, W_lg, W_r, W_rg preactivation gradients]
    gp = list(dxin[D:].view(4, D, M).unbind())
    _gate(proj, gate, dy, ds.reshape(n, D), dp, dxin[:D])
    dn = torch.mm(dp, wp)
    dwp = torch.mm(dp.t(), norm, **({"out_dtype": torch.float32} if master else {}))
    dt = torch.empty_like(tri)
    dgo, dbo = torch.zeros(D, **f32), torch.zeros(D, **f32)
    _output_ln(tri, dt, dn, mu, rs, go, dgo, dbo)
    _contract_gp(dt, ab, pre, pmask, gp, outgoing)
    # One GEMM for [dW_g; dW_l; dW_lg; dW_r; dW_rg] (five D x D products over K = M: 2x faster than
    # separate GEMMs at D256); the autograd function splits it into views outside this op.
    dws = torch.mm(dxin, xn2, **({"out_dtype": torch.float32} if master else {}))
    wcat = torch.cat((wg, wl, wlg, wr, wrg))
    dxn = torch.mm(dxin.t(), wcat)
    dx = torch.empty_like(x)
    dgi, dbi = torch.zeros(D, **f32), torch.zeros(D, **f32)
    _input_ln(x, dxn, dy, dx, gi, dgi, dbi)
    return [dx, dws, dwp, dgi, dbi, dgo, dbo]


def _backward_fake(leaves, mask, ds, saved, dy, outgoing, master=False):
    """dx, the stacked [5D, D] weight gradients (g, l, lg, r, rg), dW_p and the four LN affine gradients."""
    x, D = leaves[0], leaves[0].shape[-1]
    return [torch.empty_like(x), x.new_empty((5 * D, D), dtype=torch.float32 if master else x.dtype), torch.empty_like(leaves[6], dtype=torch.float32 if master else leaves[6].dtype),
            *(torch.empty_like(t) for t in leaves[7:])]


@opaque(fake=_backward_fake, name="trimul_h100_uni_wide_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
             saved: list[torch.Tensor], dy: torch.Tensor, outgoing: bool, master: bool = False) -> list[torch.Tensor]:
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    with T.native_context(leaves[0].device):
        return _run_backward(leaves, mask, ds, saved, dy, outgoing, master)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outgoing, *args):
        leaves = list(args[:11])
        ctx.master = any(w.dtype == torch.float32 for w in leaves[1:7])
        weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]] if ctx.master else leaves[1:7]
        if ctx.master:
            torch._foreach_copy_(weights, leaves[1:7])
        leaves = [leaves[0], *weights, *leaves[7:]]
        mask, ds = args[11:]
        y, *saved = forward(leaves, mask, ds, outgoing)
        ctx.outgoing = outgoing
        ctx.saved_count = len(saved)
        ctx.save_for_backward(*args, *saved, *weights)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        leaves = [vals[0], *vals[-6:], *vals[7:11]]
        dx, dws, dwp, *affine = backward(leaves, vals[11], vals[12], list(vals[13:13+ctx.saved_count]), dy, ctx.outgoing, ctx.master)
        dwg, dwl, dwlg, dwr, dwrg = dws.chunk(5)
        grads = [dx, dwl, dwlg, dwr, dwrg, dwg, dwp, *affine]
        grads = [g.to(t.dtype) for g,t in zip(grads, vals[:11],strict=True)]
        return (None, *grads, None, None)


def single_trimul(outgoing, *args):
    """(x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, pair_mask[n,n], dropscale[n,D]) -> y."""
    if not torch.is_grad_enabled():
        leaves = list(args[:11])
        if any(w.dtype != leaves[0].dtype for w in leaves[1:7]):
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]]
            torch._foreach_copy_(weights, leaves[1:7])
            leaves = [leaves[0], *weights, *leaves[7:]]
        return forward_nograd(leaves, *args[11:], outgoing)
    return _Training.apply(outgoing, *args)
