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

WIDTHS = (256, 384, 512)
# front K1 tiles: (BI, BJ, ring slots, K chunks per slot, min blocks per SM)
_K1 = {256: (2, 64, 4, 4, 1), 384: (2, 64, 2, 6, 1)}


def supports(width, length) -> bool:
    return width in WIDTHS and length in (384, 768)


# --------------------------------------------------------------------------- forward
def _front(x, w1, ab, mask, gi, bi, xn, pre, hidden=None):
    n, D = x.shape[1], x.shape[-1]
    hidden = D if hidden is None else hidden
    cfg = _K1[D] if D in _K1 else {128: (2, 64, 2, 2, 1), 512: (1, 64, 2, 8, 1)}[D]
    a, b, slots, sk, minb = cfg
    defines = W._defines(MW_MINB=minb, MW_K1_STREAM=0, TMN_SIGMOID_TANH=1, TMN_WSKIP=1,
                         TMN_MASK_TEMPLATE=1, FRONT_CZ=D, FRONT_CH=hidden, FRONT_BI=a, FRONT_SLOTS=slots, FRONT_SK=sk)
    smem = W._k1_smem(D, cfg, a * b // 64 * 4096)
    k = W._kernel("front_pre_d384.cu", "mw_wide_front_pre_overlap", "forward", defines, smem)
    tj = (n + b - 1) // b
    tiles = ((n + a - 1) // a) * tj
    fields = [
        W._map(x, [64, b, a], [D, n, n], [D * 2, n * D * 2]),
        W._map(w1, [64, 64], [D, 4 * hidden], [D * 2]),
        W._map(ab, [64, 1, 32], [n, n, 2 * hidden], [n * 2, n * n * 2]),
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
        W._bmm(left, right.transpose(-1, -2), out=tri)
    else:
        W._bmm(left.transpose(-1, -2), right, out=tri)


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


def saved_like(x, selected=False):
    """ab, tri, x_n, norm, proj, gate, mean_out, rstd_out, pre, bf16 pair mask."""
    n, D = x.shape[1], x.shape[-1]
    M = n * n
    f32 = dict(dtype=torch.float32)
    saved = [x.new_empty((2 * D, n, n)), x.new_empty((D, n, n)), torch.empty_like(x),
            x.new_empty((M, D)), x.new_empty((M, D)), x.new_empty((M, D)),
            x.new_empty((M,), **f32), x.new_empty((M,), **f32), x.new_empty((4 * D, M)),
            x.new_empty((n, n))]
    from ._h100_pack import fused_d256_glue
    if fused_d256_glue(D,n,selected):
        saved.append(x.new_empty((5*D,D)))
    return saved


def _run_forward(leaves, mask, ds, outgoing, save):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, D = x.shape[1], x.shape[-1]
    if not supports(D, n):
        raise ValueError(f"single-direction wide TriMul training covers D256/384/512 at L384/768, not {(D, n)}")
    M = n * n
    saved = saved_like(x)
    ab, tri, xn, norm, proj, gate, mu, rs, pre, pmask = saved[:10]
    if T.preserve_training(D):
        from ._h100_pack import dense_copy_into
        dense_copy_into(pmask,mask.reshape(n,n))
    else:
        pmask.copy_(mask.reshape(n, n))
    w1 = x.new_empty((4 * D, D))
    if len(saved)>10:
        from ._h100_pack import pack_training_pair
        pack_training_pair(w1,saved[10],(wg,wl,wlg,wr,wrg))
    else:
        W.pack_into(w1, wl, wlg, wr, wrg)
    _front(x, w1, ab, pmask, gi, bi, xn, pre)
    _triangle(ab, tri, outgoing)
    _output_norm(tri, norm, go, bo, mu, rs)
    W._mm(norm, wp.t(), out=proj)
    W._mm(xn.reshape(M, D), wg.t(), out=gate)
    y = torch.empty_like(x)
    defines = W._defines(MW_MINB=1, MW_K1_STREAM=0, TMN_SIGMOID_TANH=1, WIDTH=D, GROUPS=1, KCHUNK=1)
    k = W._kernel("output_dense.cu", "mw_wide_dense_output_epi", "forward", defines)
    W._launch(k, 1056, 256, 0, W._params(f"uni_epilogue{D}", [proj, gate, x, ds.reshape(n, D), y, M * D, n * D]))
    return [y, *saved] if save else y


def _forward_fake(leaves, mask, ds, outgoing):
    """y like x, then the saved set (``saved_like``)."""
    x = leaves[0]
    return [torch.empty_like(x), *saved_like(x, T.uni_training(x, outgoing))]


@opaque(fake=_forward_fake, name="trimul_h100_uni_wide_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool) -> list[torch.Tensor]:
    """Training forward; returns ``[y, *saved]`` (all freshly allocated)."""
    with T.native_context(leaves[0].device), T.preserved_d256_context(T.uni_training(leaves[0], outgoing)), T.preserved_uni_blas_context(leaves[0], outgoing):
        return _run_forward(leaves, mask, ds, outgoing, True)


def _forward_nograd_fake(leaves, mask, ds, outgoing):
    """y like x."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_uni_wide_dropout_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool) -> torch.Tensor:
    """Training-mode (dropout) forward without autograd."""
    with T.native_context(leaves[0].device), T.preserved_d256_context(T.uni_training(leaves[0], outgoing)), T.preserved_uni_blas_context(leaves[0], outgoing):
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
OUT_LN = {256: ("d256", 32, 3), 384: ("d384", 16, 2), 512: ("d512", 16, 2)}


def _output_ln(tri, dt, dn, mu, rs, go, dgo, dbo, defer=False):
    """Output-LN backward (dgo / dbo accumulate: zeroed by the caller)."""
    H, n = tri.shape[0], tri.shape[1]
    M = n * n
    native = T.preserve_training(H)
    family, rows, minblocks = OUT_LN[H]
    buffers = 2 if family in ("d256", "d512") else 4
    defines_ = dict(MW_MINB=1, MW_K1_STREAM=0, WIDTH=H // 2, LN_ROWS=rows, LN_DN_TMA=1, LN_FENCE=0,
                    LN_MINBLOCKS=minblocks)
    sb = rows * H * 2
    stats = buffers * sb + 128 + 4 * H
    defines = W._defines(**defines_, LN_STATS_OFFSET=stats)
    if native:
        defines += W._defines(MW_LN_PARTIALS=1)
    smem = stats + rows * 8 * (buffers // 2) + (640 if family == "d512" else 0)
    source = f"output_ln_{family}_l384.cu"
    swizzle = "64B" if rows == 32 else "32B"
    maps = [W._map(t, [rows, 64, H // 64], [M, 64, H // 64], [M * 2, M * 128], swizzle) for t in (tri, dt)]
    maps.append(W._map(dn, [64, rows, H // 64], [64, M, H // 64], [H * 2, 128]))
    k = W._kernel(source, "mw_independent_output_ln", "upstream", defines, smem)
    grid = W._resident_grid(source, "mw_independent_output_ln", "upstream", defines, 128, smem)
    if native:
        pg,pb=[torch.empty((grid,H),device=tri.device,dtype=torch.float32) for _ in range(2)]
    else:
        pg,pb=dgo,dbo
    W._launch(k, grid, 128, smem, W._params(f"uni_{source}", [*maps, dn, mu, rs, go, pg, pb, M]))
    if native:
        if defer:
            return pg,pb,dgo,dbo
        from ._h100_pack import finish_affine_partials
        finish_affine_partials(pg,pb,dgo,dbo)


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


def _input_ln(x, dxn, dy, dx, gi, dgi, dbi, defer=False):
    """Input-LN backward + identity residual (dgi / dbi accumulate: zeroed by the caller)."""
    n, D = x.shape[1], x.shape[-1]
    M = n * n
    native = T.preserve_training(D)
    defines = W._defines(WIDTH=D, INPUT_ROWS=16, INPUT_THREADS=128, INPUT_MINBLOCKS=4, INPUT_NO_REDUCE=1)
    if native:
        defines += W._defines(MW_LN_PARTIALS=1)
    smem = max(3 * 16 * D * 2 + 128, 2 * 4 * D * 4) + D * 4
    source = f"input_ln_d{D}_l384.cu"
    k = W._kernel(source, "mw_independent_input_ln", "upstream", defines, smem)
    maps = [W._map(t, [64, 16], [D, M], [D * 2]) for t in (x, dxn, dy, dx)]
    grid = W._sms() * {256: 6, 384: 5, 512: 4}[D]
    if native:
        pg,pb=[torch.empty((grid,D),device=x.device,dtype=torch.float32) for _ in range(2)]
    else:
        pg,pb=dgi,dbi
    fields = [*maps, gi, pg, pb, None, None, None, None, None, *([None] if D == 384 else []), M]
    W._launch(k, grid, 128, smem, W._params(f"uni_{source}", fields))
    if native:
        if defer:
            return pg,pb,dgi,dbi
        from ._h100_pack import finish_affine_partials
        finish_affine_partials(pg,pb,dgi,dbi)


def _run_backward(leaves, mask, ds, saved, dy, outgoing, master=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, D = x.shape[1], x.shape[-1]
    M = n * n
    ab, tri, xn, norm, proj, gate, mu, rs, pre, pmask = saved[:10]
    fused_glue = len(saved)>10
    xn2 = xn.reshape(M, D)
    dy = dy.contiguous()
    f32 = dict(device=x.device, dtype=torch.float32)
    dp = x.new_empty((M, D))
    dxin = x.new_empty((5 * D, M))   # [gate-logit gradient; W_l, W_lg, W_r, W_rg preactivation gradients]
    gp = list(dxin[D:].view(4, D, M).unbind())
    _gate(proj, gate, dy, ds.reshape(n, D), dp, dxin[:D])
    dn = W._mm(dp, wp)
    dwp = W._mm(dp.t(), norm, **({"out_dtype": torch.float32} if master else {}))
    dt = torch.empty_like(tri)
    affine_alloc = torch.empty if T.preserve_training(D) else torch.zeros
    dgo, dbo = affine_alloc(D, **f32), affine_alloc(D, **f32)
    out_affine = _output_ln(tri, dt, dn, mu, rs, go, dgo, dbo, defer=fused_glue)
    _contract_gp(dt, ab, pre, pmask, gp, outgoing)
    # One GEMM for [dW_g; dW_l; dW_lg; dW_r; dW_rg] (five D x D products over K = M: 2x faster than
    # separate GEMMs at D256); the autograd function splits it into views outside this op.
    dws = W._mm(dxin, xn2, **({"out_dtype": torch.float32} if master else {}))
    if fused_glue:
        wcat = saved[10]
    elif T.preserve_training(D):
        from ._h100_pack import stack_backward_weights
        wcat = stack_backward_weights((wg,wl,wlg,wr,wrg))
    else:
        wcat = torch.cat((wg, wl, wlg, wr, wrg))
    dxn = W._mm(dxin.t(), wcat)
    dx = torch.empty_like(x)
    dgi, dbi = affine_alloc(D, **f32), affine_alloc(D, **f32)
    in_affine = _input_ln(x, dxn, dy, dx, gi, dgi, dbi, defer=fused_glue)
    if fused_glue:
        from ._h100_pack import finish_affine_pair
        finish_affine_pair(out_affine,in_affine)
    return [dx, dws, dwp, dgi, dbi, dgo, dbo]


def _backward_fake(leaves, mask, ds, saved, dy, outgoing, master=False, native=False):
    """dx, the stacked [5D, D] weight gradients (g, l, lg, r, rg), dW_p and the four LN affine gradients."""
    x, D = leaves[0], leaves[0].shape[-1]
    return [torch.empty_like(x), x.new_empty((5 * D, D), dtype=torch.float32 if master else x.dtype), torch.empty_like(leaves[6], dtype=torch.float32 if master else leaves[6].dtype),
            *(torch.empty_like(t) for t in leaves[7:])]


@opaque(fake=_backward_fake, name="trimul_h100_uni_wide_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
             saved: list[torch.Tensor], dy: torch.Tensor, outgoing: bool, master: bool = False,
             native: bool = False) -> list[torch.Tensor]:
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    with T.native_context(leaves[0].device), T.preserved_d256_context(native), T.preserved_uni_blas_context(leaves[0], outgoing):
        return _run_backward(leaves, mask, ds, saved, dy, outgoing, master)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outgoing, *args):
        leaves = list(args[:11])
        ctx.master = any(w.dtype == torch.float32 for w in leaves[1:7])
        native = T.uni_training(leaves[0], outgoing)
        ctx.native = native
        if native:
            from ._h100_pack import cast_training_weights
            weights = cast_training_weights(leaves[1:7])
        else:
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]] if ctx.master else leaves[1:7]
        if ctx.master and not native:
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
        dx, dws, dwp, *affine = backward(leaves, vals[11], vals[12], list(vals[13:13+ctx.saved_count]), dy, ctx.outgoing, ctx.master, ctx.native)
        dwg, dwl, dwlg, dwr, dwrg = dws.chunk(5)
        grads = [dx, dwl, dwlg, dwr, dwrg, dwg, dwp, *affine]
        if ctx.native:
            from ._h100_pack import native_training_grads
            grads = native_training_grads(grads,list(vals[:11]))
        else:
            grads = [g.to(t.dtype) for g,t in zip(grads, vals[:11],strict=True)]
        return (None, *grads, None, None)


def single_trimul(outgoing, *args):
    """(x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, pair_mask[n,n], dropscale[n,D]) -> y."""
    import os
    native = T.uni_training(args[0], outgoing)
    if os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "1" and not native:
        return native_buckets(False, outgoing, *args)
    if not torch.is_grad_enabled():
        leaves = list(args[:11])
        if native:
            from ._h100_pack import cast_training_weights
            leaves = [leaves[0],*cast_training_weights(leaves[1:7]),*leaves[7:]]
        elif any(w.dtype != leaves[0].dtype for w in leaves[1:7]):
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]]
            torch._foreach_copy_(weights, leaves[1:7])
            leaves = [leaves[0], *weights, *leaves[7:]]
        return forward_nograd(leaves, *args[11:], outgoing)
    return _Training.apply(outgoing, *args)


# Native coverage candidate. Keep the selected legacy plans above as frozen
# comparison paths. This extends their Hopper K1 and fused contraction/gate
# algorithm to all buckets; auxiliary reductions are deterministic hand CUDA.
# The output-side wide decomposition still materializes norm/proj/gate and is
# not a claim of B200 K3 fusion or of a latency win.
def _native_saved(x, hidden):
    n, d = x.shape[1], x.shape[-1]
    m = n*n
    return [x.new_empty((2*hidden,n,n)), x.new_empty((hidden,n,n)), torch.empty_like(x),
            x.new_empty((m,hidden)), x.new_empty((m,d)), x.new_empty((m,d)),
            x.new_empty((m,),dtype=torch.float32), x.new_empty((m,),dtype=torch.float32),
            x.new_empty((4*hidden,m)), x.new_empty((n,n))]


def _native_forward_fake(leaves, mask, ds, bidirectional, outgoing):
    return [torch.empty_like(leaves[0]), *_native_saved(leaves[0], leaves[1].shape[0]),
            *(w.new_empty((0,)) if w.dtype == torch.bfloat16 and (i < 4 or w.is_contiguous())
              else torch.empty(w.shape, device=w.device, dtype=torch.bfloat16)
              for i,w in enumerate(leaves[1:7]))]


@opaque(fake=_native_forward_fake, name="trimul_h100_native_buckets_forward")
def native_forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
                   bidirectional: bool, outgoing: bool) -> list[torch.Tensor]:
    from ._h100_pack import cast_weights, owned_bf16_mask, pack_into_cuda
    x, _, _, _, _, _, _, gi, bi, go, bo = leaves
    n, d, h = x.shape[1], x.shape[-1], leaves[1].shape[0]
    if (d not in (128,256,384,512) or n not in (128,256,384,512,640,768)
            or h != d*(2 if bidirectional else 1)):
        raise ValueError("native TriMul buckets require D128/256/384/512 and L128..768")
    matrices = leaves[1:7]
    expected = [(h, d)] * 4 + [(d, d), (d, h)]
    if (x.dtype != torch.bfloat16 or not x.is_contiguous()
            or matrices[0].dtype not in (torch.bfloat16, torch.float32)
            or any(w.dtype != matrices[0].dtype or w.device != x.device
                   or not (w.is_contiguous() or (i < 4 and w.t().is_contiguous()))
                   or tuple(w.shape) != shape
                   for i, (w, shape) in enumerate(zip(matrices, expected, strict=True)))
            or any(w.dtype != torch.float32 or w.device != x.device or not w.is_contiguous()
                   for w in (gi, bi, go, bo))
            or ds.device != x.device or ds.dtype != torch.bfloat16
            or not ds.is_contiguous() or ds.numel() != n*d):
        raise ValueError("native TriMul requires contiguous BF16 activations/dropout scale, "
                         "uniform dense BF16/FP32 matrices and FP32 norm affine on the same device")
    with T.native_context(x.device):
        weights = cast_weights(leaves[1:7])
        wl,wlg,wr,wrg,wg,wp = weights
        saved = _native_saved(x,h)
        ab,tri,xn,norm,proj,gate,mu,rs,pre,_ = saved
        pmask = owned_bf16_mask(mask,n,x.device)
        saved[-1] = pmask
        w1 = x.new_empty((4*h,d))
        pack_into_cuda(w1,wl,wlg,wr,wrg)
        _front(x,w1,ab,pmask,gi,bi,xn,pre,hidden=h)
        left,right = ab[:h],ab[h:]
        if bidirectional:
            W._bmm(left[:d],right[:d].transpose(-1,-2),out=tri[:d])
            W._bmm(left[d:].transpose(-1,-2),right[d:],out=tri[d:])
        else:
            _triangle(ab,tri,outgoing)
        _output_norm(tri,norm,go,bo,mu,rs)
        W._mm(norm,wp.t(),out=proj)
        W._mm(xn.reshape(n*n,d),wg.t(),out=gate)
        y = torch.empty_like(x)
        defines = W._defines(MW_MINB=1,MW_K1_STREAM=0,TMN_SIGMOID_TANH=1,WIDTH=d,GROUPS=1,KCHUNK=1)
        k = W._kernel("output_dense.cu","mw_wide_dense_output_epi","forward",defines)
        W._launch(k,1056,256,0,W._params(f"native_epilogue{d}",[proj,gate,x,ds,y,n*n*d,n*d]))
        # Custom ops may not return input aliases. BF16 weights are saved by the
        # caller itself; only owned master casts need to cross this boundary.
        return [y,*saved,*(w if w is not src else w.new_empty((0,))
                           for w,src in zip(weights,leaves[1:7],strict=True))]


def _native_contract_gp(dt, ab, pre, mask, gp, bidirectional, outgoing):
    h,n = dt.shape[:2]
    d = h//2 if bidirectional else h
    direction = 0 if bidirectional else 1 if outgoing else 2
    defines = W._defines(WIDTH=d,GP_DIR=direction,GP_GENERIC_LENGTH=1)
    smem = 98304+128
    k = W._kernel("contract_gp_d384_l384.cu","mw_wide_staged_epilogue_gp","upstream",defines,smem)
    left,right = ab[:h],ab[h:]
    if bidirectional:
        a = (dt[:d],dt[:d],right[d:],left[d:])
        b = (right[:d],left[:d],dt[d:],dt[d:])
    else:
        a = (dt,dt,right,left)
        b = (right,left,dt,dt)
    def plane(t,channels):
        return W._map(t,[64,64],[n,channels*n],[n*2])
    maps = [plane(t,d) for t in (*a,*b)]
    tail = [plane(pre,4*h),plane(mask,1),*[plane(g,h) for g in gp]]
    params = [*maps,pre,mask,*gp,None,None,*tail,n]
    W._launch(k,(4 if bidirectional else 2)*d*(n//128)**2,256,smem,
              W._params(f"native_contract{d}_{direction}",params))


def _native_backward_fake(leaves, ds, saved, dy, bidirectional, outgoing, master):
    x,d,h = leaves[0],leaves[0].shape[-1],leaves[1].shape[0]
    dtype = torch.float32 if master else x.dtype
    return [torch.empty_like(x),x.new_empty((d+4*h,d),dtype=dtype),
            torch.empty_like(leaves[6],dtype=dtype),*(torch.empty_like(t) for t in leaves[7:])]


@opaque(fake=_native_backward_fake, name="trimul_h100_native_buckets_backward")
def native_backward(leaves: list[torch.Tensor], ds: torch.Tensor,
                    saved: list[torch.Tensor], dy: torch.Tensor,
                    bidirectional: bool, outgoing: bool, master: bool) -> list[torch.Tensor]:
    from ._h100_pack import deterministic_ln_backward, stack_backward_weights
    x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo = leaves
    n,d,h = x.shape[1],x.shape[-1],wl.shape[0]
    m=n*n
    ab,tri,xn,norm,proj,gate,mu,rs,pre,pmask = saved
    with T.native_context(x.device):
        if not dy.is_contiguous():
            raise ValueError("native TriMul backward requires contiguous output gradients")
        dp = x.new_empty((m,d))
        dxin = x.new_empty((d+4*h,m))
        gp = list(dxin[d:].view(4,h,m).unbind())
        _gate(proj,gate,dy,ds.reshape(n,d),dp,dxin[:d])
        dn = W._mm(dp,wp)
        dwp = W._mm(dp.t(),norm,**({"out_dtype":torch.float32} if master else {}))
        dt,dgo,dbo = deterministic_ln_backward(tri,dn,go,planes=True,stats=(mu,rs))
        _native_contract_gp(dt,ab,pre,pmask,gp,bidirectional,outgoing)
        dws = W._mm(dxin,xn.reshape(m,d),**({"out_dtype":torch.float32} if master else {}))
        wcat = stack_backward_weights((wg,wl,wlg,wr,wrg))
        dxn = W._mm(dxin.t(),wcat)
        dx,dgi,dbi = deterministic_ln_backward(x,dxn,gi,residual=dy)
        return [dx,dws,dwp,dgi,dbi,dgo,dbo]


class _NativeBuckets(torch.autograd.Function):
    @staticmethod
    def forward(ctx,bidirectional,outgoing,*args):
        leaves=list(args[:11]);mask,ds=args[11:]
        out=native_forward(leaves,mask,ds,bidirectional,outgoing)
        weights=[cast if cast.numel() else src for cast,src in zip(out[-6:],leaves[1:7],strict=True)]
        ctx.master=leaves[1].dtype==torch.float32
        ctx.bidirectional,ctx.outgoing=bidirectional,outgoing
        ctx.save_for_backward(*args,*out[1:-6],*weights)
        return out[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        vals=ctx.saved_tensors
        leaves=[vals[0],*vals[-6:],*vals[7:11]]
        dx,dws,dwp,*affine=native_backward(leaves,vals[12],list(vals[13:-6]),dy,
                                          ctx.bidirectional,ctx.outgoing,ctx.master)
        d,h=vals[0].shape[-1],vals[1].shape[0]
        dwg=dws[:d];dwl,dwlg,dwr,dwrg=dws[d:].split(h)
        from ._h100_pack import preserve_parameter_layout
        grads = preserve_parameter_layout([dwl,dwlg,dwr,dwrg,dwg,dwp], vals[1:7])
        return (None,None,dx,*grads,*affine,None,None)


def native_buckets(bidirectional,outgoing,*args):
    """Opt-in all-bucket native training implementation; no fallback or autotuning."""
    if not torch.is_grad_enabled():
        return native_forward(list(args[:11]),args[11],args[12],bidirectional,outgoing)[0]
    return _NativeBuckets.apply(bidirectional,outgoing,*args)
