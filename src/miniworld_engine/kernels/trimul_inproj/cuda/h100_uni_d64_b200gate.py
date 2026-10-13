"""Explicit B200-style D64 training gate-gradient placement on H100.

B1 stores BF16 dGout. B7 accumulates dGout @ Wg with the input projection
gradient in FP32. Constant accumulator indices avoid local array traffic.
The original normalization-statistics saves and forward path are retained.
Selected by integration only with ``Settings.trimul_h100_b200style``.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_d64_b200gate as D64

R = T.SOURCES / "d64_train"
WIDTH = 64
HIDDEN = 64
K3_TILE = (2, 64, 4, 1)


def supports(width, length) -> bool:
    return width == WIDTH and length in ((128, 256, 384, 512, 640, 768, 896, 1024)
                                         if T.cuda_training_aux_enabled() else (384, 768))


def _k3_smem(tile):
    bi, bj, slots, _ = tile
    bmt = bi * bj
    nkcz = WIDTH // 64
    nbar = 2 * nkcz + 2 + 2 * slots
    return (bmt // 64 * HIDDEN * 128 + nkcz * bmt * 128 + slots * max(WIDTH, HIDDEN) * 64
            + 8 * 16 * 64 * 2 + (2 * WIDTH + 2 * HIDDEN) * 4 + (nbar * 8 + 127) // 128 * 128)


@T.device_cache
def _k3(tile):
    bi, bj, slots, acc = tile
    defines = dict(MW_CZ=WIDTH, MW_HIDDEN=HIDDEN, MW_SAVE_PG=0, MW_PG_METHOD=-1, XHAT_FP32=-1,
                   MW_SAVE_IN=0, MW_SAVE_OUT=0, MW_STORE_METHOD=0, MW_SAVE_STATS_IN=1,
                   MW_SAVE_STATS_OUT=1, MW_BI=bi, MW_BJ=bj, MW_SLOT=slots, MW_ACC=acc,
                   TMN_K3_REGS_24_240=1, MW_SERIAL=1, MW_FUSED=1)
    cubin = T.compile(R / "save_k3_d64.cu", D64._flags(T._upstream() / "csrc", defines.items()))
    k = T.load_unit(str(cubin), "d64_save_k3").kernel("save_k3")
    smem = _k3_smem(tile)
    k.set_max_dynamic_smem(smem)
    return k, smem


B1W_SMEM, B2W_SMEM = 171008, 170560  # == b1::SMEM, b2::SMEM in uni_bwd_wg.cu
B1W_SLOT, B2W_SLOT = 4096 + 128 + 4096, 16384 + 128
FINW_OUT = B2W_SLOT + B1W_SLOT


@T.device_cache
def _bwd_wg(master=False, affine_split=False, gatepipe=True):
    defines = [("MASTER_FP32", int(master))]
    if affine_split:
        defines.append(("UNI_AFFINE_SPLIT", 1))
    # Outgoing L256 has no reproducible complete-workload improvement.
    # Full-module gains are qualified for the nonsplit affine reduction.
    # Preserve the separately forced split configuration and the L256 fallback.
    source = ("uni_bwd_wg_affinereg.cu" if not affine_split else "uni_bwd_wg_gatepipe.cu") if gatepipe else "uni_bwd_wg_b7gate.cu"
    cubin = T.compile(R / source, D64._flags(R, defines))
    unit = T.load_unit(str(cubin), "uni64_bwd_wg")
    b1, b2, fin = (unit.kernel(n) for n in ("uni64_b1w_b7gate", (("uni64_b2w_affinereg" if not affine_split else "uni64_b2w_gatepipe") if gatepipe else "uni64_b2w_b7gate"), "uni64_finw_b7gate"))
    b1.set_max_dynamic_smem(B1W_SMEM)
    b2.set_max_dynamic_smem(B2W_SMEM)
    return b1, b2, fin


# --------------------------------------------------------------------------- forward
def _front(x, w1, maskf, gi, bi, ab):
    kk, K = D64._k1()
    n = x.shape[1]
    cfg = K.lookup("sm_90a", WIDTH, HIDDEN, "b", N=n)["k1"]
    kk.k1(x[0], maskf, {"w1": w1, "ln_in_w": gi, "ln_in_b": bi}, ab, N=n, Np=n, cz=WIDTH,
          ch=HIDDEN, cfg=cfg, lnm=2, eps=1e-5, cache=_k1_cache())


@T.device_cache
def _k1_cache():
    return {}


def _bmm(a, b, *, out):
    if T.d64_uni_owned_bmm_enabled(a.shape[-1]):
        from ._h100_pack import owned_bmm
        return owned_bmm(a, b, out=out)
    return torch.bmm(a, b, out=out)


def _triangle(ab, tri, outgoing):
    left, right = ab[:HIDDEN], ab[HIDDEN:]
    if outgoing:
        _bmm(left, right.transpose(-1, -2), out=tri)
    else:
        _bmm(left.transpose(-1, -2), right, out=tri)


def _output(x, tri, wg, wp, gi, bi, go, bo, ds, y, stats):
    L = T._launch_module()
    n = x.shape[1]
    m = n * n
    bi_, bj, _, _ = K3_TILE
    k, smem = _k3(K3_TILE)
    tj = (n + bj - 1) // bj
    tiles = ((n + bi_ - 1) // bi_) * tj
    ymap = D64._map(y, [64, 16, 1], [WIDTH, n, n], [WIDTH * 2, n * WIDTH * 2])
    base = L.Struct.fixed("h100_uni64_affinereg:k3base", [
        D64._map(x, [64, bj, bi_], [WIDTH, n, n], [WIDTH * 2, n * WIDTH * 2]),
        D64._map(tri, [64, 1, 64], [n, n, HIDDEN], [n * 2, m * 2]),
        D64._map(wg, [64, 32], [WIDTH, WIDTH], [WIDTH * 2]),
        D64._map(wp, [64, 32], [HIDDEN, WIDTH], [HIDDEN * 2]),
        ymap, gi, bi, go, bo, x, y, None, n, n, tj, tiles, 1, 0, 1e-5, 0])
    params = L.Struct.fixed("h100_uni64_affinereg:k3", [
        base, ds, ymap, ymap, None, None, stats[0], stats[1], stats[2], stats[3], ymap, ymap,
        None, None])
    k.launch((min(tiles, D64._sms()), 1, 1), (384, 1, 1), [params], smem)


def _prepare(leaves, mask):
    x, wl, wlg, wr, wrg = leaves[:5]
    n = x.shape[1]
    w1 = x.new_empty((4 * HIDDEN, WIDTH))
    if T.cuda_training_aux_enabled():
        from ._h100_pack import pack_into_cuda, training_pair_mask
        pack_into_cuda(w1, wl, wlg, wr, wrg)
        return w1, training_pair_mask(mask, n, x.device)
    T.pack_into(w1, wl, wlg, wr, wrg)
    return w1, mask.reshape(n, n).to(torch.float32, copy=True)


def _run_forward(leaves, mask, ds, outgoing, save):
    prepared = len(leaves) == 12
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    n = x.shape[1]
    w1, maskf = (leaves[11], mask) if prepared else _prepare(leaves, mask)
    ab = x.new_empty((2 * HIDDEN, n, n))
    tri = x.new_empty((HIDDEN, n, n))
    stats = torch.empty((4, n * n), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    _front(x, w1, maskf, gi, bi, ab)
    _triangle(ab, tri, outgoing)
    _output(x, tri, wg, wp, gi, bi, go, bo, ds.reshape(n, WIDTH), y, stats)
    if not save:
        return y
    return [y, ab, tri, stats] if prepared else [y, ab, tri, stats, w1, maskf]


def _forward_fake(leaves, mask, ds, outgoing):
    """Output plus saved ab, tri, LN statistics [mean_in, rstd_in, mean_out, rstd_out], weight pack, FP32 mask."""
    x = leaves[0]
    n = x.shape[1]
    if len(leaves) == 12:
        return [torch.empty_like(x), x.new_empty((2*HIDDEN,n,n)), x.new_empty((HIDDEN,n,n)),
                x.new_empty((4,n*n),dtype=torch.float32)]
    return [torch.empty_like(x), x.new_empty((2 * HIDDEN, n, n)), x.new_empty((HIDDEN, n, n)),
            x.new_empty((4, n * n), dtype=torch.float32), x.new_empty((4 * HIDDEN, WIDTH)),
            x.new_empty((n, n), dtype=torch.float32)]


@opaque(fake=_forward_fake, name="trimul_h100_affinereg_uni64_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool) -> list[torch.Tensor]:
    """Training forward; returns ``[y, ab, tri, stats, w1, maskf]`` (all freshly allocated)."""
    with T.preserved_d64_context(T.d64_training(leaves[0], outgoing=outgoing)), T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, outgoing, True)


def _forward_nograd_fake(leaves, mask, ds, outgoing):
    """Output only."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_affinereg_uni64_dropout_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool) -> torch.Tensor:
    """Training-mode (dropout) forward without autograd: no saved activations are returned."""
    with T.preserved_d64_context(T.d64_training(leaves[0], outgoing=outgoing)), T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, outgoing, False)


# --------------------------------------------------------------------------- backward
def _run_backward(leaves, mask, ds, saved, dy, outgoing, master=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    ab, tri, stats, w1, maskf = saved
    L = T._launch_module()
    n = x.shape[1]
    m = n * n
    h = WIDTH
    dy = dy.contiguous()
    b1, b2, fin = _bwd_wg(master, T.d64_uni_affine_enabled(x, outgoing), gatepipe=not (outgoing and n == 256))
    sms = D64._sms()
    ntiles = m // 64
    g1 = min(sms, (ntiles + 1) // 2)
    g2 = min(sms, ntiles)
    planes = lambda t: D64._map2(t, [64, 64], [m, HIDDEN], [m * 2])
    rows = lambda t: D64._map2(t, [64, 64], [h, m], [h * 2])
    dt = torch.empty_like(tri)
    dgate = torch.empty_like(x)
    p1 = torch.empty((g1, B1W_SLOT), device=x.device, dtype=torch.float32)
    params = L.Struct.fixed("h100_uni64_affinereg:b1w", [
        planes(tri), rows(x), rows(dy), D64._map2(ds.reshape(n, h).contiguous(), [64, 64], [h, n], [h * 2]),
        D64._map2(stats, [64, 4], [m, 4], [m * 4], "none"), planes(dt), rows(dgate),
        wp, wg, go, bo, gi, bi, p1, ntiles, n, None])
    b1.launch((g1, 1, 1), (256, 1, 1), [params], B1W_SMEM)
    left, right = ab[:HIDDEN], ab[HIDDEN:]
    dl = torch.empty_like(tri)
    dr = torch.empty_like(tri)
    if outgoing:        # tri = l r^T
        _bmm(dt, right, out=dl)
        _bmm(dt.transpose(-1, -2), left, out=dr)
    else:               # tri = l^T r
        _bmm(right, dt.transpose(-1, -2), out=dl)
        _bmm(left, dt, out=dr)
    dx = torch.empty_like(x)
    p2 = torch.empty((g2, B2W_SLOT), device=x.device, dtype=torch.float32)
    params = L.Struct.fixed("h100_uni64_affinereg:b2w", [
        planes(dl), planes(dr), rows(x), rows(dgate), rows(dy),
        D64._map2(stats, [64, 2], [m, 4], [m * 4], "none"),
        D64._map2(maskf.reshape(m), [64, 1], [m, 1], [m * 4], "none"), rows(dx),
        w1, wg, gi, bi, p2, ntiles, None])
    b2.launch((g2, 1, 1), (256, 1, 1), [params], B2W_SMEM)
    grads = [torch.empty_like(w, dtype=torch.float32 if master else w.dtype) for w in (wl, wlg, wr, wrg, wg, wp)]
    f32 = dict(device=x.device, dtype=torch.float32)
    dgi, dbi = torch.empty(h, **f32), torch.empty(h, **f32)
    dgo, dbo = torch.empty(HIDDEN, **f32), torch.empty(HIDDEN, **f32)
    fin.launch(((FINW_OUT + 255) // 256, 1, 1), (256, 1, 1),
               [p1, g1, p2, g2, *grads, dgi, dbi, dgo, dbo])
    return [dx, *grads, dgi, dbi, dgo, dbo]


def _backward_fake(leaves, mask, ds, saved, dy, outgoing, master=False):
    """One gradient per differentiable leaf, same shape and dtype."""
    return [torch.empty_like(t, dtype=torch.float32 if master and 1 <= i <= 6 else t.dtype) for i, t in enumerate(leaves)]


@opaque(fake=_backward_fake, name="trimul_h100_affinereg_uni64_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
             saved: list[torch.Tensor], dy: torch.Tensor, outgoing: bool, master: bool = False) -> list[torch.Tensor]:
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    # Autograd invokes this boundary after the caller's forward scope has ended.
    with T.preserved_d64_context(T.d64_training(leaves[0], outgoing=outgoing)), T.native_context(leaves[0].device):
        return _run_backward(leaves, mask, ds, saved, dy, outgoing, master)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outgoing, *args):
        leaves = list(args[:11])
        ctx.master = any(w.dtype == torch.float32 for w in leaves[1:7])
        if T.cuda_training_fused_prepare_enabled():
            from ._h100_pack import fused_training_prepare
            packed, maskf, gate, output = fused_training_prepare(leaves, args[11])
            weights = [*leaves[1:5], gate, output]
            prepared = [leaves[0], *weights, *leaves[7:], packed]
            y, ab, tri, stats = forward(prepared, maskf, args[12], outgoing)
            saved = [ab, tri, stats, packed, maskf]
            ctx.outgoing = outgoing
            ctx.saved_count = len(saved)
            ctx.save_for_backward(*args, *saved, *weights)
            return y
        if T.cuda_training_aux_enabled():
            from ._h100_pack import cast_weights
            weights = cast_weights(leaves[1:7])
        else:
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]] if ctx.master else leaves[1:7]
        if ctx.master and not T.cuda_training_aux_enabled():
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
        saved = list(vals[13:13+ctx.saved_count])
        grads = backward(leaves, vals[11], vals[12], saved, dy, ctx.outgoing, ctx.master)
        grads = [g.to(t.dtype) for g, t in zip(grads, vals[:11], strict=True)]
        return (None, *grads, None, None)


def single_trimul(outgoing, *args):
    """(x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, pair_mask[n,n], dropscale[n,D]) -> y."""
    if not torch.is_grad_enabled():
        leaves = list(args[:11])
        if T.cuda_training_fused_prepare_enabled():
            from ._h100_pack import fused_training_prepare
            packed, maskf, gate, output = fused_training_prepare(leaves, args[11])
            prepared = [leaves[0], *leaves[1:5], gate, output, *leaves[7:], packed]
            return forward_nograd(prepared, maskf, args[12], outgoing)
        if T.cuda_training_aux_enabled():
            from ._h100_pack import cast_weights
            leaves = [leaves[0], *cast_weights(leaves[1:7]), *leaves[7:]]
        elif any(w.dtype != leaves[0].dtype for w in leaves[1:7]):
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]]
            torch._foreach_copy_(weights, leaves[1:7])
            leaves = [leaves[0], *weights, *leaves[7:]]
        return forward_nograd(leaves, *args[11:], outgoing)
    return _Training.apply(outgoing, *args)
