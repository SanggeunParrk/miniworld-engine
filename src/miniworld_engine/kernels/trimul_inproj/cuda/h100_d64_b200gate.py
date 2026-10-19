"""B200-style D64 training gate-gradient placement on H100.

B1 stores BF16 dGout. B7 accumulates dGout @ Wg with the input projection
gradient in FP32. Constant accumulator indices avoid local array traffic.
The original normalization-statistics saves and forward path are retained.
The D64 training path the integration selects (``integrations.trimul_h100``).
"""

from __future__ import annotations

import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm as _H100_BMM
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "d64_train"
WIDTH = 64
HIDDEN = 128  # both directions
K3_TILE = (2, 64, 4, 1)  # BI, BJ, ring slots, accumulator sets (K3Cfg: 128-token tiles, resident weights)


def supports(width, length) -> bool:
    """Qualified cells (tuple membership works for symbolic sizes under Dynamo)."""
    return width == WIDTH and ((length % 128 == 0 and length >= 128)
                                         if T.cuda_training_aux_enabled() else length in (384, 768))


# --------------------------------------------------------------------------- build
def _flags(include, defines=()):
    return ["-std=c++17", "-O3", "-arch=sm_90a", "--cubin", "-lineinfo", "-Xptxas=-v",
            "-I" + str(include), *("-D%s=%s" % kv for kv in defines)]


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
    cubin = T.compile(R / "save_k3_d64.cu", _flags(T._upstream() / "csrc", defines.items()))
    k = T.load_unit(str(cubin), "d64_save_k3").kernel("save_k3")
    smem = _k3_smem(tile)
    k.set_max_dynamic_smem(smem)
    return k, smem


B1W_SMEM, B2W_SMEM = 211968, 232000  # == b1::SMEM, b2::SMEM in d64_bwd_wg_gatepipe.cu / d64_bwd_wg_b7gate.cu
B1W_SLOT, B2W_SLOT = 8192 + 256 + 4096, 32768 + 128
FINW_OUT = B2W_SLOT + B1W_SLOT


@T.device_cache
def _bwd_wg(prof=False, master=False, gatepipe=True):
    defines = [("MASTER_FP32", int(master))]
    if prof:
        defines.append(("D64_PROF", 1))
    # The short bidirectional cell has no consistent F+B gain.
    source = "d64_bwd_wg_gatepipe.cu" if gatepipe else "d64_bwd_wg_b7gate.cu"
    cubin = T.compile(R / source, _flags(R, defines))
    unit = T.load_unit(str(cubin), "d64_bwd_wg")
    b1, b2, fin = (unit.kernel(n) for n in ("d64_b1w_b7gate", ("d64_b2w_gatepipe" if gatepipe else "d64_b2w_b7gate"), "d64_finw_b7gate"))
    b1.set_max_dynamic_smem(B1W_SMEM)
    b2.set_max_dynamic_smem(B2W_SMEM)
    return b1, b2, fin


@T.device_cache
def _sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@T.device_cache
def _k1():
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as K
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import _kernels

    return _kernels(), K


@T.device_cache
def _k1_cache():
    """Launch templates / descriptors of the packaged K1 (pointers are re-patched each call)."""
    return {}


# --------------------------------------------------------------------------- forward
# Batched samples (x [B, n, n, 64], B > 1: MiniWorld's templates): tokens b-major, M = B n n. K1 / K3 take the samples stacked along the slow
# token axis (rows = B n) in one launch each, the planes are [c, B n, n] so the contraction batches over c x B, the pair mask is [B n, n] and
# the row-dropout scale [B n, 64] (K3 / B1 read row s n + j for a token of sample s). B = 1 is the single-sample path unchanged.
def _map(t, box, dims, strides):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle="128B", l2="128B")


def _front(x, w1, maskf, gi, bi, ab):
    kk, K = _k1()
    bsz, n = x.shape[0], x.shape[1]
    cfg = K.lookup("sm_90a", WIDTH, HIDDEN, "b", N=n)["k1"]
    kk.k1(x[0] if bsz == 1 else x.view(bsz * n, n, WIDTH), maskf, {"w1": w1, "ln_in_w": gi, "ln_in_b": bi}, ab, N=n, Np=n,
          cz=WIDTH, ch=HIDDEN, cfg=cfg, lnm=2, eps=1e-5, cache=_k1_cache(), rows=bsz * n)


def _bmm(a, b, *, out):
    return _H100_BMM.bmm(a, b, out=out)


def fold(t):
    """[c, B n, n] -> [c B, n, n] (a view): the samples join the channel planes in cuBLAS's batch dim."""
    return t.view(-1, t.shape[-1], t.shape[-1])


def _triangle(ab, tri):
    h = WIDTH
    _bmm(fold(ab[:h]), fold(ab[2 * h:3 * h]).transpose(-1, -2), out=fold(tri[:h]))
    _bmm(fold(ab[h:2 * h]).transpose(-1, -2), fold(ab[3 * h:]), out=fold(tri[h:]))


def _output(x, tri, wg, wp, gi, bi, go, bo, ds, y, stats):
    L = T._launch_module()
    bsz, n = x.shape[0], x.shape[1]
    rows = bsz * n
    m = rows * n
    tile = K3_TILE
    bi_, bj, _, _ = tile
    k, smem = _k3(tile)
    tj = (n + bj - 1) // bj
    tiles = ((rows + bi_ - 1) // bi_) * tj
    ymap = _map(y, [64, 16, 1], [WIDTH, n, rows], [WIDTH * 2, n * WIDTH * 2])
    base = L.Struct.fixed("h100_d64_gatepipe_selected_v2:k3base", [
        _map(x, [64, bj, bi_], [WIDTH, n, rows], [WIDTH * 2, n * WIDTH * 2]),
        _map(tri, [64, 1, 64], [n, rows, HIDDEN], [n * 2, m * 2]),
        _map(wg, [64, 32], [WIDTH, WIDTH], [WIDTH * 2]),
        _map(wp, [64, 32], [HIDDEN, WIDTH], [HIDDEN * 2]),
        ymap, gi, bi, go, bo, x, y, None, n, n, tj, tiles, 1, 0, 1e-5, 0])
    params = L.Struct.fixed("h100_d64_gatepipe_selected_v3:k3", [
        base, ds, ymap, ymap, None, None, stats[0], stats[1], stats[2], stats[3], ymap, ymap,
        None, None, 0 if bsz == 1 else rows])
    k.launch((min(tiles, _sms()), 1, 1), (384, 1, 1), [params], smem)


def _prepare(leaves, mask):
    x, wl, wlg, wr, wrg = leaves[:5]
    bsz, n = x.shape[0], x.shape[1]
    w1 = x.new_empty((4 * HIDDEN, WIDTH))
    if T.cuda_training_aux_enabled():
        from ._h100_pack import pack_into_cuda, training_pair_mask
        pack_into_cuda(w1, wl, wlg, wr, wrg)
        return w1, training_pair_mask(mask, n, x.device, bsz)
    T.pack_into(w1, wl, wlg, wr, wrg)
    # An owned FP32 copy (also for FP32 callers: a saved output must not alias an input);
    # K1 reads it now and the backward reuses it.
    return w1, mask.reshape(n, n).to(torch.float32, copy=True)


def _run_forward(leaves, mask, ds, save):
    prepared = len(leaves) == 12
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    bsz, n = x.shape[0], x.shape[1]
    m = bsz * n * n
    w1, maskf = (leaves[11], mask) if prepared else _prepare(leaves, mask)
    ab = x.new_empty((2 * HIDDEN, bsz * n, n))
    tri = x.new_empty((HIDDEN, bsz * n, n))
    stats = torch.empty((4, m), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    _front(x, w1, maskf, gi, bi, ab)
    _triangle(ab, tri)
    _output(x, tri, wg, wp, gi, bi, go, bo, ds.reshape(bsz * n, WIDTH), y, stats)
    if not save:
        return y
    return [y, ab, tri, stats] if prepared else [y, ab, tri, stats, w1, maskf]


def _forward_fake(leaves, mask, ds):
    """Output plus saved ab, tri, LN statistics [mean_in, rstd_in, mean_out, rstd_out], weight pack, FP32 mask."""
    x = leaves[0]
    bsz, n = x.shape[0], x.shape[1]
    r = bsz * n
    if len(leaves) == 12:
        return [torch.empty_like(x), x.new_empty((2*HIDDEN,r,n)), x.new_empty((HIDDEN,r,n)),
                x.new_empty((4,r*n),dtype=torch.float32)]
    return [torch.empty_like(x), x.new_empty((2 * HIDDEN, r, n)), x.new_empty((HIDDEN, r, n)),
            x.new_empty((4, r * n), dtype=torch.float32), x.new_empty((4 * HIDDEN, WIDTH)),
            x.new_empty((r, n), dtype=torch.float32)]


@opaque(fake=_forward_fake, name="trimul_h100_gatepipe_selected_v2_d64_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> list[torch.Tensor]:
    """Training forward; returns ``[y, ab, tri, stats, w1]`` (all freshly allocated)."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, True)


def _forward_nograd_fake(leaves, mask, ds):
    """Output only."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_gatepipe_selected_v2_d64_dropout_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
    """Training-mode (dropout) forward without autograd: no saved activations are returned."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, False)


# --------------------------------------------------------------------------- backward
PROF = None  # development: a [2, 16] int64 CUDA tensor enables the -DD64_PROF build and receives per-phase cycles


def _map2(t, box, dims, strides, swizzle="128B"):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle=swizzle, l2="128B")


def _run_backward(leaves, mask, ds, saved, dy, master=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    ab, tri, stats, w1, maskf = saved
    L = T._launch_module()
    bsz, n = x.shape[0], x.shape[1]
    m = bsz * n * n                     # tokens b-major: B1 / B2 run over every sample's rows, the weight partials sum them
    h = WIDTH
    dy = dy.contiguous()
    b1, b2, fin = _bwd_wg(PROF is not None, master, gatepipe=n >= 256)
    sms = _sms()
    ntiles = m // 64
    g1 = min(sms, (ntiles + 1) // 2)
    g2 = min(sms, ntiles)
    planes = lambda t: _map2(t, [64, 64], [m, HIDDEN], [m * 2])
    rows = lambda t: _map2(t, [64, 64], [h, m], [h * 2])
    dt = torch.empty_like(tri)
    dgate = torch.empty_like(x)
    # one folded partial slot per B1 CTA (dWp | dgo | dbo | dWg)
    p1 = torch.empty((g1, B1W_SLOT), device=x.device, dtype=torch.float32)
    params = L.Struct.fixed("h100_d64_gatepipe_selected_v2:b1w", [
        planes(tri), rows(x), rows(dy), _map2(ds.reshape(bsz * n, h).contiguous(), [64, 64], [h, bsz * n], [h * 2]),
        _map2(stats, [64, 4], [m, 4], [m * 4], "none"), planes(dt), rows(dgate),
        wp, wg, go, bo, gi, bi, p1, ntiles, n, None if PROF is None else PROF[0]])
    b1.launch((g1, 1, 1), (256, 1, 1), [params], B1W_SMEM)
    dl = torch.empty_like(tri)
    dr = torch.empty_like(tri)
    _bmm(fold(dt[:h]), fold(ab[2 * h:3 * h]), out=fold(dl[:h]))
    _bmm(fold(dt[:h]).transpose(-1, -2), fold(ab[:h]), out=fold(dr[:h]))
    _bmm(fold(ab[3 * h:]), fold(dt[h:]).transpose(-1, -2), out=fold(dl[h:]))
    _bmm(fold(ab[h:2 * h]), fold(dt[h:]), out=fold(dr[h:]))
    dx = torch.empty_like(x)
    p2 = torch.empty((g2, B2W_SLOT), device=x.device, dtype=torch.float32)
    maskf = maskf.reshape(m)
    params = L.Struct.fixed("h100_d64_gatepipe_selected_v2:b2w", [
        planes(dl), planes(dr), rows(x), rows(dgate), rows(dy),
        _map2(stats, [64, 2], [m, 4], [m * 4], "none"),
        _map2(maskf, [64, 1], [m, 1], [m * 4], "none"), rows(dx),
        w1, wg, gi, bi, p2, ntiles, None if PROF is None else PROF[1]])
    b2.launch((g2, 1, 1), (256, 1, 1), [params], B2W_SMEM)
    grads = [torch.empty_like(w, dtype=torch.float32 if master else w.dtype) for w in (wl, wlg, wr, wrg, wg, wp)]
    f32 = dict(device=x.device, dtype=torch.float32)
    dgi, dbi = torch.empty(h, **f32), torch.empty(h, **f32)
    dgo, dbo = torch.empty(2 * h, **f32), torch.empty(2 * h, **f32)
    fin.launch(((FINW_OUT + 255) // 256, 1, 1), (256, 1, 1),
               [p1, g1, p2, g2, *grads, dgi, dbi, dgo, dbo])
    return [dx, *grads, dgi, dbi, dgo, dbo]


def _backward_fake(leaves, mask, ds, saved, dy, master=False):
    """One gradient per differentiable leaf, same shape and dtype."""
    return [torch.empty_like(t, dtype=torch.float32 if master and 1 <= i <= 6 else t.dtype) for i, t in enumerate(leaves)]


@opaque(fake=_backward_fake, name="trimul_h100_gatepipe_selected_v2_d64_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
             saved: list[torch.Tensor], dy: torch.Tensor, master: bool = False) -> list[torch.Tensor]:
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    with T.native_context(leaves[0].device):
        return _run_backward(leaves, mask, ds, saved, dy, master)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        ctx.master = any(w.dtype == torch.float32 for w in leaves[1:7])
        if T.cuda_training_fused_prepare_enabled():
            from ._h100_pack import fused_training_prepare
            packed, maskf, gate, output = fused_training_prepare(leaves, args[11])
            weights = [*leaves[1:5], gate, output]
            prepared = [leaves[0], *weights, *leaves[7:], packed]
            y, ab, tri, stats = forward(prepared, maskf, args[12])
            saved = [ab, tri, stats, packed, maskf]
            # Preserve per-forward lifetime and the backward saved-tensor contract.
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
        y, *saved = forward(leaves, mask, ds)
        ctx.saved_count = len(saved)
        ctx.save_for_backward(*args, *saved, *weights)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        leaves = [vals[0], *vals[-6:], *vals[7:11]]
        saved = list(vals[13:13+ctx.saved_count])
        grads = backward(leaves, vals[11], vals[12], saved, dy, ctx.master)
        grads = [g.to(t.dtype) for g, t in zip(grads, vals[:11], strict=True)]
        return (*grads, None, None)


def bidirectional_trimul(*args):
    """(x [B, n, n, 64], wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, mask, dropscale [B n, D]) -> y; mask: a token mask [B, n] / pair
    mask [B, n, n] or empty (the native preparation builds the FP32 pair mask [B n, n]). B > 1: samples b-major (see ``_front``)."""
    if not torch.is_grad_enabled():
        leaves = list(args[:11])
        if T.cuda_training_fused_prepare_enabled():
            from ._h100_pack import fused_training_prepare
            packed, maskf, gate, output = fused_training_prepare(leaves, args[11])
            prepared = [leaves[0], *leaves[1:5], gate, output, *leaves[7:], packed]
            return forward_nograd(prepared, maskf, args[12])
        if T.cuda_training_aux_enabled():
            from ._h100_pack import cast_weights
            leaves = [leaves[0], *cast_weights(leaves[1:7]), *leaves[7:]]
        elif any(w.dtype != leaves[0].dtype for w in leaves[1:7]):
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]]
            torch._foreach_copy_(weights, leaves[1:7])
            leaves = [leaves[0], *weights, *leaves[7:]]
        return forward_nograd(leaves, *args[11:])
    return _Training.apply(*args)
