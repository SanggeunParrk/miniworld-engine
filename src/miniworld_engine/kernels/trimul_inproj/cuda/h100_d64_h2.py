"""Bidirectional TriMul training at pair width 64 with hidden width 128 per direction (contraction channels H = 256), H100.

The module ``BidirectionalTriangleMultiplication(64, d_hidden=128)`` (AF3 / Protenix-style pair width, a hidden width twice the
pair width in each direction): K1 writes 8 D planes, K3 normalises a 4 D = 256-channel contraction output and projects it back to D.

Forward: the D64 front K1 (packaged unit, c_z 64, 256 channels), two cuBLAS contractions, the D64 K3 with the training saves at
``MW_HIDDEN = 256`` (``save_k3_d64.cu``).  Backward (``d64_h2_bwd.cu``): the registers of the hidden-128 backward of
``h100_d64_b200gate`` hold a whole 128-channel row, so the 256 channels run as two halves and only the output LayerNorm couples
them: ``b1a`` (whole row: dp, dgl, the LayerNorm row sums), ``b1b`` per half (dt, dWp, dgo, dbo), four cuBLAS gradient
contractions, ``b2w`` per half in two passes (the second adds the first's dxn and does the input-LayerNorm backward), ``fin``.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm as _H100_BMM
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "d64_train"
WIDTH = 64
HIDDEN = 256  # contraction channels: 128 outgoing | 128 incoming
HALF = HIDDEN // 2
K3_TILE = (2, 64, 4, 1)  # BI, BJ, ring slots, accumulator sets (K3Cfg: 128-token tiles, resident weights)

B1A_SMEM, B1B_SMEM, B2_SMEM = 177152, 124928, 232000  # == ha::SMEM, hb::SMEM, b2::SMEM in d64_h2_bwd.cu
B1A_SLOT, B1B_SLOT, B2_SLOT = 4096, 8448, 32768 + 128
FIN_OUT = 65536 + 128 + 4096 + 16384 + 512


def supports(width, length) -> bool:
    """Training lengths: a multiple of 128 (the B1 / B2 tiles of 64 tokens must not straddle a pair row)."""
    return width == WIDTH and length % 128 == 0 and length >= 128


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
    k = T.load_unit(str(cubin), "d64h2_save_k3").kernel("save_k3")
    smem = _k3_smem(tile)
    k.set_max_dynamic_smem(smem)
    return k, smem


@T.device_cache
def _bwd(master=False):
    """(b1a, b1b, b2 pass 0, b2 pass 1, fin) of ``d64_h2_bwd.cu``."""
    base = [("MASTER_FP32", int(master))]
    a = T.load_unit(str(T.compile(R / "d64_h2_bwd.cu", _flags(R, [*base, ("PASS", 0)]))), "d64h2_bwd_p0")
    b = T.load_unit(str(T.compile(R / "d64_h2_bwd.cu", _flags(R, [*base, ("PASS", 1)]))), "d64h2_bwd_p1")
    b1a, b1b, b2p0, fin = (a.kernel(n) for n in ("d64h2_b1a", "d64h2_b1b", "d64h2_b2w", "d64h2_fin"))
    b2p1 = b.kernel("d64h2_b2w")
    b1a.set_max_dynamic_smem(B1A_SMEM)
    b1b.set_max_dynamic_smem(B1B_SMEM)
    b2p0.set_max_dynamic_smem(B2_SMEM)
    b2p1.set_max_dynamic_smem(B2_SMEM)
    return b1a, b1b, b2p0, b2p1, fin


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
def _map(t, box, dims, strides):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="128B")


def _map2(t, box, dims, strides, swizzle="128B"):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides, swizzle=swizzle, l2="128B")


def fold(t):
    """[c, B n, n] -> [c B, n, n] (a view): the samples join the channel planes in cuBLAS's batch dim."""
    return t.view(-1, t.shape[-1], t.shape[-1])


def _front(x, w1, maskf, gi, bi, ab):
    kk, K = _k1()
    bsz, n = x.shape[0], x.shape[1]
    cfg = K.lookup("sm_90a", WIDTH, HIDDEN, "b", N=n)["k1"]
    kk.k1(x[0] if bsz == 1 else x.view(bsz * n, n, WIDTH), maskf, {"w1": w1, "ln_in_w": gi, "ln_in_b": bi}, ab, N=n, Np=n,
          cz=WIDTH, ch=HIDDEN, cfg=cfg, lnm=2, eps=1e-5, cache=_k1_cache(), rows=bsz * n)


def _triangle(ab, tri):
    """[a_out | a_in | b_out | b_in] (128 channels each): outgoing on the first 128 channels, incoming on the rest."""
    h = HALF
    _H100_BMM.bmm(fold(ab[:h]), fold(ab[2 * h:3 * h]).transpose(-1, -2), out=fold(tri[:h]))
    _H100_BMM.bmm(fold(ab[h:2 * h]).transpose(-1, -2), fold(ab[3 * h:]), out=fold(tri[h:]))


def _contraction_grads(ab, dt, dl, dr):
    """dl / dr (the gradients of the a / b planes) from dt, pairing the planes as ``_triangle``."""
    h = HALF
    _H100_BMM.bmm(fold(dt[:h]), fold(ab[2 * h:3 * h]), out=fold(dl[:h]))
    _H100_BMM.bmm(fold(dt[:h]).transpose(-1, -2), fold(ab[:h]), out=fold(dr[:h]))
    _H100_BMM.bmm(fold(ab[3 * h:]), fold(dt[h:]).transpose(-1, -2), out=fold(dl[h:]))
    _H100_BMM.bmm(fold(ab[h:2 * h]), fold(dt[h:]), out=fold(dr[h:]))


def _output(x, tri, wg, wp, gi, bi, go, bo, ds, y, stats):
    L = T._launch_module()
    bsz, n = x.shape[0], x.shape[1]
    rows = bsz * n
    m = rows * n
    bi_, bj, _, _ = K3_TILE
    k, smem = _k3(K3_TILE)
    tj = (n + bj - 1) // bj
    tiles = ((rows + bi_ - 1) // bi_) * tj
    ymap = _map(y, [64, 16, 1], [WIDTH, n, rows], [WIDTH * 2, n * WIDTH * 2])
    base = L.Struct.fixed("h100_d64h2_k3base", [
        _map(x, [64, bj, bi_], [WIDTH, n, rows], [WIDTH * 2, n * WIDTH * 2]),
        _map(tri, [64, 1, 64], [n, rows, HIDDEN], [n * 2, m * 2]),
        _map(wg, [64, 32], [WIDTH, WIDTH], [WIDTH * 2]),
        _map(wp, [64, 32], [HIDDEN, WIDTH], [HIDDEN * 2]),
        ymap, gi, bi, go, bo, x, y, None, n, n, tj, tiles, 1, 0, 1e-5, 0])
    params = L.Struct.fixed("h100_d64h2_k3", [
        base, ds, ymap, ymap, None, None, stats[0], stats[1], stats[2], stats[3], ymap, ymap,
        None, None, 0 if bsz == 1 else rows])
    k.launch((min(tiles, _sms()), 1, 1), (384, 1, 1), [params], smem)


def _prepare(leaves, mask):
    from ._h100_pack import cast_weights, pack_into_cuda, training_pair_mask
    x, wl, wlg, wr, wrg = leaves[:5]
    bsz, n = x.shape[0], x.shape[1]
    w1 = x.new_empty((4 * HIDDEN, WIDTH))
    pack_into_cuda(w1, wl, wlg, wr, wrg)
    return w1, training_pair_mask(mask, n, x.device, bsz)


def _run_forward(leaves, mask, ds, save):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    bsz, n = x.shape[0], x.shape[1]
    m = bsz * n * n
    w1, maskf = _prepare(leaves, mask)
    ab = x.new_empty((2 * HIDDEN, bsz * n, n))
    tri = x.new_empty((HIDDEN, bsz * n, n))
    stats = torch.empty((4, m), device=x.device, dtype=torch.float32)
    y = T.take_output(x)
    _front(x, w1, maskf, gi, bi, ab)
    _triangle(ab, tri)
    _output(x, tri, wg, wp, gi, bi, go, bo, ds.reshape(bsz * n, WIDTH), y, stats)
    if not save:
        return y
    return [y, ab, tri, stats, w1, maskf]


def _forward_fake(leaves, mask, ds):
    """Output plus saved ab, tri, LN statistics [mean_in, rstd_in, mean_out, rstd_out], weight pack, FP32 mask."""
    x = leaves[0]
    bsz, n = x.shape[0], x.shape[1]
    r = bsz * n
    return [torch.empty_like(x), x.new_empty((2 * HIDDEN, r, n)), x.new_empty((HIDDEN, r, n)),
            x.new_empty((4, r * n), dtype=torch.float32), x.new_empty((4 * HIDDEN, WIDTH)),
            x.new_empty((r, n), dtype=torch.float32)]


@opaque(fake=_forward_fake, name="trimul_h100_d64h2_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> list[torch.Tensor]:
    """Training forward; returns ``[y, ab, tri, stats, w1, maskf]`` (all freshly allocated)."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, True)


def _forward_nograd_fake(leaves, mask, ds):
    """y like x (the no-grad dropout forward saves nothing)."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_d64h2_dropout_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
    """Training-mode (dropout) forward without autograd: no saved activations are returned."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, False)


# --------------------------------------------------------------------------- backward
def _run_backward(leaves, mask, ds, saved, dy, master=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    ab, tri, stats, w1, maskf = saved
    L = T._launch_module()
    bsz, n = x.shape[0], x.shape[1]
    m = bsz * n * n                     # tokens b-major: every stage runs over every sample's rows, the weight partials sum them
    h = WIDTH
    dy = dy.contiguous()
    b1a, b1b, b2p0, b2p1, fin = _bwd(master)
    sms = _sms()
    ntiles = m // 64
    g1 = min(sms, (ntiles + 1) // 2)
    g2 = min(sms, ntiles)
    f32 = dict(device=x.device, dtype=torch.float32)
    planes = lambda t: _map2(t, [64, 64], [m, HIDDEN], [m * 2])
    rows = lambda t: _map2(t, [64, 64], [h, m], [h * 2])
    dt = torch.empty_like(tri)
    dp = torch.empty_like(x)
    dgate = torch.empty_like(x)
    s12 = torch.empty((2, m), **f32)
    pa = torch.empty((g1, B1A_SLOT), **f32)
    pb = [torch.empty((g1, B1B_SLOT), **f32) for _ in range(2)]
    stats_all = _map2(stats, [64, 4], [m, 4], [m * 4], "none")
    # whole row: dp, dgl, the output-LayerNorm row sums
    params = L.Struct.fixed("h100_d64h2:b1a", [
        planes(tri), rows(x), rows(dy), _map2(ds.reshape(bsz * n, h).contiguous(), [64, 64], [h, bsz * n], [h * 2]),
        stats_all, rows(dp), rows(dgate), wp, wg, go, bo, gi, bi, pa, s12, ntiles, n, m])
    b1a.launch((g1, 1, 1), (256, 1, 1), [params], B1A_SMEM)
    # per channel half: dt, dWp, dgo, dbo
    for hh in range(2):
        params = L.Struct.fixed("h100_d64h2:b1b", [
            planes(tri), rows(dp), _map2(stats, [64, 2], [m, 4], [m * 4], "none"),
            _map2(s12, [64, 2], [m, 2], [m * 4], "none"), planes(dt), wp, go, bo, pb[hh], ntiles, HALF * hh])
        b1b.launch((g1, 1, 1), (256, 1, 1), [params], B1B_SMEM)
    dl = torch.empty_like(tri)
    dr = torch.empty_like(tri)
    _contraction_grads(ab, dt, dl, dr)
    dx = T.take_input_grad(x)
    dxn = torch.empty((m, h), **f32)
    maskf = maskf.reshape(m)
    pc = [torch.empty((g2, B2_SLOT), **f32) for _ in range(2)]
    for hh, kernel in enumerate((b2p0, b2p1)):
        params = L.Struct.fixed("h100_d64h2:b2w", [
            planes(dl), planes(dr), rows(x), rows(dgate), rows(dy),
            _map2(stats, [64, 2], [m, 4], [m * 4], "none"),
            _map2(maskf, [64, 1], [m, 1], [m * 4], "none"), rows(dx),
            w1, wg, gi, bi, pc[hh], dxn, ntiles, hh])
        kernel.launch((g2, 1, 1), (256, 1, 1), [params], B2_SMEM)
    grads = [torch.empty_like(w, dtype=torch.float32 if master else w.dtype) for w in (wl, wlg, wr, wrg, wg, wp)]
    dgi, dbi = torch.empty(h, **f32), torch.empty(h, **f32)
    dgo, dbo = torch.empty(HIDDEN, **f32), torch.empty(HIDDEN, **f32)
    fin.launch(((FIN_OUT + 255) // 256, 1, 1), (256, 1, 1),
               [pa, g1, pb[0], pb[1], g1, pc[0], pc[1], g2, *grads, dgi, dbi, dgo, dbo])
    return [dx, *grads, dgi, dbi, dgo, dbo]


def _backward_fake(leaves, mask, ds, saved, dy, master=False):
    """One gradient per differentiable leaf, same shape and dtype."""
    return [torch.empty_like(t, dtype=torch.float32 if master and 1 <= i <= 6 else t.dtype) for i, t in enumerate(leaves)]


@opaque(fake=_backward_fake, name="trimul_h100_d64h2_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
             saved: list[torch.Tensor], dy: torch.Tensor, master: bool = False) -> list[torch.Tensor]:
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    with T.native_context(leaves[0].device):
        return _run_backward(leaves, mask, ds, saved, dy, master)


def _cast(leaves):
    from ._h100_pack import cast_weights
    return [leaves[0], *cast_weights(leaves[1:7]), *leaves[7:11]]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        ctx.master = any(w.dtype == torch.float32 for w in leaves[1:7])
        leaves = _cast(leaves)
        mask, ds = args[11:]
        y, *saved = forward(leaves, mask, ds)
        ctx.saved_count = len(saved)
        ctx.save_for_backward(*args, *saved, *leaves[1:7])
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        leaves = [vals[0], *vals[-6:], *vals[7:11]]
        saved = list(vals[13:13 + ctx.saved_count])
        grads = backward(leaves, vals[11], vals[12], saved, dy, ctx.master)
        grads = [g.to(t.dtype) for g, t in zip(grads, vals[:11], strict=True)]
        return (*grads, None, None)


def bidirectional_trimul(*args):
    """(x [B, n, n, 64], wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, mask, dropscale [B n, 64]) -> y; mask: a token mask [B, n] / pair mask
    [B, n, n] or empty; B > 1: samples b-major (see ``_front``).  Hidden width 128 per direction (wl.shape[0] == 256)."""
    if not torch.is_grad_enabled():
        return forward_nograd(_cast(list(args[:11])), *args[11:])
    return _Training.apply(*args)
