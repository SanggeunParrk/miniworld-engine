"""H100 bidirectional TriMul training at pair width D=64 (hidden 64 per direction, H=128).

Forward (4 launches + weight pack):
  K1   the packaged D64 inference K1 (Anthropic-derived; LN_in + four gated projections + mask)
       -> ab [256, n, n] channel-major (left | right, each [outgoing 64 | incoming 64]);
  2x   cuBLAS batched contractions -> tri [128, n, n];
  K3   ``save_k3_d64.cu`` (the D128 training K3 at CZ=64): LN_out, output projection, gate,
       dropout scale and residual in one epilogue; saves LN_in / LN_out mean and rstd.
Saved: ab, tri, the four LN statistics rows [4, n*n] fp32, the per-forward weight pack and an
FP32 copy of the pair mask.

Backward (7 launches, ``h100_sources/d64_train/d64_bwd_wg.cu``, wgmma + TMA, persistent CTAs):
  d64_b1w  output side (two independent warpgroups per CTA): LN_out / projection / gate
           recompute with the LN affines folded into the resident weights, dy -> dProj and the
           gate-logit gradient, output-LN backward -> dt [128, n, n], the whole output-gate
           backward (dxg = dgl Wg -> [n*n, 64] bf16); dW_out, dW_gate and output-LN affine
           gradients from kernel-long register sums (G = dProj^T nhat, Gg = dgl^T xhat, row sums);
  4x       cuBLAS batched contraction gradients -> dl, dr [128, n, n];
  d64_b2w  input side (warpgroup per direction): input projections recomputed in half chunks
           (the next products in flight during the gate/mask derivative stage), dx_n = dpre W1
           + dxg, dW1 = dpre^T x_n in registers, input-LN backward + identity residual -> dx;
  d64_finw fixed-order reduction of the per-CTA partials into the parameter gradients.
Nothing is cached across calls except compiled kernels, tensor-map descriptors keyed by address
and launch templates: every call owns its activations and weight pack.
B=1, BF16 pair and projection weights, FP32 LayerNorm affine, LN eps 1e-5, L in (384, 768).
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "d64_train"
WIDTH = 64
HIDDEN = 128  # both directions
K3_TILE = (2, 64, 4, 1)  # BI, BJ, ring slots, accumulator sets (K3Cfg: 128-token tiles, resident weights)


def supports(width, length) -> bool:
    """Qualified cells (tuple membership works for symbolic sizes under Dynamo)."""
    return width == WIDTH and length in (384, 768)


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


B1W_SMEM, B2W_SMEM = 220160, 227904  # == b1::SMEM, b2::SMEM in d64_bwd_wg.cu
B1W_SLOT, B2W_SLOT = 8192 + 256 + 4096, 32768 + 128
FINW_OUT = B2W_SLOT + B1W_SLOT


@T.device_cache
def _bwd_wg(prof=False):
    cubin = T.compile(R / "d64_bwd_wg.cu", _flags(R, [("D64_PROF", 1)] if prof else ()))
    unit = T.load_unit(str(cubin), "d64_bwd_wg")
    b1, b2, fin = (unit.kernel(n) for n in ("d64_b1w", "d64_b2w", "d64_finw"))
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
def _map(t, box, dims, strides):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle="128B", l2="128B")


def _front(x, w1, maskf, gi, bi, ab):
    kk, K = _k1()
    n = x.shape[1]
    cfg = K.lookup("sm_90a", WIDTH, HIDDEN, "b", N=n)["k1"]
    kk.k1(x[0], maskf, {"w1": w1, "ln_in_w": gi, "ln_in_b": bi}, ab, N=n, Np=n, cz=WIDTH,
          ch=HIDDEN, cfg=cfg, lnm=2, eps=1e-5, cache=_k1_cache())


def _triangle(ab, tri):
    h = WIDTH
    torch.bmm(ab[:h], ab[2 * h:3 * h].transpose(-1, -2), out=tri[:h])
    torch.bmm(ab[h:2 * h].transpose(-1, -2), ab[3 * h:], out=tri[h:])


def _output(x, tri, wg, wp, gi, bi, go, bo, ds, y, stats):
    L = T._launch_module()
    n = x.shape[1]
    m = n * n
    tile = K3_TILE
    bi_, bj, _, _ = tile
    k, smem = _k3(tile)
    tj = (n + bj - 1) // bj
    tiles = ((n + bi_ - 1) // bi_) * tj
    ymap = _map(y, [64, 16, 1], [WIDTH, n, n], [WIDTH * 2, n * WIDTH * 2])
    base = L.Struct.fixed("h100_d64:k3base", [
        _map(x, [64, bj, bi_], [WIDTH, n, n], [WIDTH * 2, n * WIDTH * 2]),
        _map(tri, [64, 1, 64], [n, n, HIDDEN], [n * 2, m * 2]),
        _map(wg, [64, 32], [WIDTH, WIDTH], [WIDTH * 2]),
        _map(wp, [64, 32], [HIDDEN, WIDTH], [HIDDEN * 2]),
        ymap, gi, bi, go, bo, x, y, None, n, n, tj, tiles, 1, 0, 1e-5, 0])
    params = L.Struct.fixed("h100_d64:k3", [
        base, ds, ymap, ymap, None, None, stats[0], stats[1], stats[2], stats[3], ymap, ymap,
        None, None])
    k.launch((min(tiles, _sms()), 1, 1), (384, 1, 1), [params], smem)


def _prepare(leaves, mask):
    x, wl, wlg, wr, wrg = leaves[:5]
    n = x.shape[1]
    w1 = x.new_empty((4 * HIDDEN, WIDTH))
    T.pack_into(w1, wl, wlg, wr, wrg)
    # An owned FP32 copy (also for FP32 callers: a saved output must not alias an input);
    # K1 reads it now and the backward reuses it.
    return w1, mask.reshape(n, n).to(torch.float32, copy=True)


def _run_forward(leaves, mask, ds, save):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n = x.shape[1]
    m = n * n
    w1, maskf = _prepare(leaves, mask)
    ab = x.new_empty((2 * HIDDEN, n, n))
    tri = x.new_empty((HIDDEN, n, n))
    stats = torch.empty((4, m), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    _front(x, w1, maskf, gi, bi, ab)
    _triangle(ab, tri)
    _output(x, tri, wg, wp, gi, bi, go, bo, ds.reshape(n, WIDTH), y, stats)
    return [y, ab, tri, stats, w1, maskf] if save else y


def _forward_fake(leaves, mask, ds):
    """Output plus saved ab, tri, LN statistics [mean_in, rstd_in, mean_out, rstd_out], weight pack, FP32 mask."""
    x = leaves[0]
    n = x.shape[1]
    return [torch.empty_like(x), x.new_empty((2 * HIDDEN, n, n)), x.new_empty((HIDDEN, n, n)),
            x.new_empty((4, n * n), dtype=torch.float32), x.new_empty((4 * HIDDEN, WIDTH)),
            x.new_empty((n, n), dtype=torch.float32)]


@opaque(fake=_forward_fake, name="trimul_h100_d64_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> list[torch.Tensor]:
    """Training forward; returns ``[y, ab, tri, stats, w1]`` (all freshly allocated)."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, True)


def _forward_nograd_fake(leaves, mask, ds):
    """Output only."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_d64_dropout_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
    """Training-mode (dropout) forward without autograd: no saved activations are returned."""
    with T.native_context(leaves[0].device):
        return _run_forward(leaves, mask, ds, False)


# --------------------------------------------------------------------------- backward
PROF = None  # development: a [2, 16] int64 CUDA tensor enables the -DD64_PROF build and receives per-phase cycles


def _map2(t, box, dims, strides, swizzle="128B"):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle=swizzle, l2="128B")


def _run_backward(leaves, mask, ds, saved, dy):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    ab, tri, stats, w1, maskf = saved
    L = T._launch_module()
    n = x.shape[1]
    m = n * n
    h = WIDTH
    dy = dy.contiguous()
    b1, b2, fin = _bwd_wg(PROF is not None)
    sms = _sms()
    ntiles = m // 64
    g1 = min(sms, (ntiles + 1) // 2)
    g2 = min(sms, ntiles)
    planes = lambda t: _map2(t, [64, 64], [m, HIDDEN], [m * 2])
    rows = lambda t: _map2(t, [64, 64], [h, m], [h * 2])
    dt = torch.empty_like(tri)
    dxg = torch.empty_like(x)
    # one folded partial slot per B1 CTA (dWp | dgo | dbo | dWg)
    p1 = torch.empty((g1, B1W_SLOT), device=x.device, dtype=torch.float32)
    params = L.Struct.fixed("h100_d64:b1w", [
        planes(tri), rows(x), rows(dy), _map2(ds.reshape(n, h).contiguous(), [64, 64], [h, n], [h * 2]),
        _map2(stats, [64, 4], [m, 4], [m * 4], "none"), planes(dt), rows(dxg),
        wp, wg, go, bo, gi, bi, p1, ntiles, n, None if PROF is None else PROF[0]])
    b1.launch((g1, 1, 1), (256, 1, 1), [params], B1W_SMEM)
    dl = torch.empty_like(tri)
    dr = torch.empty_like(tri)
    torch.bmm(dt[:h], ab[2 * h:3 * h], out=dl[:h])
    torch.bmm(dt[:h].transpose(-1, -2), ab[:h], out=dr[:h])
    torch.bmm(ab[3 * h:], dt[h:].transpose(-1, -2), out=dl[h:])
    torch.bmm(ab[h:2 * h], dt[h:], out=dr[h:])
    dx = torch.empty_like(x)
    p2 = torch.empty((g2, B2W_SLOT), device=x.device, dtype=torch.float32)
    maskf = maskf.reshape(m)
    params = L.Struct.fixed("h100_d64:b2w", [
        planes(dl), planes(dr), rows(x), rows(dxg), rows(dy),
        _map2(stats, [64, 2], [m, 4], [m * 4], "none"),
        _map2(maskf, [64, 1], [m, 1], [m * 4], "none"), rows(dx),
        w1, gi, bi, p2, ntiles, None if PROF is None else PROF[1]])
    b2.launch((g2, 1, 1), (256, 1, 1), [params], B2W_SMEM)
    grads = [torch.empty_like(w) for w in (wl, wlg, wr, wrg, wg, wp)]
    f32 = dict(device=x.device, dtype=torch.float32)
    dgi, dbi = torch.empty(h, **f32), torch.empty(h, **f32)
    dgo, dbo = torch.empty(2 * h, **f32), torch.empty(2 * h, **f32)
    fin.launch(((FINW_OUT + 255) // 256, 1, 1), (256, 1, 1),
               [p1, g1, p2, g2, *grads, dgi, dbi, dgo, dbo])
    return [dx, *grads, dgi, dbi, dgo, dbo]


def _backward_fake(leaves, mask, ds, saved, dy):
    """One gradient per differentiable leaf, same shape and dtype."""
    return [torch.empty_like(t) for t in leaves]


@opaque(fake=_backward_fake, name="trimul_h100_d64_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor,
             saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    with T.native_context(leaves[0].device):
        return _run_backward(leaves, mask, ds, saved, dy)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        mask, ds = args[11:]
        y, *saved = forward(leaves, mask, ds)
        ctx.save_for_backward(*args, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = backward(list(vals[:11]), vals[11], vals[12], list(vals[13:]), dy)
        return (*grads, None, None)


def bidirectional_trimul(*args):
    """(x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, pair_mask[n,n], dropscale[n,D]) -> y."""
    if not torch.is_grad_enabled():
        return forward_nograd(list(args[:11]), *args[11:])
    return _Training.apply(*args)
