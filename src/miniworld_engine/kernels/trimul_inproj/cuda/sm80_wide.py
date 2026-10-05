"""A100 (sm_80) hand-CUDA TriMul at every width: D = d_pair = 64 / 128 / 256 / 384, one direction (hidden D) or bidirectional (hidden 2 D), bf16 and fp32 (TF32 tensor cores).

The width-generic sibling of ``sm80.py`` (whose fused kernels are literals for D128): the data flow of the B200 wide path (``b200_infer.py``) on ``mma.sync``.

* forward: ``stats_rows`` (LayerNorm_in statistics of every token) -> ``k1w`` (the gated projections of the RAW input rows: LayerNorm_in is folded into the weights,
  the GEMM is a plain ``cp.async`` stream, the epilogue applies ``r (acc - mu s) + b``, the sigmoid gate and the pair mask and stores the planes channel-major) ->
  cuBLAS contraction -> ``stats_cm`` (LayerNorm_out statistics of every token, from the channel-major contraction output) -> ``k3w`` (output projection of the raw
  contraction planes and the gate GEMM of the raw input rows with both LayerNorms folded into the weights; gate, dropout row scale and residual in the epilogue).
  Training saves the planes, the contraction output, both statistics and p' / g' (bf16, the 0.5-scaled units of the fold).  bf16 with a hidden width <= 128 (D64 in both modules, D128
  one direction) computes both statistics inside k1w / k3w from the operand tiles still resident in their rings: the two statistics kernels are not launched.
* ``supports()`` is the whole gate; everything it rejects keeps the existing path.  ``MINIWORLD_TRIMUL_SM80=0`` (shared with the D128 path) or
  ``MINIWORLD_TRIMUL_SM80_WIDE=0`` turns it off.

Numerics follow the D128 path (``sm80.py``): the same bf16 rounding points, the 0.5 folded into the weights (``sigmoid(g) p = p' (1 + tanh g')``).  fp32 operands run the same stages
on TF32 MMAs (``wide_gemm32.cuh``) with fp32 storage, nothing rounded to a narrower type; the folded weights are TF32-rounded, and every cuBLAS GEMM of the fp32 path is
issued with TF32 enabled whatever ``torch.backends.cuda.matmul.allow_tf32`` says (one precision per path; the Triton path follows the global flag).
"""

import contextlib
import functools
import os
import warnings
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from . import sm80 as _d128

_dir = Path(__file__).parent / "sm80"

WIDTHS = (64, 128, 256, 384)
BIDIR, OUTGOING, INCOMING = _d128.BIDIR, _d128.OUTGOING, _d128.INCOMING


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_TRIMUL_SM80_WIDE_FLAGS: extra -D flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_TRIMUL_SM80_WIDE_FLAGS", "").split()
    import hashlib

    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"trimul_sm80_wide{tag}",
        sources=[str(_dir / "wide_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def shape_ok(shape, dtype, d_hidden: int, hs: int | None = None, mask_shape=None, mask_dtype=None) -> bool:
    """The kernels' shape / dtype requirements, device-free (the CPU tests of the gate call this): bf16 or fp32 (TF32), width in WIDTHS with hidden channels ``hs`` = D
    (one direction) or 2 D (bidirectional), a square plane with L % 16 == 0, and a [B, L] bool token mask if any."""
    if dtype not in (torch.bfloat16, torch.float32) or len(shape) != 4 or shape[0] < 1 or shape[1] != shape[2]:
        return False
    d, n = shape[-1], shape[1]
    hs = d_hidden if hs is None else hs
    if d not in WIDTHS or hs not in (d, 2 * d) or n <= 0 or n % 16 != 0:
        return False
    return mask_shape is None or (tuple(mask_shape) == (shape[0], n) and mask_dtype is torch.bool)


def supports(pair: torch.Tensor, d_hidden: int, mask: torch.Tensor | None = None, *, hs: int | None = None) -> bool:
    """The kernels' own requirements: sm_80, ``shape_ok``, one contiguous plane per call (a batch runs plane by plane in the integration)."""
    if os.environ.get("MINIWORLD_TRIMUL_SM80", "1") == "0" or os.environ.get("MINIWORLD_TRIMUL_SM80_WIDE", "1") == "0":
        return False
    if not pair.is_cuda or not pair.is_contiguous():
        return False
    if not shape_ok(tuple(pair.shape), pair.dtype, d_hidden, hs, None if mask is None else tuple(mask.shape), None if mask is None else mask.dtype):
        return False
    return _d128._is_ampere(pair.device.index if pair.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the extension; False, with one warning, when the toolchain fails (the existing path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"sm80 wide TriMul unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def available(pair: torch.Tensor, d_hidden: int, mask: torch.Tensor | None = None, *, hs: int | None = None) -> bool:
    """``supports()`` plus a successful (cached) build."""
    return supports(pair, d_hidden, mask, hs=hs) and _loads()


# ------------------------------------------------------------------------------------------------------------------------------- packs
def _pack(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, *, train: bool = False) -> dict:
    """The folded weight layouts of the forward (and, in training, the backward's unscaled packed weights and the exact LayerNorm_out row-sum vector), one launch
    per call (never cached: a captured CUDA graph must repack after an optimizer step).  The four front matrices are read through their strides."""
    hs, d = wl.shape
    dev = wl.device
    ln = [t.detach().float().contiguous() for t in (gi, bi, go, bo)]
    ws = [*(wl, wlg, wr, wrg), *(w.contiguous() for w in (wg, wo))]
    f32 = dict(dtype=torch.float32, device=dev)
    wd = wl.dtype                                            # the folded weights are stored in the activations' dtype (bf16, or TF32-rounded fp32)
    w1 = torch.empty((4 * hs, d), dtype=wd, device=dev)
    wo3 = torch.empty((d, hs), dtype=wd, device=dev)
    wg3 = torch.empty((d, d), dtype=wd, device=dev)
    vec = torch.empty((2 * 4 * hs + 4 * d + (d if train else 0),), **f32)
    vs, vb = vec[:4 * hs], vec[4 * hs:8 * hs]
    so, eo, sg, eg = vec[8 * hs:8 * hs + 4 * d].view(4, d).unbind(0)
    spx = vec[8 * hs + 4 * d:] if train else vec.new_empty((0,))
    wdx = torch.empty((4 * hs + d, d), dtype=wd, device=dev) if train else w1.new_empty((0,))
    _ext().wpack(*ws, *ln, w1, vs, vb, wo3, so, eo, wg3, sg, eg, wdx, spx)
    pk = dict(hs=hs, d=d, w1=w1, vs=vs, vb=vb, wo=wo3, wg=wg3, so=so, eo=eo, sg=sg, eg=eg, ln=ln)
    if train:
        pk.update(wdx=wdx, spx=spx)
    return pk


def _cast_weights(leaves) -> list:
    """The ten parameters with the six weight matrices in the activations' dtype (strides kept).  fp32 master parameters are cast here, inside the op and
    outside autograd, so no cast copy sits in the graph and the gradients (written in the parameters' own dtype) are never rounded through bf16."""
    dt = leaves[0].dtype
    return [*(w.detach().to(dt) for w in leaves[1:7]), *leaves[7:]]


def _pack_reference(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo) -> dict:
    """``_pack`` in torch operations (the layouts' definition; the tests compare the two)."""
    f = lambda t: t.detach().float()  # noqa: E731
    hs, d = wl.shape
    wg_all = torch.cat([f(wlg), f(wrg)], 0)                                                   # [2 Hs, D]
    wp_all = torch.cat([f(wl), f(wr)], 0)
    rows = torch.stack([wg_all.view(2 * hs // 8, 8, d), wp_all.view(2 * hs // 8, 8, d)], 1).reshape(4 * hs, d)   # 16 j + i: gate (i < 8) | projection
    wd = wl.dtype
    rnd = _tf32_round if wd is torch.float32 else (lambda t: t.to(wd))    # noqa: E731
    w1 = rnd(0.5 * rows * f(gi)[None, :])
    wo3 = rnd(0.5 * f(wo) * f(go)[None, :])
    wg3 = rnd(0.5 * f(wg) * f(gi)[None, :])
    return dict(hs=hs, d=d, w1=w1, vs=w1.float().sum(1), vb=0.5 * (rows @ f(bi)), wo=wo3, wg=wg3, so=wo3.float().sum(1), eo=0.5 * (f(wo) @ f(bo)),
                sg=wg3.float().sum(1), eg=0.5 * (f(wg) @ f(bi)))


def _tf32_round(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> the nearest TF32 value (10-bit mantissa, ties away from zero: ``cvt.rna.tf32.f32``), as an fp32 tensor."""
    return ((t.contiguous().view(torch.int32) + 0x1000) & -0x2000).view(torch.float32)


@contextlib.contextmanager
def _tf32(z: torch.Tensor):
    """fp32 operands: the cuBLAS GEMMs of this path run on the TF32 tensor cores whatever the global matmul flag says (the kernels always do; one precision per path)."""
    if z.dtype is not torch.float32:
        yield
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# ------------------------------------------------------------------------------------------------------------------------------ forward
def _contract(planes, x, hs, n, direction):
    """cuBLAS contraction of the plane pair: channels [0, h) outgoing (sum_k a[i,k] b[j,k]), the rest incoming (sum_k a[k,i] b[k,j])."""
    a, b = planes[:hs].view(hs, n, n), planes[hs:].view(hs, n, n)
    h = hs // 2 if direction == BIDIR else (hs if direction == OUTGOING else 0)
    if h:
        torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h])
    if h < hs:
        torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:])


#: bf16 with a hidden width <= 128 (D64 both modules, D128 one direction): the statistics kernels are replaced by in-kernel two-pass statistics of k1w / k3w (see ``_run_forward``); False keeps the kernels (probe switch for A / B timings).
_FUSE_STATS = True


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """The kernels move 16-byte granules (cp.async, 128-bit loads): a view that starts off a 16-byte boundary (a slice of a larger buffer) is copied once."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _run_forward(leaves, mask, ds, direction, eps_in, eps_out, *, save):
    E = _ext()
    z = leaves[0]
    n, d = z.shape[1], z.shape[-1]
    T = n * n
    zf = _aligned(z.reshape(T, d))
    ds = _aligned(ds) if ds.numel() else ds
    pk = _pack(*_cast_weights(leaves))
    hs = pk["hs"]
    # bf16, hidden width <= 128: k1w computes the input rows' statistics from its own tile and k3w the output stage's LayerNorm_out statistics -- the two statistics kernels and
    # their reads of z / X are gone (every k-tile of both operands is still in its ring slot when its GEMM ends)
    fuse_z = _FUSE_STATS and z.dtype is torch.bfloat16 and hs <= 128           # <= 4 CTAs share a token tile: the redundant per-CTA statistics are cheaper than the kernels
    fuse_x = fuse_z
    st_in = torch.empty((T, 2), dtype=torch.float32, device=z.device)
    if not fuse_z:
        E.stats_rows(zf, st_in, eps_in)
    planes = zf.new_empty((2 * hs, T))
    E.k1w(zf, pk["w1"], pk["vs"], pk["vb"], st_in, mask, planes, n, eps_in, fuse_z)
    x = zf.new_empty((hs, n, n))
    with _tf32(z):
        _contract(planes, x, hs, n, direction)
    st_out = torch.empty((T, 2), dtype=torch.float32, device=z.device)
    if not fuse_x:
        E.stats_cm(x.view(hs, T), st_out, eps_out)
    out = torch.empty_like(zf)
    if save:
        ps, gs = torch.empty_like(zf), torch.empty_like(zf)
    else:
        ps = gs = zf.new_empty((0,))
    E.k3w(x.view(hs, T), zf, pk["wo"], pk["wg"], pk["so"], pk["eo"], pk["sg"], pk["eg"], st_out, st_in, ds, out, ps, gs, n, eps_out, fuse_x)
    return out.view_as(z), ((planes, x, st_in, st_out, ps, gs) if save else None)


def _forward_nograd_fake(leaves, mask, ds, direction, eps_in, eps_out):
    """The output has the pair's shape and dtype; nothing is saved."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_sm80_wide_forward")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int, eps_in: float, eps_out: float) -> torch.Tensor:
    """z + drop_row(trimul(z)) without the backward's saves.  ``mask`` [L] uint8 or empty, ``ds`` [L, D] bf16 or empty."""
    return _run_forward(leaves, mask, ds, direction, eps_in, eps_out, save=False)[0]


def _forward_train_fake(leaves, mask, ds, direction, eps_in, eps_out):
    """[out like the pair, planes [2 Hs, T], X [Hs, L, L], LN_in statistics [T, 2], LN_out statistics [T, 2], p' [T, D], g' [T, D]], Hs the first weight's leading size."""
    z = leaves[0]
    n, d, hs = z.shape[1], z.shape[-1], leaves[1].shape[0]
    T = n * n
    f32 = torch.float32
    return [torch.empty_like(z), z.new_empty((2 * hs, T)), z.new_empty((hs, n, n)), torch.empty((T, 2), dtype=f32, device=z.device),
            torch.empty((T, 2), dtype=f32, device=z.device), z.new_empty((T, d)), z.new_empty((T, d))]


@opaque(fake=_forward_train_fake, name="trimul_sm80_wide_train_fwd")
def forward_train(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int, eps_in: float, eps_out: float) -> list[torch.Tensor]:
    """[out, planes, X, LN_in statistics, LN_out statistics, p', g']."""
    out, (planes, x, st_in, st_out, ps, gs) = _run_forward(leaves, mask, ds, direction, eps_in, eps_out, save=True)
    return [out, planes, x, st_in, st_out, ps, gs]


def trimul_inference(z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, mask, ds, direction, eps_in, eps_out):
    """z + drop_row(trimul(z)), no gradients (probe entry)."""
    leaves = (z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo)
    return forward_nograd(list(leaves), mask, ds, direction, float(eps_in), float(eps_out))


# ----------------------------------------------------------------------------------------------------------------------------- backward
def _gemm_tk(a, b):
    """a [T, M], b [T, N] (or a transposed view) -> a^T b in fp32, K = T split into chunks (see sm80._gemm_tk: cuBLAS runs these long-K shapes unsplit otherwise).
    bf16 operands accumulate in fp32 (``sm80._gemm_tk``); fp32 operands run the same chunked batched GEMM (the caller holds the TF32 context)."""
    if a.dtype is not torch.float32:
        return _d128._gemm_tk(a, b)
    t = a.shape[0]
    s = next((c for c in (32, 16, 8, 4, 2) if t % c == 0 and t // c >= 256), 1)
    if s == 1:
        return a.t() @ b
    return torch.bmm(a.unflatten(0, (s, t // s)).transpose(1, 2), b.unflatten(0, (s, t // s))).sum(0)


def _contract_bwd(planes, dx, dplanes, hs, n, direction):
    """dX [Hs, L, L] -> the plane gradients [2 Hs, L, L] (cuBLAS): X = A B^T: dA = dX B, dB = dX^T A; X = A^T B: dA = B dX^T, dB = A dX."""
    a, b = planes[:hs].view(hs, n, n), planes[hs:].view(hs, n, n)
    da, db = dplanes[:hs], dplanes[hs:]
    h = hs // 2 if direction == BIDIR else (hs if direction == OUTGOING else 0)
    if h:
        torch.bmm(dx[:h], b[:h], out=da[:h])
        torch.bmm(dx[:h].transpose(1, 2), a[:h], out=db[:h])
    if h < hs:
        torch.bmm(b[h:], dx[h:].transpose(1, 2), out=da[h:])
        torch.bmm(a[h:], dx[h:], out=db[h:])


def _run_backward(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    with _tf32(leaves[0]):
        return _run_backward_body(leaves, mask, ds, saved, dy, direction, eps_in, eps_out)


def _run_backward_body(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    E = _ext()
    z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo = leaves
    planes, x, st_in, st_out, ps, gs = saved
    n, d = z.shape[1], z.shape[-1]
    T = n * n
    hs = wl.shape[0]
    dev = z.device
    f32 = dict(dtype=torch.float32, device=dev)
    zf = _aligned(z.reshape(T, d))
    ds = _aligned(ds) if ds.numel() else ds
    cw = _cast_weights(leaves)
    pk = _pack(*cw, train=True)
    gi32, bi32, go32, bo32 = pk["ln"]
    is32 = z.dtype is torch.float32
    dyf = _aligned(dy.reshape(T, d).to(z.dtype).contiguous())
    xcm = x.view(hs, T)
    # ---- output side: gate gradients, LayerNorm_out backward
    nb = E.gate_bwd_blocks(T, d, is32)
    dcat = zf.new_empty((T, 4 * hs + d))                    # [dpre (k1w packed order) | dg of the output gate]
    dpr = torch.empty_like(zf)
    s12 = torch.empty((T, 2), **f32)
    part = torch.empty((nb, 2 * d), **f32)
    E.gate_bwd(dyf, ps, gs, ds, pk["spx"], 2.0 * pk["eo"], st_out, dpr, dcat, 4 * hs, s12, part, n)
    r01 = torch.empty(2 * d, **f32)
    E.colsum(part, r01)
    dor = torch.mm(cw[5].t(), dpr.t())                         # [Hs, T] = r_o dy_o, channel-major like X
    dt = torch.empty_like(xcm)
    part_o = torch.empty((T // (128 if is32 else 256), 2 * hs), **f32)
    E.lnout_bwd(dor, xcm, st_out, s12, go32, dt, part_o)
    del dor
    gout = torch.empty(2 * hs, **f32)
    E.colsum(part_o, gout)
    G = _gemm_tk(dpr, xcm.t())                              # [D, Hs] = dpr^T X^T
    del dpr
    # ---- contraction backward -> the plane gradients
    dplanes = planes.new_empty((2 * hs, n, n))
    _contract_bwd(planes, dt.view(hs, n, n), dplanes, hs, n, direction)
    del dt
    # ---- input side: pre-activation gradients (k1wb), weight and input gradients
    E.k1wb(zf, pk["w1"], pk["vs"], pk["vb"], st_in, mask, dplanes.view(2 * hs, T), dcat, n)
    del dplanes
    xn = torch.empty_like(zf)
    E.ln_apply(zf, st_in, gi32, bi32, xn)
    dw1 = _gemm_tk(dcat, xn)                                # [4 Hs + D, D] fp32, k1w packed order then W_g
    del xn
    dxn = dcat @ pk["wdx"]                                  # [T, D]
    del dcat
    dx = torch.empty_like(zf)
    nbi = E.lnin_bwd_blocks(T, d, is32)
    part_i = torch.empty((nbi, 2 * d), **f32)
    E.lnin_bwd(dxn, zf, dyf, st_in, gi32, dx, part_i)
    gin = torch.empty(2 * d, **f32)
    E.colsum(part_i, gin)
    # ---- the parameters' gradients, in their dtypes and strides
    g_w = [_d128._grad_like(t) for t in (wl, wlg, wr, wrg)]
    g_wg, g_wo = torch.empty_like(wg), torch.empty_like(wo)
    g_go, g_bo, g_gi, g_bi = (torch.empty_like(t) for t in (go, bo, gi, bi))
    E.wfin(dw1, G, r01, gout, gin, go32, bo32, *g_w, g_wg, g_wo, g_go, g_bo, g_gi, g_bi)
    return [dx.view_as(z), *g_w, g_wg, g_wo, g_gi, g_bi, g_go, g_bo]


def _backward_fake(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    """One gradient per leaf in the leaf's dtype: contiguous, except the four input-projection matrices, whose gradient takes the leaf's strides when it is stored
    [in, out] (``sm80._grad_like``); a compiled caller checks the strides."""
    grads = [torch.empty_like(t, memory_format=torch.contiguous_format) for t in leaves]
    grads[1:5] = [_d128._grad_like(t) for t in leaves[1:5]]
    return grads


@opaque(fake=_backward_fake, name="trimul_sm80_wide_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, saved: list[torch.Tensor], dy: torch.Tensor, direction: int, eps_in: float,
             eps_out: float) -> list[torch.Tensor]:
    """Gradients of every leaf (z, W_l, W_lg, W_r, W_rg, W_g, W_o, ln_in weight / bias, ln_out weight / bias), in the leaves' dtypes."""
    return _run_backward(leaves, mask, ds, list(saved), dy, direction, eps_in, eps_out)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        mask, ds, direction, eps_in, eps_out = args[11:]
        out, *saved = forward_train(leaves, mask, ds, direction, eps_in, eps_out)
        ctx.save_for_backward(*leaves, mask, ds, *saved)
        ctx.cfg = (direction, eps_in, eps_out)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = backward(list(vals[:11]), vals[11], vals[12], list(vals[13:]), dy.contiguous(), *ctx.cfg)
        return (*grads, None, None, None, None, None)


def trimul(z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, mask, ds, direction, eps_in, eps_out):
    """z + drop_row(trimul(z)).  ``mask`` [L] uint8 or empty (``sm80.token_mask``), ``ds`` [L, D] bf16 or empty.  Call ``available()`` first."""
    leaves = (z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return forward_nograd(list(leaves), mask, ds, direction, float(eps_in), float(eps_out))
    return _Training.apply(*leaves, mask, ds, direction, float(eps_in), float(eps_out))
