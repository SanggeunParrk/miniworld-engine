"""A100 (sm_80) hand-CUDA TriMul: D = d_pair = 64 / 128 / 256 / 384, one direction (hidden D) or bidirectional (hidden 2 D), bf16 and fp32 (TF32 tensor cores).
D512 additionally serves bf16 inference, with the same folded projections and channel-major contraction; training retains its general path.

The width-generic sibling of ``sm80.py`` (whose fused kernels are literals for D128): the data flow of the B200 wide path (``b200_infer.py``) on ``mma.sync``.

* forward: ``stats_rows`` (LayerNorm_in statistics of every token) -> ``k1w`` (the gated projections of the RAW input rows: LayerNorm_in is folded into the weights,
  the GEMM is a plain ``cp.async`` stream, the epilogue applies ``r (acc - mu s) + b``, the sigmoid gate and the pair mask and stores the planes channel-major) ->
  cuBLAS contraction -> ``stats_cm`` (LayerNorm_out statistics of every token, from the channel-major contraction output) -> ``k3w`` (output projection of the raw
  contraction planes and the gate GEMM of the raw input rows with both LayerNorms folded into the weights; gate, dropout row scale and residual in the epilogue).
  Training saves the planes, the contraction output, both statistics and p' / g' (bf16, the 0.5-scaled units of the fold).  bf16 with a hidden width <= 128 (D64 in both modules, D128
  one direction) computes both statistics inside k1w / k3w from the operand tiles still resident in their rings: the two statistics kernels are not launched.
  bf16 D64 (hidden 64 or 128) runs k1w as ``k1w64_kernel`` (``wide_k1.cuh``): the same planes and statistics bit for bit, one persistent CTA per resident slot with
  all packed weights in shared memory and the next token tile's rows double-buffered (the short per-n-tile k1w CTAs were latency-bound at this width).
* ``supports()`` is the whole gate; everything it rejects keeps the existing path.

Numerics follow the D128 path (``sm80.py``): the same bf16 rounding points, the 0.5 folded into the weights (``sigmoid(g) p = p' (1 + tanh g')``).  fp32 operands run the same stages
on TF32 MMAs (``wide_gemm32.cuh``) with fp32 storage, nothing rounded to a narrower type; the folded weights are TF32-rounded, and every cuBLAS GEMM of the fp32 path is
issued with TF32 enabled whatever ``torch.backends.cuda.matmul.allow_tf32`` says (one precision per path; the Triton path follows the global flag).
"""

import contextlib
import functools
from pathlib import Path

import torch

from ... import _capture
from ..._compile import opaque
from ..._nvcc import build_failed
from . import contract_sm80 as _cp
from . import sm80 as _d128

_dir = Path(__file__).parent / "sm80"

WIDTHS = (64, 128, 256, 384, 512)
BIDIR, OUTGOING, INCOMING = _d128.BIDIR, _d128.OUTGOING, _d128.INCOMING


@functools.lru_cache(maxsize=1)
def _ext():
    return _d128._build("trimul_sm80_wide", "wide_ops.cu", 80)


@functools.lru_cache(maxsize=1)
def _ext120():
    """The same sources for the GeForce RTX 5090 (sm_120a)."""
    return _d128._build("trimul_sm80_wide", "wide_ops.cu", 120)


def _lib():
    """The build for the current device: sm_80 (A100) or sm_120a (RTX 5090)."""
    return _ext120() if _d128.on_sm120() else _ext()


def shape_ok(shape, dtype, d_hidden: int, hs: int | None = None, mask_shape=None, mask_dtype=None) -> bool:
    """The kernels' shape / dtype requirements, device-free (the CPU tests of the gate call this): bf16 or fp32 (TF32), width in WIDTHS with hidden channels ``hs`` = D
    (one direction) or 2 D (bidirectional), a square plane with L % 16 == 0, and a [B, L] bool token mask if any."""
    if dtype not in (torch.bfloat16, torch.float32) or len(shape) != 4 or shape[0] < 1 or shape[1] != shape[2]:
        return False
    d, n = shape[-1], shape[1]
    hs = d_hidden if hs is None else hs
    if d not in WIDTHS or hs not in (d, 2 * d) or n <= 0 or n % 16 != 0:
        return False
    if d == 512 and dtype is not torch.bfloat16:
        return False
    return mask_shape is None or (tuple(mask_shape) == (shape[0], n) and mask_dtype is torch.bool)


def supports(
    pair: torch.Tensor,
    d_hidden: int,
    mask: torch.Tensor | None = None,
    *,
    hs: int | None = None,
) -> bool:
    """The kernels' own requirements: sm_80, ``shape_ok``, one contiguous plane per call (a batch runs plane by plane in the integration)."""
    if not pair.is_cuda or not pair.is_contiguous():
        return False
    # D512 extends the BF16 inference path only; its backward keeps the general path.
    if pair.ndim == 4 and pair.shape[-1] == 512 and torch.is_grad_enabled():
        return False
    if not shape_ok(
        tuple(pair.shape),
        pair.dtype,
        d_hidden,
        hs,
        None if mask is None else tuple(mask.shape),
        None if mask is None else mask.dtype,
    ):
        return False
    return _d128._is_ampere(pair.device.index if pair.device.index is not None else torch.cuda.current_device())


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the extension; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    try:
        _lib()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm80 wide TriMul", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def available(
    pair: torch.Tensor,
    d_hidden: int,
    mask: torch.Tensor | None = None,
    *,
    hs: int | None = None,
) -> bool:
    """``supports()`` plus a successful (cached) build."""
    return supports(pair, d_hidden, mask, hs=hs) and _loads()


# ------------------------------------------------------------------------------------------------------------------------------- packs
def _pack(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, *, train: bool = False, adt=None) -> dict:
    """The folded weight layouts of the forward (and, in training, the backward's unscaled packed weights and the exact LayerNorm_out row-sum vector), one launch
    per call (never cached: a captured CUDA graph must repack after an optimizer step).  The four front matrices are read through their strides."""
    hs, d = wl.shape
    dev = wl.device
    ln = [t.detach().float().contiguous() for t in (gi, bi, go, bo)]
    ws = [*(wl, wlg, wr, wrg), *(w.contiguous() for w in (wg, wo))]
    f32 = dict(dtype=torch.float32, device=dev)
    # the folded weights are stored in the activations' dtype (bf16, or TF32-rounded fp32); ``adt`` names it when the weights are fp32 masters under
    # bf16 activations (the kernel rounds them on load)
    wd = wl.dtype if adt is None else adt
    w1 = torch.empty((4 * hs, d), dtype=wd, device=dev)
    wo3 = torch.empty((d, hs), dtype=wd, device=dev)
    wg3 = torch.empty((d, d), dtype=wd, device=dev)
    vec = torch.empty((2 * 4 * hs + 4 * d + (d if train else 0),), **f32)
    vs, vb = vec[: 4 * hs], vec[4 * hs : 8 * hs]
    so, eo, sg, eg = vec[8 * hs : 8 * hs + 4 * d].view(4, d).unbind(0)
    spx = vec[8 * hs + 4 * d :] if train else vec.new_empty((0,))
    wdx = torch.empty((4 * hs + d, d), dtype=wd, device=dev) if train else w1.new_empty((0,))
    _lib().wpack(*ws, *ln, w1, vs, vb, wo3, so, eo, wg3, sg, eg, wdx, spx)
    pk = dict(
        hs=hs,
        d=d,
        w1=w1,
        vs=vs,
        vb=vb,
        wo=wo3,
        wg=wg3,
        so=so,
        eo=eo,
        sg=sg,
        eg=eg,
        ln=ln,
    )
    if train:
        pk.update(wdx=wdx, spx=spx)
    return pk


def _cast_weights(leaves) -> list:
    """The ten parameters with the six weight matrices in the activations' dtype (strides kept).  fp32 master parameters are cast here, inside the op and
    outside autograd, so no cast copy sits in the graph and the gradients (written in the parameters' own dtype) are never rounded through bf16."""
    dt = leaves[0].dtype
    return [*(w.detach().to(dt) for w in leaves[1:7]), *leaves[7:]]


def _pack_leaves(leaves) -> dict:
    """``_pack`` of the op's leaves for the forward.  Bf16 activations over six fp32 master weights go to the pack kernel as they are (it rounds on load, as the
    cast would: no cast kernels per call); every other combination takes ``_cast_weights``."""
    z, w = leaves[0], leaves[1:7]
    if z.dtype is torch.bfloat16 and all(t.dtype is torch.float32 for t in w):
        return _pack(*(t.detach() for t in w), *leaves[7:], adt=torch.bfloat16)
    return _pack(*_cast_weights(leaves))


def _pack_reference(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo) -> dict:
    """``_pack`` in torch operations (the layouts' definition; the tests compare the two)."""
    f = lambda t: t.detach().float()  # noqa: E731
    hs, d = wl.shape
    wg_all = torch.cat([f(wlg), f(wrg)], 0)  # [2 Hs, D]
    wp_all = torch.cat([f(wl), f(wr)], 0)
    rows = torch.stack([wg_all.view(2 * hs // 8, 8, d), wp_all.view(2 * hs // 8, 8, d)], 1).reshape(
        4 * hs, d
    )  # 16 j + i: gate (i < 8) | projection
    wd = wl.dtype
    rnd = _tf32_round if wd is torch.float32 else (lambda t: t.to(wd))  # noqa: E731
    w1 = rnd(0.5 * rows * f(gi)[None, :])
    wo3 = rnd(0.5 * f(wo) * f(go)[None, :])
    wg3 = rnd(0.5 * f(wg) * f(gi)[None, :])
    return dict(
        hs=hs,
        d=d,
        w1=w1,
        vs=w1.float().sum(1),
        vb=0.5 * (rows @ f(bi)),
        wo=wo3,
        wg=wg3,
        so=wo3.float().sum(1),
        eo=0.5 * (f(wo) @ f(bo)),
        sg=wg3.float().sum(1),
        eg=0.5 * (f(wg) @ f(bi)),
    )


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
    """The contraction of the plane pair: channels [0, h) outgoing (sum_k a[i,k] b[j,k]), the rest incoming (sum_k a[k,i] b[k,j]); cuBLAS, or the
    own plane GEMM on an RTX 5090 (``contract_sm80.planes``)."""
    a, b = planes[:hs].view(hs, n, n), planes[hs:].view(hs, n, n)
    h = hs // 2 if direction == BIDIR else (hs if direction == OUTGOING else 0)
    if h:
        _cp.planes(0, a[:h], b[:h], x[:h])
    if h < hs:
        _cp.planes(1, a[h:], b[h:], x[h:])


#: bf16 with a hidden width <= 128 (D64 both modules, D128 one direction): the statistics kernels are replaced by in-kernel two-pass statistics of k1w / k3w (see ``_run_forward``); False keeps the kernels (probe switch for A / B timings).
_FUSE_STATS = True
_FUSE_STATS_WIDE = (
    False  # in-loop input statistics in k1w at D = 256 / 384: measured 4-8 % SLOWER than ln_stats_rows (A100, 2026-10-30), kept off
)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """The kernels move 16-byte granules (cp.async, 128-bit loads): a view that starts off a 16-byte boundary (a slice of a larger buffer) is copied once."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


#: RTX 5090 inference past GROUP_TOKENS tokens: the plane pair [2 Hs, T] (two pair-sized buffers at Hs = D) is formed and contracted GROUP channels at a time
#: (``_grouped_front``; a group's a | b rows are 4 GROUP = 128 packed rows, one k1w column tile), so the op's transient is the pair, one [64, T] plane group, X and the output (an AF3 trunk past L 3000 on 32 GB). A fixed function of the
#: shape; per channel the same k1w rows and plane contraction as the whole call.
GROUP_TOKENS = 2048 * 2048
GROUP = 32


def _grouped(z: torch.Tensor, hs: int, direction: int, save: bool) -> bool:
    if save or z.dtype is not torch.bfloat16 or not _d128.on_sm120() or z.shape[1] * z.shape[2] <= GROUP_TOKENS:
        return False
    h = hs // 2 if direction == BIDIR else (hs if direction == OUTGOING else 0)
    return hs > GROUP and hs % GROUP == 0 and h % GROUP == 0


def _grouped_front(k1, zf, pk, st_in, mask, x, hs, n, direction, eps_in, fuse_z) -> None:
    """k1w + the plane contraction per group of GROUP channels: the group's a rows (packed rows 2 GROUP g ..) and b rows (2 Hs + 2 GROUP g ..) of the packed front
    weights in one k1w call -> planes [2 GROUP, T] (a | b of the group), then the group's channels of X (outgoing below h, incoming above)."""
    T = n * n
    h = hs // 2 if direction == BIDIR else (hs if direction == OUTGOING else 0)
    w1, vs, vb = pk["w1"], pk["vs"], pk["vb"]
    planes = zf.new_empty((2 * GROUP, T))
    a, b = planes[:GROUP].view(GROUP, n, n), planes[GROUP:].view(GROUP, n, n)
    for c0 in range(0, hs, GROUP):
        ra, rb = slice(2 * c0, 2 * c0 + 2 * GROUP), slice(2 * hs + 2 * c0, 2 * hs + 2 * c0 + 2 * GROUP)
        k1(zf, torch.cat([w1[ra], w1[rb]]), torch.cat([vs[ra], vs[rb]]), torch.cat([vb[ra], vb[rb]]), st_in, mask, planes, n, eps_in, fuse_z)
        with _tf32(zf):
            _cp.planes(0 if c0 < h else 1, a, b, x[c0 : c0 + GROUP])


def _run_forward(leaves, mask, ds, direction, eps_in, eps_out, *, save):
    E = _lib()
    z = leaves[0]
    n, d = z.shape[1], z.shape[-1]
    T = n * n
    zf = _aligned(z.reshape(T, d))
    ds = _aligned(ds) if ds.numel() else ds
    # inference (no saves, grad off): the folded weights once per weight version, read by a CUDA-graph capture as its eager warm-up made them
    # (``capture_reads_eager``: no pack kernel per replay); training packs every call (a captured step must repack after an optimizer step)
    pk = (
        _capture.param_forms(
            ("trimul_sm80_wide_pack", leaves[0].dtype), list(leaves[1:]), lambda: _pack_leaves(leaves), capture_reads_eager=True
        )
        if not save and not torch.is_grad_enabled()
        else _pack_leaves(leaves)
    )
    hs = pk["hs"]
    # bf16, hidden width <= 128: k1w computes the input rows' statistics from its own tile and k3w the output stage's LayerNorm_out statistics -- the two statistics kernels and
    # their reads of z / X are gone (every k-tile of both operands is still in its ring slot when its GEMM ends)
    fuse_x = _FUSE_STATS and z.dtype is torch.bfloat16 and hs <= 128
    # input rows: <= 4 CTAs share a token tile at hs <= 128 (the redundant per-CTA statistics are cheaper than the kernels); at D = 256 / 384 k1w accumulates them
    # while its stages go by (``K1StatHook``), so ``ln_stats_rows`` and its read of z are gone there too
    fuse_z = fuse_x or (_FUSE_STATS and _FUSE_STATS_WIDE and z.dtype is torch.bfloat16 and 128 < d <= 384)
    st_in = torch.empty((T, 2), dtype=torch.float32, device=z.device)
    if not fuse_z:
        E.stats_rows(zf, st_in, eps_in)
    k1 = E.k1w if save else E.k1w_row_mask
    x = zf.new_empty((hs, n, n))
    if _grouped(z, hs, direction, save):
        _grouped_front(k1, zf, pk, st_in, mask, x, hs, n, direction, eps_in, fuse_z)
    else:
        planes = zf.new_empty((2 * hs, T))
        k1(zf, pk["w1"], pk["vs"], pk["vb"], st_in, mask, planes, n, eps_in, fuse_z)
        with _tf32(z):
            _contract(planes, x, hs, n, direction)
        del planes
    st_out = torch.empty((T, 2), dtype=torch.float32, device=z.device)
    if not fuse_x:
        E.stats_cm(x.view(hs, T), st_out, eps_out)
    out = torch.empty_like(zf)
    if save:
        ps, gs = torch.empty_like(zf), torch.empty_like(zf)
    else:
        ps = gs = zf.new_empty((0,))
    k3 = E.k3w if save else E.k3w_tuned
    k3(
        x.view(hs, T),
        zf,
        pk["wo"],
        pk["wg"],
        pk["so"],
        pk["eo"],
        pk["sg"],
        pk["eg"],
        st_out,
        st_in,
        ds,
        out,
        ps,
        gs,
        n,
        eps_out,
        fuse_x,
    )
    return out.view_as(z), ((planes, x, st_in, st_out, ps, gs) if save else None)


def _forward_nograd_fake(leaves, mask, ds, direction, eps_in, eps_out):
    """The output has the pair's shape and dtype; nothing is saved."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_sm80_wide_forward")
def forward_nograd(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    direction: int,
    eps_in: float,
    eps_out: float,
) -> torch.Tensor:
    """z + drop_row(trimul(z)) without the backward's saves.  ``mask`` [L] uint8 or empty, ``ds`` [L, D] bf16 or empty."""
    return _run_forward(leaves, mask, ds, direction, eps_in, eps_out, save=False)[0]


def _forward_train_fake(leaves, mask, ds, direction, eps_in, eps_out):
    """[out like the pair, planes [2 Hs, T], X [Hs, L, L], LN_in statistics [T, 2], LN_out statistics [T, 2], p' [T, D], g' [T, D]], Hs the first weight's leading size."""
    z = leaves[0]
    n, d, hs = z.shape[1], z.shape[-1], leaves[1].shape[0]
    T = n * n
    f32 = torch.float32
    return [
        torch.empty_like(z),
        z.new_empty((2 * hs, T)),
        z.new_empty((hs, n, n)),
        torch.empty((T, 2), dtype=f32, device=z.device),
        torch.empty((T, 2), dtype=f32, device=z.device),
        z.new_empty((T, d)),
        z.new_empty((T, d)),
    ]


@opaque(fake=_forward_train_fake, name="trimul_sm80_wide_train_fwd")
def forward_train(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    direction: int,
    eps_in: float,
    eps_out: float,
) -> list[torch.Tensor]:
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
    """dX [Hs, L, L] -> the plane gradients [2 Hs, L, L] (cuBLAS, or the own plane GEMM on an RTX 5090): X = A B^T: dA = dX B, dB = dX^T A;
    X = A^T B: dA = B dX^T, dB = A dX."""
    a, b = planes[:hs].view(hs, n, n), planes[hs:].view(hs, n, n)
    da, db = dplanes[:hs], dplanes[hs:]
    h = hs // 2 if direction == BIDIR else (hs if direction == OUTGOING else 0)
    if h:
        _cp.planes(2, dx[:h], b[:h], da[:h])
        _cp.planes(1, dx[:h], a[:h], db[:h])
    if h < hs:
        _cp.planes(0, b[h:], dx[h:], da[h:])
        _cp.planes(2, a[h:], dx[h:], db[h:])


def _run_backward(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    with _tf32(leaves[0]):
        return _run_backward_body(leaves, mask, ds, saved, dy, direction, eps_in, eps_out)


def _run_backward_body(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    E = _lib()
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
    dcat = zf.new_empty((T, 4 * hs + d))  # [dpre (k1w packed order) | dg of the output gate]
    dpr = torch.empty_like(zf)
    s12 = torch.empty((T, 2), **f32)
    part = torch.empty((nb, 2 * d), **f32)
    E.gate_bwd(
        dyf,
        ps,
        gs,
        ds,
        pk["spx"],
        2.0 * pk["eo"],
        st_out,
        dpr,
        dcat,
        4 * hs,
        s12,
        part,
        n,
    )
    r01 = torch.empty(2 * d, **f32)
    E.colsum(part, r01)
    # [Hs, T] = r_o dy_o, channel-major like X (RTX 5090: W_o^T contiguous first, cuBLAS takes a 1.5-1.8x faster kernel for that layout there)
    dor = torch.mm(cw[5].t().contiguous() if _d128.on_sm120() else cw[5].t(), dpr.t())
    dt = torch.empty_like(xcm)
    part_o = torch.empty((T // (128 if is32 else 256), 2 * hs), **f32)
    E.lnout_bwd(dor, xcm, st_out, s12, go32, dt, part_o)
    del dor
    gout = torch.empty(2 * hs, **f32)
    E.colsum(part_o, gout)
    G = _gemm_tk(dpr, xcm.t())  # [D, Hs] = dpr^T X^T
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
    dw1 = _gemm_tk(dcat, xn)  # [4 Hs + D, D] fp32, k1w packed order then W_g
    del xn
    dxn = dcat @ pk["wdx"]  # [T, D]
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
def backward(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    direction: int,
    eps_in: float,
    eps_out: float,
) -> list[torch.Tensor]:
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
        grads = backward(
            list(vals[:11]),
            vals[11],
            vals[12],
            list(vals[13:]),
            dy.contiguous(),
            *ctx.cfg,
        )
        return (*grads, None, None, None, None, None)


def trimul(z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, mask, ds, direction, eps_in, eps_out):
    """z + drop_row(trimul(z)).  ``mask`` [L] uint8 or empty (``sm80.token_mask``), ``ds`` [L, D] bf16 or empty.  Call ``available()`` first."""
    leaves = (z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return forward_nograd(list(leaves), mask, ds, direction, float(eps_in), float(eps_out))
    return _Training.apply(*leaves, mask, ds, direction, float(eps_in), float(eps_out))
