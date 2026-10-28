"""The fp32 token stack's pair-biased attention on H100: TF32 wgmma cores (``attn_tf32.cu``) with the pair bias folded in.

``softmax(q k^T / sqrt(48) + wf . LN(pair)) v`` for fp32 q / k / v [A, 1, L, 16, 48] and an fp32 pair [1, L, L, 128 or 256]: the bias
comes out of ONE pass over the pair (``pair_bias_tok_cuda``: LayerNorm + the 16 head projections + the TF32 core's key
permutation + the key mask), the core is the TF32 forward / dq-dbias / dk-dv trio of the token DiT's fp32 block, and the
backward of the pair side is PyTorch (LayerNorm backward, two TF32 GEMMs). Operands are rounded to TF32 inside the tensor core
like every fp32 matmul of the process (``supported`` is False when TF32 is switched off: the call then keeps full-fp32 math).

Contract: sm_90, fp32 q / k / v / pair, 16 heads x 48 (d_pair 128 or 256) or 4 x 32 (d_pair 16: the dense atom stack), one batch row, L a multiple of 128, a key mask shared by all
samples (or none).
"""

from __future__ import annotations

import math
import weakref

import torch

from miniworld_engine.kernels._compile import opaque

from . import LOG2E, _is_hopper, tf32_backward, tf32_forward

NEG = -1e30
H, D = 16, 48   # the token stack; the dense atom stack is 4 x 32 (LAYOUTS)
#: (heads, head width, d_pair) served: the token stack (d_pair 128; 256 for the ESM-style stack) and the dense atom stack
LAYOUTS = {(16, 48, 128), (16, 48, 256), (4, 32, 16)}


def _tf32_on() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest")


def supported(q, pair, lnw, wb, mask) -> bool:
    """The contract above."""
    if not _tf32_on():
        return False
    if q.dim() != 5 or q.dtype is not torch.float32 or pair.dtype is not torch.float32 or not q.is_cuda:
        return False
    A, B, L, h, d = q.shape
    dp = pair.shape[-1]
    if (h, d, dp) not in LAYOUTS or B != 1 or L % 128 or tuple(pair.shape) != (1, L, L, dp) or tuple(wb.shape) != (h, dp) or tuple(lnw.shape) != (dp,):
        return False
    if mask is not None and (mask.dim() != 1 or mask.numel() != L or mask.dtype is not torch.bool):
        return False
    return _is_hopper(q.device.index if q.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


def available(q, pair, lnw, wb, mask) -> bool:
    """``supported()`` plus successful builds; never under torch.compile tracing or fake tensors."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(q, pair, lnw, wb, mask):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True  # two opaque ops in the trace; the extensions build when the op first runs
    if isinstance(q, FakeTensor) or isinstance(pair, FakeTensor):
        return False  # dev derive
    try:
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

        from . import _ext, tf32_fwd_defs

        rows32()
        from .fused_pair import _rows_ext

        _rows_ext()
        _ext("attn_tf32", tf32_fwd_defs())
    except Exception as exc:  # noqa: BLE001
        import warnings

        _BUILD_FAILED = True
        warnings.warn(f"TF32 pair-fused attention unavailable, keeping the PyTorch leg: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _fwd_fake(q, k, v, pair, lnw, wb, mask, eps, want_t):
    """Shapes of the TF32 pair-fused attention forward (outputs and saved tensors; all like q's dtype)."""
    A, _, L, H, D = q.shape
    return (
        q.new_empty((A * L, H * D)),
        q.new_empty((A, H, L)),
        q.new_empty((H, L, L)),
        q.new_empty((A * L, H * D)),
        q.new_empty((A * L, H * D)),
        q.new_empty((A * L, H * D)),
        q.new_empty((H * D, A * L) if want_t else (1,)),
        q.new_empty((H * D, A * L) if want_t else (1,)),
    )


@opaque(fake=_fwd_fake, name="augmented_attention_tf32_fwd")
def _fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None, eps: float,
    want_t: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(O, LSE, key-permuted bias [H, L, L] in log2 units, q scaled, k, v rounded to the TF32 grid [A L, 768]): the pair pass and the core."""
    A, _, L, H, D = q.shape
    return _fwd_core(q.reshape(A * L, H * D), k.reshape(A * L, H * D), v.reshape(A * L, H * D), A, L, H, D, pair, lnw, wb, mask, eps, want_t)


def _fwd_packed_fake(qkvg, pair, lnw, wb, mask, eps, want_t, A, L, H, D):
    """Shapes of the op this fake belongs to (no kernel runs); see the op's docstring."""
    return _fwd_fake(qkvg.new_empty((A, 1, L, H, D)), None, None, pair, lnw, wb, mask, eps, want_t)


@opaque(fake=_fwd_packed_fake, name="augmented_attention_tf32_fwd_packed")
def _fwd_packed(
    qkvg: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None, eps: float, want_t: bool, A: int, L: int, H: int, D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_fwd`` for q | k | v | gate packed in one contiguous [A L, 4 H D] matrix (the packed projection's result): the columns are sliced INSIDE the op, so
    ``torch.compile`` has no strided view to make contiguous before it (three 19 MB copies at L128, A = 48)."""
    n = H * D
    return _fwd_core(qkvg[:, :n], qkvg[:, n : 2 * n], qkvg[:, 2 * n : 3 * n], A, L, H, D, pair, lnw, wb, mask, eps, want_t)


def _fwd_core(q2, k2, v2, A, L, H, D, pair, lnw, wb, mask, eps, want_t):
    from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

    from .fused_pair import _rows_ext

    q = q2
    qs, kr, vr, vt, qt, kt = _rows_ext().prep_fwd32(q2, k2, v2, LOG2E / math.sqrt(D), want_t)
    wf = (wb.float() * lnw.float() * LOG2E).contiguous()
    bias_p = torch.empty(H, L, L, device=q.device, dtype=torch.float32)
    DP = pair.shape[-1]
    if (H, DP) == (16, 128):
        rows32().pair_bias_tok_cuda(pair.view(L * L, DP), wf, bias_p, mask, int(L), float(eps), True, NEG)
    else:   # d_pair 256 and the atom stack: the generic passes (the tensor-core one is built for 16 heads x 128 channels)
        _rows_ext().pair_bias_rows32(pair.view(L * L, DP), wf, bias_p, mask, int(L), float(eps), NEG)
    o, lse = tf32_forward(qs, kr, vt, bias_p, None, A, L, nh=H, dh=D)
    return o, lse, bias_p, qs, kr, vr, qt, kt


def _bwd_fake(do, o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, eps):   # (unchanged shapes for every layout)
    """Shapes of the TF32 pair-fused attention backward: (dq, dk, dv, dpair, dlnw fp32, dwb fp32)."""
    return (
        qs.new_empty(qs.shape),
        qs.new_empty(qs.shape),
        qs.new_empty(qs.shape),
        pair.new_empty(pair.shape),
        lnw.new_empty(lnw.shape, dtype=torch.float32),
        wb.new_empty(wb.shape, dtype=torch.float32),
    )


@opaque(fake=_bwd_fake, name="augmented_attention_tf32_bwd")
def _bwd(
    do: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, bias_p: torch.Tensor, qs: torch.Tensor, kr: torch.Tensor, vr: torch.Tensor, qt: torch.Tensor, kt: torch.Tensor,
    pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dq, dk, dv [A L, 768], dpair, dlnw, dwb) of ``_fwd``."""
    return _bwd_core(do, o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, eps, None)


def _bwd_core(do, o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, eps, dpack):
    from .fused_pair import _rows_ext


    A = o.shape[0] // pair.shape[1]
    H = lse.shape[1]
    D = o.shape[1] // H
    L = pair.shape[1]
    e = _rows_ext()
    dd, dd_p, lse_p, dobr, dot = e.dd_perm(do.contiguous().float(), o, lse, A, L)  # (dO^T too) # D = rowsum(dO O), its / the LSE's query-permuted copies, dO rounded
    dq, dk, dv, db_p = tf32_backward(
        qs, kr, vr, dobr, qt, kt, dot, bias_p, e.bias_transposed(bias_p, L),
        lse, dd, lse_p, dd_p, None, None, A, L, nh=H, dh=D, dpack=dpack,
    )
    wf = (wb.float() * lnw.float()).contiguous()
    DP = pair.shape[-1]
    dz, accp = e.pair_rows_bwd32(pair.view(L * L, DP), db_p, wf, L, eps, True)   # db_p: [H, L, L] in natural units, key columns still permuted
    dwf = accp.sum(0).view(H, DP)
    return dq, dk, dv, dz.view(pair.shape), (dwf * wb.float()).sum(0), dwf * lnw.float()


class _AttnTf32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pair, lnw, wb, mask, eps):  # noqa: D102
        A, _, L, _, _ = q.shape
        want_t = any(ctx.needs_input_grad)
        o, lse, bias_p, qs, kr, vr, qt, kt = _fwd(q, k, v, pair, lnw, wb, mask, eps, want_t)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb)
            ctx.eps, ctx.shape = eps, q.shape
        return o.view(A, 1, L, *q.shape[-2:])

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb = ctx.saved_tensors
        dq, dk, dv, dz, dlnw, dwb = _bwd(do.reshape(o.shape).contiguous(), o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, ctx.eps)
        shp = ctx.shape
        return dq.view(shp), dk.view(shp), dv.view(shp), dz, dlnw, dwb, None, None


def attention_pair_tf32(q, k, v, pair, lnw, wb, mask=None, eps: float = 1e-5):
    """See the module docstring; returns fp32 [A, 1, L, 16, 48]. Call ``available()`` first. A ``torch.compile`` trace sees two opaque ops."""
    return _AttnTf32.apply(q, k, v, pair, lnw, wb, mask, eps)


__all__ = ["attention_pair_tf32", "available", "gate_mul", "packed_qkvg", "packed_qkvg_ok", "proj_core_gate", "supported"]


# ------------------------------------------------------------------------------------------------ the projections around the core
class _PackedQKVG(torch.autograd.Function):
    """``q = x Wq^T + bq``, ``k = x Wk^T``, ``v = x Wv^T``, ``g = x Wg^T`` as ONE GEMM over the packed weight [4 d, d] (one launch and one read of x instead of four),
    q / k / v / g returned as column views of the packed result (the prep pass of the core reads them in place). Backward: ``dx = sum_i d_i W_i`` as an accumulating
    chain of GEMMs (``addmm_``: no separate add pass over [A L, d] per branch -- autograd's own accumulation of four branch gradients was three such passes) and the
    weight gradients straight into the rows of one [4 d, d] buffer. Gradients with respect to q / k / v / g that the graph did not need arrive as None."""

    @staticmethod
    def forward(ctx, x, wq, bq, wk, wv, wg):  # noqa: D102
        shp, n = x.shape, x.shape[-1]
        x2 = x.reshape(-1, n)
        w = torch.cat([wq, wk, wv, wg], 0)
        b = torch.nn.functional.pad(bq, (0, 3 * n))
        out = torch.addmm(b, x2, w.t())
        ctx.save_for_backward(x2, wq, wk, wv, wg)
        ctx.shp, ctx.has_bias = shp, bq.requires_grad
        return tuple(out[:, i * n : (i + 1) * n].view(shp) for i in range(4))

    @staticmethod
    def backward(ctx, *ds):  # noqa: D102
        x2, *ws = ctx.saved_tensors
        n = x2.shape[-1]
        need_x, need_wq, need_bq, need_wk, need_wv, need_wg = ctx.needs_input_grad
        dx = None
        grads_w = [None] * 4
        for i, (d, w) in enumerate(zip(ds, ws)):
            if d is None:
                continue
            d2 = d.reshape(-1, n)
            if need_x:
                dx = d2 @ w if dx is None else dx.addmm_(d2, w)
            if (need_wq, need_wk, need_wv, need_wg)[i]:
                grads_w[i] = d2.t() @ x2
        dbq = ds[0].reshape(-1, n).sum(0) if (need_bq and ds[0] is not None) else None
        return (None if dx is None else dx.view(ctx.shp)), grads_w[0], dbq, grads_w[1], grads_w[2], grads_w[3]


def packed_qkvg_ok(x, wq, bq, wk, wv, wg) -> bool:
    """The packed projection applies: fp32 TF32 on an H100, the core's own row contract (one batch row, L a multiple of 128), fp32 weights, bias on the query only."""
    if not _tf32_on() or torch.is_autocast_enabled("cuda"):
        return False
    if x.dtype is not torch.float32 or not x.is_cuda or x.dim() != 4 or x.shape[1] != 1 or x.shape[2] % 128 or bq is None:
        return False
    if any(w.dtype is not torch.float32 or tuple(w.shape) != (x.shape[-1],) * 2 for w in (wq, wk, wv, wg)) or bq.dtype is not torch.float32:
        return False
    return _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device())


def packed_qkvg(x, wq, bq, wk, wv, wg):
    """(q, k, v, g) of the four projections of the attention block, one GEMM; see :class:`_PackedQKVG`. Call ``packed_qkvg_ok`` first."""
    return _PackedQKVG.apply(x, wq, bq, wk, wv, wg)


# ------------------------------------------------------------------------------------------------ sigmoid gating in one pass
def _gm_fwd_fake(g, o):
    """Shapes of the op this fake belongs to (no kernel runs); see the op's docstring."""
    return o.new_empty(o.shape)


@opaque(fake=_gm_fwd_fake, name="augmented_attention_gate_mul_fwd")
def _gm_fwd(g: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """``sigmoid(g) * o`` for fp32 [R, n] (``g`` may be a column slice of a wider matrix)."""
    from .fused_pair import _rows_ext

    return _rows_ext().gate_mul_fwd(g, o)


def _gm_bwd_fake(g, o, dy):
    """Shapes of the op this fake belongs to (no kernel runs); see the op's docstring."""
    return o.new_empty(o.shape), o.new_empty(o.shape)


@opaque(fake=_gm_bwd_fake, name="augmented_attention_gate_mul_bwd")
def _gm_bwd(g: torch.Tensor, o: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(d gate, d o) of :func:`_gm_fwd`: one pass for both."""
    from .fused_pair import _rows_ext

    dg, dd = _rows_ext().gate_mul_bwd(g, o, dy.contiguous())
    return dg, dd


class _GateMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, o):  # noqa: D102
        n = o.shape[-1]
        g2, o2 = g.reshape(-1, n), o.reshape(-1, n).contiguous()
        ctx.save_for_backward(g2, o2)
        ctx.shp = o.shape
        return _gm_fwd(g2, o2).view(o.shape)

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        g2, o2 = ctx.saved_tensors
        dg, dd = _gm_bwd(g2, o2, dy.reshape(o2.shape))
        return dg.view(ctx.shp), dd.view(ctx.shp)


def gate_mul(g, o):
    """``sigmoid(g) * o`` (fp32 CUDA, same trailing width, ``n % 4 == 0``) as ONE kernel each way; replaces the sigmoid, the product and their three backward passes."""
    return _GateMul.apply(g, o)


# ------------------------------------------------------------------------------------------------ projection + core + gate as ONE autograd function
def _gmp_fwd_fake(qkvg, o, c0):
    """Shapes of the op this fake belongs to (no kernel runs); see the op's docstring."""
    return o.new_empty(o.shape)


@opaque(fake=_gmp_fwd_fake, name="augmented_attention_gate_mul_packed_fwd")
def _gmp_fwd(qkvg: torch.Tensor, o: torch.Tensor, c0: int) -> torch.Tensor:
    """``sigmoid(qkvg[:, c0 : c0 + n]) * o``: the gate columns are sliced inside the op (see ``_fwd_packed``)."""
    from .fused_pair import _rows_ext

    return _rows_ext().gate_mul_fwd(qkvg[:, c0 : c0 + o.shape[1]], o)


def _gmp_bwd_fake(qkvg, o, dy, c0):
    """Shapes of the op this fake belongs to (no kernel runs); see the op's docstring."""
    return o.new_empty(o.shape), o.new_empty(o.shape)


@opaque(fake=_gmp_bwd_fake, name="augmented_attention_gate_mul_packed_bwd")
def _gmp_bwd(qkvg: torch.Tensor, o: torch.Tensor, dy: torch.Tensor, c0: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(d gate, d o) of :func:`_gmp_fwd`."""
    from .fused_pair import _rows_ext

    dg, dd = _rows_ext().gate_mul_bwd(qkvg[:, c0 : c0 + o.shape[1]], o, dy.contiguous())
    return dg, dd


def _bwd_gc_fake(qkvg, o, dy, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, eps):
    """Shapes of the op this fake belongs to (no kernel runs); see the op's docstring."""
    return (
        qkvg.new_empty(qkvg.shape),
        pair.new_empty(pair.shape),
        lnw.new_empty(lnw.shape, dtype=torch.float32),
        wb.new_empty(wb.shape, dtype=torch.float32),
    )


@opaque(fake=_bwd_gc_fake, name="augmented_attention_tf32_bwd_gate")
def _bwd_gc(
    qkvg: torch.Tensor, o: torch.Tensor, dy: torch.Tensor, lse: torch.Tensor, bias_p: torch.Tensor, qs: torch.Tensor, kr: torch.Tensor, vr: torch.Tensor,
    qt: torch.Tensor, kt: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward of ``sigmoid(gate) * attention``: (the packed gradient [A L, 4 n] of q | k | v | gate, dpair, dlnw, dwb). d gate is written by the gate pass and
    dQ / dK / dV by the attention backward kernels straight into their column blocks, so the projection gradients are ONE GEMM each."""
    from .fused_pair import _rows_ext

    n = o.shape[1]
    dpack = torch.empty_like(qkvg)
    do = _rows_ext().gate_mul_bwd_into(qkvg[:, 3 * n : 4 * n], o, dy.contiguous(), dpack[:, 3 * n : 4 * n])
    _, _, _, dz, dlnw, dwb = _bwd_core(do, o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, eps, dpack)
    return dpack, dz, dlnw, dwb


_PACKED: dict = {}


def _packed_weights(wq, bq, wk, wv, wg, reuse: bool):
    """(Wcat [4 n, n], bias [4 n]) of the packed projection. With ``reuse`` (an inference call: no gradient is recorded) they are built once per parameter version and
    kept while the five tensors live: weak references to the very objects, so an address freed and reused by another module cannot hit; a new version replaces the entry;
    nothing is filled under ``torch.compile`` tracing or during a CUDA-graph capture (warm up once before capturing; a capture reads a hit)."""
    n = wq.shape[0]

    def make():
        return torch.cat([wq, wk, wv, wg], 0), torch.nn.functional.pad(bq, (0, 3 * n))

    if not reuse or torch.compiler.is_compiling():
        return make()
    ts = (wq, bq, wk, wv, wg)
    key = tuple((t.data_ptr(), t._version) for t in ts)
    hit = _PACKED.get(key)
    if hit is not None and all(r() is t for r, t in zip(hit[0], ts)):
        return hit[1]
    out = make()
    if torch.cuda.is_current_stream_capturing():
        return out
    for stale in [k for k, (refs, _) in _PACKED.items() if any(r() is None or any(r() is t for t in ts) for r in refs)]:
        del _PACKED[stale]
    _PACKED[key] = (tuple(weakref.ref(t) for t in ts), out)
    return out


def _pcg_forward(x, wq, bq, wk, wv, wg, pair, lnw, wb, mask, eps, n_head, want_t):
    """Forward of :class:`_ProjCoreGate`: (y, the tensors its backward needs or None). ``want_t``: a gradient will be asked for (the backward's transposed operands are written)."""
    A, _, L, n = x.shape
    H, D = n_head, n // n_head
    x2 = x.reshape(-1, n)
    w, b = _packed_weights(wq, bq, wk, wv, wg, not want_t)
    qkvg = torch.addmm(b, x2, w.t())
    o, lse, bias_p, qs, kr, vr, qt, kt = _fwd_packed(qkvg, pair, lnw, wb, mask, eps, want_t, A, L, H, D)
    y = _gmp_fwd(qkvg, o, 3 * n).view(A, 1, L, n)
    return y, ((x2, w, qkvg, o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb) if want_t else None)


class _ProjCoreGate(torch.autograd.Function):
    """``sigmoid(x Wg^T) * attention(x Wq^T + bq, x Wk^T, x Wv^T, pair)`` for the fp32 H100 token / atom stacks: the packed projection GEMM, the pair-fused TF32
    core and the output gate as one autograd node. Nothing between them is a view handed to a custom op. The backward writes the gradients of q | k | v | gate into
    one [A L, 4 n] matrix (the kernels' own epilogues), so ``dx = dpack Wcat`` and ``dWcat = dpack^T x`` are one GEMM each instead of four plus three adds / four."""

    @staticmethod
    def forward(ctx, x, wq, bq, wk, wv, wg, pair, lnw, wb, mask, eps, n_head):  # noqa: D102
        want_t = any(ctx.needs_input_grad)
        y, saved = _pcg_forward(x, wq, bq, wk, wv, wg, pair, lnw, wb, mask, eps, n_head, want_t)
        if want_t:
            ctx.save_for_backward(*saved)
            ctx.eps, ctx.shp, ctx.n = eps, x.shape, x.shape[-1]
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x2, w, qkvg, o, lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb = ctx.saved_tensors
        n = ctx.n
        need_x, need_wq, need_bq, need_wk, need_wv, need_wg, need_pair, need_lnw, need_wb = ctx.needs_input_grad[:9]
        dpack, dz, dlnw, dwb = _bwd_gc(qkvg, o, dy.reshape(o.shape), lse, bias_p, qs, kr, vr, qt, kt, pair, lnw, wb, ctx.eps)
        dx = (dpack @ w).view(ctx.shp) if need_x else None
        dw = dpack.t() @ x2 if (need_wq or need_wk or need_wv or need_wg) else None
        sl = lambda i, need: dw[i * n : (i + 1) * n] if need else None  # noqa: E731
        dbq = dpack[:, :n].sum(0) if need_bq else None
        return (dx, sl(0, need_wq), dbq, sl(1, need_wk), sl(2, need_wv), sl(3, need_wg), dz if need_pair else None,
                dlnw if need_lnw else None, dwb if need_wb else None, None, None, None)


def proj_core_gate(x, wq, bq, wk, wv, wg, pair, lnw, wb, mask, eps, n_head):
    """See :class:`_ProjCoreGate`; ``x`` [A, 1, L, n] fp32 (the adaLN output), ``mask`` [L] bool or None. Call ``packed_qkvg_ok`` and ``supported`` first."""
    ts = (x, wq, bq, wk, wv, wg, pair, lnw, wb)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in ts)):
        return _pcg_forward(*ts[:6], pair, lnw, wb, mask, eps, n_head, False)[0]   # inference: nothing for a backward is made or kept
    return _ProjCoreGate.apply(x, wq, bq, wk, wv, wg, pair, lnw, wb, mask, eps, n_head)
