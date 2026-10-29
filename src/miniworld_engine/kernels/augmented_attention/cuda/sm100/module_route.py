"""AugmentedAttentionPairBias on B200 (sm_100a) after its AdaLN, as ONE autograd node per call: the module route.

    out = res + sigmoid(cond Wsc^T + bsc) * ( sigmoid(g) * attention(q, k, v, pair) ) Wo^T        q | k | v | g = x [Wq; Wk; Wv; Wg]^T + [bq, 0]

``x`` is the module's AdaLN output, ``cond`` / ``res`` its cond and residual streams (outputs of the AdaLN node, so their gradients join the
AdaLN backward's row kernels instead of separate adds). Before this route the module ran everything but the attention core as PyTorch
statements and four separate projections: the gradient branches' adds, the sigmoid gates and their backward, the weight casts and the
autocast copies were ~2/3 of a training step. Here (``aa_rows.cu`` for every row pass):

  training forward   one cast launch (fp32 master weights -> bf16 [Wq; Wk; Wv; Wg], [bq, 0], Wo, Wsc, bsc; fp32 path: one packing launch)
                     | addmm q|k|v|g (cuBLAS) | [fp32: q|k|v rounded to the TF32 grid in place] | the pair pass (LayerNorm + head
                     projection in one read of the pair, ``fused_pair._pair_bias``) [+ ``aam_bias_fix``: extra_pair_bias, padding, per-sample
                     or fp32 key masks] | the tcgen05 core on q / k / v column views (attn_fwd2 / attn_fwd_tf32) | ``aam_gate_o`` | mm Wo |
                     addmm Wsc | ``aam_res_gate`` (gate + residual)
  training backward  ``aam_res_gate_bwd`` | mm dWo, mm d og | ``aam_gate_bwd`` (dO for the core, D = rowsum(dO O), d gate into the
                     q|k|v|g gradient) | attn_dkv + attn_dqb built -DDET=1 -DDQPART=1 (TF32: dK / dV straight into the q|k|v|g gradient)
                     | ``aam_finish`` (dQ partials summed in order into the q|k|v|g gradient, bf16 dK / dV converted, dbq partials) |
                     mm dx, mm dW q|k|v|g | the pair backward (``aam_dbias_fix`` first when the bias was fixed) | mm d cond, mm dWsc |
                     one fixed-order column-sum launch for dbq and dbsc
  inference (bf16)   cached weight casts (q pre-scaled into exp2 units) | addmm | pair pass (log2 units) [+ ``aam_bias_fix``] | ``attn_inf``
                     (sigmoid(g) o written over q; any A, odd A without a pad sample; any L, the bias rows padded to a multiple of 8) | mm
                     Wo (reading the q columns) | addmm Wsc | ``aam_res_gate``
  inference (fp32)   the training forward's kernels, nothing saved

QK-norm (per-head RMSNorm of q and k): ``aam_qknorm`` after the GEMM (training: normalized copies, the rstd kept; inference: in place)
and ``aam_qknorm_bwd`` after the core's backward (in place in the q|k|v|g gradient). A key mask per sample: the cores built -DKMASK=1
(the mask as bits, ``aam_pack_kmask``; masked keys' P exactly 0 in the backward).

Every gradient is bitwise repeatable (fixed-order sums, the deterministic core backward). Contract (``supported``): sm_100, q / k / v
heads at a token DiT layout (``TDIT_HEADS``), B == 1, QK-norm on or off, d_pair 128 (bf16 pair: also 256 at 16 heads), a key mask [L]
shared by the samples, [A, L] one per sample, or none, any A, any L (the training cores run L padded to a multiple of 128 and A to an even count, the pair passes L
padded to a multiple of 16: zero rows, masked keys, sliced off), an extra [L, L] logit term (any float dtype, differentiable).
Precision: bf16 under bf16 autocast or for bf16 activations; fp32 (TF32 tensor cores) for fp32 activations without autocast when TF32
matmuls are allowed. Weights of any float dtype: rounded to the compute dtype per call (inference: cached per parameter version),
gradients in the parameters' dtype (fp32 masters: unrounded fp32 GEMM outputs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

from . import TDIT_HEADS, _grid, _index, _is_blackwell, _rs, _sm100_kernel, _tdit_defs, _tm, forward, forward_tf32
from .fused_pair import NEG, _bwd_bf16, _core_defs, _finish_grid, _pair_bias, _tf32_on, _wf

BF = torch.bfloat16
F32 = torch.float32
LOG2E = 1.4426950408889634
QM = 128   # the training cores' query / key tile
PM = 16    # the pair passes' row groups


def _k(name: str, dev: int):
    return _sm100_kernel("aa_rows", name, dev, smem=0)


def _ceil(x: int, m: int) -> int:
    return -(-x // m) * m


# --------------------------------------------------------------------------------------------------- contract
def supported(x, pair, lnw, wb, mask, extra, dtype, heads: int) -> bool:
    """``x`` [A, 1, L, d] the AdaLN output, ``dtype`` the compute dtype (bf16 / fp32), ``mask`` the key mask shared by the samples [L], one
    per sample [A, L], or None, ``extra`` [L, L] or None. See the module docstring."""
    cur = settings.current()
    if not cur.augmented_attention_bf16_sm100 or cur.force_portable_kernels:
        return False
    if x.dim() != 4 or not x.is_cuda or not pair.is_cuda or x.shape[1] != 1:
        return False
    A, _, L, d = x.shape
    if A < 1 or L < 1 or d % heads or (heads, d // heads) not in TDIT_HEADS:
        return False
    dp = pair.shape[-1]
    if pair.dim() != 4 or tuple(pair.shape) != (1, L, L, dp) or tuple(wb.shape) != (heads, dp) or tuple(lnw.shape) != (dp,):
        return False
    if dtype is BF:
        ok = (pair.dtype is BF and (dp == 128 or (dp == 256 and heads == 16))) or (pair.dtype is F32 and dp == 128)
    elif dtype is F32:
        ok = pair.dtype is F32 and dp == 128 and _tf32_on()
    else:
        ok = False
    if not ok:
        return False
    if mask is not None and (tuple(mask.shape) not in ((L,), (A, L)) or mask.dtype is not torch.bool):
        return False
    if extra is not None and (extra.dim() != 2 or tuple(extra.shape) != (L, L) or extra.dtype not in (BF, F32, torch.float16)):
        return False
    return _is_blackwell(_index(x))


def _build(dev: int, H: int, DH: int, dtype: torch.dtype, pair_dtype: torch.dtype, train: bool, kmask: bool = False) -> None:
    """Load every kernel a call will launch (a build or load failure raises: B200 has no silent fallback). ``kmask``: the cores built
    for a per-sample key mask (-DKMASK=1)."""
    kd = ("KMASK=1",) if kmask else ()
    td, cd = (*_tdit_defs(H, DH), *kd), (*_core_defs(H, DH), *kd)
    for name in ("aam_gate_o_bf16", "aam_res_gate_bb", "aam_bias_fix_bf16", "aam_dbias_fix", "aam_round_qkv", "aam_finish_bf16", "aam_pack_kmask"):
        _k(name, dev)
    if dtype is BF and not train:
        _sm100_kernel("attn_inf", "augattn_inf_sm100", dev, defs=td)
    elif dtype is BF:
        _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev, defs=td)
        _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, defs=cd)
        _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=(*cd, "DQPART=1"))
        _sm100_kernel("glue", "bias_transpose_bf16", dev, smem=0)
    else:
        _sm100_kernel("attn_fwd_tf32", "augattn_fwd_tf32_sm100", dev, defs=td)
        if train:
            _sm100_kernel("attn_dkv_tf32", "augattn_dkv_tf32_sm100", dev, defs=cd)
            _sm100_kernel("attn_dqb_tf32", "augattn_dqb_tf32_sm100", dev, defs=(*cd, "DQPART=1"))
    if pair_dtype is BF:
        from ..apb import ext

        ext()
    else:
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

        rows32()
    from miniworld_engine.kernels import _cast_cuda

    _cast_cuda._ext()


def available(x, pair, lnw, wb, mask, extra, dtype, heads: int, train: bool) -> bool:
    """``supported()`` plus the builds; under a ``torch.compile`` trace the route is taken (opaque ops), fake tensors outside one are not."""
    if not supported(x, pair, lnw, wb, mask, extra, dtype, heads):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True
    if isinstance(x, FakeTensor) or isinstance(pair, FakeTensor):
        return False
    try:
        _build(_index(x), heads, x.shape[-1] // heads, dtype, pair.dtype, train, mask is not None and mask.dim() == 2)
    except Exception as exc:
        raise RuntimeError(f"sm100 AugmentedAttention module route kernels failed to build or load on B200: {exc!r}") from exc
    return True


# --------------------------------------------------------------------------------------------------- weights
_ZEROS: dict = {}


def _zeros(n: int, dev) -> torch.Tensor:
    """A cached fp32 zero vector (the zero tail of the packed bias; read only)."""
    key = (n, str(dev))
    z = _ZEROS.get(key)
    if z is None:
        z = _ZEROS[key] = torch.zeros(n, device=dev, dtype=F32)
    return z


def _weights(wq, bq, wk, wv, wg, wo, wsc, bsc, dt, qscale: float = 1.0):
    """(Wcat [4 d, d], bcat [4 d], Wo, Wsc, bsc) in ``dt``: ONE launch (``_cast_cuda.cast_into``: casts / copies into one packed buffer).
    ``qscale`` multiplies the query rows and bias (inference: exp2 units for attn_inf)."""
    from miniworld_engine.kernels import _cast_cuda

    d = wq.shape[0]
    dev = wq.device
    wcat = torch.empty(4 * d, d, device=dev, dtype=dt)
    bcat = torch.empty(4 * d, device=dev, dtype=dt)
    if qscale != 1.0:
        wq, bq = wq * qscale, bq * qscale
    srcs = [wq, wk, wv, wg, bq, _zeros(3 * d, dev)]
    dsts = [wcat[:d], wcat[d : 2 * d], wcat[2 * d : 3 * d], wcat[3 * d :], bcat[:d], bcat[d:]]
    rest = []
    for t in (wo, wsc, bsc):
        if _own_w(t, dt):
            rest.append(t)
        else:
            rest.append(torch.empty(t.shape, device=dev, dtype=dt))
            srcs.append(t)
            dsts.append(rest[-1])
    wo_, wsc_, bsc_ = rest
    _cast_cuda.cast_into([s.detach() for s in srcs], dsts)
    return wcat, bcat, wo_, wsc_, bsc_


# --------------------------------------------------------------------------------------------------- row passes
def _n4grid(n4: int, dev: int):
    return _finish_grid(n4, dev)


def _gate_o(O, g, og):
    dev = _index(O)
    M, W = O.shape
    n8 = M * W // 8
    name = "aam_gate_o_bf16" if og.dtype is BF else "aam_gate_o_f32"
    _k(name, dev)(_n4grid(n8, dev), (256, 1, 1), O, g, int(g.stride(0)), og, int(n8), int(W))


def _res_gate(res, sp, y, out, L, Lp):
    dev = _index(sp)
    M, D = sp.shape
    n8 = M * D // 8
    tag = ("b" if sp.dtype is BF else "f") + ("b" if out.dtype is BF else "f")
    _k(f"aam_res_gate_{tag}", dev)(_n4grid(n8, dev), (256, 1, 1), res, sp, y, out, int(n8), int(D), int(L), int(Lp))


RB = 8   # aa_rows.cu: rows per row block


def _nsm(dev: int) -> int:
    from miniworld_engine.kernels._device import num_sms

    return num_sms(dev)


def _row_blocks(rows: int, dev: int) -> int:
    """Row blocks of the reducing row kernels: one resident wave (a function of the shape and the GPU only)."""
    from miniworld_engine.kernels._device import num_sms

    return max(1, min(-(-rows // RB), num_sms(dev)))


_COUNTERS: dict = {}


def _counter(dev: int, slot: int) -> torch.Tensor:
    """The last-block counters of the reducing row kernels (``colsum_tail``): zero at rest, reset by the block that uses them; one slot per
    kernel of a call (launches on one stream run in order)."""
    c = _COUNTERS.get(dev)
    if c is None:
        c = _COUNTERS[dev] = torch.zeros(8, device=f"cuda:{dev}", dtype=torch.int32)
    return c[slot : slot + 1]


def _bias_fix(bias, extra, pm, out, S, H, L, Lb, Lp, xs: float = 1.0):
    dev = _index(bias)
    n = S * H * Lp
    name = "aam_bias_fix_bf16" if bias.dtype is BF else "aam_bias_fix_f32"
    ef32 = 1
    if extra is not None and extra.dtype not in (BF, F32):
        extra = extra.float()
    if extra is not None:
        ef32 = int(extra.dtype is F32)
        extra = extra.contiguous()
    _k(name, dev)((min(n, 16 * _nsm(dev)), 1, 1), (256, 1, 1), bias, extra, ef32, float(xs), pm, out, int(S), int(H), int(L), int(Lb), int(Lp),
                  float(NEG))


def _wgrad(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """g^T b with an fp32 result (a weight gradient; g [M, N], b [M, K], both contiguous). bf16: one cuBLAS GEMM with an fp32 output.
    fp32 (TF32): a gradient with few output tiles (Wo [768, 768], Wsc [768, 384]: under a wave of 128 x 128 tiles) and the long
    reduction over the A L rows runs at ~300 TF/s as one GEMM; split over the rows into s <= 8 chunks as one batched GEMM, then the
    chunks summed in order (bitwise repeatable)."""
    if g.dtype is BF:
        return torch.mm(g.t(), b, out_dtype=F32)
    M, N = g.shape
    K = b.shape[1]
    tiles = -(-N // 128) * -(-K // 128)
    nsm = _nsm(_index(g))
    s = 1
    while tiles < 128 and s < 8 and tiles * s * 2 <= 2 * nsm and M % (2 * s * 128) == 0:
        s *= 2
    if s == 1:
        return torch.mm(g.t(), b)
    return torch.bmm(g.view(s, M // s, N).transpose(1, 2), b.view(s, M // s, K)).sum(0)


def _qknorm(qkvg, nq, nk, qn, kn, rstd, H, DH, qke, qs: float):
    """QK-norm (per-head RMSNorm) of the q / k columns of q|k|v|g into ``qn`` / ``kn`` (row stride of their own: separate buffers, or
    the q / k columns themselves), q times ``qs``; ``rstd`` [M, 2 H] kept for the backward (or None). fp32: rounded to the TF32 grid."""
    dev = _index(qkvg)
    M, D = qkvg.shape[0], H * DH
    bf = qkvg.dtype is BF
    nq, nk = (w.detach().float().contiguous() for w in (nq, nk))
    _k("aam_qknorm_bf16" if bf else "aam_qknorm_f32", dev)(
        (_row_blocks(M, dev), 1, 1), (D // 8, RB, 1), qkvg, int(qkvg.stride(0)), nq, nk, qn, kn, int(qn.stride(0)), rstd, int(M), int(H), int(DH),
        float(qke[0]), float(qke[1]), float(qs)
    )


# --------------------------------------------------------------------------------------------------- the bias
def _bias(pair, lnw, wb, mask, extra, eps, bf: bool, train: bool, L: int, Lb: int, Lp: int, H: int, log2: bool = False):
    """(the core's bias [H, Lp, Lp] in its dtype, the pair statistics (fp32 pair, training) or [0, 2], whether it was fixed).
    ``Lb``: the pair passes' length (L padded to 16). ``log2``: exp2 units (attn_inf)."""
    pz = pair if Lb == L else F.pad(pair, (0, 0, 0, Lb - L, 0, Lb - L))
    fix = _fixed(mask, extra, pair.dtype, train, L, Lb, Lp)
    if mask is not None and mask.dim() == 2:
        mask = None   # per sample: the cores' key mask
    if log2:
        lnw = lnw * LOG2E
    # the pair pass folds a key mask only at its own length, and its fp32 training form takes none (folded into the fix then)
    m = mask if (mask is not None and Lb == L and not (pair.dtype is F32 and train)) else None
    bias, pst = _pair_bias(pz, lnw, wb, m, eps, bf, train)
    if not fix:
        return bias, pst, False
    pm = None if mask is None or m is not None else mask.reshape(L).contiguous()
    out = bias if Lp == Lb else torch.empty(H, Lp, Lp, device=bias.device, dtype=bias.dtype)
    _bias_fix(bias, extra, pm, out, 1, H, L, Lb, Lp, LOG2E if log2 else 1.0)
    return out, pst, True


# --------------------------------------------------------------------------------------------------- the cores (per-sample key mask)
def _kmask(mask, A, Ap, L, KW):
    """The per-sample key mask [A, L] bool as the cores' bits [Ap, KW] int32 (``aam_pack_kmask``)."""
    out = torch.empty(Ap, KW, device=mask.device, dtype=torch.int32)
    _k("aam_pack_kmask", _index(mask))((-(-Ap * KW // 256), 1, 1), (256, 1, 1), mask, out, int(A), int(L), int(Ap), int(KW))
    return out


def _km_defs(km) -> tuple[str, ...]:
    return () if km is None else ("KMASK=1",)


def _core_fwd(q, k, v, bias, A, L, H, DH, bf: bool, km):
    """The training forward core (``attn_fwd2`` / ``attn_fwd_tf32``; ``km``: the per-sample key mask bits, built -DKMASK=1):
    O [A L, W] fp32, LSE [A, H, L]."""
    if km is None:
        return (forward if bf else forward_tf32)(q, k, v, bias, A, L, H, DH)
    dev = _index(q)
    W = H * DH
    O = torch.empty(A * L, W, device=q.device, dtype=F32)
    LSE = torch.empty(A, H, L, device=q.device, dtype=F32)
    defs = (*_tdit_defs(H, DH), "KMASK=1")
    grid = _grid((A // 2) * H * (L // 128), dev)
    if bf:
        maps = (
            _tm(q, [W, A * L], _rs(q), [DH, 128]),
            _tm(k, [W, A * L], _rs(k), [DH, 64]),
            _tm(v, [W, A * L], _rs(v), [DH, 64]),
            _tm(bias, [L, H * L], L * 2, [64, 128]),
            _tm(O, [W, A * L], W * 4, [32, 128], swizzle=128, dtype="f32"),
            _tm(O, [W, A * L], W * 4, [16, 128], swizzle=64, dtype="f32"),
        )
        _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev, defs=defs)(grid, (384, 1, 1), *maps, O, LSE, int(L), int(A), km)
    else:
        f32 = dict(dtype="f32")
        dims = [W, A * L]
        maps = (
            _tm(q, dims, _rs(q), [32, 128], swizzle=128, **f32),
            _tm(q, dims, _rs(q), [16, 128], swizzle=64, **f32),
            _tm(k, dims, _rs(k), [32, 32], swizzle=128, **f32),
            _tm(k, dims, _rs(k), [16, 32], swizzle=64, **f32),
            _tm(v, dims, _rs(v), [32, 32], swizzle="128a32", **f32),
            _tm(bias, [L, H * L], L * 4, [32, 128], swizzle=128, **f32),
            _tm(O, dims, W * 4, [32, 128], swizzle=128, **f32),
            _tm(O, dims, W * 4, [16, 128], swizzle=64, **f32),
        )
        _sm100_kernel("attn_fwd_tf32", "augattn_fwd_tf32_sm100", dev, defs=defs)(grid, (384, 1, 1), *maps, LSE, int(L), int(A), km)
    return O, LSE


def _bwd_bf16_km(q2, k2, v2, dob, bias, LSE, Dd, A, L, H, DH, km):
    """``fused_pair._bwd_bf16`` with the per-sample key mask (-DKMASK=1): (dQ partials, dK, dV, dbias) fp32."""
    if km is None:
        return _bwd_bf16(q2, k2, v2, dob, bias, LSE, Dd, A, L, H, DH)
    from . import bias_transpose

    dev = _index(q2)
    W, M, nch = H * DH, A * L, L // 128
    f32 = dict(dtype="f32")
    bias_t = bias_transpose(bias)
    DQ = torch.empty(nch, M, W, device=q2.device, dtype=F32)
    DK, DV = (torch.empty(M, W, device=q2.device, dtype=F32) for _ in range(2))
    DB = torch.empty(H, L, L, device=q2.device, dtype=F32)
    dims = [W, M]
    dkv_maps = (
        _tm(q2, dims, _rs(q2), [DH, 64]),
        _tm(k2, dims, _rs(k2), [DH, 128]),
        _tm(v2, dims, _rs(v2), [DH, 128]),
        _tm(dob, dims, _rs(dob), [DH, 64]),
        _tm(bias_t, [L, H * L], L * 2, [64, 128]),
        _tm(DK, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DK, dims, W * 4, [16, 128], swizzle=64, **f32),
        _tm(DV, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DV, dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    dq_dims = [W, nch * M]
    dqb_maps = (
        *(_tm(t, dims, _rs(t), [DH, 128]) for t in (q2, k2, v2, dob)),
        _tm(bias, [L, H * L], L * 2, [64, 128]),
        _tm(DQ, dq_dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DQ, dq_dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    cd = (*_core_defs(H, DH), "KMASK=1")
    _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, defs=cd)(
        _grid(A * H * (L // 128), dev), (384, 1, 1), *dkv_maps, LSE, Dd, DK, DV, None, int(L), int(A), km
    )
    _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=(*cd, "DQPART=1"))(
        _grid(H * (L // 128) * nch, dev), (384, 1, 1), *dqb_maps, LSE, Dd, DQ, DB, int(L), int(A), km
    )
    return DQ, DK, DV, DB


# --------------------------------------------------------------------------------------------------- training forward
def _dims(A, L):
    return A + (A & 1), _ceil(L, QM), _ceil(L, PM)


def _fixed(mask, extra, pair_dtype, train: bool, L: int, Lb: int, Lp: int) -> bool:
    """Whether the pair pass's bias goes through ``aam_bias_fix`` (``_bias``): padding, the extra term, or a shared key mask the pair
    pass does not fold (the fp32 training pass takes none; a padded pair's mask is folded with the padding). A per-sample key mask is
    the cores' (``KMASK``), never the bias's."""
    if mask is not None and mask.dim() == 2:
        mask = None
    fold = mask is not None and Lb == L and not (pair_dtype is F32 and train)
    return Lp != Lb or Lb != L or extra is not None or (mask is not None and not fold)


def _fwd_fake(x, cond, res, pair, mask, extra, params, eps, heads, train, out_dtype, qke):
    A, _, L, D = x.shape
    dt = x.dtype
    if not train:
        return [x.new_empty((A, 1, L, D), dtype=out_dtype), *(x.new_empty((0, 1)) for _ in range(16))]
    Ap, Lp, Lb = _dims(A, L)
    Mp, M = Ap * Lp, A * L
    e = x.new_empty
    st = (Lb * Lb, 2) if (train and pair.dtype is F32) else (0, 2)
    padded = (Ap, Lp) != (A, L)
    return [
        e((A, 1, L, D), dtype=out_dtype),
        e((Mp if padded else 0, D), dtype=dt),         # x padded
        e((Mp, 4 * D), dtype=dt),                       # q|k|v|g
        e((Mp, D), dtype=F32),                          # O
        e((Ap, heads, Lp), dtype=F32),                  # LSE
        e((Mp, D), dtype=dt),                           # og
        e((heads, Lp, Lp), dtype=dt),                   # bias
        e((Mp, D), dtype=dt),                           # y
        e((M, D), dtype=dt),                            # sp
        e(st, dtype=F32),                               # pair statistics
        e((4 * D, D), dtype=dt),                        # Wcat
        e((0 if _own_w(params[5], dt) else D, D), dtype=dt),                         # Wo (empty: the fp32 parameter itself)
        e((0 if _own_w(params[6], dt) else D, cond.shape[-1]), dtype=dt),             # Wsc
        e((Ap, Lp // 32) if (mask is not None and mask.dim() == 2) else (0,), dtype=torch.int32),   # per-sample key mask bits
        e((Mp if qke else 0, D), dtype=dt),             # QK-norm: q normalized
        e((Mp if qke else 0, D), dtype=dt),             # k normalized
        e((Mp if qke else 0, 2 * heads), dtype=F32),    # their rstd
    ]


def _own_w(w, dt) -> bool:
    """The fp32 path uses an fp32 contiguous weight as it is (``_weights``)."""
    return dt is F32 and w.dtype is F32 and w.is_contiguous()


@opaque(fake=_fwd_fake, name="augmented_attention_module_sm100_fwd")
def _fwd(
    x: torch.Tensor, cond: torch.Tensor, res: torch.Tensor | None, pair: torch.Tensor, mask: torch.Tensor | None, extra: torch.Tensor | None,
    params: list[torch.Tensor], eps: float, heads: int, train: bool, out_dtype: torch.dtype, qke: list[float],
) -> list[torch.Tensor]:
    """[out, *saved]: see the module docstring (training forward); ``train`` False keeps nothing for a backward (empty saved slots).
    ``qke``: the QK-norm epsilons (q, k) with its weights the last two ``params``, or empty (no QK-norm)."""
    from miniworld_engine.kernels._tf32 import tf32_matmul

    wq, bq, wk, wv, wg, wo, wsc, bsc, lnw, wb = params[:10]
    A, _, L, D = x.shape
    H = heads
    DH = D // H
    dt = x.dtype
    bf = dt is BF
    dev = x.device
    Ap, Lp, Lb = _dims(A, L)
    M, Mp = A * L, Ap * Lp
    padded = (Ap, Lp) != (A, L)
    with tf32_matmul(not bf), torch.autocast("cuda", enabled=False):
        wcat, bcat, wo_, wsc_, bsc_ = _weights(wq, bq, wk, wv, wg, wo, wsc, bsc, dt)
        x2 = x.reshape(M, D)
        xp = F.pad(x.reshape(A, L, D), (0, 0, 0, Lp - L, 0, Ap - A)).view(Mp, D) if padded else x2
        qkvg = torch.addmm(bcat, xp, wcat.t())
        if not bf:   # the TF32 core's operands on the TF32 grid (QK-norm: q / k rounded by the norm pass)
            c0 = 2 * D if qke else 0
            n4 = Mp * (3 * D - c0) // 4
            _k("aam_round_qkv", _index(x))(_n4grid(n4, _index(x)), (256, 1, 1), qkvg[:, c0:], int(4 * D), int(n4), int(3 * D - c0))
        bias, pst, _ = _bias(pair, lnw, wb, mask, extra, eps, bf, train, L, Lb, Lp, H)
        km = _kmask(mask, A, Ap, L, Lp // 32) if (mask is not None and mask.dim() == 2) else None
        q, k, v, g = (qkvg[:, i * D : (i + 1) * D] for i in range(4))
        qn = kn = rq = None
        if qke:
            qn, kn = (torch.empty(Mp, D, device=dev, dtype=dt) for _ in range(2))
            rq = torch.empty(Mp, 2 * H, device=dev, dtype=F32) if train else None
            _qknorm(qkvg, params[10], params[11], qn, kn, rq, H, DH, qke, 1.0)
            q, k = qn, kn
        O, LSE = _core_fwd(q, k, v, bias, Ap, Lp, H, DH, bf, km)
        og = torch.empty(Mp, D, device=dev, dtype=dt)
        _gate_o(O, g, og)
        y = torch.mm(og, wo_.t())
        sp = torch.addmm(bsc_, cond.reshape(M, cond.shape[-1]), wsc_.t())
        out = torch.empty(A, 1, L, D, device=dev, dtype=out_dtype)
        _res_gate(None if res is None else res.reshape(M, D), sp, y, out.view(M, D), L, Lp)
    e = x.new_empty
    if not train:
        return [out, *(e((0, 1)) for _ in range(16))]
    # custom-op outputs may not alias the inputs: the fp32 path's Wo / Wsc are the parameters themselves (the caller keeps those)
    wo_c = wo_ if wo_ is not wo else e((0, D))
    wsc_c = wsc_ if wsc_ is not wsc else e((0, wsc.shape[1]))
    kmo = km if km is not None else e((0,), dtype=torch.int32)
    qx = [qn, kn, rq] if qke else [e((0, D)), e((0, D)), e((0, 2 * H), dtype=F32)]
    return [out, xp if padded else e((0, D)), qkvg, O, LSE, og, bias, y, sp, pst, wcat, wo_c, wsc_c, kmo, *qx]


# --------------------------------------------------------------------------------------------------- training backward
def _bwd_fake(dout, x, xp, cond, pair, extra, lnw, wb, qkvg, O, LSE, og, bias, y, sp, pst, wcat, wo, wsc, km, qn, kn, rq, nq, nk, eps,
              has_res, fixed):
    A, _, L, D = x.shape
    dt = x.dtype
    e = x.new_empty
    dh = D // LSE.shape[1]
    return [
        e((dh if qn.numel() else 0,), dtype=F32),
        e((dh if qn.numel() else 0,), dtype=F32),
        e((A, 1, L, D), dtype=dt),
        e(cond.shape, dtype=dt),
        e(pair.shape, dtype=pair.dtype),
        e(lnw.shape, dtype=F32),
        e(wb.shape, dtype=F32),
        e((4 * D, D), dtype=F32),
        e((D,), dtype=F32),
        e((D, D), dtype=F32),
        e(wsc.shape, dtype=F32),
        e((D,), dtype=F32),
        e((L, L) if extra is not None else (0,), dtype=F32),
    ]


def _bwd_tf32_into(q, k, v, do, bias, LSE, Dd, A, L, H, DH, dk, dv, km=None):
    """The TF32 core's deterministic backward with dK / dV written straight into ``dk`` / ``dv`` (column views of the q|k|v|g gradient):
    (dQ partials [L / 64, A L, W], dbias [H, L, L]) fp32."""
    dev = _index(q)
    W, M, nch = H * DH, A * L, L // 64
    f32 = dict(dtype="f32")
    DQ = torch.empty(nch, M, W, device=q.device, dtype=F32)
    DB = torch.empty(H, L, L, device=q.device, dtype=F32)
    dims = [W, M]
    a = lambda t, rows: _tm(t, dims, _rs(t), [32, rows], swizzle=128, **f32)  # noqa: E731
    b = lambda t, rows: _tm(t, dims, _rs(t), [16, rows], swizzle=64, **f32)  # noqa: E731
    m = lambda t, rows: _tm(t, dims, _rs(t), [32, rows], swizzle="128a32", **f32)  # noqa: E731
    dkv_maps = (
        a(q, 32), b(q, 32), m(q, 32), a(do, 32), b(do, 32), m(do, 32),
        a(k, 128), b(k, 128), a(v, 128), b(v, 128),
        _tm(bias, [L, H * L], L * 4, [32, 32], swizzle=0, **f32),
        a(dk, 128), b(dk, 128), a(dv, 128), b(dv, 128),
    )
    dqb_maps = (
        a(q, 128), b(q, 128), a(do, 128), b(do, 128), a(v, 64), b(v, 64), a(k, 64), b(k, 64), m(k, 64),
        _tm(bias, [L, H * L], L * 4, [32, 128], swizzle=128, **f32),
    )
    cd = (*_core_defs(H, DH), *_km_defs(km))
    kx = () if km is None else (km,)
    _sm100_kernel("attn_dkv_tf32", "augattn_dkv_tf32_sm100", dev, defs=cd)(
        _grid(A * H * (L // 128), dev), (384, 1, 1), *dkv_maps, LSE, Dd, None, int(L), int(A), int(W), *kx
    )
    _sm100_kernel("attn_dqb_tf32", "augattn_dqb_tf32_sm100", dev, defs=(*cd, "DQPART=1"))(
        _grid(H * (L // 128) * nch, dev), (384, 1, 1), *dqb_maps, LSE, Dd, DQ, DB, int(L), int(A), 1, int(W), *kx
    )
    return DQ, DB


@opaque(fake=_bwd_fake, name="augmented_attention_module_sm100_bwd")
def _bwd(
    dout: torch.Tensor, x: torch.Tensor, xp: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, extra: torch.Tensor | None,
    lnw: torch.Tensor, wb: torch.Tensor, qkvg: torch.Tensor, O: torch.Tensor, LSE: torch.Tensor, og: torch.Tensor, bias: torch.Tensor,
    y: torch.Tensor, sp: torch.Tensor, pst: torch.Tensor, wcat: torch.Tensor, wo: torch.Tensor, wsc: torch.Tensor, km: torch.Tensor,
    qn: torch.Tensor, kn: torch.Tensor, rq: torch.Tensor, nq: torch.Tensor | None, nk: torch.Tensor | None, eps: float, has_res: bool,
    fixed: bool,
) -> list[torch.Tensor]:
    """[dnq, dnk (QK-norm weights, or empty), dx, dcond, dpair, dlnw, dwb, dWcat, dbq, dWo, dWsc, dbsc, dextra] (weight gradients fp32)."""
    from miniworld_engine.kernels._tf32 import tf32_matmul

    A, _, L, D = x.shape
    Ap, H, Lp = LSE.shape
    _, _, Lb = _dims(A, L)
    DH = D // H
    dt = x.dtype
    bf = dt is BF
    dev = _index(x)
    M, Mp = A * L, Ap * Lp
    padded = (Ap, Lp) != (A, L)
    DC = cond.shape[-1]
    with tf32_matmul(not bf), torch.autocast("cuda", enabled=False):
        do2 = dout.reshape(M, D)
        if not do2.is_contiguous():
            do2 = do2.contiguous()
        # the output gate: dy (padded rows zero), d sp and its column-sum partials
        dy = torch.empty(Mp, D, device=x.device, dtype=dt)
        dsp = torch.empty(M, D, device=x.device, dtype=dt)
        nbr = _row_blocks(Mp, dev)
        part = torch.empty(nbr, D, device=x.device, dtype=F32)
        dbsc = torch.empty(D, device=x.device, dtype=F32)
        tag = ("b" if bf else "f") + ("b" if do2.dtype is BF else "f")
        _k(f"aam_res_gate_bwd_{tag}", dev)(
            (nbr, 1, 1), (D // 8, RB, 1), do2, sp, y, dy, dsp, part, dbsc, _counter(dev, 0), int(Mp), int(D), int(L), int(Lp), int(A)
        )
        dWo = _wgrad(dy, og)
        dog = torch.mm(dy, wo)
        # the attention gate and the core's dO prep
        dqkvg = torch.empty(Mp, 4 * D, device=x.device, dtype=dt)
        dob = torch.empty(Mp, D, device=x.device, dtype=dt)
        dd = torch.empty(Ap, H, Lp, device=x.device, dtype=F32)
        g = qkvg[:, 3 * D :]
        _k("aam_gate_bwd_bf16" if bf else "aam_gate_bwd_f32", dev)(
            (_row_blocks(Mp, dev), 1, 1), (D // 8, RB, 1), dog, O, g, int(g.stride(0)), dob, dd, dqkvg[:, 3 * D :], int(4 * D), int(Mp), int(Lp),
            int(H), int(DH)
        )
        q, k, v = (qkvg[:, i * D : (i + 1) * D] for i in range(3))
        qk = qn.numel() > 0
        if qk:
            q, k = qn, kn
        dbq = torch.empty(D, device=x.device, dtype=F32)
        kmb = km if km.numel() else None
        if bf:
            DQ, DK, DV, DB = _bwd_bf16_km(q, k, v, dob, bias, LSE, dd, Ap, Lp, H, DH, kmb)
            _k("aam_finish_bf16", dev)(
                (nbr, 1, 1), (D // 8, RB, 1), DQ, int(DQ.shape[0]), int(Mp * D), DK, DV, dqkvg, int(4 * D), int(Mp), int(D), part, dbq, _counter(dev, 1)
            )
        else:
            DQ, DB = _bwd_tf32_into(q, k, v, dob, bias, LSE, dd, Ap, Lp, H, DH, dqkvg[:, D : 2 * D], dqkvg[:, 2 * D : 3 * D], kmb)
            _k("aam_finish_f32", dev)(
                (nbr, 1, 1), (D // 8, RB, 1), DQ, int(DQ.shape[0]), int(Mp * D), dqkvg, int(4 * D), int(Mp), int(D), part, dbq, _counter(dev, 1)
            )
        del DQ
        dnq, dnk = (x.new_empty((0,), dtype=F32) for _ in range(2))
        if qk:   # dq / dk of the normalized q / k -> of the raw ones, in place; dbq and the norm weights' gradients from its partials
            nbq = _row_blocks(Mp, dev)
            qpart = torch.empty(nbq, 3 * D, device=x.device, dtype=F32)
            nq32, nk32 = (w.detach().float().contiguous() for w in (nq, nk))
            _k("aam_qknorm_bwd_bf16" if bf else "aam_qknorm_bwd_f32", dev)(
                (nbq, 1, 1), (D // 8, RB, 1), qkvg, dqkvg, int(4 * D), nq32, nk32, rq, qpart, int(Mp), int(H), int(DH)
            )
            sums = qpart.sum(0)   # fixed-order reduction over the block partials
            dbq = sums[:D]
            dnq = sums[D : 2 * D].view(H, DH).sum(0)
            dnk = sums[2 * D :].view(H, DH).sum(0)
        xin = xp if padded else x.reshape(M, D)
        dxp = torch.mm(dqkvg, wcat)
        dWcat = _wgrad(dqkvg, xin)
        dx = dxp.view(Ap, Lp, D)[:A, :L].contiguous().view(A, 1, L, D) if padded else dxp.view(A, 1, L, D)
        # the pair side (dbias in natural units; masked keys carry zero dbias)
        dxt = x.new_empty((0,), dtype=F32)
        if Lp != Lb or extra is not None:   # (a fix that only folded the key mask has nothing to undo: masked keys carry zero dbias)
            dpb = DB if (Lp == Lb) else torch.empty(H, Lb, Lb, device=x.device, dtype=F32)
            if extra is not None:
                dxt = torch.empty(L, L, device=x.device, dtype=F32)
            _k("aam_dbias_fix", dev)((Lb, 1, 1), (256, 1, 1), DB, dpb, dxt if extra is not None else None, 1, int(H), int(L), int(Lb), int(Lp))
            DB = dpb
        dp = pair.shape[-1]
        pz = pair if Lb == L else F.pad(pair, (0, 0, 0, Lb - L, 0, Lb - L))
        p2 = pz.reshape(Lb * Lb, dp)
        if pair.dtype is BF:
            from ..apb import acc_n, ext, pair_bias_bwd, width

            acc = torch.empty(ext().pair_bwd_blocks(pz, H, Lb, 1), acc_n(H, 384, dp), device=x.device, dtype=F32)
            dz, _, _ = pair_bias_bwd(p2, DB, _wf(wb, lnw, BF), Lb, eps, acc, 384)
            off = 2 * 384 + width(H, 384)
            dwf = acc[:, off : off + H * dp].sum(0).view(H, dp)
        else:
            from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

            R = rows32()
            dz = torch.empty_like(p2)
            pwf = torch.empty(R.partial_rows(Mp), H, dp, device=x.device, dtype=F32)
            nwf = R.pair_bias_bwd_tc_cuda(DB.view(H, Lb * Lb), p2, pst, _wf(wb, lnw, F32), dz, pwf, 1)
            dwf = pwf[:nwf].sum(0)
        dpair = dz.view(1, Lb, Lb, dp)
        dpair = dpair[:, :L, :L].contiguous() if Lb != L else dpair
        dlnw = (dwf * wb.float()).sum(0)
        dwb = dwf * lnw.float()
        # the output gate's cond projection
        c2 = cond.reshape(M, DC)
        dcond = torch.mm(dsp, wsc)
        dWsc = _wgrad(dsp, c2 if c2.is_contiguous() else c2.contiguous())
    return [dnq, dnk, dx, dcond.view(cond.shape), dpair, dlnw, dwb, dWcat, dbq, dWo, dWsc, dbsc, dxt]


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, res, pair, mask, extra, eps, heads, out_dtype, qke, *params):  # noqa: D102
        A, _, L, D = x.shape
        out, xp, qkvg, O, LSE, og, bias, y, sp, pst, wcat, wo, wsc, km, qn, kn, rq = _fwd(
            x, cond, res, pair, mask, extra, list(params), eps, heads, True, out_dtype, list(qke)
        )
        if wo.numel() == 0:
            wo = params[5]
        if wsc.numel() == 0:
            wsc = params[6]
        Ap, Lp, Lb = _dims(A, L)
        fixed = _fixed(mask, extra, pair.dtype, True, L, Lb, Lp)
        nqk = (params[10], params[11]) if qke else (None, None)
        ctx.save_for_backward(x, xp, cond, pair, extra, params[8], params[9], qkvg, O, LSE, og, bias, y, sp, pst, wcat, wo, wsc, km, qn, kn, rq, *nqk)
        ctx.meta = (eps, res is not None, fixed, [p.dtype for p in params], extra is not None and ctx.needs_input_grad[5])
        return out

    @staticmethod
    def backward(ctx, dout):  # noqa: D102
        x, xp, cond, pair, extra, lnw, wb, qkvg, O, LSE, og, bias, y, sp, pst, wcat, wo, wsc, km, qn, kn, rq, nq, nk = ctx.saved_tensors
        eps, has_res, fixed, pdt, xgrad = ctx.meta
        dnq, dnk, dx, dcond, dpair, dlnw, dwb, dWcat, dbq, dWo, dWsc, dbsc, dxt = _bwd(
            dout, x, xp, cond, pair, extra, lnw, wb, qkvg, O, LSE, og, bias, y, sp, pst, wcat, wo, wsc, km, qn, kn, rq, nq, nk, eps, has_res,
            fixed
        )
        D = x.shape[-1]
        grads = (dWcat[:D], dbq, dWcat[D : 2 * D], dWcat[2 * D : 3 * D], dWcat[3 * D :], dWo, dWsc, dbsc, dlnw, dwb,
                 *((dnq, dnk) if nq is not None else ()))
        pg = tuple(gr if gr.dtype is dt else gr.to(dt) for gr, dt in zip(grads, pdt))
        dres = dout if has_res else None
        return (dx, dcond, dres, dpair if ctx.needs_input_grad[3] else None, None,
                (dxt.to(extra.dtype) if xgrad else None), None, None, None, None, *pg)


# --------------------------------------------------------------------------------------------------- inference (bf16): attn_inf
def _inf_bf16_fake(x, cond, res, pair, mask, extra, params, eps, heads, out_dtype, qke):
    return x.new_empty(x.shape, dtype=out_dtype)


@opaque(fake=_inf_bf16_fake, name="augmented_attention_module_sm100_inf_bf16")
def _inf_bf16(
    x: torch.Tensor, cond: torch.Tensor, res: torch.Tensor | None, pair: torch.Tensor, mask: torch.Tensor | None, extra: torch.Tensor | None,
    params: list[torch.Tensor], eps: float, heads: int, out_dtype: torch.dtype, qke: list[float],
) -> torch.Tensor:
    """The bf16 inference update (see the module docstring); QK-norm (``qke``): q / k normalized in place, q scaled after it."""
    from miniworld_engine.kernels import _capture

    wq, bq, wk, wv, wg, wo, wsc, bsc, lnw, wb = params[:10]
    A, _, L, D = x.shape
    H, DH = heads, D // heads
    M = A * L
    dev = _index(x)
    qs = LOG2E * DH**-0.5
    ws = (wq, bq, wk, wv, wg, wo, wsc, bsc)
    wscale = 1.0 if qke else qs   # QK-norm: the scale follows the norm (in the norm pass)
    wcat, bcat, wo_, wsc_, bsc_ = _capture.param_forms(("aa_sm100_inf", wscale), ws, lambda: _weights(*ws, BF, qscale=wscale))
    Lb = _ceil(L, PM)   # the pair pass's length and the bias rows' pitch (a multiple of 16: the 16-byte TMA stride holds)
    with torch.autocast("cuda", enabled=False):
        qkvg = torch.addmm(bcat, x.reshape(M, D), wcat.t())
        if qke:
            _qknorm(qkvg, params[10], params[11], qkvg[:, :D], qkvg[:, D : 2 * D], None, H, DH, qke, qs)
        bias, _, _ = _bias(pair, lnw, wb, mask, extra, eps, True, False, L, Lb, Lb, H, log2=True)
        km = None
        if mask is not None and mask.dim() == 2:
            km = _kmask(mask, A, A, L, 2 * -(-L // 64))
        _attn_inf(qkvg, bias, A, L, Lb, H, DH, dev, km)
        y = torch.mm(qkvg[:, :D], wo_.t())
        sp = torch.addmm(bsc_, cond.reshape(M, cond.shape[-1]), wsc_.t())
        out = torch.empty(A, 1, L, D, device=x.device, dtype=out_dtype)
        _res_gate(None if res is None else res.reshape(M, D), sp, y, out.view(M, D), L, L)
    return out


_INF_RUNS: dict = {}


def _attn_inf(qkvg, bias, S, L, Lb, H, DH, dev, km=None):
    """attn_inf over q|k|v|g [S L, 4 W] bf16 (exp2 units) and the bias [H, Lb, Lb] (exp2 units, Lb >= L, keys past L masked in the
    kernel): sigmoid(g) o written over q. Launch bindings cached by the buffers' addresses."""
    from miniworld_engine.kernels._device import num_sms

    Mw = qkvg.shape[1]
    W = Mw // 4
    es = 2
    rs = Mw * es
    key = (dev, qkvg.data_ptr(), bias.data_ptr(), S, L, Lb, H, DH, None if km is None else km.data_ptr())
    run = _INF_RUNS.get(key)
    if run is None:
        q, k, v, g = (qkvg[:, i * W : (i + 1) * W] for i in range(4))
        t3 = lambda t, rows: _tm(t, [W, L, S], [rs, L * rs], [DH, rows, 1])  # noqa: E731
        mq, mk, mv, mg = t3(q, 128), t3(k, 64), t3(v, 64), t3(g, 128)
        mb = _tm(bias, [Lb, L, H], [Lb * es, Lb * Lb * es], [64, 128, 1])   # columns L .. Lb: the masked padding
        for m in (mq, mk, mv, mg, mb):
            m.keep = None
        grid = (min(num_sms(dev), ((S + 1) // 2) * H * -(-L // 128)), 1, 1)
        kx = () if km is None else (km, int(km.shape[1]))
        run = _sm100_kernel("attn_inf", "augattn_inf_sm100", dev, defs=(*_tdit_defs(H, DH), *_km_defs(km))).bind(
            grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S), *kx
        )
        if len(_INF_RUNS) >= 64:
            _INF_RUNS.clear()
        _INF_RUNS[key] = run
    run()


# --------------------------------------------------------------------------------------------------- entry
def module_update(x, cond, res, pair, mask, extra, params, eps: float, heads: int, out_dtype: torch.dtype, qke=()):
    """``res + delta`` (``res`` None: ``delta``) for the module's AdaLN output ``x`` [A, 1, L, d] (the compute dtype), ``cond`` [A, 1, L, dc]
    (same dtype), ``res`` the residual stream, ``params`` = (to_query.weight, to_query.bias, to_key.weight, to_value.weight,
    to_gate.weight, to_out.weight, to_scale.weight, to_scale.bias, ln_pair.weight, to_bias.weight). Call ``available()`` first."""
    train = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x, cond, res, pair, extra, *params)
    )
    if mask is not None:
        mask = mask.to(torch.bool)
        mask = (mask.reshape(-1) if mask.dim() == 1 or mask.shape[0] == 1 else mask.reshape(mask.shape[0], -1)).contiguous()
    if not train:
        if x.dtype is BF:
            return _inf_bf16(x, cond, res, pair, mask, extra, list(params), eps, heads, out_dtype, list(qke))
        return _fwd(x, cond, res, pair, mask, extra, list(params), eps, heads, False, out_dtype, list(qke))[0]
    return _Route.apply(x, cond, res, pair, mask, extra, eps, heads, out_dtype, tuple(qke), *params)


__all__ = ["available", "module_update", "supported"]
