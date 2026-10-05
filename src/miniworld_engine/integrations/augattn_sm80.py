"""The A100 (sm_80) path of ``AugmentedAttentionPairBias`` -- hand-written CUDA and cuBLAS from the AdaLN's output to the module's result
(``kernels/augmented_attention/cuda/sm80.py``).

``AugmentedAttentionPairBias.forward`` / ``delta`` call ``serves()`` first; when it accepts, the module runs its AdaLN (``ada_ln_in``, the conditioned LayerNorm: its
own kernels) and hands the normalised stream to ONE autograd Function here, whose forward and backward are each one opaque op (``kernels/_compile.opaque``):

    forward   qkvg = x [Wq; Wk; Wv; Wg]^T + [bq, 0, 0, 0]      cuBLAS, one GEMM (the four weights packed once per parameter version)
              bias  = pair bias of the pair tensor, head-major [B, H, Lp, Lp] in the core's RAW units (natural x sqrt(hd))
                        16 pair channels / 4 heads (the atom width): ``aux_sm80.cuh`` (LayerNorm + projection, one read of the pair)
                        128 pair channels, 8 / 12 / 16 heads: the AttentionPairBias kernel (``apb_pair_bias_sm80.cuh``, tensor-core projection)
                        any other width: ATen LayerNorm + cuBLAS
              o     = the attention core (``sm80.pgate_forward`` / ``plain_forward``: mma.sync, cp.async ring, online softmax, the gate fused in inference)
              y     = (sigmoid(g) o) Wo^T                        cuBLAS
              g2    = cond Ws^T + bs                             cuBLAS
              out   = res + sigmoid(g2) y                        one row pass (``glue_sm80.cuh``); ``res`` is the module's residual (``forward``) or absent (``delta``)
    backward  the same steps reversed: ``res_gate_bwd``, GEMMs for dog / dWo, ``gate_bwd``, the core's backward (dq / dk / dv written straight into the projection's
              gradient, the bias gradient summed over the samples from bf16 partials in a fixed order), GEMMs for dx / dW | dcond / dWs, the pair bias's backward.

A key mask [B, L] shared by the samples is folded into the bias (masked keys carry a very negative FINITE bias: a sample with no valid key gets the uniform softmax); a
mask that differs per sample [A, B, L] goes through the kernels' per-key penalties (a sample with no valid key gets a zero output, as the Triton kernel's finite guard).
L is padded to a multiple of 128 inside the call (zero rows, the padded keys masked), B > 1 runs one core call per batch element on b-major staged streams.

``serves()`` is the whole gate: the engine's kernel backend (implementation TRITON or MINIWORLD), A100 (sm_80), single / cond / pair and module weights all bf16 or all fp32
(``compute_dtype`` None or that dtype; bf16 activations also take fp32 master parameters: cast to bf16 inside the ops, unrounded fp32 weight gradients), no QK-norm, head dim 32 or 48, any A, B and L, a key mask [B, L] or [A, B, L] (bool) or none, LayerNorm eps 1e-5. Anything else
keeps the module path. ``MINIWORLD_AUGATTN_SM80=0`` turns the path off, ``MINIWORLD_AUGATTN_SM80_FUSED=0`` keeps the earlier composition (bf16 only: the module's own
projections and PyTorch gates around the core; the pair-bias cache and the fused atom pair bias stay), and a failed extension build warns once and keeps the module path.
Numerics and timings: ``docs/gpus/a100/atom_dit/atom_dit.md``, ``docs/gpus/a100/token_dit/token_dit.md``, ``docs/gpus/a100/a100.md``.

fp32 (all four tensors and the weights): the same flow with the TF32 tensor-core core (``kernels/augmented_attention/cuda/sm80/*_tf32_sm80.cuh``: fp32 operands rounded to
TF32 inside the mma, fp32 accumulation and softmax -- what the Triton kernel does), the bias in natural units (fp32, masked keys -1e4), and the GEMMs on cuBLAS under the
caller's ``allow_tf32``.
"""

from __future__ import annotations

import functools
import math
import os
import warnings

import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

BF = torch.bfloat16
F32 = torch.float32
EPS = 1e-5
#: a masked or padded key's bias in natural units (bf16 -9984; the core's raw units multiply it by sqrt(hd)): its weight is 2^(-9984 log2 e + ...) = 0 in fp32 for
#: every row with a valid key
MASKED = -1e4
_ATOM = (4, 32, 16)                       # (heads, head dim, pair channels) of the fused pair bias
_FAILED = False
_NEG_INF = float("-inf")


def _sm80():
    from miniworld_engine.kernels.augmented_attention.cuda import sm80
    return sm80


def _pad(length: int) -> int:
    return -(-length // 128) * 128


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the extensions of the path (the core, the gates, the AttentionPairBias pair kernels); False, with one warning, when the toolchain fails
    (the module path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _FAILED
    if _FAILED:
        return False
    try:
        sm80 = _sm80()
        sm80._ext()
        sm80._glue_ext()
        _apb().ext()
    except Exception as exc:  # a toolchain problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_80 AugmentedAttentionPairBias kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


@torch.compiler.assume_constant_result
def _loads_tf32() -> bool:
    """:func:`_loads` for the fp32 path: the core's TF32 extension, the shared pair-bias / gate kernels (no AttentionPairBias kernel: its width is bf16 only)."""
    global _FAILED
    if _FAILED:
        return False
    try:
        sm80 = _sm80()
        sm80._ext()
        sm80._glue_ext()
        sm80._tf32_ext()
    except Exception as exc:  # a toolchain problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_80 AugmentedAttentionPairBias kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _apb():
    from miniworld_engine.kernels.augmented_attention.cuda import apb
    return apb


def _fused() -> bool:
    return os.environ.get("MINIWORLD_AUGATTN_SM80_FUSED", "1") != "0"


def _generic_kernel() -> bool:
    """The streaming pair-bias kernel for the widths without a dedicated one (MINIWORLD_AUGATTN_SM80_PBGEN=0: ATen LayerNorm + cuBLAS)."""
    return os.environ.get("MINIWORLD_AUGATTN_SM80_PBGEN", "1") != "0"


def _mask_kind(mask, a: int, b: int, length: int) -> int | None:
    """0 no mask, 1 a key mask shared by the samples ([B, L], or a [A, B, L] view with stride 0 over the samples), 2 one per sample ([A, B, L]); None: not served."""
    if mask is None:
        return 0
    if mask.dtype is not torch.bool:
        return None
    if mask.ndim == 2 and tuple(mask.shape) == (b, length):
        return 1
    if mask.ndim == 3 and tuple(mask.shape) == (a, b, length):
        return 1 if mask.stride(0) == 0 else 2
    return None


def serves(module, single, pair, mask, compute_dtype=None, cond=None) -> bool:
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend
    if module._backend != KernelBackend.TRITON or module.use_qk_norm:
        return False
    dt = single.dtype
    if not single.is_cuda or dt not in (BF, F32) or pair.dtype is not dt or compute_dtype not in (None, dt):
        return False
    if module.to_query.weight.dtype not in _weight_dtypes(dt) or module.to_key.weight.dtype not in _weight_dtypes(dt):
        return False
    if single.ndim != 4 or pair.ndim != 4 or pair.shape[0] != single.shape[1] or tuple(pair.shape[1:3]) != (single.shape[2], single.shape[2]):
        return False
    a, b, length, d = single.shape
    if _mask_kind(mask, a, b, length) is None:
        return False
    heads = module.n_head
    if d % heads or d // heads not in (32, 48) or d % 8 or module.to_bias.weight.shape != (heads, pair.shape[-1]):
        return False
    if module.ln_pair.eps != EPS:
        return False
    fused = _fused()
    if dt is F32 and not fused:                                                        # the earlier composition is the bf16 path
        return False
    if not fused and (module.to_query.weight.dtype is not dt or not _legacy_serves(module, single, pair, mask)):
        return False
    if fused and (not _fused_params_ok(module, dt) or (cond is not None and (cond.dtype is not dt or cond.ndim != 4 or tuple(cond.shape[:3]) != (a, b, length) or cond.shape[-1] % 8))):
        return False
    sm80 = _sm80()
    idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
    if not (sm80.supported_plain(BF, _pad(length), d, heads, idx) if dt is BF else sm80.supported_tf32(F32, _pad(length), d, heads, idx)):
        return False
    if not fused:
        return True
    return _loads() if dt is BF else _loads_tf32()


def _weight_dtypes(dt: torch.dtype) -> tuple[torch.dtype, ...]:
    """The parameter dtypes served over ``dt`` activations: ``dt`` itself, and fp32 masters over bf16 (AMP's bf16-mixed: the op casts them to bf16 outside autograd and hands
    back fp32 weight gradients)."""
    return (BF, F32) if dt is BF else (dt,)


def _fused_params_ok(module, dt: torch.dtype = BF) -> bool:
    """The fused op reads the module's weights as ``dt`` tensors (or fp32 masters over bf16) in the nn.Linear layout."""
    ok = _weight_dtypes(dt)
    linears = (module.to_query, module.to_key, module.to_value, module.to_gate, module.to_out, module.to_scale, module.to_bias)
    if any(lin.weight.dtype not in ok for lin in linears) or module.to_query.bias is None or module.to_scale.bias is None:
        return False
    if module.to_key.bias is not None or module.to_value.bias is not None or module.to_gate.bias is not None or module.to_out.bias is not None or module.to_bias.bias is not None:
        return False
    return not (module.to_query.bias.dtype not in ok or module.to_scale.bias.dtype not in ok or module.ln_pair.weight is None or module.ln_pair.bias is not None)


# ---------------------------------------------------------------------------------------------------------------------------------- the fused op
# staging: the module's [A, B, L, c] streams -> b-major [B A Lp, c] rows (zero rows after L): a view when B == 1 and L is a multiple of 128
def _stage(t: torch.Tensor, batch: int, padded: int) -> torch.Tensor:
    a, _, length, c = t.shape
    if batch == 1 and length == padded:
        return t.reshape(a * length, c)
    t = t.permute(1, 0, 2, 3)
    if padded != length:
        t = F.pad(t, (0, 0, 0, padded - length))
    return t.reshape(batch * a * padded, c)


def _unstage(t: torch.Tensor, a: int, batch: int, length: int, padded: int, c: int) -> torch.Tensor:
    if batch == 1 and length == padded:
        return t.view(a, 1, length, c)
    return t.view(batch, a, padded, c)[:, :, :length].permute(1, 0, 2, 3).contiguous()


def _penalties(kmask: torch.Tensor, a: int, batch: int, length: int, padded: int) -> torch.Tensor:
    """The per-sample key penalties of the kernels: fp32 [B, A, Lp], 0 where a key is valid, -inf where it is masked (and past L)."""
    pen = torch.full((batch, a, padded), _NEG_INF, device=kmask.device, dtype=torch.float32)
    pen[:, :, :length].masked_fill_(kmask.permute(1, 0, 2), 0.0)
    return pen


_PACKS: dict = {}


def _pack(params: list[torch.Tensor], heads: int, dt: torch.dtype):
    """The module's weights in the kernels' layouts (``dt``), rebuilt only when a parameter changes (an optimizer step bumps ``_version``): W q | k | v | g [4 d, d] and its bias
    [4 d] (bq, then zeros), then the folded pair-bias weights: wf = to_bias.weight x ln_pair.weight (natural units, ``dt``), wfr = wf x sqrt(hd) (the core's raw units, ``dt``) and
    w32 = the same product in fp32 (the atom width's kernels).  ``params`` are the module's own parameters (any dtype): the cache keys on them, not on casts (fresh tensors per
    call for fp32), and is scoped to the CUDA-graph capture (``kernels._capture``: a capture packs once, recorded, and replays repack the current weights)."""
    wq, bq, wk, wv, wg, _, _, _, lnw, wb = params

    def build():
        d = wq.shape[0]
        hd = d // heads
        wqkvg = torch.cat([wq.to(dt), wk.to(dt), wv.to(dt), wg.to(dt)])
        bvec = torch.cat([bq.to(dt), bq.new_zeros(3 * d, dtype=dt)])
        w32 = (wb.float() * lnw.float()[None]).contiguous()
        return (wqkvg, bvec, w32.to(dt), (w32 * math.sqrt(hd)).to(dt), w32, w32.t().contiguous())

    key = (wq.device.index, heads, dt, *((t.data_ptr(), t._version) for t in (wq, bq, wk, wv, wg, lnw, wb)))
    return _capture.lookup(_PACKS, key, build, limit=64)


def _kind(dp: int, heads: int, dt: torch.dtype = BF) -> str:
    """Which pair-bias producer: the atom width's (bf16 or fp32), AttentionPairBias's 128-channel kernel (bf16, 8 / 12 / 16 heads), or the generic ATen + cuBLAS one."""
    return "atom" if (dp, heads) == (_ATOM[2], _ATOM[0]) else "apb" if dp == 128 and heads in (8, 12, 16) and dt is BF else "generic"


def _bias_into(out: torch.Tensor, pair_b: torch.Tensor, pack, heads: int, hd: int, length: int, padded: int, mask_b) -> None:
    """The pair bias of one batch element into ``out`` [H, Lp, Lp] (bf16: the core's raw units, natural x sqrt(hd); fp32: natural units): ``pair_b`` [L, L, dp], ``mask_b`` a
    bool [L] key mask folded into the bias or None; keys past L carry the masked fill, query rows past L are finite."""
    sm80, dp = _sm80(), pair_b.shape[-1]
    f32 = out.dtype is F32
    kind = _kind(dp, heads, out.dtype)
    if kind == "atom":
        sm80._ext().pair_bias_fwd(pair_b.reshape(length, length, sm80.PAIR_C), pack[4], out, sm80._kv(mask_b, pair_b.device), padded, length, EPS,
                                  1.0 if f32 else math.sqrt(sm80.PAIR_HD))
    elif kind == "apb":
        _apb().ext().pair_bias_fwd(pair_b.reshape(length * length, dp), pack[3], mask_b, out, length, EPS, MASKED * math.sqrt(hd))
    elif sm80.pair_bias_generic_supported(dp, heads, pair_b.dtype) and pair_b.dtype is out.dtype and _generic_kernel():   # any width: LayerNorm + projection streamed in one pass
        sm80.pair_bias_generic(pair_b, pack[5], mask_b, padded, EPS, oscale=1.0 if f32 else math.sqrt(hd), out=out)
    else:
        z, _, _ = torch.native_layer_norm(pair_b.reshape(length * length, dp), (dp,), None, None, EPS)
        raw = torch.mm(pack[2] if f32 else pack[3], z.t()).view(heads, length, length)
        fill = MASKED if f32 else MASKED * math.sqrt(hd)
        out.fill_(fill)
        out[:, :length, :length].copy_(raw)
        if mask_b is not None:
            out[:, :length, :length].masked_fill_(~mask_b.reshape(1, 1, length), fill)
        if padded != length:
            out[:, length:, :].zero_()


def _fwd_fake(x, cond, pair, res, kmask, bias_in, params, a, batch, length, heads, mask_kind, save):
    padded = _pad(length)
    d = x.shape[-1]
    m = a * batch * padded
    outs = [x.new_empty((a, batch, length, d))]
    if save:
        outs += [x.new_empty((m, 4 * d)), x.new_empty((m, d)), x.new_empty((batch * a, heads, padded), dtype=torch.float32), x.new_empty((m, d)), x.new_empty((m, d)),
                 x.new_empty((m, d)), x.new_empty((batch, heads, padded, padded))]
    return outs


@opaque(fake=_fwd_fake, name="augmented_attention_sm80_module_fwd")
def _fwd(x: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, res: torch.Tensor | None, kmask: torch.Tensor | None, bias_in: torch.Tensor | None,
         params: list[torch.Tensor], a: int, batch: int, length: int, heads: int, mask_kind: int, save: bool) -> list[torch.Tensor]:
    """The module's update after the AdaLN: x [A, B, L, d] (the AdaLN's output), cond [A, B, L, dc], pair [B, L, L, dp], res [A, B, L, d] or None (the residual), kmask
    a bool [B, L] / [A, B, L] key mask or None (``mask_kind`` 1 / 2 / 0), bias_in a pair bias made earlier (the compute dtype [1, H, Lp, Lp]: ``cache_pair_bias``) or None, params =
    (Wq, bq, Wk, Wv, Wg, Wo, Ws, bs, ln_pair.weight, to_bias.weight). Returns ``[out]``, or with ``save`` also the activations of the backward: qkvg, o, lse, og, y, g2, bias."""
    sm80 = _sm80()
    dt = x.dtype
    f32 = dt is F32
    d = x.shape[-1]
    hd = d // heads
    padded = _pad(length)
    m = a * batch * padded
    n = a * padded                                                             # rows of one batch element
    dev = x.device
    wo, ws, bs = params[5].to(dt), params[6].to(dt), params[7].to(dt)             # no copies for bf16 parameters (the autograd function hands cast ones for fp32)
    pack = _pack(params, heads, dt)
    wqkvg, bvec = pack[0], pack[1]
    xs, cs = _stage(x, batch, padded), _stage(cond, batch, padded)
    rs = None if res is None else _stage(res, batch, padded)
    qkvg = torch.addmm(bvec, xs, wqkvg.t())                                   # [M, 4 d]
    q, k, v, g = (qkvg[:, i * d:(i + 1) * d] for i in range(4))
    if bias_in is not None:
        bias = bias_in
    else:
        bias = torch.empty(batch, heads, padded, padded, device=dev, dtype=dt)
        for b in range(batch):
            _bias_into(bias[b], pair[b], pack, heads, hd, length, padded, kmask[b] if mask_kind == 1 else None)
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    if save:
        o = torch.empty(m, d, device=dev, dtype=dt)
        lse = torch.empty(batch * a, heads, padded, device=dev, dtype=torch.float32)
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        kp = None if kpen is None else kpen[b]
        if f32 and save:
            sm80.tf32_forward(q[rows], k[rows], v[rows], bias[b], a, padded, heads, hd, kpen=kp, out=o[rows], lse=lse[b * a:(b + 1) * a])
        elif f32:
            sm80.tf32_forward(q[rows], k[rows], v[rows], bias[b], a, padded, heads, hd, gate=g[rows], out=q[rows], kpen=kp)
        elif save:
            sm80.plain_forward(q[rows], k[rows], v[rows], bias[b], a, padded, heads, hd, save_lse=True, kpen=kp, out=o[rows], lse=lse[b * a:(b + 1) * a])
        else:
            sm80.pgate_forward(q[rows], k[rows], v[rows], g[rows], bias[b], a, padded, heads, hd, kpen=kp, out=q[rows])
    if save:
        og = torch.empty(m, d, device=dev, dtype=dt)
        sm80.gate_rows(o, g, og)
    else:
        og = q                                                                 # sigmoid(g) o, written over q's columns
    y = torch.mm(og, wo.t())
    g2 = torch.addmm(bs, cs, ws.t())
    out = torch.empty(m, d, device=dev, dtype=dt)
    sm80.res_gate(y, g2, rs, out)
    outs = [_unstage(out, a, batch, length, padded, d)]
    if save:
        outs += [qkvg, o, lse, og, y, g2, bias]
    return outs


def _mm32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` with an fp32 result: bf16 operands accumulate in fp32 and round once at the end of the output (``out_dtype``), fp32 operands are the plain GEMM."""
    return torch.mm(a, b, out_dtype=F32) if a.dtype is BF else torch.mm(a, b)


def _bwd_fake(dout, x, cond, pair, kmask, params, saved, a, batch, length, heads, mask_kind):
    return [torch.empty_like(x), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(p) for p in params)]


@opaque(fake=_bwd_fake, name="augmented_attention_sm80_module_bwd")
def _bwd(dout: torch.Tensor, x: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, kmask: torch.Tensor | None, params: list[torch.Tensor], saved: list[torch.Tensor],
         a: int, batch: int, length: int, heads: int, mask_kind: int) -> list[torch.Tensor]:
    """The backward of ``_fwd`` (``save=True``): [d x, d cond, d pair, *d params], each in its input's dtype (the residual's gradient is dout itself: the caller returns it)."""
    sm80 = _sm80()
    qkvg, o, lse, og, y, g2, bias = saved
    dt = x.dtype
    f32 = torch.float32
    d = x.shape[-1]
    hd = d // heads
    padded = _pad(length)
    m = a * batch * padded
    n = a * padded
    dev = x.device
    dp = pair.shape[-1]
    lnw, wb = params[8], params[9]
    wo, ws = params[5].to(dt), params[6].to(dt)
    pack = _pack(params, heads, dt)
    wqkvg, wf, w32 = pack[0], pack[2], pack[4]
    xs, cs = _stage(x, batch, padded), _stage(cond, batch, padded)
    ds = _stage(dout, batch, padded)
    dy = torch.empty(m, d, device=dev, dtype=dt)
    dg2 = torch.empty(m, d, device=dev, dtype=dt)
    sm80.res_gate_bwd(ds, y, g2, dy, dg2)
    dog = dy @ wo                                                              # [M, d]
    dwo = _mm32(dy.t(), og)
    dob = torch.empty(m, d, device=dev, dtype=dt)
    dqkvg = torch.empty(m, 4 * d, device=dev, dtype=dt)
    sm80.gate_bwd(dog, o, qkvg[:, 3 * d:], dob, dqkvg[:, 3 * d:])
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    kind = _kind(dp, heads, dt)
    acc = torch.zeros(_apb().acc_n(heads, 384), device=dev) if kind == "apb" else None
    dzs, dw_sum = [], None
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        core_bwd = sm80.tf32_backward if dt is F32 else functools.partial(sm80.plain_backward, db_natural=True)
        _, _, _, db = core_bwd(qkvg[rows, :d], qkvg[rows, d:2 * d], qkvg[rows, 2 * d:3 * d], dob[rows], bias[b], o[rows], lse[b * a:(b + 1) * a], a, padded, heads, hd,
                               kpen=None if kpen is None else kpen[b], dq=dqkvg[rows, :d], dk=dqkvg[rows, d:2 * d], dv=dqkvg[rows, 2 * d:3 * d])
        mask_b = kmask[b] if mask_kind == 1 else None
        pair_b = pair[b]
        if kind == "atom":
            dz, dw = sm80.pair_bias_backward_w(pair_b.unsqueeze(0), w32, db, None if mask_b is None else mask_b.unsqueeze(0), EPS)
            dzs.append(dz.view(length, length, dp))
            dw_sum = dw if dw_sum is None else dw_sum + dw
        elif kind == "apb":
            dz, _, _ = _apb().pair_bias_bwd(pair_b.reshape(length * length, dp), db, wf, length, EPS, acc, 384)
            dzs.append(dz.view(length, length, dp))
        elif sm80.pair_bias_generic_bwd_supported(dp, heads, pair_b.dtype) and _generic_kernel():   # any supported width: dz and dW' in one pass over the pair
            dz, dw = sm80.pair_bias_generic_backward(pair_b, pack[5], db, mask_b, padded, EPS)
            dzs.append(dz.view(length, length, dp))
            dw_sum = dw if dw_sum is None else dw_sum + dw
        else:
            db_c = db[:, :length, :length]
            if mask_b is not None:
                db_c = db_c.masked_fill(~mask_b.reshape(1, 1, length), 0.0)           # a masked key has no gradient
            dbb = db_c.reshape(heads, length * length).to(dt)
            p2 = pair_b.reshape(length * length, dp)
            zz, mean, rstd = torch.native_layer_norm(p2, (dp,), None, None, EPS)
            dwf_b = _mm32(dbb, zz)                                             # [H, dp]: the gradient of the natural-unit folded weight
            dzh = torch.mm(dbb.t(), wf)                                        # [L L, dp]
            dzs.append(torch.ops.aten.native_layer_norm_backward(dzh, p2, (dp,), mean, rstd, None, None, [True, False, False])[0].view(length, length, dp))
            dw_sum = dwf_b if dw_sum is None else dw_sum + dwf_b
    dpair = dzs[0].view(pair.shape) if batch == 1 else torch.stack(dzs)
    dx = torch.mm(dqkvg, wqkvg)                                                # [M, d]
    dwqkvg = _mm32(dqkvg.t(), xs)                                              # [4 d, d]
    dbq = torch.sum(dqkvg[:, :d], dim=0, dtype=f32)
    dcond = torch.mm(dg2, ws)                                                  # [M, dc]
    dws = _mm32(dg2.t(), cs)                                                   # [d, dc]
    dbs = torch.sum(dg2, dim=0, dtype=f32)
    # the pair bias's parameter gradients: d wf (the folded weight, natural units) -> d to_bias.weight = d wf x ln_pair.weight, d ln_pair.weight = sum_h d wf x to_bias.weight
    if kind == "apb":
        o_ = 2 * 384 + _apb().width(heads, 384)
        dw_sum = acc[o_:o_ + heads * dp].view(heads, dp)
    dlnw = (dw_sum * wb.float()).sum(0)
    dwb = dw_sum * lnw.float()[None]
    grads = [dwqkvg[:d], dbq, dwqkvg[d:2 * d], dwqkvg[2 * d:3 * d], dwqkvg[3 * d:], dwo, dws, dbs, dlnw, dwb]
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(outs, [g_.reshape(p.shape) for g_, p in zip(grads, params, strict=True)])
    return [_unstage(dx, a, batch, length, padded, d), _unstage(dcond, a, batch, length, padded, cond.shape[-1]), dpair, *outs]


class _Fused(torch.autograd.Function):
    """x (the AdaLN's output), cond, pair, res (the residual or None), a key mask and the ten parameters -> the module's update (+ the residual)."""

    @staticmethod
    def forward(ctx, x, cond, pair, res, kmask, a, batch, length, heads, mask_kind, *params):
        out, *saved = _fwd(x, cond, pair, res, kmask, None, list(params), a, batch, length, heads, mask_kind, True)
        ctx.save_for_backward(x, cond, pair, *params, *saved)
        ctx.meta = (kmask, a, batch, length, heads, mask_kind, len(params), res is not None)
        return out

    @staticmethod
    def backward(ctx, dout):
        kmask, a, batch, length, heads, mask_kind, npar, has_res = ctx.meta
        vals = ctx.saved_tensors
        x, cond, pair = vals[:3]
        params, saved = list(vals[3:3 + npar]), list(vals[3 + npar:])
        dout = dout.contiguous()
        dx, dc, dpair, *pg = _bwd(dout, x, cond, pair, kmask, params, saved, a, batch, length, heads, mask_kind)
        need = ctx.needs_input_grad
        return (dx if need[0] else None, dc if need[1] else None, dpair if need[2] else None, dout if has_res and need[3] else None, None, None, None, None, None, None,
                *(g if need[10 + i] else None for i, g in enumerate(pg)))


def _params(module) -> list[torch.Tensor]:
    return [module.to_query.weight, module.to_query.bias, module.to_key.weight, module.to_value.weight, module.to_gate.weight, module.to_out.weight, module.to_scale.weight,
            module.to_scale.bias, module.ln_pair.weight, module.to_bias.weight]


def _fused_update(module, x, cond, pair, mask, res):
    """The fused path on the AdaLN's output ``x`` (the caller has checked ``serves``): ``res`` + the update, or the update alone when ``res`` is None."""
    a, batch, length, _ = x.shape
    kind = _mask_kind(mask, a, batch, length)
    kmask = None if kind == 0 else (mask[0] if mask.ndim == 3 and kind == 1 else mask)
    params = _params(module)
    x, cond, pair = x.to(cond.dtype).contiguous(), cond.contiguous(), pair.contiguous()       # the AdaLN of fp32 masters may hand back fp32
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, cond, pair, *([] if res is None else [res]), *params)):
        return _Fused.apply(x, cond, pair, res, kmask, a, batch, length, module.n_head, kind, *params)
    bias_in = None
    if batch == 1 and kind in (0, 1) and (module.n_head, x.shape[-1] // module.n_head, pair.shape[-1]) == _ATOM:
        cached = _cached_bias(module, pair, mask)
        bias_in = None if cached is None else cached.unsqueeze(0)
    return _fwd(x, cond, pair, res, kmask, bias_in, params, a, batch, length, module.n_head, kind, False)[0]




# ================================================================================================================================ the whole op (projected attention)
# ``ops.augmented_attention_pair_bias(q, k, v, bias, mask)`` -- the attention core on already-projected q / k / v [A, B, H, L, D] (head-major), a natural-unit pair bias [B, H, L, L]
# and a key mask [A, B, L] (the registry's ``projected_attention`` rows): the same sm_80 core, with the layouts staged (token-major, head dim 24 run 32 wide with zero columns,
# L padded to 128, b-major) and the bias packed into the core's raw units in one CUDA pass.
def _dpad(dim: int) -> int:
    return 32 if dim == 24 else dim


#: The whole-op path stages q / k / v / the bias / the output around the core (layout copies, a bias pack); the Triton kernel reads the head-major tensors in place. Measured on the A100 (2026-10-04,
#: ``bench.py target=augmented_attention``, the ``projected_attention`` rows): inference is 13-71 % slower than Triton at every registry size; training is faster from A * B * H * L^2 >= 2e6
#: (16 x 768, A = 48: 1.3-1.95x) and at head dim 24 (1.1-1.6x), slower below that at head dim 48 (8 x 384, A = 1, L <= 384: 9-13 %).  So the default (``auto``) serves the calls that need gradients above
#: that size and the Triton kernel keeps the rest; ``MINIWORLD_AUGATTN_SM80_OPS=all`` serves every call, ``0`` none.
_OPS_MIN_WORK = 2_000_000


def _ops_wanted(query, key, value, bias, work: int, dim: int) -> bool:
    mode = os.environ.get("MINIWORLD_AUGATTN_SM80_OPS", "auto")
    if mode in ("all", "0"):
        return mode == "all"
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value, bias))):
        return False
    return dim == 24 or work >= _OPS_MIN_WORK


def serves_ops(query, key, value, bias, mask, kernel_type="compute_efficient") -> bool:
    """The gate of the whole-op path: A100, engine backend not forced to Triton, bf16 q / k / v / bias, [A, B, H, L, D] with head dim 24 / 32 / 48, a bool key mask [A, B, L] / [B, L]
    or none, the compute-efficient kernel type, and (``MINIWORLD_AUGATTN_SM80_OPS``, default ``auto``) a call that needs gradients and is large enough for the core to win (see ``_OPS_MIN_WORK``)."""
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0" or settings.current().engine_backend == "triton" or kernel_type != "compute_efficient":
        return False
    if not query.is_cuda or query.ndim != 5 or any(t.dtype is not BF for t in (query, key, value, bias)):
        return False
    a, b, h, length, dim = query.shape
    if key.shape != query.shape or value.shape != query.shape or tuple(bias.shape) != (b, h, length, length) or dim not in (24, 32, 48):
        return False
    if _mask_kind(mask, a, b, length) is None:
        return False
    if not _ops_wanted(query, key, value, bias, a * b * h * length * length, dim):
        return False
    idx = query.device.index if query.device.index is not None else torch.cuda.current_device()
    return _sm80().supported_plain(BF, _pad(length), h * _dpad(dim), h, idx) and _loads()


def _stage_heads(t: torch.Tensor, a: int, batch: int, length: int, padded: int, heads: int, dim: int) -> torch.Tensor:
    """[A, B, H, L, D] -> b-major token rows [B A Lp, H Dp] (zero past L and past D)."""
    dpad = _dpad(dim)
    t = t.permute(1, 0, 3, 2, 4)
    if padded != length or dpad != dim:
        t = F.pad(t, (0, dpad - dim, 0, 0, 0, padded - length))
    return t.reshape(batch * a * padded, heads * dpad)


def _unstage_heads(t: torch.Tensor, a: int, batch: int, length: int, padded: int, heads: int, dim: int) -> torch.Tensor:
    return t.view(batch, a, padded, heads, _dpad(dim))[:, :, :length, :, :dim].permute(1, 0, 3, 2, 4).contiguous()


def _pack_bias(bias: torch.Tensor, kmask, length: int, padded: int, heads: int, dim: int) -> torch.Tensor:
    """bf16 [B, H, Lp, Lp] in the core's raw units from the natural-unit bias (masked / padded keys at the masked fill, padded query rows 0)."""
    scale, fill = math.sqrt(dim), MASKED * math.sqrt(dim)
    bias = bias.contiguous()
    if length % 8 == 0:
        return _sm80().bias_pack(bias, kmask, length, padded, heads, scale, fill)
    batch = bias.shape[0]
    raw = (bias.float() * scale).to(BF)
    if kmask is not None:
        raw = raw.masked_fill(~kmask[:, None, None, :], fill)
    out = torch.zeros(batch, heads, padded, padded, device=bias.device, dtype=BF)
    out[:, :, :length, :] = fill
    out[:, :, :length, :length] = raw
    return out


def _ops_fwd_fake(q, k, v, bias, kmask, a, batch, length, heads, dim, mask_kind, save):
    padded, dpad = _pad(length), _dpad(dim)
    m = a * batch * padded
    outs = [torch.empty_like(q)]
    if save:
        outs += [q.new_empty((m, heads * dpad)), q.new_empty((m, heads * dpad)), q.new_empty((m, heads * dpad)), q.new_empty((batch, heads, padded, padded)),
                 q.new_empty((m, heads * dpad)), q.new_empty((batch * a, heads, padded), dtype=torch.float32)]
    return outs


@opaque(fake=_ops_fwd_fake, name="augmented_attention_sm80_ops_fwd")
def _ops_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, kmask: torch.Tensor | None, a: int, batch: int, length: int, heads: int, dim: int,
             mask_kind: int, save: bool) -> list[torch.Tensor]:
    """softmax(q k^T / sqrt(D) + bias) v over [A, B, H, L, D] q / k / v (bf16), bias [B, H, L, L], a bool key mask [B, L] / [A, B, L] (``mask_kind`` 1 / 2) or none. Returns
    ``[out]``, or with ``save`` also the staged operands the backward reads: q, k, v rows, the raw-unit bias, o rows, the log-sum-exp."""
    sm80 = _sm80()
    padded, dpad = _pad(length), _dpad(dim)
    m = a * batch * padded
    n = a * padded
    dev = q.device
    qs, ks, vs = (_stage_heads(t, a, batch, length, padded, heads, dim) for t in (q, k, v))
    braw = _pack_bias(bias, kmask if mask_kind == 1 else None, length, padded, heads, dim)
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    o = torch.empty(m, heads * dpad, device=dev, dtype=BF)
    lse = torch.empty(batch * a, heads, padded, device=dev, dtype=torch.float32) if save else None
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        sm80.plain_forward(qs[rows], ks[rows], vs[rows], braw[b], a, padded, heads, dpad, save_lse=save, sm_scale=dim ** -0.5, kpen=None if kpen is None else kpen[b], out=o[rows],
                           lse=None if lse is None else lse[b * a:(b + 1) * a])
    outs = [_unstage_heads(o, a, batch, length, padded, heads, dim)]
    if save:
        outs += [qs, ks, vs, braw, o, lse]
    return outs


def _ops_bwd_fake(dout, qs, ks, vs, braw, o, lse, kmask, a, batch, length, heads, dim, mask_kind, bias_dtype):
    return [torch.empty_like(dout), torch.empty_like(dout), torch.empty_like(dout), dout.new_empty((batch, heads, length, length), dtype=bias_dtype)]


@opaque(fake=_ops_bwd_fake, name="augmented_attention_sm80_ops_bwd")
def _ops_bwd(dout: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, braw: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, kmask: torch.Tensor | None,
             a: int, batch: int, length: int, heads: int, dim: int, mask_kind: int, bias_dtype: torch.dtype) -> list[torch.Tensor]:
    """The backward of ``_ops_fwd``: ``[dq, dk, dv, dbias]`` (q / k / v's dtype and layout, the bias's dtype)."""
    sm80 = _sm80()
    padded, dpad = _pad(length), _dpad(dim)
    m = a * batch * padded
    n = a * padded
    dev = dout.device
    dob = _stage_heads(dout, a, batch, length, padded, heads, dim)
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    dq, dk, dv = (torch.empty(m, heads * dpad, device=dev, dtype=BF) for _ in range(3))
    dbias = torch.empty(batch, heads, length, length, device=dev, dtype=bias_dtype)
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        _, _, _, db = sm80.plain_backward(qs[rows], ks[rows], vs[rows], dob[rows], braw[b], o[rows], lse[b * a:(b + 1) * a], a, padded, heads, dpad, db_natural=True,
                                          sm_scale=dim ** -0.5, kpen=None if kpen is None else kpen[b], dq=dq[rows], dk=dk[rows], dv=dv[rows])
        dbias[b].copy_(db[:, :length, :length])
    return [_unstage_heads(t, a, batch, length, padded, heads, dim) for t in (dq, dk, dv)] + [dbias]


class _OpsAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, kmask, a, batch, length, heads, dim, mask_kind):
        out, *saved = _ops_fwd(q, k, v, bias, kmask, a, batch, length, heads, dim, mask_kind, True)
        ctx.save_for_backward(*saved)
        ctx.meta = (kmask, a, batch, length, heads, dim, mask_kind, bias.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        kmask, a, batch, length, heads, dim, mask_kind, bias_dtype = ctx.meta
        qs, ks, vs, braw, o, lse = ctx.saved_tensors
        dq, dk, dv, dbias = _ops_bwd(dout.contiguous(), qs, ks, vs, braw, o, lse, kmask, a, batch, length, heads, dim, mask_kind, bias_dtype)
        need = ctx.needs_input_grad
        return dq if need[0] else None, dk if need[1] else None, dv if need[2] else None, dbias if need[3] else None, None, None, None, None, None, None, None


def attention_ops(query, key, value, bias, mask):
    """The whole-op attention on the sm_80 core (call ``serves_ops`` first): [A, B, H, L, D] in, [A, B, H, L, D] out (differentiable in q, k, v and the bias)."""
    a, batch, heads, length, dim = query.shape
    kind = _mask_kind(mask, a, batch, length)
    kmask = None if kind == 0 else (mask[0] if mask.ndim == 3 and kind == 1 else mask)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value, bias)):
        return _OpsAttention.apply(query, key, value, bias, kmask, a, batch, length, heads, dim, kind)
    return _ops_fwd(query, key, value, bias, kmask, a, batch, length, heads, dim, kind, False)[0]
# ================================================================================================================================ the earlier composition
# (kept: MINIWORLD_AUGATTN_SM80_FUSED=0 -- the module's own projections and PyTorch gates around the core; the atom width's fused pair bias, the pair-bias cache)
# ---------------------------------------------------------------------------------------------------------------------------------- opaque ops
def _attn_fwd_fake(q, k, v, bias, samples, length, heads, head_dim, save_lse):
    lse = q.new_empty((samples, heads, length), dtype=torch.float32) if save_lse else q.new_empty((0,), dtype=torch.float32)
    return [q.new_empty((samples * length, heads * head_dim)), lse]


@opaque(fake=_attn_fwd_fake, name="augmented_attention_sm80_fwd")
def _attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, samples: int, length: int, heads: int, head_dim: int,
              save_lse: bool) -> list[torch.Tensor]:
    """``softmax(q k^T / sqrt(hd) + bias) v``: q / k / v token-major bf16 ``[A L, H hd]`` (L a multiple of 128), bias head-major bf16 ``[H, L, L]``;
    returns ``[o bf16 [A L, H hd], lse fp32 [A, H, L] or empty]``."""
    o, lse = _sm80().plain_forward(q, k, v, bias, samples, length, heads, head_dim, save_lse=save_lse)
    return [o, lse]


def _attn_bwd_fake(q, k, v, dob, bias, o, lse, samples, length, heads, head_dim, db_natural):
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), bias.new_empty(bias.shape, dtype=torch.float32)]


@opaque(fake=_attn_bwd_fake, name="augmented_attention_sm80_bwd")
def _attn_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dob: torch.Tensor, bias: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, samples: int,
              length: int, heads: int, head_dim: int, db_natural: bool) -> list[torch.Tensor]:
    """The backward: returns ``[dq, dk, dv bf16 [A L, H hd], db fp32 [H, L, L]]``; ``db`` is the gradient with respect to the natural-unit bias (``db_natural``) or
    with respect to the raw-unit ``bias`` that was passed in."""
    dq, dk, dv, db = _sm80().plain_backward(q, k, v, dob, bias, o, lse, samples, length, heads, head_dim, db_natural=db_natural)
    return [dq, dk, dv, db]


def _pb_fwd_fake(z, ln_w, bias_w, kv, n):
    return z.new_empty((bias_w.shape[0], n, n))


@opaque(fake=_pb_fwd_fake, name="augmented_attention_sm80_pair_bias")
def _pb_fwd(z: torch.Tensor, ln_w: torch.Tensor, bias_w: torch.Tensor, kv: torch.Tensor, n: int) -> torch.Tensor:
    """The atom width's pair bias, head-major bf16 ``[4, n, n]``; ``kv`` a bool ``[NZ]`` key mask or an empty tensor."""
    return _sm80().pair_bias_forward(z, ln_w, bias_w, kv if kv.numel() else None, n)


def _pb_bwd_fake(z, ln_w, bias_w, db, kv):
    return [torch.empty_like(z), ln_w.new_empty(ln_w.shape, dtype=torch.float32), bias_w.new_empty(bias_w.shape, dtype=torch.float32)]


@opaque(fake=_pb_bwd_fake, name="augmented_attention_sm80_pair_bias_bwd")
def _pb_bwd(z: torch.Tensor, ln_w: torch.Tensor, bias_w: torch.Tensor, db: torch.Tensor, kv: torch.Tensor) -> list[torch.Tensor]:
    """The backward of ``_pb_fwd``: ``[dz (z's dtype), d ln weight fp32, d bias weight fp32]``."""
    dz, dln, dbw = _sm80().pair_bias_backward(z, ln_w, bias_w, db, kv if kv.numel() else None)
    return [dz, dln, dbw]


# ---------------------------------------------------------------------------------------------------------------------------------- autograd
def _tokens(x: torch.Tensor, samples: int, length: int, padded: int) -> torch.Tensor:
    """``[A, 1, L, H, hd]`` (or ``[A, 1, L, H hd]``) -> token-major ``[A Lp, H hd]``, zero rows after L."""
    x = x.reshape(samples, length, -1)
    if padded != length:
        x = F.pad(x, (0, 0, 0, padded - length))
    return x.reshape(samples * padded, -1)


def _untokens(x: torch.Tensor, samples: int, length: int, padded: int, heads: int, head_dim: int) -> torch.Tensor:
    x = x.view(samples, padded, heads * head_dim)
    if padded != length:
        x = x[:, :length]
    return x.reshape(samples, 1, length, heads, head_dim)


class _Attention(torch.autograd.Function):
    """The core over (q, k, v, bias): q / k / v ``[A, 1, L, H, hd]`` bf16, bias head-major bf16 ``[H, Lp, Lp]`` in raw units (``bias_natural sqrt(hd)``; padded keys
    masked, Lp = L padded to 128); the gradient it returns for ``bias`` is the one with respect to that raw-unit tensor."""

    @staticmethod
    def forward(ctx, q, k, v, bias, samples, length, heads, head_dim, need):
        padded = bias.shape[-1]
        q2, k2, v2 = (_tokens(t, samples, length, padded) for t in (q, k, v))
        need = need and any(ctx.needs_input_grad[:4])
        o, lse = _attn_fwd(q2, k2, v2, bias, samples, padded, heads, head_dim, need)
        if need:
            ctx.save_for_backward(q2, k2, v2, bias, o, lse)
        ctx.meta = (samples, length, padded, heads, head_dim)
        return _untokens(o, samples, length, padded, heads, head_dim)

    @staticmethod
    def backward(ctx, dout):
        q2, k2, v2, bias, o, lse = ctx.saved_tensors
        samples, length, padded, heads, head_dim = ctx.meta
        dob = _tokens(dout.to(BF), samples, length, padded).contiguous()
        dq, dk, dv, db = _attn_bwd(q2, k2, v2, dob, bias, o, lse, samples, padded, heads, head_dim, False)
        return (*(_untokens(t, samples, length, padded, heads, head_dim) for t in (dq, dk, dv)), db, None, None, None, None, None)


class _AtomAttention(torch.autograd.Function):
    """The core with the atom width's pair bias inside: q / k / v ``[A, 1, L, 4, 32]``, pair ``[1, L, L, 16]``, the LayerNorm weight ``[16]`` and ``to_bias``'s
    ``[4, 16]``, a bool key mask ``[L]`` or an empty tensor; the bias never leaves the Function (its gradient goes from the attention backward straight into the
    pair-bias backward, in fp32)."""

    @staticmethod
    def forward(ctx, q, k, v, pair, ln_w, bias_w, kv, samples, length, need):
        padded = _pad(length)
        pair = pair.contiguous()
        bias = _pb_fwd(pair, ln_w, bias_w, kv, padded)
        q2, k2, v2 = (_tokens(t, samples, length, padded) for t in (q, k, v))
        need = need and any(ctx.needs_input_grad[:6])
        o, lse = _attn_fwd(q2, k2, v2, bias, samples, padded, 4, 32, need)
        if need:
            ctx.save_for_backward(q2, k2, v2, bias, o, lse, pair, ln_w, bias_w, kv)
        ctx.meta = (samples, length, padded)
        return _untokens(o, samples, length, padded, 4, 32)

    @staticmethod
    def backward(ctx, dout):
        q2, k2, v2, bias, o, lse, pair, ln_w, bias_w, kv = ctx.saved_tensors
        samples, length, padded = ctx.meta
        dob = _tokens(dout.to(BF), samples, length, padded).contiguous()
        dq, dk, dv, db = _attn_bwd(q2, k2, v2, dob, bias, o, lse, samples, padded, 4, 32, True)
        dz, dln, dbw = _pb_bwd(pair, ln_w, bias_w, db, kv)
        grads = [_untokens(t, samples, length, padded, 4, 32) for t in (dq, dk, dv)]
        return (*grads, dz if ctx.needs_input_grad[3] else None, dln.to(ln_w.dtype) if ctx.needs_input_grad[4] else None,
                dbw.to(bias_w.dtype) if ctx.needs_input_grad[5] else None, None, None, None, None)


def _bias(module, pair, mask, length, head_dim):
    """The pair bias, head-major bf16 ``[H, Lp, Lp]`` in the attention core's raw units (``bias_natural sqrt(hd)``: the units of ``q . k``), padded keys and masked keys at
    ``MASKED`` (natural units): ``LayerNorm(pair) Wb^T`` through the module's own layers (autograd does its backward), permuted and padded.  The scale rides on the
    projection's weight (a ``[H, d_pair]`` tensor), so the GEMM's one bf16 rounding is the bias's only rounding."""
    padded = _pad(length)
    scale = math.sqrt(head_dim)
    w = module.to_bias.weight * scale
    z = module.ln_pair(pair).reshape(length * length, -1).to(w.dtype)
    b = torch.mm(w, z.t()).view(module.n_head, length, length)                          # head-major straight out of the GEMM: no permute copy
    if mask is not None:
        b = torch.where(mask[0][None, None, :], b, torch.full_like(b, MASKED * scale))
    if padded != length:
        b = F.pad(b, (0, padded - length, 0, padded - length))
        pad_cols = torch.zeros(padded, device=b.device, dtype=b.dtype)
        pad_cols[length:] = MASKED * scale
        b = b + pad_cols[None, None, :]
    return b.to(BF).contiguous()


# ---------------------------------------------------------------------------------------------------------------------------------- the opt-in pair-bias cache
# A sampling loop calls a block with the SAME pair tensor and weights at every step, and at the atom width the pair-bias pass is a third of an inference call.  The cache is
# explicit: ``cache_pair_bias(module, pair, mask)`` computes the bias into a buffer the module owns (the same buffer on every refresh: a CUDA graph that captured a call
# keeps reading it), and a later no-grad call with the same tensors (data pointer, shape, strides, version of ``pair``, ``mask`` and the two weights) reads it instead of
# recomputing.  Nothing is cached unless asked.  An in-place change of ``pair`` or a weight through PyTorch bumps its version and the call recomputes; a CUDA-graph
# REPLAY does not run this code, so after changing the graph's static ``pair`` buffer the caller refreshes the cache (one eager ``cache_pair_bias``) before replaying.
_CACHE = "_augattn_sm80_pair_bias"


def _ident(t):
    return None if t is None else (t.data_ptr(), tuple(t.shape), t.stride(), t._version)


def _cache_key(module, pair, mask):
    return (_ident(pair), _ident(mask), _ident(module.ln_pair.weight), _ident(module.to_bias.weight))


def _atom_pair(module, pair) -> bool:
    """The fused pair bias (and so the cache) is the atom width: 4 heads of 32, 16 pair channels, a bf16 or fp32 [1, N, N, 16] pair tensor on an A100."""
    heads = module.n_head
    width = module.to_query.weight.shape[0] // heads
    if not ((heads, width, pair.shape[-1]) == _ATOM and pair.is_cuda and pair.dtype in (BF, F32) and pair.ndim == 4 and pair.shape[0] == 1 and pair.shape[1] == pair.shape[2]):
        return False
    idx = pair.device.index if pair.device.index is not None else torch.cuda.current_device()
    sm80 = _sm80()
    return sm80.supported_plain(BF, _pad(pair.shape[1]), heads * width, heads, idx) if pair.dtype is BF else sm80.supported_tf32(F32, _pad(pair.shape[1]), heads * width, heads, idx)


def cache_pair_bias(module, pair, mask=None) -> bool:
    """Compute and keep the pair bias of ``pair`` / ``mask`` for ``module`` (see above); returns False, caching nothing, when the call would not take this path (not the
    atom width, not an A100, not bf16 / fp32)."""
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0" or not _atom_pair(module, pair):
        return False
    length = pair.shape[1]
    padded = _pad(length)
    entry = getattr(module, _CACHE, None)
    buf = entry[1] if entry is not None and entry[1].shape[-1] == padded and entry[1].device == pair.device and entry[1].dtype is pair.dtype else None
    with torch.no_grad():
        buf = _sm80().pair_bias_forward(pair.contiguous(), module.ln_pair.weight, module.to_bias.weight, mask, padded, out=buf)
    setattr(module, _CACHE, (_cache_key(module, pair, mask), buf))
    return True


def clear_pair_bias(module) -> None:
    """Drop the cached pair bias of ``module`` (its buffer is freed once nothing else holds it)."""
    if hasattr(module, _CACHE):
        delattr(module, _CACHE)


def _cached_bias(module, pair, mask):
    """The cached bias when this no-grad, eager call is the one that was cached, else None."""
    entry = getattr(module, _CACHE, None)
    if entry is None or torch.is_grad_enabled() or torch.compiler.is_compiling() or entry[0] != _cache_key(module, pair, mask):
        return None
    return entry[1]



def _legacy_serves(module, single, pair, mask) -> bool:
    """The gate of the earlier composition: B == 1, a key mask [1, L] or none."""
    if single.ndim != 4 or single.shape[1] != 1 or tuple(pair.shape[:3]) != (1, single.shape[2], single.shape[2]):
        return False
    return mask is None or (mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2]) and mask.dtype is torch.bool)


def _delta_unfused(module, single, cond, pair, mask):
    """``AugmentedAttentionPairBias.delta`` with the sm_80 core (and, at the atom width, the fused pair bias) inside the module's own composition."""
    from einops import rearrange

    from miniworld_engine.modules.functional import sigmoid_gate

    in_dtype = single.dtype
    x = module.ada_ln_in(single, cond)
    query, key, value, gate = module.to_query(x), module.to_key(x), module.to_value(x), module.to_gate(x)
    samples, _, length, width = query.shape
    heads = module.n_head
    head_dim = width // heads
    q, k, v = (t.view(samples, 1, length, heads, head_dim) for t in (query, key, value))
    if (heads, head_dim, pair.shape[-1]) == _ATOM:
        cached = _cached_bias(module, pair, mask)
        if cached is not None:                                                         # inference with the bias cached by cache_pair_bias()
            padded = cached.shape[-1]
            o, _ = _attn_fwd(*(_tokens(t, samples, length, padded) for t in (q, k, v)), cached, samples, padded, heads, head_dim, False)
            out = _untokens(o, samples, length, padded, heads, head_dim)
        else:
            kv = torch.empty(0, device=pair.device, dtype=torch.bool) if mask is None else mask.reshape(-1).contiguous()
            need = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, pair, module.ln_pair.weight, module.to_bias.weight))
            out = _AtomAttention.apply(q, k, v, pair, module.ln_pair.weight, module.to_bias.weight, kv, samples, length, need)
    else:
        bias = _bias(module, pair, mask, length, head_dim)
        need = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, bias))
        out = _Attention.apply(q, k, v, bias, samples, length, heads, head_dim, need)
    out = rearrange(out.to(in_dtype), "A B L H D -> A B L (H D)")
    out = sigmoid_gate(gate, out)
    out = module.to_out(out)
    return sigmoid_gate(module.to_scale(cond), out)


# ================================================================================================================================ entry points
def delta(module, single, cond, pair, mask):
    """``AugmentedAttentionPairBias.delta`` on the sm_80 path (the update alone); call ``serves`` first."""
    if not _fused():
        return _delta_unfused(module, single, cond, pair, mask)
    return _fused_update(module, module.ada_ln_in(single, cond), cond, pair, mask, None)


def forward(module, single, cond, pair, mask):
    """``AugmentedAttentionPairBias.forward`` on the sm_80 path: ``single + delta`` with the residual inside the last pass; call ``serves`` first."""
    if not _fused():
        return single + _delta_unfused(module, single, cond, pair, mask)
    return _fused_update(module, module.ada_ln_in(single, cond), cond, pair, mask, single.contiguous())


__all__ = ["attention_ops", "cache_pair_bias", "clear_pair_bias", "delta", "forward", "serves", "serves_ops"]
