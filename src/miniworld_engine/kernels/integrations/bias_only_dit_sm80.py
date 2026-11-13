"""The A100 (sm_80) path of ``TokenDiTMiniWorldBiasOnly``: hand-written CUDA and cuBLAS, forward and backward, bf16 / master / fp32 (TF32).

The block is its two parts, each on its sm_80 path:

    xa   = AdaLN(single, cond)                    ``integrations/adaln_sm80.py`` (LN(cond) w, one GEMM, LN(x) + sigmoid gate + sum)
    a    = attention(xa, cond, pair) + single      one autograd Function here, its forward and backward each one opaque op
    out  = ConditionedTransition(a, cond)          ``integrations/conditioned_transition_sm80.py``

The attention of the Function (no query, no key: the logits are the pair bias, one softmax for every sample):

    forward   [v | g] = xa [Wv; Wg]^T                         cuBLAS, one GEMM (the two weights packed once per parameter version)
              bias = LN(pair) Wf^T, head-major [H, L, L]      the generic sm_80 pair-bias kernel (``augmented_attention/cuda/sm80.pair_bias_generic``)
              P = softmax(bias) over the keys (mask: 0)        the family's row kernels (``bias_only_dit_rows.cu`` / fp32 ``bias_only_dit_f32_rows.cu``;
                                                               training also writes P^T beside P)
              a = sigmoid(g) (P v) per head and sample        the sm_80 bias-only core (``bias_only_dit/cuda/sm80``: ``PvGate`` / TF32 ``PvGateTF32``)
              y = a Wo^T ; g2 = cond Ws^T + bs                cuBLAS
              out = single + sigmoid(g2) y                    one row pass (``augmented_attention/cuda/sm80/glue_sm80.cuh``)
    backward  dy / dg2 (the residual gate's backward), da = dy Wo, then do / dg / D from a = sigmoid(g) o (the pre-gate o is never stored), dv = P^T do
              (the core without the gate), dbias = P o (sum_a do v^T - D) (the sm_80 ``Dpb`` / ``Dpb32``), d pair and dWf (the generic pair-bias backward), GEMMs for dxa / dW,
              dcond / dWs.

fp32 (single, cond, pair and the block's weights fp32, no autocast) runs the same flow on the TF32 kernels, the GEMMs on cuBLAS forced to TF32. bf16 activations also take fp32 master
parameters (bf16 autocast, AMP's bf16-mixed): the op casts them outside autograd and hands back unrounded fp32 weight gradients.

``refusal()`` is the whole gate (None = served): A100, bf16 / master / fp32, single [A, 1, L, d] with L a multiple of 128 (training: at most 1024), the heads as 12 / 16 / 24 of width
32 / 48 / 64 with 768 or 1024 attention channels, the pair [1, L, L, dp] (dp 64 / 128 / 256 / 512), a bool key mask [1, L] or none, no hoisted bias, LayerNorm eps 1e-5, and the AdaLN /
ConditionedTransition sm_80 paths serving their parts.
"""

from __future__ import annotations

import contextlib
from typing import cast

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.integrations.adaln_sm80 import _tf32, _wgrad, cached

BF = torch.bfloat16
F32 = torch.float32
EPS = 1e-5
#: (heads, head width) of the sm_80 core / row kernels: 768 attention channels as 16 x 48, 24 x 32, 12 x 64; 1024 as 16 x 64
LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))


def _a100(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) in ((8, 0), (12, 0))  # 12.0: RTX 5090, sm_120a build (kernels/_sm120.py)


def _core():
    from miniworld_engine.kernels.bias_only_dit.cuda import sm80

    return sm80


def _sm80():
    from miniworld_engine.kernels.augmented_attention.cuda import sm80

    return sm80


def _rows():
    from miniworld_engine.kernels.bias_only_dit import cuda as C

    return C


def _rows32():
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32

    return tf32.rows32()


def refusal(module, single, cond, pair, mask, bias, grad: bool) -> str | None:
    """None when the call runs on the sm_80 path, else why not (``module`` the block, ``grad`` whether the call needs a backward)."""
    from miniworld_engine.kernels.integrations import adaln_sm80
    from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80

    at = module.attention
    dt = single.dtype
    if bias is not None:
        return "A100: a hoisted pair bias (bias=; the sm_80 block takes the pair)"
    if pair is None or pair.dtype is not dt or cond.dtype is not dt or dt not in (BF, F32):
        return f"A100: dtypes {single.dtype} / {cond.dtype} / {None if pair is None else pair.dtype} (all bfloat16 or all float32)"
    if torch.is_autocast_enabled("cuda") and (torch.get_autocast_dtype("cuda") is not BF or dt is not BF):
        return "A100: autocast: bfloat16 activations under bfloat16 autocast only (float32 activations run without autocast, on TF32)"
    wdt = (BF, F32) if dt is BF else (F32,)
    if any(p.dtype not in wdt for p in module.parameters()):
        return f"A100: parameters must be {' / '.join(str(w) for w in wdt)} for {dt} activations"
    if single.ndim != 4 or single.shape[1] != 1 or cond.ndim != 4:
        return f"A100: single / cond must be [A, 1, L, d], got {tuple(single.shape)} / {tuple(cond.shape)}"
    a, _, length, _ = single.shape
    if length % 128 or (grad and length > 1024):
        return f"A100: L {length} (a multiple of 128; training at most 1024)"
    if tuple(cond.shape[1:3]) != (1, length) or cond.shape[0] not in (1, a):
        return f"A100: cond must be [A or 1, 1, L, d_cond], got {tuple(cond.shape)}"
    heads, da = at.n_head, at.to_value.weight.shape[0]
    if da % heads or (heads, da // heads) not in LAYOUTS:
        return (
            f"A100: {heads} heads x {da // heads if da % heads == 0 else da / heads} (the sm_80 core: 16 x 48, 24 x 32, 12 x 64, 16 x 64)"
        )
    if grad and dt is F32 and da // heads == 64 and torch.cuda.get_device_capability(single.device) == (12, 0):
        return "RTX 5090: fp32 training at head width 64 (the TF32 bias-gradient core needs 131 KB of shared memory; the card has 99 KB a block)"
    dp = at.ln_pair.weight.shape[0]
    if tuple(pair.shape) != (1, length, length, dp):
        return f"A100: pair must be [1, L, L, d_pair], got {tuple(pair.shape)}"
    sm80 = _sm80()
    if not (sm80.pair_bias_generic_supported(dp, heads, dt) and sm80.pair_bias_generic_bwd_supported(dp, heads, dt)):
        return f"A100: d_pair {dp} with {heads} heads (the sm_80 pair-bias kernels: 64 / 128 / 256 / 512 wide, 8 / 12 / 16 / 24 heads)"
    if mask is not None and (mask.dtype is not torch.bool or tuple(mask.shape) != (1, length)):
        return "A100: mask must be a bool [1, L] key mask"
    if any(n.eps != EPS for n in (at.ln_pair, at.ada_ln_in.ln_in, at.ada_ln_in.ln_cond)):
        return "A100: LayerNorm eps != 1e-5"
    sm120 = torch.cuda.get_device_capability(single.device) == (12, 0)
    if sm120 and not adaln_sm80.serves(at.ada_ln_in, single, cond, dt, grad):
        return "the AdaLN: kernels/integrations/adaln_sm80 has no sm_120a build yet (the AdaLN / ConditionedTransition port), or the call is outside its range"
    if sm120 and not ct_sm80.serves(module.transition, single, cond, dt, grad):
        return "the transition: kernels/integrations/conditioned_transition_sm80 has no sm_120a build yet (the ConditionedTransition port), or the call is outside its range"
    if not adaln_sm80.serves(at.ada_ln_in, single, cond, dt, grad):
        return "A100: outside the sm_80 AdaLN kernels' range (widths 128 / 384 / 768, sigmoid AdaLN without a shift bias)"
    if not ct_sm80.serves(module.transition, single, cond, dt, grad):
        return (
            "A100: outside the sm_80 ConditionedTransition kernels' range (bf16 / fp32, widths of their table, sigmoid AdaLN without a shift "
            "bias, bias-free SwiGLU)"
        )
    return None


def _pack(params, dt):
    """([Wv; Wg] in ``dt``, Wo / Ws / bs in ``dt``, the folded pair weight Wf = to_bias.weight x ln_pair.weight fp32 channel-major [dp, H]), per parameter version."""
    wv, wg, wo, ws, bs, lnw, wb = params

    def build():
        return (
            torch.cat([wv.to(dt), wg.to(dt)]).contiguous(),
            wo.to(dt).contiguous(),
            ws.to(dt).contiguous(),
            bs.to(dt).contiguous(),
            (wb.float() * lnw.float()[None]).t().contiguous(),
        )

    return cached(f"bias_only_dit_sm80_{dt}", tuple(params), build)


def _softmax(bias, mask, transpose: bool):
    """P = softmax over the keys of ``bias`` [H, L, L] (masked keys 0), and P^T [H, L, L] (``transpose``) or None."""
    h, length = bias.shape[0], bias.shape[1]
    m = None if mask is None else mask.reshape(length).contiguous()
    P = torch.empty_like(bias)
    Pt = torch.empty_like(bias) if transpose else None
    flat = (bias.view(h * length, length), P.view(h * length, length))
    if bias.dtype is F32:
        if transpose:
            _rows32().softmax_t_cuda(*flat, cast(torch.Tensor, Pt).view(h * length, length), m, 1, 1.0)
        else:
            _rows32().softmax_rows_cuda(*flat, m)
    elif transpose:
        _rows().softmax_t(*flat, cast(torch.Tensor, Pt).view(h * length, length), m)
    else:
        _rows().softmax_rows(*flat, m)
    return P, Pt


def _fwd_fake(x, cond, pair, res, mask, params, heads, save):
    m, d = x.shape
    da = params[0].shape[0]
    length = pair.shape[1]
    outs = [torch.empty_like(x)]
    if save:
        e = x.new_empty
        outs += [e((m, 2 * da)), e((m, da)), e((m, d)), e((m, d)), e((heads, length, length)), e((heads, length, length))]
    return outs


@opaque(fake=_fwd_fake, name="bias_only_dit_sm80_attention_fwd")
def _fwd(
    x: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    res: torch.Tensor,
    mask: torch.Tensor | None,
    params: list[torch.Tensor],
    heads: int,
    save: bool,
) -> list[torch.Tensor]:
    """The attention after the AdaLN: x [A L, d] (the AdaLN's output), cond [A L, dc], pair [1, L, L, dp], res [A L, d] (the residual), mask a bool [1, L] or None, params =
    (Wv, Wg, Wo, Ws, bs, ln_pair.weight, to_bias.weight). Returns [res + update], with ``save`` also vg, a, y, g2, P, P^T."""
    dt = x.dtype
    f32 = dt is F32
    length = pair.shape[1]
    samples = x.shape[0] // length
    eval120 = not save and _eval120(x, x.shape[0])
    if eval120:  # RTX 5090 inference: no packed [Wv; Wg] (see _mm_halves)
        from miniworld_engine.kernels.integrations.adaln_sm80 import cast_params

        wv, wg, wo, ws, bs = cast_params(tuple(params[:5]), dt, True)
        lnw, wb = params[5], params[6]
        wcm = cached("bias_only_dit_sm80_wcm", (lnw, wb), lambda: (wb.float() * lnw.float()[None]).t().contiguous())
        da = wv.shape[0]
    else:
        wvg, wo, ws, bs, wcm = _pack(params, dt)
        da = wvg.shape[0] // 2
    with torch.cuda.device(x.device), _tf32(f32):
        if _sm120(x) and SM120_SIDE:
            # RTX 5090: the pair bias and its softmax (memory-bound) on a side stream beside the two GEMMs that do not need them (tensor-bound:
            # [v | g] and the residual gate's logits); the same kernels and values, joined before the core
            main, side = torch.cuda.current_stream(), _side_stream(x.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                bias = _sm80().pair_bias_generic(pair[0], wcm, None, length, EPS, oscale=1.0)  # [H, L, L], natural units
                P, Pt = _softmax(bias, mask, save)
                del bias
            vg = _mm_halves(x, wv, wg) if eval120 else torch.mm(x, wvg.t())  # [M, 2 da]
            g2: torch.Tensor | None = torch.addmm(bs, cond, ws.t())
            main.wait_stream(side)
            for t in (P, Pt) if save else (P,):
                t.record_stream(main)
            pair.record_stream(side)
            wcm.record_stream(side)
            if mask is not None:
                mask.record_stream(side)
        else:
            vg = torch.mm(x, wvg.t())  # [M, 2 da]
            bias = _sm80().pair_bias_generic(pair[0], wcm, None, length, EPS, oscale=1.0)  # [H, L, L], natural units
            P, Pt = _softmax(bias, mask, save)
            del bias
            g2 = None
        a = torch.empty(x.shape[0], da, device=x.device, dtype=dt)
        core = _core()
        (core.PvGateTF32 if f32 else core.PvGate)(heads, da // heads)(vg[:, :da], P.view(heads * length, length), a, samples, g=vg[:, da:])
        y = _mm_pick(a, wo, _pick(a, wo.shape[0])) if eval120 else torch.mm(a, wo.t())
        if g2 is None:
            g2 = torch.addmm(bs, cond, ws.t())
        out = torch.empty_like(res)
        _sm80().res_gate(y, g2, res, out)
    return [out, vg, a, y, g2, P, cast(torch.Tensor, Pt)] if save else [out]  # (save: P^T was made)


def _bwd_fake(dout, x, cond, pair, params, saved, heads):
    return [torch.empty_like(x), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(p) for p in params)]


@opaque(fake=_bwd_fake, name="bias_only_dit_sm80_attention_bwd")
def _bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    params: list[torch.Tensor],
    saved: list[torch.Tensor],
    heads: int,
) -> list[torch.Tensor]:
    """The backward of ``_fwd`` (``save=True``): [dx, dcond, dpair, *d params] (the residual's gradient is dout itself: the caller returns it)."""
    vg, a, y, g2, P, Pt = saved
    dt = x.dtype
    f32 = dt is F32
    m, length = x.shape[0], pair.shape[1]
    samples = m // length
    lnw, wb = params[5], params[6]
    wvg, wo, ws, _, wcm = _pack(params, dt)
    da = wvg.shape[0] // 2
    hd = da // heads
    dev = x.device
    core = _core()
    with torch.cuda.device(dev), _tf32(f32):
        dy, dg2 = torch.empty_like(y), torch.empty_like(g2)
        _sm80().res_gate_bwd(dout, y, g2, dy, dg2)
        dA = torch.mm(dy, wo)  # [M, da]
        dwo = _wgrad(dy, a)  # [d, da] fp32
        dvg = torch.empty_like(vg)
        do = torch.empty_like(a)
        dd = torch.empty(samples, heads, length, device=dev, dtype=F32)
        if f32:
            _rows32().gate_bwd_cuda(dA, a, vg[:, da:], do, dvg[:, da:], dd, length)
        else:
            _rows().gate_bwd_rows(dA, a, vg[:, da:], do, dvg[:, da:], dd, length)
        (core.PvGateTF32 if f32 else core.PvGate)(heads, hd)(do, Pt.view(heads * length, length), dvg[:, :da], samples)  # dv = P^T do
        db = torch.empty_like(P)
        (core.Dpb32 if f32 else core.Dpb)(heads, hd)(
            do, vg[:, :da], P.view(heads * length, length), dd, db.view(heads * length, length), samples
        )
        dz, dwf = _sm80().pair_bias_generic_backward(pair[0], wcm, db if f32 else db.float(), None, length, EPS)
        dx = torch.mm(dvg, wvg)
        dwvg = _wgrad(dvg, x)  # [2 da, d] fp32
        dcond = torch.mm(dg2, ws)
        dws = _wgrad(dg2, cond)
        dbs = torch.sum(dg2, dim=0, dtype=F32)
        dlnw = (dwf * wb.float()).sum(0)
        dwb = dwf * lnw.float()[None]
    grads = [dwvg[:da], dwvg[da:], dwo, dws, dbs, dlnw, dwb]
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(outs, [g.reshape(p.shape) for g, p in zip(grads, params, strict=True)])  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
    return [dx, dcond, dz.view(pair.shape), *outs]


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, pair, res, mask, heads, *params):
        out, *saved = _fwd(x, cond, pair, res, mask, list(params), heads, True)
        ctx.save_for_backward(x, cond, pair, *params, *saved)
        ctx.meta = (heads, len(params))
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        heads, npar = ctx.meta
        vals = ctx.saved_tensors
        x, cond, pair = vals[:3]
        params, saved = list(vals[3 : 3 + npar]), list(vals[3 + npar :])
        dx, dc, dz, *pg = _bwd(dout.contiguous(), x, cond, pair, params, saved, heads)
        need = ctx.needs_input_grad
        return (
            dx if need[0] else None,
            dc if need[1] else None,
            dz if need[2] else None,
            dout if need[3] else None,
            None,
            None,
            *(g if need[6 + i] else None for i, g in enumerate(pg)),
        )


def _params(at) -> list[torch.Tensor]:
    return [
        at.to_value.weight,
        at.to_gate.weight,
        at.to_out.weight,
        at.to_scale.weight,
        at.to_scale.bias,
        at.ln_pair.weight,
        at.to_bias.weight,
    ]


#: RTX 5090, bf16 inference: the token-row counts (A x L, half-open ranges) at which the block's GEMMs go to cuBLASLt (torch's
#: ``preferred_blas_library``). The cuBLAS heuristic picks a 32 x 32 wmma kernel for several of the block's bf16 GEMMs on sm_120 (the
#: transition's [M, 768] x [768, 3072]: 114 us at M = 1920, cuBLASLt 59 us; [M, 768] x [768, 1536] at M = 3840: 114 vs 61 us); at M = 640 and
#: 2560 cuBLAS is as fast or faster (640: 29 vs 43 us on the transition GEMM; 2560: [M, 768] x [768, 768] 30 vs 45 us). Measured per GEMM, L2 flushed.
SM120_LT_ROWS = ((1024, 2048), (3072, 1 << 40))


@contextlib.contextmanager
def _sm120_blas(single: torch.Tensor, grad: bool):
    """cuBLASLt for the block's GEMMs on an RTX 5090 (bf16 inference with a row count in ``SM120_LT_ROWS``); the caller's choice restored after."""
    rows = single.shape[0] * single.shape[2]
    if grad or single.dtype is not BF or not any(lo <= rows < hi for lo, hi in SM120_LT_ROWS) or not _sm120(single):
        yield
        return
    old = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(old)


def _sm120(t: torch.Tensor) -> bool:
    return torch.cuda.get_device_capability(t.device) == (12, 0)


#: RTX 5090: the attention's pair bias + softmax on a side stream beside the [v | g] and gate GEMMs (_fwd)
SM120_SIDE = True
#: RTX 5090 inference: the block's own AdaLN / transition / [v | g] compositions (below: no packed weights at small M); fp32 at every row
#: count, bf16 from HALVES_MAX_ROWS on (below it the bf16 casts of the unpacked weights cost what the pack saves: L 128 master 190 vs 188 us)
SM120_EVAL = True


def _eval120(t: torch.Tensor, rows: int) -> bool:
    return SM120_EVAL and _sm120(t)
_SIDE: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    """The RTX 5090 forward's second stream (one per device)."""
    if device not in _SIDE:
        _SIDE[device] = torch.cuda.Stream(device=device)
    return _SIDE[device]


def block(module, single, cond, pair, mask, grad: bool) -> torch.Tensor:
    """The block on the sm_80 path (``refusal`` must have returned None): [A, 1, L, d] -> [A, 1, L, d]."""
    from miniworld_engine.kernels.integrations import adaln_sm80
    from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80

    at = module.attention
    dt = single.dtype
    a, _, length, d = single.shape
    eval120 = not grad and _eval120(single, a * length)
    with torch.autocast("cuda", enabled=False), _sm120_blas(single, grad):
        xa = _adaln_eval_sm120(at.ada_ln_in, single, cond, dt) if eval120 else adaln_sm80.update(at.ada_ln_in, single, cond, dt, grad)
        cs = cond.to(dt).expand(a, 1, length, cond.shape[-1]).reshape(a * length, cond.shape[-1]).contiguous()
        xs, rs, pr = xa.reshape(a * length, d).contiguous(), single.reshape(a * length, d).contiguous(), pair.contiguous()
        params = _params(at)
        if grad:
            out = _Attention.apply(xs, cs, pr, rs, mask, at.n_head, *params)
        else:
            out = _fwd(xs, cs, pr, rs, mask, params, at.n_head, False)[0]
        if eval120:
            return _ct_eval_sm120(module.transition, out.view(a, 1, length, d), cond, dt)
        return ct_sm80.update(module.transition, out.view(a, 1, length, d), cond, True, dt, grad)


# ------------------------------------------------------------------------------------------------ RTX 5090 inference compositions
# The inference graph replays every per-parameter-version pack (``cached``: a capture records the pack, so that an in-place weight update is
# seen), and the packs of the AdaLN ([Ws; Wb]), the transition ([Ws1; Wb1], [Wa; Wb]) and the attention ([Wv; Wg]) copy 19 MB of fp32
# weights a call (18 us of the L 128 fp32 block). On the RTX 5090 inference these GEMMs instead write the column halves of one output from the
# unpacked weights (cuBLAS takes the row-major view with its leading dimension as it is): the same kernels around them, the same products.
#: the row count below which the two half GEMMs replace the packed one (measured: M = 640 fp32 block 223 -> 207 us; at M = 1920 the two
#: GEMMs into strided halves cost more than the pack saves, fp32 610 -> 655 us)
HALVES_MAX_ROWS = 1024


#: RTX 5090 inference: the GEMM library per (dtype, M, K, N) of the block's GEMMs where the measured best (graph replay, L2 flushed) beats the
#: default by >= 5 %: "lt" cuBLASLt, "cublas", or ("g", tile) the extension's gemm120 (bias_only_dit/cuda/sm80/gemm_sm120.cuh). Not listed:
#: torch.mm under the block's BLAS choice (_sm120_blas). Values in us, default -> pick.
GEMM_PICK: dict = {
    # bf16, M = 640 (L 128 x 5): cuBLAS's own choices elsewhere
    ("bf16", 640, 1536, 768): ("g", 2),  # 24.3 -> 20.5
    ("bf16", 640, 384, 1536): "lt",  # 14.1 -> 12.0
    ("bf16", 640, 768, 1536): "lt",  # 22.5 -> 16.4
    ("bf16", 640, 768, 768): ("g", 2),  # 14.3 -> 12.3
    # bf16, M = 1280 (cuBLASLt for the block)
    ("bf16", 1280, 384, 1536): ("g", 0),  # 24.6 -> 18.4
    ("bf16", 1280, 768, 1536): ("g", 2),  # 43.0 -> 30.4
    # bf16, M = 2560 (cuBLAS for the block)
    ("bf16", 2560, 768, 3072): ("g", 0),  # 92.1 -> 83.7
    ("bf16", 2560, 384, 1536): "lt",  # 26.6 -> 24.6
    # fp32: cuBLAS throughout (gemm120's TF32 build, the operands' bits as the hardware truncates them, was up to 12 % faster on five shapes
    # but turned eight fp32 inference cases of the accuracy run from GREEN to YELLOW: 8e-4 relative against cuBLAS's 3e-4)
}


def _pick(a: torch.Tensor, n: int):
    return GEMM_PICK.get(("fp32" if a.dtype is F32 else "bf16", a.shape[0], a.shape[1], n))


def _mm_pick(a: torch.Tensor, w: torch.Tensor, choice=None) -> torch.Tensor:
    """``a w^T`` [M, N] (fp32: TF32) on ``choice`` (see GEMM_PICK; None: torch.mm as the block runs it)."""
    with _tf32(a.dtype is F32):
        if isinstance(choice, tuple):
            from miniworld_engine.kernels.bias_only_dit.cuda.sm80 import gemm120

            return gemm120(a, w, a.new_empty(a.shape[0], w.shape[0]), cfg=choice[1])
        if choice is None:
            return torch.mm(a, w.t())
        old = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library("cublaslt" if choice == "lt" else "cublas")
        try:
            return torch.mm(a, w.t())
        finally:
            torch.backends.cuda.preferred_blas_library(old)


def _mm_halves(a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``[a w1^T | a w2^T]`` [M, n1 + n2] in a's dtype (fp32: TF32 tensor cores): on the GEMM_PICK choice for the packed ``[w1; w2]`` (kept
    per parameter version, as the sm_80 paths keep theirs) when there is one or from ``HALVES_MAX_ROWS`` rows on, else (fp32 below it) without
    a pack: two GEMMs into the column halves."""
    n1 = w1.shape[0]
    choice = _pick(a, n1 + w2.shape[0])
    if choice is not None or a.shape[0] >= HALVES_MAX_ROWS or a.dtype is not F32:
        return _mm_pick(a, cached("bias_only_dit_sm80_halves", (w1, w2), lambda: torch.cat([w1, w2])), choice)
    out = a.new_empty(a.shape[0], n1 + w2.shape[0])
    with _tf32(True):
        torch.mm(a, w1.t(), out=out[:, :n1])
        torch.mm(a, w2.t(), out=out[:, n1:])
    return out


def _cond_rows(x: torch.Tensor, cond: torch.Tensor, dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(x rows [M, d], cond rows [P, dc]) as ``adaln_sm80.update`` / ``conditioned_transition_sm80.update`` make them for a no-grad call."""
    from miniworld_engine.kernels.integrations.adaln_sm80 import _period

    xs = x.to(dt).reshape(-1, x.shape[-1]).contiguous()
    cs = cond if _period(x, cond, False) == xs.shape[0] else cond[0]
    return xs, cs.to(dt).reshape(-1, cond.shape[-1]).contiguous()


def _adaln_eval_sm120(module, x: torch.Tensor, cond: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """``adaln_sm80.update`` (no grad) for the block's 768-wide AdaLN: cond LayerNorm, the [scale | bias] GEMM, the epilogue."""
    from miniworld_engine.kernels.integrations.adaln_sm80 import _kernels, cast_params

    k = _kernels()
    xs, cs = _cond_rows(x, cond, dt)
    ws, sb, wb = cast_params((module.to_scale.weight, module.to_scale.bias, module.to_bias.weight), dt, True)
    aff, _ = k.cond_ln(cs, module.ln_cond.weight, dt, float(module.ln_cond.eps))
    y, _ = k.epilogue(xs, _mm_halves(aff, ws, wb), sb, float(module.ln_in.eps))
    return y.reshape(x.shape)


def _ct_eval_sm120(module, x: torch.Tensor, cond: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """``conditioned_transition_sm80.update`` (no grad, residual) for the block's 768-wide transition: the gate GEMM on the side stream, cond
    LayerNorm, [scale | bias], the epilogue, [a | b], SwiGLU, the squeeze GEMM, gate + residual."""
    from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct
    from miniworld_engine.kernels.integrations.adaln_sm80 import Branch, _mm_op, cast_params

    ka, kt = ct._adaln(), ct._tail()
    xs, cs = _cond_rows(x, cond, dt)
    ps = ct._params(module)
    ws1, sb1, wb1, wa, wb, ws, wsc, bsc = cast_params(tuple(ps[1:]), dt, True)
    br = Branch(xs.device)
    g = br.run(lambda: ct._gate_logits(cs, wsc, bsc), cs, wsc, bsc)
    aff, _ = ka.cond_ln(cs, ps[0], dt, float(module.ada_ln_in.ln_cond.eps))
    xa, _ = ka.epilogue(xs, _mm_halves(aff, ws1, wb1), sb1, float(module.ada_ln_in.ln_in.eps))
    h = kt.swiglu_fwd(_mm_halves(xa, wa, wb))
    z = _mm_pick(h, ws, _pick(h, ws.shape[0]))
    br.join()
    return kt.gate_res_fwd(xs, z, g).reshape(x.shape)


__all__ = ["block", "refusal"]
