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
    wvg, wo, ws, bs, wcm = _pack(params, dt)
    da = wvg.shape[0] // 2
    with torch.cuda.device(x.device), _tf32(f32):
        if _sm120(x):
            # RTX 5090: the pair bias and its softmax (memory-bound) on a side stream beside the two GEMMs that do not need them (tensor-bound:
            # [v | g] and the residual gate's logits); the same kernels and values, joined before the core
            main, side = torch.cuda.current_stream(), _side_stream(x.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                bias = _sm80().pair_bias_generic(pair[0], wcm, None, length, EPS, oscale=1.0)  # [H, L, L], natural units
                P, Pt = _softmax(bias, mask, save)
                del bias
            vg = _mm120(x, wvg)  # [M, 2 da]
            g2: torch.Tensor | None = _mm120(cond, ws, bs)
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
        y = _mm120(a, wo) if _sm120(x) else torch.mm(a, wo.t())
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


def _sm120(t: torch.Tensor) -> bool:
    return torch.cuda.get_device_capability(t.device) == (12, 0)


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
    with torch.autocast("cuda", enabled=False):
        if not grad and _fused_eval(module, single, cond):
            return _block_eval_sm120(module, single, cond, pair, mask)
        xa = adaln_sm80.update(at.ada_ln_in, single, cond, dt, grad)
        cs = cond.to(dt).expand(a, 1, length, cond.shape[-1]).reshape(a * length, cond.shape[-1]).contiguous()
        xs, rs, pr = xa.reshape(a * length, d).contiguous(), single.reshape(a * length, d).contiguous(), pair.contiguous()
        params = _params(at)
        if grad:
            out = _Attention.apply(xs, cs, pr, rs, mask, at.n_head, *params)
        else:
            out = _fwd(xs, cs, pr, rs, mask, params, at.n_head, False)[0]
        return ct_sm80.update(module.transition, out.view(a, 1, length, d), cond, True, dt, grad)


# ---------------------------------------------------------------------------------------------------- RTX 5090 inference block
# The whole block's forward on the RTX 5090 when no gradient is needed: every GEMM is gemm120 (bias_only_dit/cuda/sm80/gemm_sm120.cuh) with the
# block's row pass fused into its epilogue -- the AdaLNs (sigmoid(S + sb) LN(x) + B), [v | g], the residual gates (r + sigmoid(g) acc), the
# transition's SwiGLU -- so no [S | B], [a | b], y or z is written and read back and no weight is packed (the GEMM reads [W1; W2] as one
# operand). The cond-only work (the pair bias + softmax, the two gate GEMMs, the transition's cond LayerNorm) runs on a side stream beside the
# AdaLN and [v | g]. bf16 activations (with or without fp32 master weights) only, on mma.sync bf16. The attention core and the row
# statistics are the sm_80 path's kernels.
# FP32_FUSED (not routed): gemm120's fp32 form (TF32, operands rounded to nearest: 2-4e-4 relative per GEMM, as cuBLAS) ran the fp32 block
# 2-8 % faster than triton at L 256 / 384 / 512 / 768 (5 % slower at L 640), but its maximum error crossed PyTorch's own (cuBLAS TF32) in 3 of
# the 24 fp32 inference cases (GREEN -> YELLOW at 3.5-4.0e-4 against 3.4-3.8e-4) while 5 others went YELLOW -> GREEN; fp32 keeps the sm_80 path.

#: the CTA tile (gemm120 cfg) per (dtype, epilogue kind, M, N, K): measured (graph replay, L2 flushed) over the nine tiles, the paired
#: epilogues (AdaLN, SwiGLU) over the six that hold 64 output columns; any shape not listed: the largest tile that still fills the SMs
_TILE_PAIR = {  # (M, N, K) -> (bf16, fp32)
    (640, 768, 384): (1, 1), (640, 768, 768): (1, 1), (640, 1536, 768): (0, 0),
    (1280, 768, 384): (0, 10), (1280, 768, 768): (0, 10), (1280, 1536, 768): (6, 6),
    (1920, 768, 384): (6, 6), (1920, 768, 768): (6, 6), (1920, 1536, 768): (6, 6),
    (2560, 768, 384): (6, 6), (2560, 768, 768): (6, 6), (2560, 1536, 768): (5, 5),
    (3200, 768, 384): (5, 5), (3200, 768, 768): (5, 5), (3200, 1536, 768): (5, 5),
    (3840, 768, 384): (6, 6), (3840, 768, 768): (6, 6), (3840, 1536, 768): (6, 6),
}
_TILE_PLAIN = {
    (640, 1536, 768): (8, 1), (640, 768, 384): (2, 2), (640, 768, 768): (4, 2), (640, 768, 1536): (2, 2),
    (1280, 1536, 768): (8, 8), (1280, 768, 384): (3, 1), (1280, 768, 768): (8, 1), (1280, 768, 1536): (8, 1),
    (1920, 1536, 768): (8, 8), (1920, 768, 384): (8, 8), (1920, 768, 768): (8, 8), (1920, 768, 1536): (8, 8),
    (2560, 1536, 768): (8, 6), (2560, 768, 384): (8, 8), (2560, 768, 768): (8, 8), (2560, 768, 1536): (8, 8),
    (3200, 1536, 768): (5, 5), (3200, 768, 384): (0, 0), (3200, 768, 768): (6, 0), (3200, 768, 1536): (0, 0),
    (3840, 1536, 768): (8, 8), (3840, 768, 384): (8, 8), (3840, 768, 768): (8, 8), (3840, 768, 1536): (8, 8),
}
_SMS = 170
#: the token length from which the cond work stays on the main stream (L 640 fp32: the side stream's pair bias + softmax + cond work 330 us
#: against the main stream's 200; at L 128 moving it to the main stream cost 3-5 %)
SIDE_COND_MAX_L = 512


def _tile(paired: bool, a: torch.Tensor, n: int) -> int:
    m, k = a.shape
    hit = (_TILE_PAIR if paired else _TILE_PLAIN).get((m, n, k))
    if hit is not None:
        return hit[a.dtype is F32]
    if m >= 3840 and m % 64 == 0:  # the training forward's rows (48 L): the 64-row RB-64 tiles, as at M = 3840
        return 6 if paired else 8
    for cfg, (bm, bn) in ((0, (128, 64 if paired else 128)), (1, (64, 64 if paired else 128)), (4, (32, 64 if paired else 128))):
        if m % bm == 0 and (m // bm) * (n // bn) >= _SMS:
            return cfg
    return 4 if m % 32 == 0 else 1


def _mm120(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """RTX 5090: ``a w^T (+ bias)`` of the training forward (_fwd) on gemm120 (cuBLAS picks 32 x 32 wmma kernels for several of these
    shapes on sm_120); rows not a multiple of 32: cuBLAS."""
    if a.dtype is not BF or a.shape[0] % 32 or not a.is_contiguous() or not w.is_contiguous():
        return torch.mm(a, w.t()) if bias is None else torch.addmm(bias, a, w.t())
    from miniworld_engine.kernels.bias_only_dit.cuda.sm80 import EPI_PLAIN, gemm120

    return gemm120(EPI_PLAIN, a, w, a.new_empty(a.shape[0], w.shape[0]), bias=bias, cfg=_tile(False, a, w.shape[0]))


def _fused_eval(module, single: torch.Tensor, cond: torch.Tensor) -> bool:
    """The RTX 5090 inference block serves this call: an sm_120 device, the block's widths (768 / cond 384 or 768 / transition n = 2), one
    conditioning row per token (as the block's AdaLN reads it), A L a multiple of 32."""
    if not _sm120(single) or single.dtype is not BF:  # fp32: see FP32_FUSED
        return False
    at, ct = module.attention, module.transition
    a, _, length, d = single.shape
    dc = cond.shape[-1]
    return (
        d == 768
        and dc in (384, 768)
        and at.to_value.weight.shape[0] == 768
        and ct.expand_a.weight.shape == (1536, 768)
        and tuple(cond.shape[:-1]) == (a, 1, length)
        and (a * length) % 32 == 0
    )


def _block_eval_sm120(module, single, cond, pair, mask) -> torch.Tensor:
    from miniworld_engine.kernels.bias_only_dit.cuda.sm80 import EPI_ADALN, EPI_GATERES, EPI_PLAIN, EPI_SWIGLU, gemm120, row_stats120
    from miniworld_engine.kernels.integrations.adaln_sm80 import _kernels, cast_params

    def gemm(epi, a, w1, n, **kw):
        out = a.new_empty(a.shape[0], n)
        return gemm120(epi, a, w1, out, cfg=_tile(epi in (EPI_ADALN, EPI_SWIGLU), a, n), **kw)

    at, ct = module.attention, module.transition
    ada, cada = at.ada_ln_in, ct.ada_ln_in
    dt = single.dtype
    a_, _, length, d = single.shape
    m = a_ * length
    ka = _kernels()
    xs = single.reshape(m, d).contiguous()
    cs = cond.to(dt).reshape(m, cond.shape[-1]).contiguous()
    ws, sb, wb, wv, wg, wo, wsg, bs = cast_params(
        (ada.to_scale.weight, ada.to_scale.bias, ada.to_bias.weight, at.to_value.weight, at.to_gate.weight, at.to_out.weight,
         at.to_scale.weight, at.to_scale.bias), dt, True)  # fmt: skip
    ws1, sb1, wb1, wa, wbb, wsq, wsc, bsc = cast_params(
        (cada.to_scale.weight, cada.to_scale.bias, cada.to_bias.weight, ct.expand_a.weight, ct.expand_b.weight, ct.squeeze.weight,
         ct.to_scale.weight, ct.to_scale.bias), dt, True)  # fmt: skip
    lnw, wbias = at.ln_pair.weight, at.to_bias.weight
    wcm = cached("bias_only_dit_sm80_wcm", (lnw, wbias), lambda: (wbias.float() * lnw.float()[None]).t().contiguous())
    heads, da = at.n_head, wv.shape[0]
    pr = pair.contiguous()
    main, side = torch.cuda.current_stream(), _side_stream(single.device)
    with torch.cuda.device(single.device):
        side.wait_stream(main)
        def cond_work():  # the gate logits of the attention's and the transition's residual gates, the transition's cond LayerNorm
            return (gemm(EPI_PLAIN, cs, wsg, d, bias=bs), gemm(EPI_PLAIN, cs, wsc, d, bias=bsc),
                    ka.cond_ln(cs, cada.ln_cond.weight, dt, float(cada.ln_cond.eps))[0])  # fmt: skip

        # below SIDE_COND_MAX_L the side stream also takes the cond work; from it on the pair bias alone outlasts the main stream's AdaLN and [v | g]
        with torch.cuda.stream(side):
            bias = _sm80().pair_bias_generic(pr[0], wcm, None, length, EPS, oscale=1.0)  # [H, L, L], natural units
            P, _ = _softmax(bias, mask, False)
            del bias
            if length < SIDE_COND_MAX_L:
                g2, gc, aff2 = cond_work()
        aff, _ = ka.cond_ln(cs, ada.ln_cond.weight, dt, float(ada.ln_cond.eps))
        xa = gemm(EPI_ADALN, aff, ws, d, w2=wb, x=xs, xst=row_stats120(xs, float(ada.ln_in.eps)), sb=sb)
        vg = gemm(EPI_PLAIN, xa, wv, 2 * da, w2=wg)
        if length >= SIDE_COND_MAX_L:
            g2, gc, aff2 = cond_work()
        main.wait_stream(side)
        for t in (P, g2, gc, aff2):
            t.record_stream(main)
        for t in (pr, cs, wcm, wsg, bs, wsc, bsc):
            t.record_stream(side)
        if mask is not None:
            mask.record_stream(side)
        att = torch.empty(m, da, device=single.device, dtype=dt)
        core = _core()
        (core.PvGateTF32 if dt is F32 else core.PvGate)(heads, da // heads)(vg[:, :da], P.view(heads * length, length), att, a_, g=vg[:, da:])
        out = gemm(EPI_GATERES, att, wo, d, r=xs, g=g2)
        xa2 = gemm(EPI_ADALN, aff2, ws1, d, w2=wb1, x=out, xst=row_stats120(out, float(cada.ln_in.eps)), sb=sb1)
        h = gemm(EPI_SWIGLU, xa2, wa, wa.shape[0], w2=wbb)
        y = gemm(EPI_GATERES, h, wsq, d, r=out, g=gc)
    return y.view(single.shape)


__all__ = ["block", "refusal"]
