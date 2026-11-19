"""MSAPairWeightedAveraging on the B200 (sm_100a): the ``cuda`` backend's route at capability (10, 0).

Two builds of the same boundaries, both from the B200 branch (``origin/b200/ladder-final``: ``docs/gpus/b200/pwa/pwa.md``):

MiniWorld's MSA side (d_msa 64, 8 heads x 32: MiniWorld / Boltz-2 / RoseTTAFold3 at d_pair 128; Chai-1's d_pair 256 in training),
`csrc/sm100/pwa_sm100.cu`:
  forward   pair_fwd (LN_z folded into proj_z, key mask, softmax over j -> w [H, N, N] bf16), ln_vg (LN_m + value projection, v
            head-major; y kept for training), cuBLAS bmm o = w . v, pwa_gate_out (gate, out-projection, the row dropout, + msa)
  backward  pwa_glue2 (do, dyg = dgp . Wg, the dWo | dWg slab: dgp stays on chip), cuBLAS bmm dv = w^T . do and dw = do . v^T,
            dv_bwd (dm, dWv, the LayerNorm gradients, the dWo / dWg finish), pair_bwd (dz, dWb, dgamma_z)
The other widths (d_msa 64 / 128 at 8 x 8 / 16, d_msa 64 at 8 x 32 in inference (Chai-1), IntelliFold-v2's 256 / 512 / 8 x 32),
pair_fwd / pair_bwd of `pwa_sm100.cu` and `csrc/sm100/pwa_wide_sm100.cu` built per (d_msa, d_hidden):
  forward   pair_fwd, ln_vg_w, cuBLAS bmm, gate_out_w
  backward  glue_w (do head-major, dgp -> dgv, go, the dropout's dout), cuBLAS bmm dv / dw, dvb_w (dm, the LayerNorm gradients),
            cuBLAS dWo = dout^T go and [dWg; dWv] = dgv^T y, pair_bwd

The depth S and the length N are padded to 128 (pad tokens masked as keys; zero MSA rows change no real row; the pad's backward drops
their gradient); N after the pad <= 2048 (pair_fwd holds a row of logits). A batch runs one stack at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from miniworld_engine.kernels import _cast_cuda
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.integrations.opm_sm100 import _bf16_params, _side_stream

H, C, D = 8, 32, 64
NMAX = 2048  # pair_fwd holds a row of up to 2048 logits (N padded to 128)
D_PAIR = (128, 256, 384, 512)
WIDE = ((64, 8), (128, 8), (128, 16))  # (d_msa, d_hidden) of the wide build at d_pair 128 / 256 / 384, inference and training
INTELLIFOLD2 = (256, 512, 32)  # (d_msa, d_pair, d_hidden): the wide build's streamed-weight form (glue_ws / dvb_ws), inference and training
_EXT: dict[str, Any] = {}


def _mw_side(d_msa: int, d_hidden: int) -> bool:
    return (d_msa, d_hidden) == (D, C)


def refusal(msa: torch.Tensor, pair: torch.Tensor, d_msa: int, d_pair: int, n_head: int, d_hidden: int) -> str | None:
    """None if the B200 kernels serve this call, else why not."""
    if not msa.is_cuda or torch.cuda.get_device_capability(msa.device) != (10, 0):
        return "the sm_100a MSAPairWeightedAveraging kernels run on a B200"
    if n_head != H:
        return f"the kernels take {H} heads, got {n_head}"
    ok = (
        (_mw_side(d_msa, d_hidden) and d_pair in D_PAIR)
        or ((d_msa, d_hidden) in WIDE and d_pair in (128, 256, 384))
        or (d_msa, d_pair, d_hidden) == INTELLIFOLD2
    )
    if not ok:
        return (
            f"the sm_100a kernels serve (d_msa 64, 8 x 32) at d_pair in {D_PAIR}, (d_msa, d_hidden) in {WIDE} at d_pair 128 / 256 / 384 "
            f"and {INTELLIFOLD2}; got ({d_msa}, {d_pair}, {n_head} x {d_hidden})"
        )
    s, n = msa.shape[1], msa.shape[2]
    if n == 0 or s == 0:
        return f"empty MSA stack, got N={n}, S={s}"
    if -(-n // 128) * 128 > NMAX:
        return f"the sm_100a pair kernels hold one row of logits: N <= {NMAX} (after the pad to 128), got N={n}"
    return None


def _k():
    """`csrc/sm100/pwa_sm100.cu` (built by `pwa_train._k100`, the prebuild's entry)."""
    from miniworld_engine.kernels.integrations import pwa_train

    return pwa_train._k100()["pwa"]


def _value(d_msa: int, d_hidden: int):
    """ln_vg_w / gate_out_w / glue_w / dvb_w of `csrc/sm100/pwa_wide_sm100.cu`, built for (d_msa, d_hidden)."""
    name = f"miniworld_pwa_wide_sm100_{d_msa}_{d_hidden}"
    if name not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        csrc = Path(__file__).with_name("csrc") / "sm100"
        _EXT[name] = load(
            name=name,
            sources=[str(csrc / "pwa_wide_sm100.cu")],
            extra_include_paths=[str(csrc)],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--use_fast_math",
                f"-DPW_D={d_msa}",
                f"-DPW_C={d_hidden}",
            ],
            extra_cflags=["-O3"],
        )
    return _EXT[name]


def builds():
    """Every build this route launches (the prebuild's entries for target sm100)."""
    import functools

    yield _k
    for d_msa, d_hidden in (*WIDE, (D, C), (INTELLIFOLD2[0], INTELLIFOLD2[2])):
        yield functools.partial(_value, d_msa, d_hidden)


def _keep_thr(p_drop: float) -> float:
    """keep = rand > p with rand a bf16 tensor compares in bf16: the threshold the MiniWorld kernels compare the drawn bf16 rand against,
    so their keep is exactly (rand > p)."""
    return float(torch.tensor(p_drop, dtype=torch.bfloat16).float())


def _uses_mw(args: list[torch.Tensor], train: bool) -> bool:
    """MiniWorld's kernels (`pwa_sm100.cu`) serve the call: d_msa 64 x 8 x 32 in training at every d_pair, in inference at d_pair 64 / 128
    (Chai-1's d_pair 256 inference runs the wide build at (64, 32))."""
    d_msa, hc, d_pair = args[0].shape[-1], args[5].shape[0], args[1].shape[-1]
    return _mw_side(d_msa, hc // H) and (train or d_pair in (64, 128))


# ---------------------------------------------------------------------------------------------------------------------------- forward
#: Inference: pair_fwd on a side stream beside the MSA side's weight cast + ln_vg (two branches of the CUDA graph; B200, 2026-11-17, graph
#: timing, side / serial: 0.96-0.98 at L 128-768). A / B switch.
_PAIR_SIDE = [True]
#: The same in training for MiniWorld's MSA side (d_msa 64, 8 x 32): pair_fwd + its fp32 softmax copy beside the weight cast + ln_vg
#: (B200 2026-11-17, train step graph, side / serial: L128 D384 167.0 -> 162.0 us, L384 323.8 -> 319.6). The wide build loses with it
#: (128 / 8 x 8 / 256 L128 158.8 -> 166.0 us; 64 / 8 x 8 0.99-1.004) and keeps the serial order. A / B switch.
_PAIR_SIDE_TRAIN = [True]
#: The SMs of the two branches, disjoint: {N up to: pair_fwd CTAs}, ln_vg takes the other SMs; past the last N each kernel asks for the
#: whole GPU. Measured (2026-11-17, five widths, split / whole GPU each): N 128 0.94-0.97, N 256 0.94-0.98, N 384 0.94-0.98 (0.96 with
#: 32 at 128 / 8 / 256), N 512 0.93-1.01; N 768 from 0.93 to 1.08 with the width, so not split. A pair_fwd narrower than these starves
#: (16 CTAs at N 384: 1.2-1.3), and a narrowed pair_fwd beside a whole-GPU ln_vg queues behind it (N 128: 1.1-1.3).
_PAIR_CTAS: dict[int, int] = {256: 32, 512: 48}
#: Overrides `_PAIR_CTAS` when set: (pair_fwd CTAs, ln_vg CTAs), 0 = the whole GPU (A / B knob).
_SPLIT: list[int] = []
#: Inference on MiniWorld's kernels at d_pair 128, N <= 512 (where `_split` gives disjoint grids): pair_fwd and ln_vg as ONE launch
#: (`pwa_dual`: CTAs [0, npair) run pair_fwd's body, the rest ln_vg's) instead of two kernels on two streams joined before the bmm.
#: B200 2026-11-17, graph, L2 flushed, 64 / 8 x 32 / 128: L128 depth 384 44.9 -> 40.8 us, depth 1024 73.7 -> 71.6, L256 depth 384
#: 70.6 -> 63.4 (+16), L512 depth 384 131.5 -> 122.9 (+16); the same bits. A / B switch.
_DUAL = [True]
#: CTAs moved from the ln_vg part to the pair part of `pwa_dual`, per N (measured: L128 +8 / +16 level, L256 / L512 +16 best).
_DUAL_EXTRA: dict[int, int] = {128: 8, 256: 16, 384: 0, 512: 16}
#: Inference at N 128 after `pwa_dual`: the contraction o = w . v and the gate pass in ONE launch (`pwa_ctr_gate`, a PDL dependent of
#: `pwa_dual`: phase 0 computes the CTA's share of o on tcgen05 in place over v, a self-resetting grid counter, then the gate pass)
#: instead of cuBLAS bmm + `pwa_gate_out`; the same bits. B200 2026-11-18, graph, L2 flushed, 64 / 8 x 32 / 128 L128: depth 128 32.6 ->
#: 28.5 us, 256 34.8 -> 34.6, 384 42.8 -> 38.9, 1024 73.5 -> 71.4. A / B switch.
_CTR_GATE = [True]
_CTR: dict[tuple[int, int], torch.Tensor] = {}


def _ctr(device: torch.device) -> torch.Tensor | None:
    """`pwa_ctr_gate`'s grid counter (int32 zeros, left zero by every launch), one per (device, stream): launches on one stream are
    ordered, launches on two streams must not share it. Made only outside a capture (None inside one when missing: the caller keeps the
    bmm path)."""
    st = torch.cuda.current_stream(device)
    key = (st.device_index, st.cuda_stream)
    c = _CTR.get(key)
    if c is None and not torch.cuda.is_current_stream_capturing():
        c = _CTR[key] = torch.zeros(4, dtype=torch.int32, device=device)
    return c


def _split(n: int, d_msa: int, device: torch.device, side: bool) -> tuple[int, int]:
    """(pair_fwd CTAs, ln_vg CTAs) for N tokens; (0, 0): the whole GPU each (always at d_msa 256, IntelliFold-v2: its MSA side is the
    long one and loses with fewer SMs: split / whole 1.00-1.06 at N 128-512)."""
    if not side or (d_msa >= INTELLIFOLD2[0] and not _SPLIT):
        return 0, 0
    if _SPLIT:
        return _SPLIT[0], _SPLIT[1]
    for nmax, ctas in _PAIR_CTAS.items():
        if n <= nmax:
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            return ctas, max(sms - ctas, 1)
    return 0, 0


def _join(main: torch.cuda.Stream, aux: torch.cuda.Stream, w16: torch.Tensor) -> None:
    """The main stream waits for the side stream's pair_fwd; w16 (allocated there) is then used on the main stream."""
    if aux is not main:
        main.wait_stream(aux)
        w16.record_stream(main)


def _forward(args: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float, train: bool) -> list[torch.Tensor]:
    """args = [msa [1, S, N, d_msa], pair [1, N, N, d_pair], key mask bool [N], lnm_w, lnm_b, wv, wg, lnz_w, wb, wo] (S, N % 128)
    -> [out [S, N, d_msa] (msa + update), w, v, y, o, dropout draw | empty] (training) or [out]."""
    msa, pair, km, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo = args
    bf = torch.bfloat16
    m, z = msa[0], pair[0]
    n, d_msa = m.shape[1], m.shape[2]
    k = _k()
    w32 = torch.empty((H, n, n), device=m.device, dtype=torch.float32) if train else None  # the fp32 softmax for the backward (pair_dw_fix)
    # the pair side (pair_fwd: z, the key mask -> w) and the MSA side (ln_vg: m -> v) are independent until the bmm: in inference
    # pair_fwd runs on a side stream beside the weight cast and ln_vg (two branches of the CUDA graph), on SMs of its own
    main = torch.cuda.current_stream(m.device)
    if not train and _DUAL[0] and _uses_mw(args, False) and pair.shape[-1] == 128:
        pc, mc = _split(n, d_msa, m.device, True)
        same = len({t.dtype for t in (lnm_w, lnm_b, lnz_w, wb)}) == 1 and lnm_w.dtype in (torch.float32, bf)
        if pc > 0 and same and all(w.dtype == bf for w in (wv, wg, wo)):
            # pair_fwd + ln_vg in ONE launch on disjoint CTAs (`pwa_dual`): one graph branch, no fork / join edges; the same bits
            ex = _DUAL_EXTRA.get(n, 0)
            lnw, lnb = lnm_w.detach().contiguous(), lnm_b.detach().contiguous()
            w16, v = k.pwa_dual(
                z,
                km,
                lnz_w.detach().contiguous(),
                eps_z,
                wb.detach().contiguous(),
                m,
                lnw,
                lnb,
                wv.detach().contiguous(),
                eps_m,
                pc + ex,
                mc - ex,
            )
            ctr = _ctr(m.device) if _CTR_GATE[0] and n == 128 else None
            if ctr is not None:
                return [k.pwa_ctr_gate(w16, v, m, lnw, lnb, eps_m, wg.detach().contiguous(), wo.detach().contiguous(), ctr)]
            o = torch.bmm(w16, v)
            return [k.pwa_gate_out(o, m, lnw, lnb, eps_m, wg.detach().contiguous(), wo.detach().contiguous(), None, 1.0, 0.5)]
    aux = _side_stream(m.device) if _PAIR_SIDE[0] and (not train or (_PAIR_SIDE_TRAIN[0] and _uses_mw(args, True))) else main
    pc, mc = _split(n, d_msa, m.device, aux is not main)
    if aux is not main:
        aux.wait_stream(main)
    with torch.cuda.stream(aux):
        w16 = k.pair_fwd(z, km, lnz_w.detach().contiguous(), eps_z, wb.detach().contiguous(), w32, pc)
    if aux is not main:
        for t in (z, km, *(() if w32 is None else (w32,))):
            t.record_stream(aux)
    lnw, lnb = lnm_w.detach().contiguous(), lnm_b.detach().to(lnm_w.dtype).contiguous()
    if _uses_mw(args, train):
        if any(w.dtype != bf for w in (wv, wg, wo)):  # the three bf16 copies in one launch
            wv16, wg16, wo16 = k.pwa_wcast(wv.detach().contiguous(), wg.detach().contiguous(), wo.detach().contiguous())
        else:
            wv16, wg16, wo16 = (w.detach().contiguous() for w in (wv, wg, wo))
        v, y = k.ln_vg(m, lnw, lnb, wv16, eps_m, bool(train), mc)
        _join(main, aux, w16)
        o = torch.bmm(w16, v)  # head-major [H, N, S*C], cuBLAS
        draw = None
        if p_drop > 0:  # the module's drop_msa: one keep per (token, channel) over the MSA rows; the kernels keep rand > p
            draw = torch.rand(n, d_msa, device=m.device, dtype=bf)
        out = k.pwa_gate_out(
            o, m, lnw, lnb, eps_m, wg16, wo16, draw, 1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0, _keep_thr(p_drop) if p_drop > 0 else 0.5
        )
    else:
        kv = _value(d_msa, wv.shape[0] // H)
        (wv16, wg16, wo16), _ = _bf16_params([wv, wg, wo])  # the three bf16 copies in one launch (opm_cast_bf16)
        v, y = kv.ln_vg_w(m, lnw, lnb, wv16, eps_m, bool(train), mc)
        _join(main, aux, w16)
        o = torch.bmm(w16, v)
        draw = None
        if p_drop > 0:
            draw = (torch.rand(n, d_msa, device=m.device, dtype=bf) > p_drop).to(bf)
        out = kv.gate_out_w(
            o,
            m,
            lnw,
            lnb,
            eps_m,
            wg16,
            wo16,
            draw,
            1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0,
        )
    if not train or w32 is None:
        return [out]
    return [out, w16, v, y, o, draw if draw is not None else m.new_empty((0,)), w32]


def _fwd_fake(args, eps_m, eps_z, p_drop):
    m = args[0]
    _, s, n, d = m.shape
    c = args[5].shape[0] // H
    dev, bf = m.device, torch.bfloat16
    return [
        torch.empty((s, n, d), device=dev, dtype=bf),
        torch.empty((H, n, n), device=dev, dtype=bf),
        torch.empty((H, n, s * c), device=dev, dtype=bf),
        torch.empty((s, n, d), device=dev, dtype=bf),
        torch.empty((H, n, s * c), device=dev, dtype=bf),
        torch.empty((n, d) if p_drop else (0,), device=dev, dtype=bf),
        torch.empty((H, n, n), device=dev, dtype=torch.float32),
    ]


@opaque(fake=_fwd_fake, name="pwa_sm100_fwd")
def _fwd_op(args: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float) -> list[torch.Tensor]:
    return _forward(args, eps_m, eps_z, p_drop, True)


def _infer_fake(args, eps_m, eps_z):
    return torch.empty_like(args[0][0])


@opaque(fake=_infer_fake, name="pwa_sm100_infer")
def _infer_op(args: list[torch.Tensor], eps_m: float, eps_z: float) -> torch.Tensor:
    return _forward(args, eps_m, eps_z, 0.0, False)[0]


# --------------------------------------------------------------------------------------------------------------------------- backward
def _mm32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b for bf16 operands with an fp32 result (the weight gradients of fp32 master parameters)."""
    return torch.mm(a, b, out_dtype=torch.float32)


def _dw(d_o: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """dw [H, N, N] = do . v^T (K = S*C) on cuBLAS, fp32 out: `pair_dw_fix` rewrites it in place for the softmax backward with the
    forward's fp32 w. (With the bf16 w and a bf16 dw, the module's own roundings plus the bf16 softmax w, the pair-side gradients dz,
    dgamma_z, dWb were 1.0-1.5x the module's error at L 128-768: YELLOW; measured 2026-11-16, docs/b200/msa.md.)"""
    return torch.bmm(d_o, v.transpose(1, 2), out_dtype=torch.float32)


def _bwd_fake(args, kept, dres, eps_m, eps_z, p_drop):
    return [torch.empty_like(args[i]) for i in (0, 1, 3, 4, 5, 6, 7, 8, 9)]


@opaque(fake=_bwd_fake, name="pwa_sm100_bwd")
def _bwd_op(
    args: list[torch.Tensor], kept: list[torch.Tensor], dres: torch.Tensor, eps_m: float, eps_z: float, p_drop: float
) -> list[torch.Tensor]:
    """[dmsa, dpair, dlnm_w, dlnm_b, dWv, dWg, dlnz_w, dWb, dWo] from dres [S, N, d_msa] (the residual's gradient is dres itself)."""
    msa, pair, _km, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo = args
    w16, v, y, o, draw, w32 = kept
    bf = torch.bfloat16
    m, z = msa[0], pair[0]
    s, n, d_msa = m.shape
    hc = wv.shape[0]
    k = _k()
    dres = dres.contiguous()
    dscale = 1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0
    if _uses_mw(args, True):
        # pwa_wprep reads the (fp32 or bf16) parameters itself: Wo^T, [Wg; Wv]^T, Wg / 2 in bf16, one launch; glue2 keeps dgp on chip
        wot, wgv_t, wgh = k.pwa_wprep(wg.detach().contiguous(), wv.detach().contiguous(), wo.detach().contiguous())
        d_o, dyg, gpart = k.pwa_glue2(o, y, dres, wgh, wot, draw if p_drop > 0 else None, dscale, _keep_thr(p_drop) if p_drop > 0 else 0.5)
        dv = torch.bmm(w16.transpose(1, 2), d_o)  # head-major [H, N, S*C]
        dw = _dw(d_o, v)
        dm, dwv, dlw, dlb, dwo, dwg = k.dv_bwd(
            dv,
            dyg,
            y,
            m,
            dres,
            wgv_t,
            lnm_w.detach().contiguous(),
            eps_m,
            int(wv.dtype == bf),
            int(lnm_w.dtype == bf),
            gpart,
            int(wo.dtype == bf),
        )
    else:
        kv = _value(d_msa, hc // H)
        # the three bf16 copies in one launch (opm_cast_bf16; was three .to, 2026-11-17)
        (wg16, wv16, wo16), _ = _bf16_params([wg, wv, wo])
        d_o, dgv, go, dout = kv.glue_w(o, y, dres, wg16, wo16, draw if p_drop > 0 else None, dscale)
        dv = torch.bmm(w16.transpose(1, 2), d_o)
        dw = _dw(d_o, v)
        dm, dlw, dlb = kv.dvb_w(dv, dgv, m, dres, wg16, wv16, lnm_w.detach().contiguous(), eps_m)
        rows = s * n
        dwo = _mm32(dout.view(rows, d_msa).t(), go.view(rows, hc))  # [D, HC]
        dwgv = _mm32(dgv.view(rows, 2 * hc).t(), y.view(rows, d_msa))  # [2 HC, D]: dWg over dWv
        dwg, dwv = dwgv[:hc], dwgv[hc:]
    k.pair_dw_fix(w16, w32, dw)  # the softmax backward with the fp32 w (in place on dw)
    dz, dwb, dzw = k.pair_bwd(
        z, w16, dw, lnz_w.detach().contiguous(), eps_z, wb.detach().contiguous(), int(wb.dtype == bf), int(lnz_w.dtype == bf)
    )
    return [
        dm.view(1, s, n, d_msa),
        dz.view(1, n, n, z.shape[-1]),
        # the parameter gradients in their parameters' dtypes, new tensors (a custom op's returns may not alias each other), ONE launch
        *_cast_cuda.cast([dlw, dlb, dwv, dwg, dzw, dwb, dwo], [p.dtype for p in (lnm_w, lnm_b, wv, wg, lnz_w, wb, wo)], force=True),
    ]


class _PwaSm100Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = list(args[:10])
        ctx.eps_m, ctx.eps_z, ctx.p_drop = args[10:]
        out, *kept = _fwd_op(tensors, ctx.eps_m, ctx.eps_z, ctx.p_drop)
        ctx.save_for_backward(*tensors, *kept)
        return out[None]

    @staticmethod
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        g = _bwd_op(list(vals[:10]), list(vals[10:]), dy[0].to(torch.bfloat16), ctx.eps_m, ctx.eps_z, ctx.p_drop)
        return (g[0], g[1], None, *g[2:], None, None, None)


# ------------------------------------------------------------------------------------------------------------------------------ entry
def _pad(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None):
    """S and N padded to the kernels' 128-row tiles -> (msa, pair, key mask bool [N'], S, N) with the original S / N. Zero MSA rows change
    no real row (the rows are independent); the pad tokens are masked as keys, so no real token averages over them."""
    s, n = msa.shape[1], msa.shape[2]
    ps, pn = -s % 128, -n % 128
    km = torch.ones(n, dtype=torch.bool, device=msa.device) if mask is None else mask.reshape(n).to(torch.bool)
    if ps or pn:
        msa = torch.nn.functional.pad(msa, (0, 0, 0, pn, 0, ps))
    if pn:
        pair = torch.nn.functional.pad(pair, (0, 0, 0, pn, 0, pn))
        km = torch.nn.functional.pad(km, (0, pn), value=False)
    return msa.contiguous(), pair.contiguous(), km.contiguous(), s, n


#: Inference reads bf16 forms of to_value / to_gate / to_out cast once per parameter version (no cast launch per call or per graph
#: replay; B200 2026-11-17, graph, L128 depth 1024: 128 / 8 x 8 63.3 -> 61.2 us, 64 / 8 x 8 44.9 -> 44.8: there pair_fwd on its 32 SMs
#: is then the longer branch). The same key as the TriMul weight forms (`modules/triangle.py`). A / B switch.
_EAGER_W16 = [True]


def _infer_weights(module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(Wv, Wg, Wo) for an inference call: the parameters themselves when they are bf16, else their bf16 casts, kept on the module and
    keyed by (storage, version, dtype) of the three parameters, so an in-place update or a new parameter casts again. A miss while a
    CUDA graph is being captured is not kept (the buffer would live in the graph's pool): that call casts inside the graph as before."""
    ps = (module.to_value.weight, module.to_gate.weight, module.to_out.weight)
    if not _EAGER_W16[0] or all(p.dtype == torch.bfloat16 for p in ps):
        return ps
    key = tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in ps)
    hit = module.__dict__.get("_b200_w16")
    if hit is not None and hit[0] == key:
        return hit[1]
    if torch.cuda.is_current_stream_capturing():
        return ps
    a, b, c = _bf16_params(list(ps))[0]
    w16 = (a, b, c)
    module.__dict__["_b200_w16"] = (key, w16)
    return w16


def pair_weighted_averaging(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, *, train: bool) -> torch.Tensor:
    """msa + MSAPairWeightedAveraging(msa, pair, mask) (the residual is inside the kernels); ``train``: the autograd form, the module's row
    dropout when it is in training mode."""
    if msa.shape[0] > 1:  # one stack at a time
        return torch.cat(
            [
                pair_weighted_averaging(module, msa[b : b + 1], pair[b : b + 1], None if mask is None else mask[b : b + 1], train=train)
                for b in range(msa.shape[0])
            ]
        )
    eps_m, eps_z = float(module.ln_msa.eps), float(module.ln_pair.eps)
    wv, wg, wo = (module.to_value.weight, module.to_gate.weight, module.to_out.weight) if train else _infer_weights(module)
    m, z, km, s0, n0 = _pad(msa, pair, mask)
    tensors = [
        m,
        z,
        km,
        module.ln_msa.weight,
        module.ln_msa.bias,
        wv,
        wg,
        module.ln_pair.weight,
        module.to_bias.weight,
        wo,
    ]
    if train:
        p_drop = float(module.drop_msa.p_drop) if module.training else 0.0
        out = _PwaSm100Fn.apply(*tensors, eps_m, eps_z, p_drop)
    else:
        out = _infer_op(tensors, eps_m, eps_z)[None]
    return out[:, :s0, :n0] if tuple(out.shape[1:3]) != (s0, n0) else out
