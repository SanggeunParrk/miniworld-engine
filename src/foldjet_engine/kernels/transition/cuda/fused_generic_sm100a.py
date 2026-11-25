"""Hand-CUDA sm_100a (B200) Transition for every call the tuned ``fused_sm100a`` / ``fused_wide_sm100a`` builds do not take.

``y = x + Ws (silu(Wa LN(x)) * (Wb LN(x)))`` with

* any channel width ``D`` that is a multiple of 64 up to 1024, any hidden width that is a multiple of 128 (the expansion factor is read
  off the weights),
* any row count (TMA zero-fills the tail of the last 128-row tile on the way in and clips it on the way out: no padding copy),
* bf16 activations on the bf16 tensor cores (``kind::f16``), or fp32 activations on the TF32 tensor cores (``kind::tf32``; only where
  torch allows TF32, the switch cuBLAS follows -- strict fp32 keeps the PyTorch reference); fp32 accumulation either way.

The kernels (``sm100/generic/``): ``tg_gemm.cu`` is one persistent warp-specialised tcgen05 GEMM (TMA producer warp, MMA warp,
four epilogue warps; accumulators double-buffered in tensor memory) built per epilogue, and ``tg_rows.cu`` the row passes.

    forward    tg_ln_fwd (xn, mean, rstd) -> GEMM SWIGLU (h = silu(xn Wa^T) (xn Wb^T)) -> GEMM RESID (y = h Ws^T + x)
    backward   GEMM GATE (dh = dy Ws and a, b recomputed from xn in one tile: h, [dA | dB]) -> GEMM PLAIN (d_xn = [dA | dB] [Wa; Wb], fp32)
               -> tg_ln_bwd (dx with the residual, dgamma / dbeta partials) -> tg_dw (dWs = dy^T h, [dWa; dWb] = [dA | dB]^T xn, fp32, and
               the dgamma / dbeta sum); at many rows tg_dw runs split over the rows right after GATE and tg_wsum adds its slabs and the
               dgamma / dbeta partials last (round 3 part 5; cuBLAS before). Launches but tg_prep are programmatic dependent ones.

At a few row tiles (a single track [B, L, D]: the GEMM grids would leave most SMs idle, each CTA bound by its operand intake) the output
tile widths come from ``_plan``, the backward's d_xn GEMM splits K into fp32 slabs that the LayerNorm backward adds in order (tg_ln_bwds),
the squeeze runs on cuBLAS ``addmm`` (it splits K inside one launch), and a training forward keeps a, b (SWIGLU_AB) so the backward's gate
reads them (GATE_AB, one operand phase) instead of recomputing them.

Elsewhere the forward keeps nothing but xn and the statistics (a and b are recomputed by the backward's GATE tile, which already streams xn). The
parameters keep their dtype: the kernels get activation-dtype copies made outside autograd (``tg_prep``; fp32 on the TF32 grid), and the
parameters get fp32 gradients in their own dtype. Every reduction has one order: a replay is bit-identical.

``supported()`` is the whole gate. Every launch is an opaque op with a fake, so the module traces under ``torch.compile``.
"""

from __future__ import annotations

import functools

import torch

from foldjet_engine.kernels._device import num_sms

from ..._compile import opaque
from ..._nvcc import build_failed
from . import fused_small_sm100a
from .fused_sm100a import _ext, _is_b200, build_cubins

MAX_D = 1024
#: Output-tile widths (columns of N per tile) each GEMM epilogue is built at, widest first: SWIGLU / CTGATE keep two accumulators, GATE
#: three, so their double-buffered accumulators fit the 512 tensor-memory columns. A bf16 output is staged in 64-column (128-B) groups, so
#: 32-column tiles exist only where the output is fp32. ``_pick_bn`` takes the widest that still gives every SM a tile (small M: a
#: single-track [1, L, D] activation is a few row tiles).
_BNS = {
    "swiglu": (128, 64),
    "resid": (128, 64),
    "plain": (128, 64, 32),
    "gate": (64,),
    "ctgate": (128, 64),
    "swiglu_ab": (128, 64),
    "gate_ab": (128, 64),
}
_BNS_F32 = {
    "swiglu": (128, 64, 32),
    "resid": (128, 64, 32),
    "plain": (128, 64, 32),
    "gate": (64, 32),
    "ctgate": (128, 64, 32),
    "swiglu_ab": (128, 64, 32),
    "gate_ab": (128, 64, 32),
}
_MODE = {"swiglu": 0, "resid": 1, "plain": 2, "gate": 3, "ctgate": 4, "swiglu_ab": 5, "gate_ab": 6}
#: The CTA-pair builds (-DCTA2: cta_group::2, M = 256 products, each CTA bringing half of the B tile) and their output tile widths, taken
#: by ``_K.gemm`` where the pair tiles fill the card at least twice: half the B-operand intake per output of the 1-CTA 128-row tiles, whose
#: TMA intake bound the D768 training GEMMs (the SWIGLU_AB producer alone streamed its operands in 795 of its 1543 us). Kernels alone,
#: bf16 [147 456 rows] d768 n4, us (1-CTA -> pair): SWIGLU_AB 1543 -> 1262-1290, RESID 626 -> 438 (cuBLAS addmm 503). Measured slower
#: as pairs and kept on single CTAs: SWIGLU 1018 -> 1042, GATE_AB 1058 -> 1110 (epilogue-bound).
_BNS2 = {"swiglu_ab": 128, "resid": 256}
_BNS2_F32 = {"swiglu_ab": 128, "resid": 256}
#: A/B switch for the pair builds
CTA2 = True


def _tf32_allowed() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def _specs(f32: bool) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """(name, source, flags) of every cubin of one activation dtype (``_load`` builds them; the sm100 prebuild plan lists them)."""
    specs = [("rows", "generic/tg_rows.cu", (f"-DF32={int(f32)}",))]
    for name, mode in _MODE.items():
        for bn in (_BNS_F32 if f32 else _BNS)[name]:
            specs.append((f"{name}{bn}", "generic/tg_gemm.cu", (f"-DMODE={mode}", f"-DF32={int(f32)}", f"-DBN={bn}")))
    for name, bn in (_BNS2_F32 if f32 else _BNS2).items():
        specs.append((f"{name}{bn}x2", "generic/tg_gemm.cu", (f"-DMODE={_MODE[name]}", f"-DF32={int(f32)}", f"-DBN={bn}", "-DCTA2=1")))
    nts = DW_NTS_F32 if f32 else DW_NTS  # fp32: the -DSPL builds (dy as two TF32 values; the extra A slot leaves room for NT <= 128)
    specs += [(f"dw{nt}", "generic/tg_dw.cu", (f"-DF32={int(f32)}", f"-DNT={nt}", f"-DSPL={int(f32)}")) for nt in nts]
    if f32:
        specs += [(f"fusedh{d}", "generic/tg_fused_h16.cu", (f"-DD={d}",)) for d in FUSED_H16_WIDTHS]
        specs += [(f"fusedh{d}s", "generic/tg_fused_h16.cu", (f"-DD={d}", "-DSAVE_BUILD=1")) for d in FUSED_H16_WIDTHS]  # training
        specs += [("fusedh128e", "generic/tg_fused_h16.cu", ("-DD=128", "-DEP3=1"))]  # (S90) the epilogue split over both warp groups
    return tuple(specs)


#: fp32 INFERENCE widths whose forward runs on the fp16 tensor cores in three launches where neither the one-launch ``tg_fused_h16`` nor
#: the small-M chains take it (round 3 part 4): ``tsx_ln -DH16`` (xn fp16 / s_x) -> ``tg_gemm -DH16`` SWIGLU (h fp16 / s_h) -> RESID (fp32
#: residual and out), the scales of ``_h16_forms`` -- the H100 runs fp32 D 512 this way (``transition_lnsg_f32`` + ``squeeze_res_h16``).
#: D 512 only: at D 768 (rows 37 float32 L 1536 / 2048, 12 / 16 tiles) the fp16 products measured YELLOW 1.0054 / 1.0067 x tol against
#: the Triton / H100 tokens' 1.0036 / 1.0031 (accuracy rule), so those keep the TF32 forward.
#: Round 3 part 7: D 768 too (above the fp16 chain's 12 tiles: row 37 float32 L 2048 55.3 -> 38.9 us; YELLOW, inside the engine
#: standard).
#: GENERIC_H16 = False: the TF32 chain (A/B).
GENERIC_H16_WIDTHS = (512, 768)
GENERIC_H16 = True


def _specs_h16() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """The fp16-operand fp32 forward's cubins (``GENERIC_H16_WIDTHS``)."""
    specs = [(f"ln{d}", "widths/tsx_ln.cu", (f"-DDIM={d}", "-DH16=1")) for d in GENERIC_H16_WIDTHS]
    for name in ("swiglu", "resid"):
        for bn in _BNS[name]:
            specs.append((f"{name}{bn}", "generic/tg_gemm.cu", (f"-DMODE={_MODE[name]}", "-DH16=1", f"-DBN={bn}")))
    specs.append(("resid256x2", "generic/tg_gemm.cu", (f"-DMODE={_MODE['resid']}", "-DH16=1", "-DBN=256", "-DCTA2=1")))
    return tuple(specs)


@functools.lru_cache(maxsize=1)
def _load_h16():
    ext = _ext()
    for name, path in build_cubins("generic_h16", _specs_h16()).items():
        ext.load_cubin(f"generic_h16_{name}", path)
    return ext


@functools.lru_cache(maxsize=4)
def _load(f32: bool):
    """The cubins of one activation dtype, registered with the sm_100a extension (loaded on the current device: a cubin the driver
    rejects fails here, inside ``available()``)."""
    ext = _ext()
    tag = "f32" if f32 else "bf16"
    for name, path in build_cubins(f"generic_{tag}", _specs(f32)).items():
        ext.load_cubin(f"generic_{tag}_{name}", path)
    return ext


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """sm_100 (B200), bf16 (or fp32 with TF32 allowed), D a multiple of 64 up to 1024, hidden a multiple of 128, weights bf16 or fp32
    (cast to the activation dtype inside; the gradients come back in the weights' own dtype). Any row count."""

    if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float32) or x.numel() == 0:
        return False
    if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (wa, ws)):
        return False
    if x.dtype is torch.float32 and not _tf32_allowed():
        return False
    if not _is_b200(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d = x.shape[-1]
    if d % 64 or d > MAX_D:
        return False
    hid = wa.shape[0]
    return hid % 128 == 0 and tuple(wa.shape) == (hid, d) and tuple(ws.shape) == (d, hid)


_BUILD_FAILED = False


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    return any(isinstance(t, FakeTensor) for t in tensors)


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful (cached) build; a build failure raises (B200 has no Triton / silent PyTorch fallback)."""
    if _BUILD_FAILED or not supported(x, wa, ws):
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _load(x.dtype is torch.float32)
    except Exception as exc:
        build_failed("generic sm100a Transition", exc)
        return False
    return True


# ---------------------------------------------------------------------------------------------------------------- launches
def _ln_variant(d: int, f32: bool) -> tuple[str, int]:
    """(suffix of the tg_ln_fwd / tg_ln_bwd instantiation for width ``d``, rows per warp): 16-B vectors and as many lanes per row as
    divide the row (see tg_rows.cu)."""
    if f32:
        vec, tpr = (4, 16) if d == 64 else ((4, 32) if d % 128 == 0 else (2, 32))
    elif d == 64:
        vec, tpr = 8, 8
    elif d == 128:
        vec, tpr = 8, 16
    else:
        vec, tpr = (8 if d % 256 == 0 else 4 if d % 128 == 0 else 2), 32
    return f"v{vec}t{tpr}n{d // tpr}", 32 // tpr


def _pick_bn(widths: tuple[int, ...], m: int, n: int, nsm: int) -> int:
    """The widest tile with at least one tile per SM, else the narrowest."""
    for bn in widths:
        if -(-m // 128) * -(-n // bn) >= nsm:
            return bn
    return widths[-1]


#: Per-SM TMA intake the small-M plan below assumes (bytes per microsecond; ~52 B / clock), and the rate a split-K GEMM's fp32 partials
#: are written and read back at (bytes per microsecond, the whole GPU)
_INTAKE = 1.0e5
_PARTIAL_RATE = 4.0e6
#: A/B override of the forward squeeze at small M: None (plan), "split", "resid", "cublas"
SQUEEZE_ROUTE = None
#: bf16 inference: the cuBLAS squeeze also where the RESID tiles leave SMs idle (round 3 part 8; False: A/B)
SQUEEZE_SMALL_CUBLAS = True
#: A/B override of the backward d_xn GEMM: None (plan), "nosplit", "cublas"
DXN_ROUTE = None
#: K splits a PLAIN GEMM may take at small M
_SPLITS = (1, 2, 3, 4, 6, 8, 12, 16)


def _plan(name: str, m: int, n: int, ks: tuple[int, ...], f32: bool, nsm: int, split: bool) -> tuple[int, int]:
    """(output tile width, K splits) of one GEMM. ``ks``: the K of each operand phase (GATE: two). Where the widest tile already gives every
    SM a tile, the widest (no split). Below that (a single track: a few row tiles) each CTA's time is its operand intake -- the A rows and
    the B columns of every K-block it walks, per tile it takes -- so the plan minimises waves x per-tile intake over the tile widths and,
    with ``split`` (PLAIN), over K splits (whose fp32 partials cost a write and a read: the consumer adds them). Measured (fp32 TF32,
    [1, 768, 384] n4, kernels alone): GATE 32 -> 64 columns 20.1 -> 13.5 us (one wave instead of two); d_xn split 4 ways 9.9 us against
    cuBLAS 10.8 (the K-serial GEMM: 19.6 at D512 n2). A model, not a tuning: the ratios hold, the absolute times are ~2x it (latency)."""
    widths = (_BNS_F32 if f32 else _BNS)[name]
    mt = -(-m // 128)
    if mt * -(-n // widths[0]) >= nsm:
        return widths[0], 1
    esz, bke = (4, 32) if f32 else (2, 64)
    nb = {"swiglu": (2,), "resid": (1,), "plain": (1,), "gate": (1, 2), "ctgate": (1, 1), "swiglu_ab": (2,), "gate_ab": (1,)}[name]
    best = None
    for bn in widths:
        tiles = mt * -(-n // bn)
        for s in _SPLITS if split else (1,):
            if s > 1 and any((k // bke) % s or k // bke < 2 * s for k in ks):  # no split always plans (D 64 bf16: one K-block)
                continue
            per_tile = sum((128 + b * bn) * (k // s) * esz for b, k in zip(nb, ks, strict=True))
            if name == "gate_ab":  # the saved a, b rows of the tile
                per_tile += 2 * 128 * bn * esz
            cost = -(-tiles * s // nsm) * per_tile / _INTAKE
            if s > 1:
                cost += 2 * s * (mt * 128) * n * 4 / _PARTIAL_RATE
            if best is None or cost < best[0] - 1e-9:
                best = (cost, bn, s)
    return best[1], best[2]


def _ab(m: int, hid: int, nsm: int, d: int = 0, f32: bool = True) -> bool:
    """Whether a training call keeps a, b from the forward (SWIGLU_AB -> GATE_AB) instead of recomputing them in the backward's GATE: at
    a few row tiles (the gate's 64-column output tiles under one wave), where the GATE grid is bound by its per-CTA operand intake and the
    recompute doubles it, while the stores are small. Measured (module training step, CUDA graph): fp32 [1, 154, 768] n2 104.0 -> 96.6 us,
    bf16 [1, 500, 1024] n2 114.8 -> 109.0, [1, 300, 768] n4 110.7 -> 107.3; [1, 768, 384] n4 fp32 even (88.4 / 89.1).
    And for bf16 at d >= 768 at any row count: the recompute is 2 M d hid more tensor-core work (a third of the step at n = 4) against
    one write and one read of [a | b]. Measured, bf16 [1, 384, 384, 768] n4 (with the cuBLAS d_xn of ``_dxn_cublas``): 8189 -> 6378 us
    (min of 50 replays; the recompute alone 8189 -> 6772)."""
    if AB_FORCE is not None:
        return AB_FORCE
    return -(-m // 128) * -(-hid // 64) < nsm or d >= (_AB_F32_MIN_D if f32 else _AB_WIDE)


#: bf16 widths from which training keeps a, b (``_ab``) and takes d_xn from cuBLAS (``_dxn_cublas``) at every row count
_AB_WIDE = 768
#: fp32 widths from which training keeps a, b too (S90, 2026-11-18; the TF32 recompute GATE tile with three accumulators ran at ~30 % of the
#: cuBLAS TF32 rate; module training step, graph, L2 flushed, us, recompute -> keep a, b: D 512 n 2 [16 384 / 65 536 / 147 456 / 262 144 / 409 600 /
#: 589 824, 512] 533 / 1800 / 3923 / 7078 / 11015 / 15583 -> 440 / 1432 / 3074 / 5384 / 8354 / 12093; D 256 n 2 (with the TF32 forward
#: instead of the fp16 one, `_train_h16`) 196.6 / 612.4 / 1243.2 / 3297 -> 188.5 / 563.2 / 1157.3 / 3087). D 128 keeps the fp16 forward and the recompute
#: (keep a, b + TF32 forward: 276 -> 303 us at 65 536 rows). `_AB_F32_MIN_D` = 1 << 30: the old rule (A/B).
_AB_F32_MIN_D = 256


def _dxn_cublas(d: int, f32: bool) -> bool:
    """The backward's d_xn = [dA | dB] [Wa; Wb] (K = 2 hid) by cuBLAS at bf16 d >= 768: the PLAIN tile ran it at ~0.85 PF where cuBLAS'
    2-CTA tiles run ~1.5 (bf16 [1, 384, 384, 768] n4, kept a, b: 6772 -> 6378 us).
    fp32 at D in ``_DXN_CUBLAS_F32`` too (round 3 part 8: [65 536 / 147 456 / 262 144, 256] n 2 fp32 training 657 / 1471 / 2425 -> 629 / 1384 /
    2271 us, the PLAIN TF32 tile against cuBLAS TF32)."""
    return d in _DXN_CUBLAS_F32 if f32 else d >= _AB_WIDE


#: fp32 widths whose d_xn is cuBLAS (TF32) at one K split
#: (S90) 128 and 512 added: [16 384 / 147 456 / 262 144 / 409 600 / 589 824, 128] n 2 fp32 98.3 / 546.8 / 917.2 / 1398.9 / 1994.6 -> 96.3 / 522.9 / 876.5 /
#: 1337.3 / 1902.6 us; D 512 (with a, b kept) 7 -> 4.85 ms at 262 144 rows beside the AB change above
_DXN_CUBLAS_F32 = (128, 256, 512)


#: fp32 inference widths whose whole forward is the one kernel ``tg_fused_h16`` (2026-11-17, round 3; it replaced the TF32 ``tg_fused_f32``): the products
#: on the fp16 tensor cores (fp16 = TF32's mantissa, rounded to nearest; twice the TF32 rate, half the operand bytes), as the H100's
#: f32io forward runs them; device scales keep every fp16 operand in range (``_h16_forms``). FUSED_H16 = False runs the TF32 GEMM chain (A/B).
FUSED_H16_WIDTHS = (64, 128, 256)
FUSED_H16 = True
#: the fp16 operands' range: each scale is max(bound / _F16_BOUND, 1) (the H100's fused_wide_sm90a._F16_BOUND)
_F16_BOUND = 30000.0
#: the bound of h (round 3 part 4): Cauchy-Schwarz (the rows' L2 norms; ~1000 for N(0, 1 / D) weights at D 384 - 768, so s_h = 1) instead of
#: the element bound times the rows' L1 norms (~2e5 - 7e5 there: s_h 6 - 22, which pushed the small h of the single-track rows into fp16's
#: subnormals: rows 35 / 37 float32 measured YELLOW 1.05 - 1.15 x tol with it). Both are rigorous; False: the L1 form (A/B).
H16_BOUND_L2 = True


#: (2026-11-17, round 3 part 4) the D 256 cap of 2^20 rows is gone: with the L1 h bound tg_fused_h16 had measured YELLOW 1.050 / 1.161 x tol
#: at 1536^2 / 2048^2 rows (H100 token 1.040 / absent, Triton 1.000 / 1.002); with the Cauchy-Schwarz bound (``H16_BOUND_L2``) GREEN 0.9998 / 0.9995.


#: (S90) tg_fused_h16 at D 128 with the y epilogue split over the SwiGLU and the LayerNorm warps and no staging (EP3): from this many rows (fine us, routed -> EP3: [98 304, 128] n 2 45.0 -> 43.6,
#: [262 144] 103.1 -> 91.2, [589 824] 223.2 -> 196.5, [1 048 576] 399.6 -> 353.3; below it level or slower: [65 536] 32.3 -> 32.5, [16 384] 11.7 -> 12.6)
EP3_F32 = True
EP3_MIN_ROWS = 98304


def _fused_h16(d: int, hid: int, m: int) -> bool:
    return FUSED_H16 and d in FUSED_H16_WIDTHS and hid % 64 == 0


def _h16_forms_fake(gamma, beta, wa, wb, ws):
    h, d = wa.shape
    f16 = dict(dtype=torch.float16, device=wa.device)
    return (
        torch.empty((h, d), **f16),
        torch.empty((h, d), **f16),
        torch.empty((d, h), **f16),
        torch.empty((4,), dtype=torch.float32, device=wa.device),
    )


@opaque(fake=_h16_forms_fake, name="transition_generic_h16_forms_sm100a")
def _h16_forms(
    gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(Wa16, Wb16, Ws16 [D, H], sc fp32 [4] = [1 / s_x, s_x s_w, 1 / s_h, s_h s_ws]) for ``tg_fused_h16``. Bounds (the H100 f32io forms'):
    |xn| <= sqrt(D) max|gamma| + max|beta|; |a|, |b| <= the L2 norm bound of xn times the largest row L2 norm of Wa, Wb (``H16_BOUND_L2``); |h| <= |a|max |b|max. Each scale
    is 1 unless its bound passes ``_F16_BOUND``. Device arithmetic only (no host sync); made once per parameter version (inference)."""
    if _is_fake(wa):
        return _h16_forms_fake(gamma, beta, wa, wb, ws)
    g, b, wa, wb, ws = (t.float() for t in (gamma, beta, wa, wb, ws))
    d = wa.shape[1]
    one = torch.ones((), dtype=torch.float32, device=wa.device)
    xb = d**0.5 * g.abs().max() + b.abs().max()
    s_x = torch.maximum(xb / _F16_BOUND, one)
    s_w = torch.maximum(torch.maximum(wa.abs().max(), wb.abs().max()) / _F16_BOUND, one)
    if H16_BOUND_L2:  # |a_j| = |xn . Wa_j| <= ||xn||_2 ||Wa_j||_2, ||xn||_2 <= sqrt(D) max|gamma| + ||beta||_2 (||xhat||_2 = sqrt(D))
        x2 = d**0.5 * g.abs().max() + b.norm()
        hb = x2 * wa.norm(dim=1).max() * x2 * wb.norm(dim=1).max()
    else:
        hb = xb * wa.abs().sum(1).max() * xb * wb.abs().sum(1).max()
    s_h = torch.maximum(hb / _F16_BOUND, one)
    s_ws = torch.maximum(ws.abs().max() / _F16_BOUND, one)
    sc = torch.stack((1.0 / s_x, s_x * s_w, 1.0 / s_h, s_h * s_ws)).contiguous()
    return (wa / s_w).half().contiguous(), (wb / s_w).half().contiguous(), (ws / s_ws).half().contiguous(), sc


def _fwd_h16_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps):
    return torch.empty_like(x)


@opaque(fake=_fwd_h16_fake, name="transition_generic_fwd_h16_sm100a")
def _fwd_h16_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa16: torch.Tensor,
    wb16: torch.Tensor,
    ws16: torch.Tensor,
    sc: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """x fp32 [M, D] -> x + transition(x) (fp32), one ``tg_fused_h16`` launch. The fp16 forms go to TMA as 2-byte maps (bf16-typed:
    TMA copies bytes)."""
    if _is_fake(x, wa16):
        return _fwd_h16_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps)
    k = _K(x)
    m, d = x.shape
    hid = wa16.shape[0]
    out = torch.empty_like(x)
    t = k.ext.tmap
    v = lambda w: w.view(torch.bfloat16)  # noqa: E731
    ep3 = EP3_F32 and d == 128 and m % 128 == 0 and m >= EP3_MIN_ROWS
    k.ext.launch_kernel(
        f"generic_f32_fusedh{d}" + ("e" if ep3 else ""),
        "tg_fused_h16_sm100",
        max(1, min(k.nsm, -(-m // 128))),
        448,
        1,
        [
            t(x, d, m, 32, 128),
            t(v(wa16), d, hid, 64, 64),
            t(v(wb16), d, hid, 64, 64),
            t(v(ws16), hid, d, 64, d),
            t(out, d, m, 32, 32),
            x,
            gamma,
            beta,
            sc,
            float(eps),
            m,
            hid,
            out,
            out,
            out,
            0,
        ],
    )
    return out


#: A/B override of ``_ab`` (None: automatic)
AB_FORCE = None

#: (round 3 part 5) the backward's weight gradients dWs = dy^T h, [dWa; dWb] = [dA | dB]^T xn on ``tg_dw.cu`` (one launch for both, the
#: rows split over S slabs that ``tg_wsum`` adds in order beside the dgamma | dbeta sum) instead of two cuBLAS GEMMs (+ cuBLAS' split-K
#: reduce launch). Output tile widths built; DW_ROUTE = "cublas" restores cuBLAS (A/B), DW_FORCE = (NT, S) pins the plan.
DW_NTS = (64, 128, 256)
DW_NTS_F32 = (64, 128)
#: (round 3 part 5) the generic GEMM / row launches as programmatic dependent launches (every kernel waits for its predecessor's outputs
#: with griddepcontrol.wait, then releases the next launch's prologue); False: plain launches (A/B)
PDL = True
DW_ROUTE = None
DW_FORCE = None
#: per-SM operand intake (bytes / ns) and the slab traffic rate of the whole card (bytes / ns) of the plan's cost model, and the cost of
#: the slab pass (ns) beyond its bytes
_DW_SM_BPNS = 150.0
_DW_L2_BPNS = 8000.0
_DW_SLAB_NS = 1500.0
#: bf16 rows above which tg_dw keeps cuBLAS (fp32: DW_MAX_F32_ELEMS) (training step, graph, us, cuBLAS -> tg_dw): bf16 16 384 rows D 256 n 2
#: 137 -> 131, n 4 188 -> 184, D 512 n 2 260 -> 248; 147 456 rows D 384 n 2 976 -> 1092, D 512 n 4 (65 536) 1255 -> 1334; fp32 D 512 n 2
#: 147 456 rows 4224 -> 4279; fp32 D 256 n 2 262 144 rows 2998 -> 2475, D 128 n 2 65 536 408 -> 289
DW_MAX_ROWS = 16384
#: (round 3 part 5, takeover) fp32 (TF32): tg_dw only up to this many x elements (rows x D); measured (graph, us, tg_dw -> cuBLAS, `b200cfg/p6dw.sh`):
#: D 512 16 384 rows 708 -> 539, D 256 16 384 227 -> 232, 65 536 760 -> 703, 262 144 2848 -> 2637, 589 824 6372 -> 5618; D 128 16 384 104 -> 115,
#: 65 536 305 -> 309, 147 456 633 -> 629, 589 824 2353 -> 2227 (the dy split pass and the slab traffic grow with M; cuBLAS' fp32 GEMMs do not).
#: 2026-11-17 night: 2^22 -> 2^21 with cuBLAS over 8 row slabs (WGRAD_SPLIT): fp32 [16 384, 256] n 2 training step 204.8 -> 194.6 us (tg_dw
#: 59.9 + tg_wsum 10.2 + the dy split 10.1 against two slab bmm 22.7 + 16.4, tg_round_tf32 8.2); D 128 at 16 384 rows (2^21) keeps tg_dw
#: (96.2 either way)
DW_MAX_F32_ELEMS = 1 << 21


def _dw_tiles(d: int, hid: int, nt: int) -> int:
    return (d // 128) * (hid // nt) + (2 * hid // 128) * (d // nt)


def _dw_plan(m: int, d: int, hid: int, nsm: int, f32: bool) -> tuple[int, int] | None:
    """(output tile width NT, row splits S) of ``tg_dw``, or None for cuBLAS. Each CTA streams its rows' A (128 columns) and B (NT
    columns) operands: the plan minimises waves x per-CTA intake (+ the slab pass when S > 1)."""
    if DW_ROUTE == "cublas" or d % 128 or hid % 64:
        return None
    if DW_FORCE is not None:
        return DW_FORCE
    if DW_ROUTE is None and (m * d > DW_MAX_F32_ELEMS if f32 else m > DW_MAX_ROWS):
        return None  # cuBLAS' 2-CTA tiles win once the output tiles alone fill the card a few times (DW_MAX_ROWS / DW_MAX_F32_ELEMS)
    esz = 4 if f32 else 2
    nkb = -(-m // 64)
    best = None
    for nt in DW_NTS_F32 if f32 else DW_NTS:
        if hid % nt or d % nt:
            continue
        tiles = _dw_tiles(d, hid, nt)
        for s in range(1, nkb + 1):
            if s > 1 and tiles * s > nsm:
                break
            waves = -(-tiles * s // nsm)
            cost = waves * -(-nkb // s) * (128 + nt) * 64 * esz / _DW_SM_BPNS
            if s > 1:
                cost += s * 3 * d * hid * 4 / _DW_L2_BPNS + _DW_SLAB_NS
            if best is None or cost < best[0]:
                best = (cost, nt, s)
    return None if best is None else (best[1], best[2])


class _K:
    """The kernels of one activation dtype on the current device."""

    def __init__(self, x: torch.Tensor, h16: bool = False):
        self.h16 = h16  # the fp16-operand fp32 forward (``_specs_h16``): bf16-shaped tiles, fp32 RESID output
        self.f32 = x.dtype is torch.float32 and not h16
        self.tag = "h16" if h16 else "f32" if self.f32 else "bf16"
        self.ext = _load_h16() if h16 else _load(self.f32)
        self.nsm = num_sms(x.device)
        self.bke = 32 if self.f32 else 64  # elements in a 128-byte K-block row

    def rows(self, kernel, grid, args, block=256):
        # tg_prep stays a plain launch: the small-M chains load its weight copies before their own griddepcontrol.wait
        self.ext.launch_kernel(f"generic_{self.tag}_rows", kernel, grid, block, 1, list(args), PDL and not kernel.startswith("tg_prep"))

    def gemm(
        self,
        name,
        m,
        n,
        a0,
        b0,
        o0,
        *,
        a1=None,
        b1=None,
        b2=None,
        o1=None,
        res=None,
        bias=None,
        k0=0,
        k1=0,
        ldres=0,
        hoff=0,
        bn=None,
        ks=1,
        mstride=0,
    ):
        """C[m, n] by the ``name`` epilogue. a* [m, K] and b* [n, K] (K-major), o* the outputs; every map is built here. ``bn``: the
        output tile width (None: ``_pick_bn``); ``ks`` > 1: split K, slice s storing its fp32 partial at rows s ``mstride`` + m of o0."""
        if bn is None:
            bn = _pick_bn((_BNS_F32 if self.f32 else _BNS)[name], m, n, self.nsm)
        bn2 = {"resid": 256}.get(name) if self.h16 else (_BNS2_F32 if self.f32 else _BNS2).get(name)
        pair = CTA2 and ks == 1 and bn2 is not None and n % bn2 == 0 and -(-m // 256) * (n // bn2) >= self.nsm
        if pair:
            bn = bn2
        bb = bn // 2 if pair else bn  # B rows each CTA brings
        ocg = 32 if (self.f32 or name == "plain" or (self.h16 and name == "resid")) else 64
        t = self.ext.tmap
        ma0 = t(a0, k0, m, self.bke, 128)
        ma1 = t(a1, k1, m, self.bke, 128) if a1 is not None else ma0
        mb0 = t(b0, b0.shape[1], b0.shape[0], self.bke, bb)
        mb1 = t(b1, b1.shape[1], b1.shape[0], self.bke, bb) if b1 is not None else mb0
        mb2 = t(b2, b2.shape[1], b2.shape[0], self.bke, bb) if b2 is not None else mb0
        if name == "gate_ab":  # the epilogue TMA-loads the saved [a | b] (res) in its staging groups: OCG x 32 boxes
            mb1 = t(res, res.shape[1], res.shape[0], ocg, 32)
        mo0 = t(o0, o0.shape[1], o0.shape[0], ocg, 32)
        mo1 = t(o1, o1.shape[1], o1.shape[0], ocg, 32) if o1 is not None else mo0
        if pair:
            tiles = -(-m // 256) * (n // bn)
            self.ext.launch_kernel(
                f"generic_{self.tag}_{name}{bn}x2",
                "tg_gemm_sm100",
                2 * max(1, min(self.nsm // 2, tiles)),
                320,
                2,
                [
                    ma0,
                    ma1,
                    mb0,
                    mb1,
                    mb2,
                    mo0,
                    mo1,
                    res if res is not None else o0,
                    bias if bias is not None else o0,
                    m,
                    n,
                    k0,
                    k1,
                    ldres,
                    hoff,
                    ks,
                    mstride,
                ],
                PDL,
            )
            return
        tiles = -(-m // 128) * -(-n // bn) * ks
        self.ext.launch_kernel(
            f"generic_{self.tag}_{name}{bn}",
            "tg_gemm_sm100",
            max(1, min(self.nsm, tiles)),
            320,
            1,
            [
                ma0,
                ma1,
                mb0,
                mb1,
                mb2,
                mo0,
                mo1,
                res if res is not None else o0,
                bias if bias is not None else o0,
                m,
                n,
                k0,
                k1,
                ldres,
                hoff,
                ks,
                mstride,
            ],
            PDL,
        )


def _prep_fake(wa, wb, ws, f32, bwd):
    """Output structure of ``_prep``: (wa, wb, ws) and, with ``bwd``, Ws^T and [Wa; Wb]^T (1 x 1 placeholders without)."""
    h, d = wa.shape
    dtype = torch.float32 if f32 else torch.bfloat16
    e = lambda *s: wa.new_empty(s, dtype=dtype)  # noqa: E731
    return (e(h, d), e(h, d), e(d, h)) + ((e(h, d), e(d, 2 * h)) if bwd else (e(1, 1), e(1, 1)))


@opaque(fake=_prep_fake, name="transition_generic_prep_sm100a")
def _prep(
    wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, f32: bool, bwd: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(wa, wb, ws) in fp32 on the TF32 grid (``f32``) or bf16 and, with ``bwd``, Ws^T [H, D] and [Wa; Wb]^T [D, 2H]; one launch."""
    if _is_fake(wa, wb, ws):
        return _prep_fake(wa, wb, ws, f32, bwd)
    dtype = torch.float32 if f32 else torch.bfloat16
    h, d = wa.shape
    src = wa.dtype
    wa, wb, ws = (w.to(src).contiguous() for w in (wa, wb, ws))
    e = lambda *s: torch.empty(s, dtype=dtype, device=wa.device)  # noqa: E731
    oa, ob, os_ = e(h, d), e(h, d), e(d, h)
    ost, oabt = (e(h, d), e(d, 2 * h)) if bwd else (e(1, 1), e(1, 1))
    k = _K(oa)
    k.rows(
        "tg_prep",
        2 * (h // 32) * (d // 32) if bwd else min(-(-h * d // 256), k.nsm * 8),
        (wa, wb, ws, oa, ob, os_, ost, oabt, h, d, int(src is torch.float32), int(bwd)),
    )
    return oa, ob, os_, ost, oabt


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure of ``_fwd_launch``: (out, xn, mean, rstd, ab); mean / rstd are 1-element placeholders unless ``save``, ab the
    forward's [a | b] [M, 2H] where the backward reads it (``save`` and ``_ab``), else a 1 x 1 placeholder."""
    m, d = x.shape
    f32 = dict(dtype=torch.float32, device=x.device)
    rows = m if save else 1
    ab = (
        x.new_empty((m, 2 * wa.shape[0]))
        if save and _ab(m, wa.shape[0], num_sms(x.device), d, x.dtype is torch.float32)
        else x.new_empty((1, 1))
    )
    return torch.empty_like(x), torch.empty_like(x), torch.empty((rows,), **f32), torch.empty((rows,), **f32), ab


@opaque(fake=_fwd_launch_fake, name="transition_generic_fwd_sm100a")
def _fwd_launch(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float, save: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D]; the weights in x's dtype (fp32: TF32 grid). Returns (out, xn, mean, rstd, ab), out = transition(x) + x; see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    k = _K(x)
    m, d = x.shape
    hid = wa.shape[0]
    if fused_small_sm100a.chain_takes(m, d, hid, k.f32) and (not save or _ab(m, hid, k.nsm, d, k.f32)):
        # a few row tiles of a small-M cell (round 3): the programmatic dependent chain, which keeps [a | b] for GATE_AB with save
        return fused_small_sm100a.chain(x, gamma, beta, wa, wb, ws, eps, save)
    f32 = dict(dtype=torch.float32, device=x.device)
    xn = torch.empty_like(x)
    mean = torch.empty((m if save else 1,), **f32)
    rstd = torch.empty_like(mean)
    sfx, rpw = _ln_variant(d, k.f32)
    # the squeeze + residual: at a few row tiles (where the plan would split K) cuBLAS addmm, which splits K inside one launch --
    # measured faster than the split-K PLAIN GEMM + tg_splitsum pair (an extra launch); else the RESID epilogue
    _, ks = _plan("plain", m, d, (hid,), k.f32, k.nsm, True)
    # bf16 inference whose RESID tiles leave SMs idle (round 3 part 8): cuBLAS too -- bf16 [1, 2048, 384] n4 the RESID squeeze (48 tiles
    # of K 1536) 10.8 us against cuBLAS (64 x 96 two-CTA tiles) 5.7, call 28.7 -> 22.5 us
    small = SQUEEZE_SMALL_CUBLAS and not save and not k.f32 and -(-m // 128) * -(-d // 128) < k.nsm
    route = SQUEEZE_ROUTE or ("cublas" if ks > 1 or small else "resid")
    # the cuBLAS squeeze accumulates in place onto a copy of x that the LayerNorm launch writes while it has the row in registers
    # (torch.addmm(x, ...) copied x in a launch of its own: 1.6-1.8 us + a launch gap of a 22-30 us call)
    out = torch.empty_like(x) if route == "cublas" else xn
    k.rows(
        f"tg_ln_fwd_{sfx}",
        max(1, min(-(-m // (8 * rpw)), k.nsm * 8)),
        (x, gamma, beta, xn, mean, rstd, m, d, float(eps), int(save), out, int(route == "cublas")),
    )
    h = torch.empty((m, hid), dtype=x.dtype, device=x.device)
    if save and _ab(m, hid, k.nsm, d, k.f32):  # a few row tiles, or a wide bf16 d: keep a, b for the backward's GATE_AB
        ab = torch.empty((m, 2 * hid), dtype=x.dtype, device=x.device)
        k.gemm("swiglu_ab", m, hid, xn, wa, h, b1=wb, o1=ab, k0=d, hoff=hid, bn=_plan("swiglu_ab", m, hid, (d,), k.f32, k.nsm, False)[0])
    else:
        ab = x.new_empty((1, 1))
        k.gemm("swiglu", m, hid, xn, wa, h, b1=wb, k0=d, bn=_plan("swiglu", m, hid, (d,), k.f32, k.nsm, False)[0])
    if route == "cublas":
        out.addmm_(h, ws.t())  # y = x + h Ws^T: out holds x (tg_ln_fwd copy)
    elif route == "split":
        bn, ks = _plan("plain", m, d, (hid,), k.f32, k.nsm, True)
        mp = -(-m // 128) * 128
        part = torch.empty((ks * mp, d), dtype=torch.float32, device=x.device)
        out = torch.empty_like(x)
        k.gemm("plain", m, d, h, ws, part, k0=hid, bn=bn, ks=max(ks, 2), mstride=mp)
        k.rows("tg_splitsum", max(1, min(-(-m * d // 1024), k.nsm * 4)), (part, x, out, m, d, max(ks, 2), mp * d))
    else:
        out = torch.empty_like(x)
        k.gemm("resid", m, d, h, ws, out, res=x, k0=hid, ldres=d, bn=_plan("resid", m, d, (hid,), k.f32, k.nsm, False)[0])
    return out, xn, mean, rstd, ab


def _generic_h16(d: int, m: int) -> bool:
    return GENERIC_H16 and d in GENERIC_H16_WIDTHS


@opaque(fake=_fwd_h16_fake, name="transition_generic_fwd_h16x3_sm100a")
def _fwd_h16x3_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa16: torch.Tensor,
    wb16: torch.Tensor,
    ws16: torch.Tensor,
    sc: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """x fp32 [M, D] -> x + transition(x) (fp32) in three launches on the fp16 tensor cores (``GENERIC_H16_WIDTHS``): tsx_ln -DH16 ->
    SWIGLU -DH16 -> RESID -DH16 (the fp16 tensors go to TMA as bf16-typed 2-byte maps; ``sc`` rides in the ``bias`` slot)."""
    if _is_fake(x, wa16):
        return _fwd_h16_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps)
    k = _K(x, h16=True)
    m, d = x.shape
    hid = wa16.shape[0]
    v = lambda w: w.view(torch.bfloat16)  # noqa: E731
    xn = torch.empty((m, d), dtype=torch.float16, device=x.device)
    h = torch.empty((m, hid), dtype=torch.float16, device=x.device)
    out = torch.empty_like(x)
    k.ext.launch_kernel(f"generic_h16_ln{d}", "transition_sx_ln", (m + 7) // 8, 256, 1, [x, gamma, beta, xn, sc, m, float(eps)], True)
    k.gemm("swiglu", m, hid, v(xn), v(wa16), v(h), b1=v(wb16), bias=sc, k0=d, bn=_plan("swiglu", m, hid, (d,), False, k.nsm, False)[0])
    k.gemm("resid", m, d, v(h), v(ws16), out, res=x, bias=sc, k0=hid, ldres=d, bn=_plan("resid", m, d, (hid,), False, k.nsm, False)[0])
    return out


# ---------------------------------------------------------------------------------- fp32 TRAINING forward on fp16 (round 3 part 7)
#: (2026-11-17, round 3 part 7) the H100's ``F32_TRAIN_SHAPES``: the fp32 training forward on the inference's fp16-operand build with saves
#: (``tg_fused_h16`` at D 64 / 128 / 256 also writing xn on the TF32 grid, mean and rstd), the step's fp16 weight forms and scales made by
#: ``tg_prep16`` in ONE launch beside tg_prep's TF32 copies; the backward is the generic TF32 recompute backward, unchanged (the H100's
#: structure). From TRAIN_H16_MIN_ROWS rows: below, the step measured even or slower (graph, us, TF32 -> fp16: D 128 16 384 rows 104.4 ->
#: 104.5, D 256 16 384 227.3 -> 229.4; D 256 147 456 rows 1594 -> 1537): tg_prep16's scale reduction costs ~4 us over tg_prep, the fused
#: forward saves ~8 us there. The small-M fp16 chains with saves (D 384 / 768) were built and measured slower (fp32 [1, 128, 384] 51.2 ->
#: 53.2, [1, 128, 768] 71.7 -> 79.8, [1, 512, 768] 92.1 -> 96.3: the forward saves 3.5 - 6 us, tg_prep16 costs 5 - 12 us more than
#: tg_prep) and removed. TRAIN_H16 = False: the TF32 forward (A/B).
TRAIN_H16 = True
TRAIN_H16_MIN_ROWS = 65536


def _train_h16(d: int, hid: int, m: int) -> bool:
    # (S90) from D 256 the TF32 forward keeps a, b (_AB_F32_MIN_D): no fp16 forward there
    return TRAIN_H16 and _fused_h16(d, hid, m) and m >= TRAIN_H16_MIN_ROWS and d < _AB_F32_MIN_D


_BARS: dict[int, torch.Tensor] = {}


def _grid_bar(device: torch.device) -> torch.Tensor:
    """``tg_prep16``'s ticket counter (zeroed once per device; the last block of every launch puts it back to 0)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bar = _BARS.get(idx)
    if bar is None:
        bar = _BARS[idx] = torch.zeros((2,), dtype=torch.int32, device=device)
    return bar


def _prep16_fake(wa, wb, ws, gamma, beta):
    h, d = wa.shape
    e = lambda *s, dt=torch.float32: wa.new_empty(s, dtype=dt)  # noqa: E731
    f16 = torch.float16
    return (e(h, d), e(h, d), e(h, d), e(d, 2 * h), e(h, d, dt=f16), e(h, d, dt=f16), e(d, h, dt=f16), e(4))


@opaque(fake=_prep16_fake, name="transition_generic_prep16_sm100a")
def _prep16(
    wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(wa, wb on the TF32 grid, Ws^T, [Wa; Wb]^T on the TF32 grid, Wa16, Wb16, Ws16, sc) in one ``tg_prep16`` launch (``_prep`` +
    ``_h16_forms``). The fp16 forward needs no TF32 Ws."""
    if _is_fake(wa, wb, ws):
        return _prep16_fake(wa, wb, ws, gamma, beta)
    h, d = wa.shape
    src = wa.dtype
    wa, wb, ws = (w.to(src).contiguous() for w in (wa, wb, ws))
    e = lambda *s, dt=torch.float32: torch.empty(s, dtype=dt, device=wa.device)  # noqa: E731
    oa, ob, ost, oabt = e(h, d), e(h, d), e(h, d), e(d, 2 * h)
    a16, b16, s16 = e(h, d, dt=torch.float16), e(h, d, dt=torch.float16), e(d, h, dt=torch.float16)
    sc = e(4)
    k = _K(oa)
    nt = 2 * (h // 32) * (d // 32)  # tg_prep's grid: one block per 32 x 32 tile
    part = e(nt * 4)
    bar = _grid_bar(wa.device)
    args = (wa, wb, ws, gamma, beta, oa, ob, ost, oabt, a16, b16, s16, sc, part, bar, h, d, int(src is torch.float32))
    k.rows("tg_prep16", nt, args)
    return oa, ob, ost, oabt, a16, b16, s16, sc


def _fwd_h16_train_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps):
    m, d = x.shape
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty_like(x), torch.empty((m,), **f32), torch.empty((m,), **f32)


@opaque(fake=_fwd_h16_train_fake, name="transition_generic_fwd_h16_train_sm100a")
def _fwd_h16_train(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa16: torch.Tensor,
    wb16: torch.Tensor,
    ws16: torch.Tensor,
    sc: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fp32 training forward on the fp16 tensor cores, one ``tg_fused_h16`` launch with saves: (out, xn fp32 on the TF32 grid, mean,
    rstd) as ``_fwd_launch`` with save (the backward's GATE recomputes a, b from xn)."""
    if _is_fake(x, wa16):
        return _fwd_h16_train_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps)
    k = _K(x)
    m, d = x.shape
    hid = wa16.shape[0]
    f32 = dict(dtype=torch.float32, device=x.device)
    out, xn = torch.empty_like(x), torch.empty_like(x)
    mean, rstd = torch.empty((m,), **f32), torch.empty((m,), **f32)
    t = k.ext.tmap
    v = lambda w: w.view(torch.bfloat16)  # noqa: E731
    k.ext.launch_kernel(
        f"generic_f32_fusedh{d}s",
        "tg_fused_h16_sm100",
        max(1, min(k.nsm, -(-m // 128))),
        448,
        1,
        [
            t(x, d, m, 32, 128),
            t(v(wa16), d, hid, 64, 64),
            t(v(wb16), d, hid, 64, 64),
            t(v(ws16), hid, d, 64, d),
            t(out, d, m, 32, 32),
            x,
            gamma,
            beta,
            sc,
            float(eps),
            m,
            hid,
            xn,
            mean,
            rstd,
            1,
        ],
    )
    return out, xn, mean, rstd


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A weight-gradient GEMM with an fp32 result (cuBLAS; bf16 operands accumulate in fp32 and are not rounded through bf16)."""
    if a.dtype is torch.float32:
        return torch.mm(a, b)
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


#: (round 3, 2026-11-17 night) the fp32 (TF32) weight-gradient GEMMs on cuBLAS over many rows: the rows (the GEMM's K) split into
#: WGRAD_SPLIT slabs, one torch.bmm of the slab products, then their sum over the slabs (a fixed reduction: deterministic). cuBLAS'
#: TF32 kernels (cutlass3x 256 x 256 two-SM tiles) do not split K: dWab = dab^T xn [1024, 256] is 4 output tiles over 147 456 rows.
#: Measured (graph, L2 flushed, us, mm -> 16 slabs; error vs fp64 equal, 3e-4): [147 456, 256] n 2 dWab 290.8 -> 135.2, dWs 98.4 -> 94.2;
#: [262 144, 256] 370.7 -> 215.6, 159.6 -> 143.4; [65 536, 256] 90.1 -> 77.8, 61.4 -> 55.3; [65 536, 128] 51.2 -> 47.1, 38.9 -> 36.8
#: (`b200cfg/skp.py`). Below WGRAD_SPLIT_MIN_ROWS (or a row count the slabs do not divide) the single mm.
WGRAD_SPLIT = 16
WGRAD_SPLIT_MIN_ROWS = 65536
#: 8 slabs from WGRAD_SPLIT_SMALL_ROWS up to WGRAD_SPLIT_MIN_ROWS ([16 384, 256]: dWs 30.7 -> 26.6, dWab 45.0 -> 36.9; [16 384, 512]: 49.2 -> 43.0,
#: 67.6 -> 67.7)
WGRAD_SPLIT_SMALL_ROWS = 16384


def _wgrad_slabs(m: int, f32: bool) -> int:
    """Slabs of the split cuBLAS weight gradients at m rows (0: one torch.mm each)."""
    if not f32 or m < WGRAD_SPLIT_SMALL_ROWS:
        return 0
    s = WGRAD_SPLIT if m >= WGRAD_SPLIT_MIN_ROWS else 8
    return 0 if m % s else s


def _slab_products(a: torch.Tensor, b: torch.Tensor, s: int) -> torch.Tensor:
    """[s, P, Q]: slab j = a[rows j]^T b[rows j] for row-major a [M, P], b [M, Q] split into s row slabs (one torch.bmm)."""
    m = a.shape[0]
    return torch.bmm(a.view(s, m // s, -1).transpose(1, 2), b.view(s, m // s, -1))


def _ln_bwd_blocks(m: int, rows_per_warp: int, nsm: int) -> int:
    """LayerNorm-backward blocks: one row group per warp (8 warps) up to four blocks per SM; a few row tiles then spread over ~M / 8
    blocks instead of 64-row blocks each walking eight rows in sequence (with the slot-parallel block reduction, [768, 384] fp32 alone:
    15.6 -> 3.9 us; [90 000, 256] unchanged)."""
    return max(1, min(-(-m // (8 * rows_per_warp)), nsm * 4))


#: (round 3, training) the backward of the small-M chain cells in ONE launch (``generic/tg_bwd_small.cu``: dy split, GATE_AB, the d_xn split-K
#: tiles beside the weight-gradient tiles, the LayerNorm backward and the dgamma | dbeta sum as phases of one persistent grid with software
#: grid barriers) instead of 5-6 launches of a few us of fixed latency each. False: the generic launches (A/B).
SMALL_BWD = True
#: d_xn tile widths / dW tile widths the one-launch backward is built at
_SB_BNPS = (32, 64, 128, 256)
_SB_NTS = (64, 128)
#: the plan's per-tile fixed costs (us), the dW tile's store rate (bytes / us) and one load round of the LayerNorm phase (us)
_SB_DW_FIX, _SB_P_FIX, _SB_STORE, _SB_ROUND = 0.8, 0.4, 1.0e5, 0.7
#: analysis builds only (-DPROBE=1): the per-CTA phase stamps of the last call land in SB_STAMPS ([G, 10] int64, %globaltimer ns)
SB_PROBE = False
SB_STAMPS = None
#: extra nvcc flags of the one-launch backward (A/B builds)
SB_FLAGS: tuple[str, ...] = ()
#: fp32: dy as it is in the one-launch backward (the MMAs truncate it to TF32; no dy_lo term in dWs, no split phase): YELLOW 1.36-1.59 x tol on
#: rows 217 / 219 float32 (the split: 1.00-1.08); False: the TF32 split (A/B)
SB_DYRAW = True


def _small_bwd(m: int, d: int, hid: int, f32: bool, ab: torch.Tensor) -> bool:
    """Whether ``_bwd_launch`` runs ``tg_bwd_small``: a small-M chain cell whose forward kept [a | b]."""
    return (
        SMALL_BWD
        and fused_small_sm100a.chain_takes(m, d, hid, f32)
        and ab.shape[0] == m
        and ab.shape[1] == 2 * hid
        and d % 128 == 0
        and hid % 128 == 0
    )


def _small_plan(m: int, d: int, hid: int, f32: bool, nsm: int) -> tuple[int, int, int, int, int]:
    """(gate tile width, d_xn tile width, d_xn K slices, dW tile width, d_xn CTAs) of ``tg_bwd_small`` on ``nsm`` CTAs. Phase 2 splits the
    grid between the d_xn tiles and the weight-gradient tiles: the split, the K slices and the dW width minimise the longer of the two
    (waves x (per-tile operand bytes at ``_INTAKE`` + a fixed cost + the dW tile's fp32 store)) plus the extra load rounds the d_xn slabs
    cost the LayerNorm phase beyond its first four."""
    esz, bke = (4, 32) if f32 else (2, 64)
    lo = 2 if f32 and not SB_DYRAW else 1  # fp32 with the dy split: the dy_lo operand of the dWs tiles too
    bng = _plan("gate_ab", m, hid, (d,), f32, nsm, False)[0]
    mt = m // 128
    nkb = 2 * hid // bke
    best = None
    for ntw in _SB_NTS if f32 else (*_SB_NTS, 256):
        dwt = (d // 128) * (hid // ntw) + (2 * hid // 128) * (d // ntw)
        dw_tile = (128 + ntw) * m * esz * lo / _INTAKE + _SB_DW_FIX + 128 * ntw * 4 / _SB_STORE
        for bnp in _SB_BNPS if f32 else _SB_BNPS[1:]:  # (MN-major B: whole 128-B N blocks, 64 bf16 columns)
            if d % bnp:
                continue
            for ks in _SPLITS:
                if nkb % ks or nkb // ks < 2:
                    continue
                pt = mt * (d // bnp) * ks
                p_tile = (128 + bnp) * (2 * hid // ks) * esz / _INTAKE + _SB_P_FIX
                ln = _SB_ROUND * -(-max(ks - 4, 0) // 3)
                for npc in range(1, min(pt, nsm - 1) + 1):
                    cost = max(-(-pt // npc) * p_tile, -(-dwt // (nsm - npc)) * dw_tile) + ln
                    if best is None or cost < best[0] - 1e-9:
                        best = (cost, ntw, ks, npc, bnp)
    assert best is not None
    return bng, best[4], best[2], best[1], best[3]


#: (round 3, training) the gate's dh = dy Ws split over K in ``tg_bwd_small`` (fp32 slabs, then the gate elementwise over all CTAs): the
#: per-CTA operand intake of the unsplit GATE_AB tile (128 rows x the whole D of dy per CTA) bounded that phase. False: GATE_AB tiles (A/B)
SB_GSPLIT = True
#: A/B override of the phase-2 plan: None, or (d_xn tile width, d_xn K slices, dW tile width, d_xn CTAs)
SB_PLAN_FORCE = None
#: (round 3, training part 3) measured phase-2 plans that beat ``_small_plan`` on 148 SMs (``b200cfg/sbsweep.py``: the probe build's
#: last-CTA end, us, auto -> tuned): bf16 [512 / 640 / 768, 768] n 2 30.3 -> 29.4, 36.8 -> 32.7, 34.5 -> 34.1 (the model's K split
#: ignores the slabs' write + LayerNorm-phase read traffic; the 256-wide dW tiles take fewer operand bytes per output)
SB_PLAN_TUNED: dict[tuple[int, int, int, bool], tuple[int, int, int, int]] = {
    (512, 768, 1536, False): (256, 3, 256, 36),
    (640, 768, 1536, False): (256, 2, 256, 30),
    (768, 768, 1536, False): (256, 2, 256, 36),
}
#: measured gate plans (tile width, K slices; 1: the unsplit GATE_AB tiles; ``b200cfg/sbsweep.py`` GATES=, last-CTA end, us, auto ->
#: tuned): bf16 [256 / 384 / 512 / 640 / 768, 768] n 2 22.1 -> 21.3, 24.3 -> 22.2, 26.8 -> 23.8, 28.5 -> 26.8, 29.1 -> 27.1 (the split's
#: elementwise pass over the dh slabs costs more than the unsplit tiles' per-CTA intake from 384 rows)
SB_GATE_TUNED: dict[tuple[int, int, int, bool], tuple[int, int]] = {
    (128, 768, 1536, True): (64, 6),  # fp32 [128, 768]: 22.5 -> 21.2
    (256, 768, 1536, False): (64, 3),
    (384, 768, 1536, False): (64, 1),
    (512, 768, 1536, False): (64, 1),
    (640, 768, 1536, False): (64, 1),
    (768, 768, 1536, False): (64, 1),
}


def _gate_split(m: int, d: int, hid: int, f32: bool, nsm: int) -> tuple[int, int]:
    """(tile width, K slices) of the split gate products: the least per-CTA intake (waves x (128 + width) x K slice) plus the slabs' cost."""
    esz, bke = (4, 32) if f32 else (2, 64)
    mt, nkb = m // 128, d // bke
    best = None
    for bn in (32, 64, 128) if f32 else (64, 128):
        if hid % bn:
            continue
        tiles = mt * (hid // bn)
        for ks in _SPLITS:
            if nkb % ks or (ks > 1 and (nkb // ks < 2 or tiles * ks > nsm)):
                continue
            cost = -(-tiles * ks // nsm) * ((128 + bn) * (d // ks) * esz / _INTAKE + _SB_P_FIX) + 0.05 * ks
            if best is None or cost < best[0] - 1e-9:
                best = (cost, bn, ks)
    assert best is not None
    return best[1], best[2]


def _small_spec(f32: bool, d: int, hid: int, bng: int, bnp: int, ntw: int, probe: bool = False, flags: tuple[str, ...] = ()):
    """(name, source, flags) of one ``tg_bwd_small`` cubin."""
    name = f"bs{d}_{hid}_{bng}_{bnp}_{ntw}{'p' if probe else ''}{''.join(flags).replace('-D', '_').replace('=', '')}"
    dims = (f"-DF32={int(f32)}", f"-DDIM={d}", f"-DHID={hid}", f"-DBNG={bng}", f"-DBNP={bnp}", f"-DNTW={ntw}", f"-DPROBE={int(probe)}")
    return name, "generic/tg_bwd_small.cu", (*dims, *flags)


def _small_config(m: int, d: int, hid: int, f32: bool, nsm: int) -> tuple[int, int, int, int, int, int, tuple[str, ...]]:
    """(gate width, d_xn width, d_xn K slices, dW width, d_xn CTAs, gate K slices (0: GATE_AB tiles), build flags) of one call."""
    bng, bnp, ksp, ntw, npc = _small_plan(m, d, hid, f32, nsm)
    if SB_PLAN_FORCE is not None:  # (A/B) (d_xn width, K slices, dW width, d_xn CTAs)
        bnp, ksp, ntw, npc = SB_PLAN_FORCE
    elif nsm == 148 and (m, d, hid, f32) in SB_PLAN_TUNED:
        bnp, ksp, ntw, npc = SB_PLAN_TUNED[(m, d, hid, f32)]
    ksg = 0
    if SB_GSPLIT:
        bs, ks = _gate_split(m, d, hid, f32, nsm)
        if nsm == 148 and (m, d, hid, f32) in SB_GATE_TUNED:
            bs, ks = SB_GATE_TUNED[(m, d, hid, f32)]
            bng = bs
        if ks > 1:  # (unsplit: the GATE_AB tiles, which need no elementwise phase)
            bng, ksg = bs, ks
    flags = (("-DDYRAW=0",) if f32 and not SB_DYRAW else ()) + (() if ksg else ("-DGSPLIT=0",))
    if -(-m // nsm) > 2 or d % 768:  # (several warps per LayerNorm row: D 768 at <= 2 rows per CTA only; measured slower at D 384
        # [384, 384] fp32 +0.7 us and, with the build serving it, at D 768 L 640 / 768 +2 us)
        flags += ("-DLNW=0",)
    return bng, bnp, ksp, ntw, npc, ksg, flags


def small_specs(f32: bool, nsm: int = 148) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Every ``tg_bwd_small`` cubin the small-M chain cells of one activation dtype take (the sm100 prebuild plan)."""
    out = {}
    for (d, n, cf), tiles in fused_small_sm100a.CELLS.items():
        if cf != f32:
            continue
        for t in range(1, tiles + 1):
            if _small_cell(128 * t, d, n * d, f32, nsm):
                bng, bnp, _ks, ntw, _npc, _kg, flags = _small_config(128 * t, d, n * d, f32, nsm)
                s = _small_spec(f32, d, n * d, bng, bnp, ntw, False, flags)
                out[s[0]] = s
    return tuple(out.values())


@functools.lru_cache(maxsize=16)
def _load_small(f32: bool, d: int, hid: int, bng: int, bnp: int, ntw: int, probe: bool = False, flags: tuple[str, ...] = ()):
    ext = _ext()
    tag = "f32" if f32 else "bf16"
    spec = _small_spec(f32, d, hid, bng, bnp, ntw, probe, flags)
    for n, path in build_cubins(f"generic_{tag}", (spec,)).items():
        ext.load_cubin(f"generic_{tag}_{n}", path)
    return ext, f"generic_{tag}_{spec[0]}"


_SBARS: dict[int, torch.Tensor] = {}


def _small_bar(device: torch.device) -> torch.Tensor:
    """``tg_bwd_small``'s barrier counters (zeroed once per device; the last CTA of every launch puts them back to 0)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bar = _SBARS.get(idx)
    if bar is None:
        bar = _SBARS[idx] = torch.zeros((8,), dtype=torch.int32, device=device)
    return bar


#: (round 3, training) the small cells' weight forms made by the LayerNorm launch of the forward chain (``fused_small_sm100a.chain(cast=)``)
#: instead of a ``tg_prep`` launch of their own, where the parameters are fp32 masters. False: ``_prep_small`` (A/B)
SB_LN_CAST = True


def _fwd_small_train_fake(x, gamma, beta, wa, wb, ws, eps):
    m, d = x.shape
    h = wa.shape[0]
    f = dict(dtype=torch.float32, device=x.device)
    e = lambda *s: x.new_empty(s)  # noqa: E731
    return torch.empty_like(x), torch.empty_like(x), torch.empty((m,), **f), torch.empty((m,), **f), e(m, 2 * h), e(2 * h, d), e(d, h)


@opaque(fake=_fwd_small_train_fake, name="transition_generic_fwd_small_train_sm100a")
def _fwd_small_train(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The small cells' training forward from fp32 masters: (out, xn, mean, rstd, [a | b], [Wa; Wb] [2H, D], Ws [D, H]), the forms (fp32 on
    the TF32 grid or bf16, x's dtype) written by the chain's LayerNorm launch."""
    if _is_fake(x, wa):
        return _fwd_small_train_fake(x, gamma, beta, wa, wb, ws, eps)
    h, d = wa.shape
    wab = torch.empty((2 * h, d), dtype=x.dtype, device=x.device)
    ws_ = torch.empty((d, h), dtype=x.dtype, device=x.device)
    out, xn, mean, rstd, ab = fused_small_sm100a.chain(x, gamma, beta, wab[:h], wab[h:], ws_, eps, True, cast=(wa, wb, ws))
    return out, xn, mean, rstd, ab, wab, ws_


def _small_cell(m: int, d: int, hid: int, f32: bool, nsm: int) -> bool:
    """Whether a training call's backward will be ``tg_bwd_small`` (the forward then makes ``_prep_small``'s forms: no transposes)."""
    return SMALL_BWD and fused_small_sm100a.chain_takes(m, d, hid, f32) and _ab(m, hid, nsm, d, f32) and d % 128 == 0 and hid % 128 == 0


def _prep_small_fake(wa, wb, ws, f32):
    h, d = wa.shape
    dtype = torch.float32 if f32 else torch.bfloat16
    return wa.new_empty((2 * h, d), dtype=dtype), wa.new_empty((d, h), dtype=dtype)


@opaque(fake=_prep_small_fake, name="transition_generic_prep_small_sm100a")
def _prep_small(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, f32: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """([Wa; Wb] [2H, D], Ws [D, H]) in fp32 on the TF32 grid (``f32``) or bf16, one ``tg_prep`` launch without the transposes:
    ``tg_bwd_small`` reads the weights MN-major."""
    if _is_fake(wa, wb, ws):
        return _prep_small_fake(wa, wb, ws, f32)
    dtype = torch.float32 if f32 else torch.bfloat16
    h, d = wa.shape
    src = wa.dtype
    wa, wb, ws = (w.to(src).contiguous() for w in (wa, wb, ws))
    wab = torch.empty((2 * h, d), dtype=dtype, device=wa.device)
    os_ = torch.empty((d, h), dtype=dtype, device=wa.device)
    k = _K(os_)
    one = os_.new_empty((1, 1))
    k.rows("tg_prep", min(-(-h * d // 256), k.nsm * 8), (wa, wb, ws, wab[:h], wab[h:], os_, one, one, h, d, int(src is torch.float32), 0))
    return wab, os_


def _bwd_small(k, dy, x, xn, mean, rstd, gamma, ws_, wab, ab):
    """The one-launch backward (``tg_bwd_small``): (dx, [dgamma | dbeta], [dWa; dWb], dWs) as ``_bwd_launch``."""
    m, d = x.shape
    hid = ws_.shape[1]
    f32 = k.f32
    bng, bnp, ksp, ntw, npc, ksg, flags = _small_config(m, d, hid, f32, k.nsm)
    raw = f32 and SB_DYRAW
    ext, name = _load_small(f32, d, hid, bng, bnp, ntw, SB_PROBE, SB_FLAGS + flags)
    g = k.nsm
    fl = dict(dtype=torch.float32, device=x.device)
    if f32 and not raw:
        dyg, dylo = torch.empty_like(dy), torch.empty_like(dy)
    else:
        dyg = dylo = dy
    h = torch.empty((m, hid), dtype=x.dtype, device=x.device)
    dab = torch.empty((m, 2 * hid), dtype=x.dtype, device=x.device)
    mp = m if ksp > 1 else 0
    dxn = torch.empty((ksp * m, d), **fl)
    dws, dwab, dgb = torch.empty((d, hid), **fl), torch.empty((2 * hid, d), **fl), torch.empty((2 * d,), **fl)
    part = torch.empty((g * 2 * d + (g * 80 if SB_PROBE else 0),), **fl)
    gp = torch.empty((max(ksg, 1) * m, hid), **fl) if ksg else part
    dx = torch.empty_like(x)
    t = ext.tmap
    bke = k.bke
    ocg = 32 if f32 else 64
    e, sw = 128 // x.element_size(), int(f32)
    tw = lambda a, inner: t(a, inner, m, e, 64, -1, sw)  # noqa: E731
    args = [
        t(dyg, d, m, bke, 128),
        t(ws_, hid, d, bke, bke, -1, sw),  # Ws [D][H]: the gate B operand MN-major (no Ws^T)
        t(ab, 2 * hid, m, ocg, 32),
        t(h, hid, m, ocg, 32),
        t(dab, 2 * hid, m, ocg, 32),
        t(dab, 2 * hid, m, bke, 128),
        t(wab, d, 2 * hid, bke, bke, -1, sw),  # [Wa; Wb] [2H][D]: the d_xn B operand MN-major
        t(dxn, d, ksp * m, 32, 32),
        tw(dyg, d),
        tw(h, hid),
        tw(dab, 2 * hid),
        tw(xn, d),
        tw(dylo, d),
        dy,
        dyg,
        dylo,
        ab,
        x,
        mean,
        rstd,
        gamma,
        dx,
        dxn,
        dws,
        dwab,
        dgb,
        part,
        _small_bar(x.device),
        h,
        dab,
        m,
        ksp,
        mp,
        npc,
        gp,
        max(ksg, 1),
    ]
    ext.launch_kernel(name, "tg_bwd_small_sm100", g, 320, 1, args, PDL)
    if SB_PROBE:
        global SB_STAMPS
        SB_STAMPS = part[g * 2 * d :].view(torch.int64).view(g, 40)
    return dx, dgb, dwab, dws


def _bwd_launch_fake(dy, x, xn, mean, rstd, gamma, wa, wb, wst, wabt, ab):
    """Output structure of ``_bwd_launch``: (dx, [dgamma | dbeta], [dWa; dWb], dWs), the weight gradients fp32."""
    h, d = wa.shape
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty((2 * d,), **f32), torch.empty((2 * h, d), **f32), torch.empty((d, h), **f32)


@opaque(fake=_bwd_launch_fake, name="transition_generic_bwd_sm100a")
def _bwd_launch(
    dy: torch.Tensor,
    x: torch.Tensor,
    xn: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    wst: torch.Tensor,
    wabt: torch.Tensor,
    ab: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, [dgamma | dbeta], [dWa; dWb] [2H, D], dWs [D, H]); ``dx`` carries the residual branch, the weight gradients are fp32. ``ab``: the
    forward's [a | b] where it kept them (GATE_AB), else a placeholder (GATE recomputes a, b from xn)."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, mean, rstd, gamma, wa, wb, wst, wabt, ab)
    k = _K(x)
    m, d = x.shape
    hid = wa.shape[0]
    if _small_bwd(m, d, hid, k.f32, ab):
        return _bwd_small(k, dy, x, xn, mean, rstd, gamma, wst, wabt, ab)
    dyg, dylo = dy, None
    if k.f32:  # dy as the GATE GEMM's A operand on the TF32 grid (the MMA would truncate it); the residual branch keeps dy itself
        dyg = torch.empty_like(dy)
        grid = max(1, min(-(-dy.numel() // 1024), k.nsm * 8))
        if _dw_plan(m, d, hid, k.nsm, True) is not None:  # and its TF32 residual for tg_dw's dWs (-DSPL)
            dylo = torch.empty_like(dy)
            k.rows("tg_split_tf32", grid, (dy, dyg, dylo, dy.numel() // 4))
        else:
            k.rows("tg_round_tf32", grid, (dy, dyg, dy.numel() // 4))
    h = torch.empty((m, hid), dtype=x.dtype, device=x.device)
    dab = torch.empty((m, 2 * hid), dtype=x.dtype, device=x.device)
    if ab.shape[0] == m and ab.shape[1] == 2 * hid:
        k.gemm(
            "gate_ab",
            m,
            hid,
            dyg,
            wst,
            h,
            o1=dab,
            res=ab,
            ldres=2 * hid,
            k0=d,
            hoff=hid,
            bn=_plan("gate_ab", m, hid, (d,), k.f32, k.nsm, False)[0],
        )
    else:
        k.gemm(
            "gate",
            m,
            hid,
            dyg,
            wst,
            h,
            a1=xn,
            b1=wa,
            b2=wb,
            o1=dab,
            k0=d,
            k1=d,
            hoff=hid,
            bn=_plan("gate", m, hid, (d, d), k.f32, k.nsm, False)[0],
        )
    plan = _dw_plan(m, d, hid, k.nsm, k.f32)
    f32 = dict(dtype=torch.float32, device=x.device)
    nsl = _wgrad_slabs(m, k.f32) if plan is None else 0
    if nsl:  # cuBLAS over row slabs (WGRAD_SPLIT); tg_wsum adds them in slab order at the end, with the dgamma | dbeta sum
        pws, pab = _slab_products(dy, h, nsl), _slab_products(dab, xn, nsl)  # [S, D, H], [S, 2H, D]
        dws, dwab = torch.empty((d, hid), **f32), torch.empty((2 * hid, d), **f32)
        nt, ns, wsp = 0, 1, dy
    elif plan is None:  # cuBLAS (D not a multiple of 128, or DW_ROUTE = "cublas")
        dws, dwab = _mm_f32(dy.t(), h), _mm_f32(dab.t(), xn)  # [D, H], [2H, D]: dWa then dWb
        nt, ns, wsp = 0, 1, dy
    else:
        nt, ns = plan
        dws, dwab = torch.empty((d, hid), **f32), torch.empty((2 * hid, d), **f32)
        wsp = torch.empty((ns * 3 * d * hid,), **f32) if ns > 1 else dws
        if ns > 1:  # right after the gate (its slabs are added by tg_wsum at the end)
            _dw_launch(k, dyg, h, dab, xn, dws, dwab, wsp, nt, ns, dylo=dylo)
    bn, ks = _plan("plain", m, d, (2 * hid,), k.f32, k.nsm, True)
    if DXN_ROUTE == "cublas" or (DXN_ROUTE is None and ks == 1 and _dxn_cublas(d, k.f32)):  # (a few row tiles keep the split-K PLAIN)
        dxn, ks, mp = _mm_f32(dab, wabt.t()), 1, 0
    else:
        if DXN_ROUTE == "nosplit":
            ks = 1
        mp = -(-m // 128) * 128 if ks > 1 else 0  # split K: fp32 partial slabs of mp rows, added in order by the LayerNorm backward
        dxn = torch.empty((ks * mp if ks > 1 else m, d), **f32)
        k.gemm("plain", m, d, dab, wabt, dxn, k0=2 * hid, bn=bn, ks=ks, mstride=mp)
    sfx, rpw = _ln_variant(d, k.f32)
    nb = _ln_bwd_blocks(m, rpw, k.nsm)
    part = torch.empty((nb, 2 * d), **f32)
    dx = torch.empty_like(x)
    k.rows(f"tg_ln_bwd{'s' if ks > 1 else ''}_{sfx}", nb, (dxn, x, dy, mean, rstd, gamma, dx, part, m, d, ks, mp * d))
    dgb = torch.empty((2 * d,), **f32)
    if plan is not None and ns == 1:  # last: the weight gradients with the dgamma | dbeta sum folded in (no tg_colsum launch)
        _dw_launch(k, dyg, h, dab, xn, dws, dwab, wsp, nt, 1, part, dgb, nb, dylo=dylo)
    elif ns > 1:  # the dW slabs (slab order) and the dgamma | dbeta partials in one launch
        _wsum(k, wsp, dws, dwab, ns, part, dgb, nb)
    elif nsl:  # the cuBLAS slab products (slab order) and the dgamma | dbeta partials in one launch (no tg_colsum)
        _wsum(k, pws, dws, dwab, nsl, part, dgb, nb, pab)
    else:
        k.rows("tg_colsum", -(-2 * d // 32), (part, dgb, nb, 2 * d), block=1024)  # 32 warps over the partial rows
    return dx, dgb, dwab, dws  # the caller splits the packed ones (a slice made here is copied: outputs of a custom op must not alias)


def _dw_launch(k, dy, h, dab, xn, dws, dwab, ws, nt: int, ns: int, part=None, dgb=None, nb: int = 0, dylo=None) -> None:
    """One ``tg_dw`` launch: dWs = dy^T h and [dWa; dWb] = [dA | dB]^T xn (fp32) into ``dws`` / ``dwab`` (``ns`` = 1) or the slabs ``ws``;
    with ``part`` (``ns`` = 1) also dgb = the column sums of the [nb, 2 D] dgamma | dbeta partials."""
    m, d = xn.shape
    hid = h.shape[1]
    e, sw = 128 // dy.element_size(), int(dy.dtype is torch.float32)  # fp32: the 32-B-atom swizzle of MN-major tf32 operands
    t = lambda a, inner: k.ext.tmap(a, inner, m, e, 64, -1, sw)  # noqa: E731
    args = [t(dy, d), t(h, hid), t(dab, 2 * hid), t(xn, d), t(dylo if dylo is not None else dy, d), ws, dws, dwab, m, d, hid, ns]
    args += [part if part is not None else ws, dgb if dgb is not None else ws, nb]
    k.ext.launch_kernel(f"generic_{k.tag}_dw{nt}", "tg_dw_sm100", _dw_tiles(d, hid, nt) * ns, 256, 1, args, True)


def _wsum(k, ws, dws, dwab, ns: int, part=None, dgb=None, nb: int = 0, ws2=None) -> None:
    """``tg_wsum``: the ``ns`` dW slabs added in slab order into dws / dwab, and (``part``) the dgamma | dbeta column sums. ``ws``: tg_dw's
    [ns][dWs | dWab] slabs, or with ``ws2`` the [ns][D][H] dWs slabs and ``ws2`` the [ns][2H][D] dWab ones."""
    d, hid = dws.shape
    n1, n4 = d * hid // 4, 3 * d * hid // 4
    ws2, sa, sb = (ws[d * hid :], n4, n4) if ws2 is None else (ws2, n1, n4 - n1)  # in 16-B vectors
    nbw = max(1, min(-(-n4 // 256), k.nsm * 4))
    nc = 2 * d if part is not None else 0
    k.rows(
        "tg_wsum",
        nbw + -(-nc // 32),
        (ws, dws, dwab, ns, n1, n4, part if part is not None else ws, dgb if dgb is not None else ws, nb, nc, nbw, ws2, sa, sb),
    )


def weight_grads(
    dy: torch.Tensor,
    h: torch.Tensor,
    dab: torch.Tensor,
    xn: torch.Tensor,
    part: torch.Tensor | None = None,
    dgb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(dWs [D, H], [dWa; dWb] [2H, D]) fp32 on ``tg_dw`` (+ ``tg_wsum`` when split over the rows) for another sm_100a backward (the wide
    builds' cuBLAS dW), or None where ``_dw_plan`` keeps cuBLAS. bf16 / fp32 (on the TF32 grid) row-major operands. With ``part`` / ``dgb``
    (round 3 part 8) the same launch also writes dgb = the column sums of the [nb, 2 D] dgamma | dbeta partials ``part`` (fixed order)."""
    m, d = xn.shape
    hid = h.shape[1]
    k = _K(xn)
    plan = _dw_plan(m, d, hid, k.nsm, k.f32)
    if plan is None:
        return None
    nt, ns = plan
    f32 = dict(dtype=torch.float32, device=xn.device)
    dws, dwab = torch.empty((d, hid), **f32), torch.empty((2 * hid, d), **f32)
    ws = torch.empty((ns * 3 * d * hid,), **f32) if ns > 1 else dws
    nb = part.shape[0] if part is not None else 0
    if ns > 1:
        _dw_launch(k, dy, h, dab, xn, dws, dwab, ws, nt, ns)
        _wsum(k, ws, dws, dwab, ns, part, dgb, nb)
    else:
        _dw_launch(k, dy, h, dab, xn, dws, dwab, ws, nt, 1, part, dgb, nb)
    return dws, dwab


class _GenericTransitionSM100A(torch.autograd.Function):
    """``y = transition(x) + x``; the forward saves x, xn and the statistics. The parameters keep their dtype (bf16, fp32 masters): the
    kernels get copies in the activation dtype made here, outside autograd, and the parameters get fp32 gradients in their own dtype."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        if x.dtype is torch.float32 and _train_h16(flat.shape[1], wa.shape[0], flat.shape[0]):
            # (round 3 part 7) the forward on the fp16 tensor cores; the backward's TF32 copies made in the same launch as the fp16 forms
            wa_, wb_, wst, wabt, wa16, wb16, ws16, sc = _prep16(wa, wb, ws, gf, bf)
            out, xn, mean, rstd = _fwd_h16_train(flat, gf, bf, wa16, wb16, ws16, sc, float(eps))
            ab = flat.new_empty((1, 1))
        else:
            f32 = x.dtype is torch.float32
            hid = wa.shape[0]
            small = _small_cell(flat.shape[0], flat.shape[1], hid, f32, num_sms(x.device))
            if small and SB_LN_CAST and all(w.dtype is torch.float32 and w.is_contiguous() for w in (wa, wb, ws)):
                # (round 3, training) the weight forms from the chain's LayerNorm launch; the one-launch backward reads Ws and [Wa; Wb] as
                # they are (MN-major): no transposed copies, no prep launch
                out, xn, mean, rstd, ab, wab, ws_ = _fwd_small_train(flat, gf, bf, wa, wb, ws, float(eps))
                wa_, wb_, wst, wabt = wab[:hid], wab[hid:], ws_, wab
            else:
                if small:
                    wab, ws_ = _prep_small(wa, wb, ws, f32)
                    wa_, wb_, wst, wabt = wab[:hid], wab[hid:], ws_, wab
                else:
                    wa_, wb_, ws_, wst, wabt = _prep(wa, wb, ws, f32, True)
                out, xn, mean, rstd, ab = _fwd_launch(flat, gf, bf, wa_, wb_, ws_, float(eps), True)
        ctx.save_for_backward(flat, xn, mean, rstd, gf, wa_, wb_, wst, wabt, ab)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, mean, rstd, gf, wa_, wb_, wst, wabt, ab = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgb, dwab, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).to(flat.dtype).contiguous(), flat, xn, mean, rstd, gf, wa_, wb_, wst, wabt, ab
        )
        hid = wa_.shape[0]
        d = flat.shape[1]
        return (dx.reshape(ctx.shape), dgb[:d].to(gdt), dgb[d:].to(bdt), dwab[:hid].to(adt), dwab[hid:].to(bwdt), dws.to(sdt), None)


def transition_generic_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm100a.transition_fused_sm100a``. Call ``available()`` first: this raises rather
    than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the saves, with the weight copies
    made once per parameter version (``_capture.param_forms``)."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        from foldjet_engine.kernels import _capture

        shape = x.shape
        if x.dtype is torch.float32 and _fused_h16(x.shape[-1], wa.shape[0], x.numel() // x.shape[-1]):
            g, b, ww = _capture.param_forms(
                "transition_generic_h16_sm100a",
                (gamma, beta, wa, wb, ws),
                lambda: (gamma.float().contiguous(), beta.float().contiguous(), _h16_forms(gamma, beta, wa, wb, ws)),
                capture_reads_eager=True,
            )
            return _fwd_h16_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps)).reshape(shape)
        if x.dtype is torch.float32 and _generic_h16(x.shape[-1], x.numel() // x.shape[-1]):
            g, b, ww = _capture.param_forms(
                "transition_generic_h16x3_sm100a",
                (gamma, beta, wa, wb, ws),
                lambda: (gamma.float().contiguous(), beta.float().contiguous(), _h16_forms(gamma, beta, wa, wb, ws)),
                capture_reads_eager=True,
            )
            return _fwd_h16x3_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps)).reshape(shape)
        g, b, ww = _capture.param_forms(
            f"transition_generic_sm100a_{x.dtype}",
            (gamma, beta, wa, wb, ws),
            lambda: (gamma.float().contiguous(), beta.float().contiguous(), _prep(wa, wb, ws, x.dtype is torch.float32, False)[:3]),
            capture_reads_eager=True,
        )
        out, *_ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps), False)
        return out.reshape(shape)
    return _GenericTransitionSM100A.apply(x, gamma, beta, wa, wb, ws, eps)
