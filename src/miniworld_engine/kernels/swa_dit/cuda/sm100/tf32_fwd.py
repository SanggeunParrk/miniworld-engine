"""sm_100a (B200) forward of the fused SWA atom DiT block in fp32: every stage on hand-written tcgen05 / TMEM / TMA kernels with TF32
tensor-core MMAs (``tcgen05.mma kind::tf32``, fp32 accumulation), every elementwise step in fp32. No Triton, no PyTorch kernels
besides the cached weight rounding.

    weights     [Wqkv; Wg], Wo, Wab, Wd rounded to TF32 (one launch)     wprep_tf32.cu     (swa_wprep_tf32_sm100; Wmod:
                                                                                         swa_round_tf32_sm100)
    modulation  silu(c) Wmod^T                                         mod_fwd_tf32.cu   (swa_mod_fwd_tf32_sm100)
    qkvg        RMS-adaLN, q | k | v | gate projections, head RMS, RoPE qkvg_fwd_tf32.cu  (swa_qkvg_fwd_tf32_sm100)
    attention   window attention |i - j| <= 64, fp32 softmax, lse      attn_fwd_tf32.cu  (swa_attn_fwd_tf32_sm100)
    ffn         gated out-projection, residual, RMS-adaLN, SwiGLU,      ffn_fwd_tf32.cu   (swa_ffn_fwd_tf32_sm100)
                residual

The same equations as the Triton fp32 path (``triton/forward_fp32.py``) with two differences of precision, both toward fp32: the window
attention runs on TF32 Q / K / V / P (the Triton path rounds them to bf16, as FlashAttention-4 does), and every GEMM is single-pass
TF32 with operands ROUNDED to the nearest TF32 (the kernels round their activation operands with ``cvt.rna``; ``wprep_tf32.cu``
rounds the weights once per weight version -- once per replay inside a CUDA graph unless the weights are declared static,
``kernels._capture``) -- the tensor core alone would truncate the low 13 mantissa bits, a bias toward zero that
does not average out over K (that truncation is why the Triton path needs "tf32x3" for its FFN GEMMs).

Saved for the backward (``save``), all fp32: Qh, Kh, Vh head-major [N, 4, S, 32] (after head RMS + RoPE, TF32-valued: exactly the
attention's operands), G [N S, C] (raw gate, pre-sigmoid), O [N S, C] (attention output, pre-gate), lse [N, 4, S] (natural log of the
sum of exp of the scaled scores; 0 on rows without a valid key), q1, X (the TF32-rounded qkvg MMA operand), PQ, PK (raw projections,
pre head RMS), Att (raw out-projection, pre-gate_a), Y (fp32, unrounded), FF (raw down-projection) [N S, C] -- the meanings of the
bf16 sm_100a saves (``sm100.block_fwd``).

Served (:func:`supported_tf32`): B200, fp32, C = 128 with 4 heads of 32, SwiGLU hidden 256, half window 64 (not global), fp32 eps, the
atom count a multiple of 128 (else the Triton fp32 path serves the call). The cubins are built on first use by the newest nvcc that knows
sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached under ``settings.engine_jit_root`` (``swa_dit_sm100_tf32/``);
the launches go through the CUDA driver on torch's current stream (programmatic dependent launch between the stages), CUDA-graph
capturable. Page: ``docs/gpus/b200/swa_atom_dit/swa_atom_dit.md`` ("fp32 (TF32) path").
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels._compile import device_constant
from miniworld_engine.kernels._tf32 import tf32_grid
from miniworld_engine.kernels.swa_dit.cuda import sm100 as _sm

C, H, D, NHID, HW = _sm.C, _sm.H, _sm.D, _sm.NHID, _sm.HW
EPS = _sm.EPS
PDL = True  # programmatic dependent launch between the forward kernels
#: round 3 part 4 (swa-global, development switch): fp32 global INFERENCE on the F32IO attention build (gattn_fwd32: the fp32
#: q rounded to bf16 in the kernel, O written fp32) -- the q cast and O16.float() gone
F32IO_GLOBAL = True
#: ... up to this many rows N S (B200 2026-11-17, `_tools/ab4.py`, block alone: S5 L1024 67.5 -> 65.4 us, L4096 225 -> 219, embedder
#: L4096 278 -> 275, L6144 352 -> 348; slower above: S5 L6144 410 -> 434 (measurer), S5 L8192 584 -> 621)
F32IO_MAX_ROWS = 20480
#: round 3 part 16 (development switch): the F32IO build for training's global forward too (lse saved for the native backward)
F32IO_TRAIN = __import__("os").environ.get("SWA_F32IO_TRAIN", "1") == "1"
_dir = Path(__file__).parent

#: cubin name -> (source stem, extra nvcc flags)
CUBINS = {
    # forward
    "wprep_tf32": ("wprep_tf32", ()),
    "mod_fwd_tf32": ("mod_fwd_tf32", ()),
    "qkvg_fwd_tf32": ("qkvg_fwd_tf32", ()),
    # round 3 part 7: fp32 inference of one augment -- shift_a / scale_a made inside qkvg (MODF), the modulation kernel beside it (LATE_TRIGGER)
    "qkvg_fwd_tf32_modf": ("qkvg_fwd_tf32", ("-DMODF=1",)),
    "mod_fwd_tf32_lt": ("mod_fwd_tf32", ("-DLATE_TRIGGER=1",)),
    "attn_fwd_tf32": ("attn_fwd_tf32", ()),
    "ffn_fwd_tf32": ("ffn_fwd_tf32", ()),
    # backward (host side: tf32_bwd.py)
    "wprep_bwd_tf32": ("wprep_bwd_tf32", ()),
    "ffn_bwd_gate_tf32": ("ffn_bwd_gate_tf32", ()),
    # round 3 part 6: dy = [da | db] Wab inside the gate kernel (per-item partials, single-buffered accumulators)
    "ffn_bwd_gate_tf32_dyf": ("ffn_bwd_gate_tf32", ("-DDYF=1",)),
    "adaln_bwd_tf32": ("adaln_bwd_tf32", ()),
    "oproj_bwd_fused_tf32": ("oproj_bwd_fused_tf32", ()),
    "attn_dq_tf32": ("attn_dq_tf32", ()),
    "attn_dkv_tf32": ("attn_dkv_tf32", ()),
    # r3-swa part 21: the qkvg backward's dx = dP W as eight per-head partials in the two passes' epilogues (one item a CTA)
    "attn_dq_tf32_x": ("attn_dq_tf32", ("-DDXP=1",)),
    "attn_dkv_tf32_x": ("attn_dkv_tf32", ("-DDXP=1",)),
    "mod_bwd_tf32": ("mod_bwd_tf32", ()),
    "dw_tf32": ("dw_tf32", ()),
    "dw_tf32_direct": ("dw_tf32", ("-DDW_DIRECT=1",)),  # round 3 part 14: coalesced direct stores (development A/B, DwTf32.DIRECT_STORE)
    "dw_tf32_b3": ("dw_tf32", ("-DDW_BOX3=1",)),  # round 3 part 14: two 3-D-map boxes a ring stage (DwTf32.BOX3)
    "dw_reduce_tf32": ("dw_tf32", ()),
    # global attention glue (B200 2026-11-17, round 3): the casts around the bf16 global attention and the backward's head-wise tail
    "gattn_tf32": ("gattn_tf32", ()),
    # round 3 part 16: the few-tile backward tail (dx GEMM -> adaLN -> dc GEMM -> silu') on one 8-CTA cluster per tile (tf32_bwd.DcxTf32)
    "dcx_tf32": ("dcx_tf32", ()),
    "dcx_tf32_tr": ("dcx_tf32", ("-DTRACE=1",)),
    "dcx_tf32_sta": ("dcx_tf32", ("-DSTA=1",)),
    "dcx_tf32_sta_tr": ("dcx_tf32", ("-DSTA=1", "-DTRACE=1")),
    "dcx_tf32_st2": ("dcx_tf32", ("-DSTA=2",)),
    "dcx_tf32_st2_tr": ("dcx_tf32", ("-DSTA=2", "-DTRACE=1")),
    "dcx_tf32_st3": ("dcx_tf32", ("-DSTA=3",)),
    "dcx_tf32_st3_tr": ("dcx_tf32", ("-DSTA=3", "-DTRACE=1")),
    # r3-swa part 19: the c gradient's GEMM split at column 2C -- the early columns off the chain, silu' in the late launch (tf32_bwd.DcgTf32)
    "dcg_tf32": ("dcg_tf32", ()),
}
#: few-tile builds (B200 2026-11-17, round 3 part 4): -DROWS=64 / 32 tile slots with a deeper weight ring (:func:`tile_rows`)
ROWS_STEMS = ("ffn_fwd_tf32",)
for _s in ROWS_STEMS:
    for _r in (64, 32):
        CUBINS[f"{_s}_r{_r}"] = (_s, (f"-DROWS={_r}",))
        CUBINS[f"{_s}_r{_r}p"] = (_s, (f"-DROWS={_r}", "-DPAIR=1"))  # round 3 part 11: a CTA pair per tile (FfnFwdTf32)
#: dynamic shared memory per kernel (the sources' SMEM / SMEM_BYTES)
SMEM = {
    "wprep_tf32": 0,
    "mod_fwd_tf32": 14 * 16384 + 512,
    "qkvg_fwd_tf32": 8 * 16384 + 6 * 8192 + 16384 + 2 * 16384 + 512,
    "qkvg_fwd_tf32_modf": 8 * 16384 + 6 * 8192 + 16384 + 2 * 16384 + 512,
    "mod_fwd_tf32_lt": 14 * 16384 + 512,
    "attn_fwd_tf32": 2 * 81920 + 2048 + 2 * 8192 + 256,
    "ffn_fwd_tf32": 8 * 16384 + 8 * 8192 + 2048 + 4 * 8192 + 512,
    "wprep_bwd_tf32": 0,
    "ffn_bwd_gate_tf32": 6 * 16384 + 10 * 8192 + 6 * 8192 + 512,
    "ffn_bwd_gate_tf32_dyf": 6 * 16384 + 10 * 8192 + 6 * 8192 + 512,
    "adaln_bwd_tf32": 0,
    "oproj_bwd_fused_tf32": 4 * 16384 + 8 * 16384 + 2 * 16384 + 512,
    "attn_dq_tf32": 2 * 32768 + 12 * 12288 + 16384 + 64 * 16 + 512,
    "attn_dkv_tf32": 2 * 32768 + 7 * 17408 + 2 * 16384 + 64 * 16 + 512,
    "attn_dq_tf32_x": 2 * 32768 + 12 * 12288 + 16384 + 64 * 16 + 512,
    "attn_dkv_tf32_x": 2 * 32768 + 7 * 17408 + 2 * 16384 + 64 * 16 + 512,
    "mod_bwd_tf32": 0,
    "dw_tf32": 5 * 20 * 2048 + 256,
    "dw_tf32_direct": 5 * 20 * 2048 + 256,
    "dw_tf32_b3": 5 * 20 * 2048 + 256,
    "dw_reduce_tf32": 0,
    "gattn_tf32": 0,
    "dcx_tf32": 229376 + 256,
    "dcx_tf32_tr": 229376 + 256,
    "dcx_tf32_sta": 229376 + 256,
    "dcx_tf32_sta_tr": 229376 + 256,
    "dcx_tf32_st2": 229376 + 256,
    "dcx_tf32_st2_tr": 229376 + 256,
    "dcx_tf32_st3": 229376 + 256,
    "dcx_tf32_st3_tr": 229376 + 256,
    "dcg_tf32": 225280 + 256,
}
SMEM.update(
    {
        "ffn_fwd_tf32_r64": 8 * 8192 + 18 * 8192 + 2048 + 4 * 4096 + 1024,
        "ffn_fwd_tf32_r32": 8 * 4096 + 23 * 8192 + 2048 + 4 * 2048 + 1024,
        "ffn_fwd_tf32_r64p": 8 * 8192 + 16 * 8192 + 2048 + 4 * 4096 + 64 * 256 + 1024,
        "ffn_fwd_tf32_r32p": 8 * 4096 + 22 * 8192 + 2048 + 4 * 2048 + 32 * 256 + 1024,
    }
)
#: kernel function per cubin
FUNCS = {
    "wprep_tf32": "swa_wprep_tf32_sm100",
    "mod_fwd_tf32": "swa_mod_fwd_tf32_sm100",
    "qkvg_fwd_tf32": "swa_qkvg_fwd_tf32_sm100",
    "qkvg_fwd_tf32_modf": "swa_qkvg_fwd_tf32_sm100",
    "mod_fwd_tf32_lt": "swa_mod_fwd_tf32_sm100",
    "attn_fwd_tf32": "swa_attn_fwd_tf32_sm100",
    "ffn_fwd_tf32": "swa_ffn_fwd_tf32_sm100",
    "wprep_bwd_tf32": "swa_wprep_bwd_tf32_sm100",
    "ffn_bwd_gate_tf32": "swa_ffn_bwd_gate_tf32_sm100",
    "ffn_bwd_gate_tf32_dyf": "swa_ffn_bwd_gate_tf32_sm100",
    "adaln_bwd_tf32": "swa_adaln_bwd_tf32_sm100",
    "oproj_bwd_fused_tf32": "swa_oproj_bwd_fused_tf32_sm100",
    "attn_dq_tf32": "swa_attn_dq_tf32_sm100",
    "attn_dkv_tf32": "swa_attn_dkv_tf32_sm100",
    "attn_dq_tf32_x": "swa_attn_dq_tf32_sm100",
    "attn_dkv_tf32_x": "swa_attn_dkv_tf32_sm100",
    "mod_bwd_tf32": "swa_mod_silu_tf32_sm100",
    "dw_tf32": "swa_dw_tf32_sm100",
    "dw_tf32_direct": "swa_dw_tf32_sm100",
    "dw_tf32_b3": "swa_dw_tf32_sm100",
    "dw_reduce_tf32": "swa_dw_reduce_tf32_sm100",
    "gattn_tf32": "swa_gcast_tf32_sm100",
    "dcx_tf32": "swa_dcx_tf32_sm100",
    "dcx_tf32_tr": "swa_dcx_tf32_sm100",
    "dcx_tf32_sta": "swa_dcx_tf32_sm100",
    "dcx_tf32_sta_tr": "swa_dcx_tf32_sm100",
    "dcx_tf32_st2": "swa_dcx_tf32_sm100",
    "dcx_tf32_st2_tr": "swa_dcx_tf32_sm100",
    "dcx_tf32_st3": "swa_dcx_tf32_sm100",
    "dcx_tf32_st3_tr": "swa_dcx_tf32_sm100",
    "dcg_tf32": "swa_dcg_tf32_sm100",
}
for _s in ROWS_STEMS:
    for _r in (64, 32):
        FUNCS[f"{_s}_r{_r}"] = FUNCS[_s]
        FUNCS[f"{_s}_r{_r}p"] = FUNCS[_s]
#: round 3 part 15: the dW kernel on one cluster of CL CTAs per output tile, the split-K partials summed over DSMEM (no workspace, no
#: reduce launch); r = reduction rows a ring stage; x = development builds (1: no DSMEM sends, 2: no reduction rows)
for _c in (4, 8, 16):
    for _r in (16, 32, 64):
        for _x in (0, 1, 2):
            _n = f"dw_tf32_cl{_c}_r{_r}" + (f"x{_x}" if _x else "")
            CUBINS[_n] = ("dw_tf32", ("-DDW_BOX3=1", f"-DDW_CL={_c}", f"-DDW_BKR={_r}", f"-DDW_CLX={_x}"))
            SMEM[_n] = 5 * 20 * 2048 + 256
            FUNCS[_n] = "swa_dw_tf32_sm100"
            if not _x and _c <= 8 and _r == 32:  # r3-swa part 21: the partials through a global workspace (DwTf32.GX)
                CUBINS[_n + "g"] = ("dw_tf32", ("-DDW_BOX3=1", f"-DDW_CL={_c}", f"-DDW_BKR={_r}", "-DDW_GX=1"))
                SMEM[_n + "g"] = SMEM[_n]
                FUNCS[_n + "g"] = FUNCS[_n]

#: N2 (2026-11-18): the modulation's weight gradient dWmod = dmod^T silu(c) on the cluster dW kernel's one-problem build (DWM=1: six 128-row tiles of 8 CTAs; tf32_bwd.DwModTf32)
CUBINS["dw_tf32_cl8_r32g_dwm"] = ("dw_tf32", ("-DDW_BOX3=1", "-DDW_CL=8", "-DDW_BKR=32", "-DDW_GX=1", "-DDWM=1"))
SMEM["dw_tf32_cl8_r32g_dwm"] = SMEM["dw_tf32_cl8_r32g"]
FUNCS["dw_tf32_cl8_r32g_dwm"] = FUNCS["dw_tf32_cl8_r32g"]


# --------------------------------------------------------------------------------------------------- build / load
@functools.lru_cache(maxsize=None)
def cubin(name: str) -> str:
    """Path of the cubin ``name`` (see CUBINS), built on first use; rebuilt only when a source, a flag or the toolchain changes."""
    from miniworld_engine.kernels import _nvcc

    stem, extra = CUBINS[name]
    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
        *extra,
    )
    return str(_nvcc.cubin(f"swa_dit_sm100_tf32/{name}", flags, source=_dir / f"{stem}.cu", toolchain="sm100"))


def _load(name: str, func: str | None = None, cluster: int | None = None):
    """The kernel of cubin ``name`` (its FUNCS entry, or ``func`` of the same cubin), loaded in the current device's context."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.Kernel(cubin(name), func or FUNCS[name], SMEM[name], cluster=cluster, pdl=PDL)


def _f32map(t, dims, strides, box, swizzle=128):
    return _sm._map(t, dims, strides, box, swizzle=swizzle, dtype="f32")


#: round 3 part 11 (B200 2026-11-17): few-tile FFN stages on a CTA pair per tile when 2 x tiles fit the SMs (ffn_fwd_tf32 -DPAIR: rank r
#: runs hidden chunks 4 r .. 4 r + 3, the partial ffn halves exchanged over DSMEM)
FFN_PAIR = True
#: development override of :func:`tile_rows` (None: the rule)
ROWS_FORCE: int | None = None


def tile_rows(A: int, B: int, S: int) -> int:
    """ROWS of the row kernels' build for a call: one augment (A = 1) whose tiles of 32 / 64 atoms all fit one wave of CTAs takes the
    few-tile builds (smaller tile slots, a deeper weight ring: a tile's time there is the weight stream's, not the rows'); else 128."""
    if ROWS_FORCE is not None:
        return ROWS_FORCE
    if A != 1:
        return 128
    if FFN_PAIR:  # round 3 part 11: the first tile size whose tiles fit the SMs as CTA pairs (ESMFold2 fp32 inference L 3072: 32-row
        # tiles alone 46.9 us, 64-row tiles on pairs 44.9)
        for rows in (32, 64):
            if 2 * B * ((S + rows - 1) // rows) <= num_sms():
                return rows
    for rows in (32, 64):
        if B * ((S + rows - 1) // rows) <= num_sms():
            return rows
    return 128


def tiling(A: int, rows: int = 128) -> tuple[int, int]:
    """(SP augments, AT atoms) of a <= ``rows``-row tile: SP = min(A, 16), AT = rows // SP (the modulation rows are read from global
    memory, so any AT works: A = 1 takes ``rows`` atoms per tile)."""
    SP = min(A, 16)
    return SP, rows // SP


def _b16(t) -> int:
    """1 for a bf16 tensor (round 3 part 6: master training reads x / c / dy and writes out / dx / dc as bf16 in place), else 0."""
    return int(t.dtype == torch.bfloat16)


def _rows_map_any(t, A, B, S, SP, AT):
    """:func:`_rows_map` of fp32 rows, or of bf16 rows: box (32 channels, AT atoms, 1, SP augments) of 64-B rows, 64-B swizzle (the
    first half of a k-block slot; the kernels convert)."""
    if t.dtype != torch.bfloat16:
        return _rows_map(t, A, B, S, SP, AT)
    return _sm._map(t, [C, S, B, A], [C * 2, S * C * 2, B * S * C * 2], [32, AT, 1, SP], swizzle=64, dtype="bf16")


def _rows_map(t, A, B, S, SP, AT):
    """[A, B, S, 128] fp32 rows -> 4-D TMA map, box (32 channels, AT atoms, 1, SP augments), 128-B swizzle (one k-block per load)."""
    return _f32map(t, [C, S, B, A], [C * 4, S * C * 4, B * S * C * 4], [32, AT, 1, SP])


def _out_map(t, A, B, S, SP, AT):
    """[A, B, S, 128] fp32 rows -> 4-D TMA store map, box (16 channels, AT atoms, 1, SP augments), 64-B swizzle (a staged output tile)."""
    if t.dtype == torch.bfloat16:  # bf16 rows (round 3 part 6): box (16 channels, AT atoms, 1, SP augments) of 32-B rows, unswizzled
        return _sm._map(t, [C, S, B, A], [C * 2, S * C * 2, B * S * C * 2], [16, AT, 1, SP], swizzle=0, dtype="bf16")
    return _f32map(t, [C, S, B, A], [C * 4, S * C * 4, B * S * C * 4], [16, AT, 1, SP], swizzle=64)


def _mod_map(mod, AT):
    """mod [B S, 6C] fp32 -> 2-D TMA map, box (32 columns, AT rows), 128-B swizzle: one modulation k-block of a tile's atoms."""
    return _f32map(mod, [6 * C, mod.shape[0]], 6 * C * 4, [32, AT])


def _rope_map(t, AT):
    """cos / sin [B S, D/2] fp32 -> 2-D TMA map, box (D/2 = 16, AT rows), 64-B swizzle (qkvg_fwd_tf32's sw64)."""
    return _f32map(t, [D // 2, t.shape[0]], D // 2 * 4, [D // 2, AT], swizzle=64)


# --------------------------------------------------------------------------------------------------- kernels (host side)
#: the packed weight forms of wprep_tf32.cu: W [512, 128] | WO [128, 128] | WAB [512, 128] | WD [128, 256] (fp32 elements)
WPACK = (4 * C * C, C * C, 2 * NHID * C, C * NHID)


class WPrepTf32:
    """The weight forms in one launch each (wprep_tf32.cu): the block's four (:meth:`block`) and a plain rounding (:meth:`round`)."""

    def __init__(self):
        self.kb = _load("wprep_tf32")
        self.kr = _load("wprep_tf32", "swa_round_tf32_sm100")

    def block(self, wqkv, wg, wo, wu, wd):
        """[Wqkv; Wg] [512, C], Wo [C, C], Wab [2 NHID, C] (rows per 32-unit hidden chunk j: [Wu[32 j ..]; Wu[NHID + 32 j ..]]),
        Wd [C, NHID], all TF32-rounded fp32: views of one packed buffer."""
        wqkv, wg, wo, wu, wd = (w.float().contiguous() for w in (wqkv, wg, wo, wu, wd))
        out = torch.empty(sum(WPACK), device=wqkv.device, dtype=torch.float32)
        self.kb(
            ((sum(WPACK) // 4 + 255) // 256, 1, 1),
            (256, 1, 1),
            wqkv,
            wg,
            wo,
            wu,
            wd,
            out,
        )
        o = (
            0,
            WPACK[0],
            WPACK[0] + WPACK[1],
            WPACK[0] + WPACK[1] + WPACK[2],
            sum(WPACK),
        )
        return (
            out[o[0] : o[1]].view(4 * C, C),
            out[o[1] : o[2]].view(C, C),
            out[o[2] : o[3]].view(2 * NHID, C),
            out[o[3] : o[4]].view(C, NHID),
        )

    def round(self, w):
        """w rounded to TF32 (fp32, same shape); numel a multiple of 4."""
        w = w.float().contiguous()
        out = torch.empty_like(w)
        n4 = w.numel() // 4
        assert n4 * 4 == w.numel()
        self.kr(((n4 + 255) // 256, 1, 1), (256, 1, 1), w, out, int(n4))
        return out


def mod_groups(R: int, nsm: int) -> int:
    """NG, the channel groups of mod_fwd_tf32's items (6 / NG blocks of 128 channels each): the least work on the busiest CTA,
    ceil(items / CTAs) x (6 / NG + 1/2) -- a channel block's MMAs + stores, plus an item's c load and silu (each group recomputes its
    rows' silu) -- ties to the smaller NG."""
    ntile = (R + 127) // 128
    best = None
    for ng in (1, 2, 3, 6):
        items = ntile * ng
        cost = -(-items // min(nsm, items)) * (6 // ng + 0.5)
        if best is None or cost < best[0]:
            best = (cost, ng)
    return best[1]


class ModFwdTf32:
    """silu(c) Wmod^T: c [R, C] fp32 contiguous, Wmod [6C, C] fp32 (TF32-rounded) -> [R, 6C] fp32. Persistent: grid <= #SMs over
    (128-row tile, channel group) items (:func:`mod_groups`)."""

    def __init__(self):
        self.k = _load("mod_fwd_tf32")

    def bind(self, c, wmod, out=None, k=None, ng=None):
        """``k``: another build of the kernel (``mod_fwd_tf32_lt``, round 3 part 7), ``ng`` its channel groups (None: :func:`mod_groups`)."""
        R = c.shape[0]
        k = self.k if k is None else k
        out = torch.empty(R, 6 * C, device=c.device, dtype=torch.float32) if out is None else out
        cmap = _sm._map(c, [C, R], C * 2, [32, 128], swizzle=64, dtype="bf16") if _b16(c) else _f32map(c, [C, R], C * 4, [32, 128])
        maps = (
            cmap,
            _f32map(wmod, [C, 6 * C], C * 4, [32, 128]),
            _f32map(out, [6 * C, R], 6 * C * 4, [32, 128]),
        )
        ntile = (R + 127) // 128
        ng = mod_groups(R, num_sms()) if ng is None else ng
        grid = (min(num_sms(), ntile * ng), 1, 1)

        def run():
            k(grid, (512, 1, 1), *maps, int(ntile), int(ng), _b16(c))

        run.keep = maps
        return run, out


class QkvgFwdTf32:
    """q [N, S, C] fp32 (N = A B), mod [B S, 6C] fp32, cos / sin [B S, D/2] fp32, W = [Wqkv; Wg] [4C, C] fp32 (TF32-rounded) ->
    Qh, Kh, Vh [N, H, S, D] fp32 (TF32-valued), G [N S, C] fp32 and, with save, X, PQ, PK [N S, C] fp32."""

    def __init__(self):
        self.k = _load("qkvg_fwd_tf32")
        self.kf = None  # the MODF build (round 3 part 7), loaded on first use

    def bind(self, q, mod, cos, sin, W, A, B, save=False, cw=None):
        """``cw`` = (c [B S, C] fp32, Wmod [6C, C] TF32-rounded): the MODF build (:func:`modf_ok`) -- shift_a / scale_a made inside from c,
        ``mod`` (written beside it by ``mod_fwd_tf32_lt``) not read."""
        N, S, _ = q.shape
        SP, AT = tiling(A)
        dev = q.device
        f32 = torch.float32
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=f32) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=f32)
        X, PQ, PK = (torch.empty(N * S, C, device=dev, dtype=f32) for _ in range(3)) if save else (G, G, G)
        assert mod.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()

        def head_map(t):  # [A, B, H, S, D] head-major -> 5-D store box (32, AT, 1, 1, SP), 128-B swizzle
            return _f32map(t, [D, S, H, B, A], [D * 4, S * D * 4, H * S * D * 4, B * H * S * D * 4], [D, AT, 1, 1, SP])

        maps = (
            _rows_map_any(q, A, B, S, SP, AT),
            _f32map(W, [C, 4 * C], C * 4, [32, 64]),
            _mod_map(mod, AT),
            _rope_map(cos, AT),
            _rope_map(sin, AT),
            head_map(Qh),
            head_map(Kh),
            head_map(Vh),
            _rows_map(G.view(-1, C), A, B, S, SP, AT),
            _rows_map(PQ, A, B, S, SP, AT),
            _rows_map(PK, A, B, S, SP, AT),
        )
        k = self.k
        if cw is not None:
            assert A == 1 and SP == 1 and AT == 128 and not _b16(q)
            if self.kf is None:
                self.kf = _load("qkvg_fwd_tf32_modf")
            k = self.kf
            extra = (_f32map(cw[0], [C, cw[0].shape[0]], C * 4, [32, AT]), _f32map(cw[1], [C, 6 * C], C * 4, [32, 64]))
        else:
            extra = maps[:2]  # (ignored by the plain build)
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        NG = 1  # projection groups: spread small problems over the SMs
        while NG < 4 and ntile * NG * 2 <= num_sms():
            NG *= 2
        if cw is not None:
            NG = 4  # one projection per item, one item per CTA (the MODF TMEM plan)
            assert ntile * NG <= num_sms()
        grid = (min(num_sms(), ntile * NG), 1, 1)

        def run():
            k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                int(NG),
                float(EPS),
                float(EPS),
                int(save),
                X,
                _b16(q),
                *extra,
            )

        run.keep = (maps, W, mod, cos, sin, Qh, Kh, Vh, G, X, PQ, PK) + ((cw, extra) if cw is not None else ())
        return run, (Qh, Kh, Vh, G, X, PQ, PK)


class AttnFwdTf32:
    """Window attention: Qh, Kh, Vh head-major [N, H, S, D] fp32, seqused [N] int32 -> O [N S, C] fp32, LSE [N, H, S] fp32."""

    def __init__(self):
        self.k = _load("attn_fwd_tf32")

    def bind(self, Qh, Kh, Vh, seqused, O=None, LSE=None):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        O = torch.empty(N * S, C, device=Qh.device, dtype=torch.float32) if O is None else O
        LSE = torch.empty(N, H, S, device=Qh.device, dtype=torch.float32) if LSE is None else LSE
        rows = N * H * S
        maps = (
            _f32map(Qh.view(rows, D), [D, rows], D * 4, [D, 128]),
            _f32map(Kh.view(rows, D), [D, rows], D * 4, [D, 256]),
            _f32map(Vh.view(rows, D), [D, rows], D * 4, [D, 256], swizzle="128a32"),
        )  # MN-major PV operand
        items = N * (S // 128) * H
        grid = (min(num_sms(), items), 1, 1)
        omap = _f32map(O, [C, N * S], C * 4, [16, 128], swizzle=64)  # O [N S, C] rows, store box (16 channels, 128 rows)

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, omap, LSE, int(S), int(N), float(D**-0.5))

        run.keep = (maps, omap, O, LSE)
        return run, O, LSE


class FfnFwdTf32:
    """q, G, O [N S, C] fp32, mod [B S, 6C] fp32, Wo [C, C], Wab [2 NHID, C] (rows [Wu[32 j ..]; Wu[NHID + 32 j ..]] per 32-unit
    chunk j), Wd [C, NHID] (all TF32-rounded) -> out and, with save, q1, att, y, ffn [N S, C] fp32."""

    def __init__(self):
        self.k = _load("ffn_fwd_tf32")
        self.kr = {128: self.k}
        self.trace = None  # a TRACE=1 build's clock buffer (_tools/tr32.py)

    def kernel(self, rows, pair=False):
        key = (rows, pair) if pair else rows
        if key not in self.kr:
            self.kr[key] = _load(f"ffn_fwd_tf32_r{rows}p", cluster=2) if pair else _load(f"ffn_fwd_tf32_r{rows}")
        return self.kr[key]

    def bind(self, q, g, o, mod, wo, wab, wd, A, B, save=False, out16=None):
        """q fp32 or bf16 (read in place); ``out16``: the output in bf16 (None: q's dtype). Saves fp32."""
        M = q.numel() // C
        S = M // (A * B)
        rows = tile_rows(A, B, S)
        SP, AT = tiling(A, rows)
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        pair = FFN_PAIR and rows < 128 and 2 * ntile <= num_sms()  # round 3 part 11: a CTA pair per tile (hidden halves)
        k = self.kernel(rows, pair)
        dev = q.device
        out = torch.empty(M, C, device=dev, dtype=q.dtype if out16 is None else (torch.bfloat16 if out16 else torch.float32))
        Q1, Att, Y, Ff = (torch.empty(M, C, device=dev, dtype=torch.float32) for _ in range(4)) if save else (out, out, out, out)
        maps = (
            _rows_map(g, A, B, S, SP, AT),
            _rows_map(o, A, B, S, SP, AT),
            _rows_map_any(q, A, B, S, SP, AT),
            _f32map(wo, [C, C], C * 4, [32, 64]),
            _f32map(wab, [C, 2 * NHID], C * 4, [32, 64]),
            _f32map(wd, [NHID, C], NHID * 4, [32, 64]),
            _mod_map(mod, AT),
            *(_out_map(t, A, B, S, SP, AT) for t in (out, Q1, Att, Y, Ff)),
        )
        grid = (2 * ntile if pair else min(num_sms(), ntile), 1, 1)

        def run():
            k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                float(EPS),
                int(save),
                _b16(q) | 2 * _b16(out),
                *(() if self.trace is None else (self.trace,)),
            )

        run.keep = (maps, mod, out, Q1, Att, Y, Ff)
        return run, (out, Q1, Att, Y, Ff)


class _Kernels:
    """Every TF32 forward kernel, loaded on one device."""

    def __init__(self) -> None:
        self.prep, self.mod, self.qkvg, self.attn, self.ffn = (
            WPrepTf32(),
            ModFwdTf32(),
            QkvgFwdTf32(),
            AttnFwdTf32(),
            FfnFwdTf32(),
        )
        self.mod_lt = None  # mod_fwd_tf32 -DLATE_TRIGGER (round 3 part 7), loaded on first use

    def stats(self) -> dict[str, tuple[int, int]]:
        """kernel -> (registers per thread, local memory bytes per thread: > 0 means spills)."""
        return {
            name: (k.regs, k.lmem)
            for name, k in (
                ("wprep_tf32", self.prep.kb),
                ("round_tf32", self.prep.kr),
                ("mod_fwd_tf32", self.mod.k),
                ("qkvg_fwd_tf32", self.qkvg.k),
                ("attn_fwd_tf32", self.attn.k),
                ("ffn_fwd_tf32", self.ffn.k),
            )
        }


@functools.lru_cache(maxsize=16)
def kernels_tf32(index: int) -> _Kernels:
    """Build and load every TF32 forward kernel on device ``index`` (raises on a toolchain or driver failure)."""
    with torch.cuda.device(index):
        return _Kernels()


#: round 3 part 16 (development switch): the global glue's tail and delta with eight threads per (row, head) (gattn_tf32.cu *8 kernels)
GLUE8 = __import__("os").environ.get("SWA_GLUE8", "1") == "1"
#: N1 round 4 (development switch): swa_gprep triggers its dependent launch (gattn_bwd) at once
#: N1 round 4 master switch (A/B): the three global-glue changes below (swa_gprep, swa_gcast early trigger, gprep launched programmatically)
N1_GLUE = __import__("os").environ.get("SWA_N1_GLUE", "1") == "1"
GPREP_PDL = __import__("os").environ.get("SWA_GPREP_PDL", "1") == "1"
GTAIL_PDL = __import__("os").environ.get("SWA_GTAIL_PDL", "0") == "1"  # N1 round 4: swa_gtail8 launched programmatically (starts under the attention backward, waits on griddepcontrol.wait)
GCAST_PDL = __import__("os").environ.get("SWA_GCAST_PDL", "1") == "1"  # N1 round 4: swa_gcast triggers its dependent launch (the attention) at once
GCAST_IN_PDL = __import__("os").environ.get("SWA_GCAST_IN_PDL", "0") == "1"  # ... and is itself launched programmatically
#: ... and is itself launched programmatically (starts before the out-projection backward ends; waits on griddepcontrol.wait)
GPREP_IN_PDL = __import__("os").environ.get("SWA_GPREP_IN_PDL", "1") == "1"


class GlobalGlue:
    """The glue of the fp32 block's global attention (gattn_tf32.cu; B200 2026-11-17, round 3): the fp32 -> bf16 casts of the attention's
    operands with the zero fill of its outputs in one launch, and the backward's head-wise tail in one launch (were ~10 / ~50 torch launches
    per block: at one sample of a few thousand atoms the fp32 global training step was mostly those)."""

    def __init__(self):
        from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

        self.kc = driver.Kernel(cubin("gattn_tf32"), "swa_gcast_tf32_sm100", 0)
        self.kt = driver.Kernel(cubin("gattn_tf32"), "swa_gtail_tf32_sm100", 0)
        self.kd = driver.Kernel(cubin("gattn_tf32"), "swa_gdelta_tf32_sm100", 0)
        # round 3 part 16: eight threads per (row, head) (GLUE8)
        self.kt8 = driver.Kernel(cubin("gattn_tf32"), "swa_gtail8_tf32_sm100", 0)
        self.kt8p = driver.Kernel(cubin("gattn_tf32"), "swa_gtail8_tf32_sm100", 0, pdl=True)
        self.kd8 = driver.Kernel(cubin("gattn_tf32"), "swa_gdelta8_tf32_sm100", 0)
        self.kp = driver.Kernel(cubin("gattn_tf32"), "swa_gprep_tf32_sm100", 0)  # N1 round 4: cast + delta in one launch
        self.kp2 = driver.Kernel(cubin("gattn_tf32"), "swa_gprep_pdl_tf32_sm100", 0)
        self.kp2p = driver.Kernel(cubin("gattn_tf32"), "swa_gprep_pdl_tf32_sm100", 0, pdl=True)  # ... triggering the dependent launch (gattn_bwd) first
        self.kc2 = driver.Kernel(cubin("gattn_tf32"), "swa_gcast_pdl_tf32_sm100", 0)  # triggers the dependent attention launch first
        self.kc2p = driver.Kernel(cubin("gattn_tf32"), "swa_gcast_pdl_tf32_sm100", 0, pdl=True)

    def cast(self, srcs, nz: int) -> torch.Tensor:
        """[len(srcs) + nz, n] bf16: the fp32 arrays ``srcs`` (each n elements, contiguous, n a multiple of 8) rounded to bf16 (nearest
        even, as ``.to(bfloat16)``), then ``nz`` zero arrays."""
        n = srcs[0].numel()
        assert n % 8 == 0 and all(s.numel() == n and s.dtype == torch.float32 and s.is_contiguous() for s in srcs) and len(srcs) <= 5
        buf = torch.empty(len(srcs) + nz, n, device=srcs[0].device, dtype=torch.bfloat16)
        ptrs = [*srcs, *([srcs[0]] * (5 - len(srcs)))]
        total = (n // 8) * (len(srcs) + nz)
        (self.kc2p if GCAST_IN_PDL else self.kc2 if (GCAST_PDL and N1_GLUE) else self.kc)((min((total + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), *ptrs, buf, int(n // 8), len(srcs), int(nz))
        return buf

    def prep(self, srcs, o: torch.Tensor, do: torch.Tensor, n: int, s: int):
        """:meth:`cast` of three fp32 head arrays and :meth:`delta` in one launch (swa_gprep; bitwise the two kernels): (buf [3, n_el] bf16, dO16, D)."""
        ne = srcs[0].numel()
        assert len(srcs) == 3 and ne % 8 == 0 and all(x.numel() == ne and x.dtype == torch.float32 and x.is_contiguous() for x in srcs)
        assert o.is_contiguous() and do.is_contiguous() and o.dtype == torch.float32 and do.dtype == torch.float32
        dev = o.device
        buf = torch.empty(3, ne, device=dev, dtype=torch.bfloat16)
        do16 = torch.empty(n * s, C, device=dev, dtype=torch.bfloat16)
        dd = torch.empty(n, H, s, device=dev, dtype=torch.float32)
        tot = 3 * (ne // 8) + 8 * n * s * H
        (self.kp2p if GPREP_IN_PDL else self.kp2 if GPREP_PDL else self.kp)((min((tot + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), *srcs, buf, o, do, do16, dd, int(ne // 8), int(n), int(s))
        return buf, do16, dd

    def delta(self, o: torch.Tensor, do: torch.Tensor, n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(dO16 [N S, C] bf16, D [N, H, S] fp32) from the fp32 O (bf16 values) and dO: D of the rounded dO (gattn_tf32.cu swa_gdelta)."""
        assert o.is_contiguous() and do.is_contiguous() and o.dtype == torch.float32 and do.dtype == torch.float32
        do16 = torch.empty(n * s, C, device=o.device, dtype=torch.bfloat16)
        dd = torch.empty(n, H, s, device=o.device, dtype=torch.float32)
        total = n * s * H
        if GLUE8:
            self.kd8((min((8 * total + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), o, do, do16, dd, int(n), int(s))
            return do16, dd
        total = n * s * H
        self.kd((min((total + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), o, do, do16, dd, int(n), int(s))
        return do16, dd

    def tail(self, dqh, dkh, dvh, pq, pk, cos, sin, dp, eps: float) -> None:
        """dpq | dpk | dpv into ``dp`` [N S, 4C] columns 0 .. 3C from the bf16 head gradients [N, H, S, D] (``tf32_bwd._head_tail``'s
        equations for q and k, TF32-rounded; the v columns as they are)."""
        n, _, s, _ = dqh.shape
        assert cos.shape[-1] == D // 2 and cos.numel() == n * s * (D // 2) and cos.dtype == torch.float32 and sin.dtype == torch.float32
        total = n * s * H
        if GLUE8:
            (self.kt8p if GTAIL_PDL else self.kt8)(
                (min((8 * total + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), dqh, dkh, dvh, pq, pk, cos, sin, dp, int(n), int(s), float(eps)
            )
            return
        self.kt(
            (min((total + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), dqh, dkh, dvh, pq, pk, cos, sin, dp, int(n), int(s), float(eps)
        )


@functools.lru_cache(maxsize=16)
def global_glue(index: int) -> GlobalGlue:
    with torch.cuda.device(index):
        return GlobalGlue()


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


@device_constant
def is_b200(device: torch.device) -> bool:
    """Whether ``device`` is a B200 (sm_100). A constant to the compiled graph (``device_constant``): the capability query is not
    traced."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0)


def supported_tf32(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> bool:
    """The calls the TF32 kernels serve: fp32 q [N, S, 128] on B200 with S a multiple of 128, SwiGLU hidden 256, half window 64 (so
    not global attention), fp32 eps. Tensor metadata and the per-device constant :func:`is_b200` only: traceable under
    ``torch.compile(fullgraph=True)`` and safe on fake tensors. Whether the kernels LOAD is ``dispatch._tf32_ready``."""
    return (
        q.is_cuda
        and q.dtype == torch.float32
        and q.dim() == 3
        and q.shape[-1] == C
        and q.shape[1] > 0
        and q.shape[1] % 128 == 0
        and nhid == NHID
        and half_window == HW
        and eps == EPS
        and is_b200(q.device)
    )


def supported_tf32_global(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> bool:
    """:func:`supported_tf32` with global attention (``is_global(half_window)``): the TF32 row stages around the bf16 global attention
    (``block_fwd_tf32``; the backward ``tf32_bwd.block_bwd_tf32`` takes the global attention's backward and the head-wise tail itself)."""
    from miniworld_engine.kernels.swa_dit._common import is_global

    return is_global(half_window) and supported_tf32(q, nhid, HW, eps)


# --------------------------------------------------------------------------------------------------- weight forms (cached per version)
def _round_tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 holding the nearest TF32 value, ties away from zero (what ``cvt.rna.tf32.f32`` gives): the weights' MMA operand."""
    return tf32_grid(t.contiguous())


def _tf32_block_forms(wqkv, wg, wo, wu, wd):
    """(W = [Wqkv; Wg], WO, WAB, WD), TF32-rounded, in one launch (wprep_tf32.cu); WAB's rows per 32-unit hidden chunk j are
    [Wu[32 j .. 32 j + 31]; Wu[NHID + 32 j ..]] (dispatch._pack_ffn's wab)."""
    return kernels_tf32(_index(wqkv)).prep.block(wqkv, wg, wo, wu, wd)


def _tf32_wmod(wmod):
    """Wmod rounded to TF32, in one launch."""
    return kernels_tf32(_index(wmod)).prep.round(wmod)


def _forms(fn, *ts, save: bool = True):
    """fn(*ts) (a weight form) cached per weight version: an inference call (``save`` false, grad off) through ``dispatch.eval_form`` -- a
    CUDA-graph capture reads the form its eager warm-up made, so a replay runs no wprep / round launch (B200 2026-11-17: 9.5 us of an
    fp32 L 1024 block) -- else scoped to the capture (``_sm._cached``)."""
    if not save and not torch.is_grad_enabled():
        from miniworld_engine.kernels.swa_dit.dispatch import eval_form

        return eval_form(("swa_tf32_b200", fn.__name__), list(ts), lambda: fn(*ts))
    return _sm._cached(fn, *ts)


# --------------------------------------------------------------------------------------------------- the block
def block_fwd_tf32(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window=HW, out16=None):
    """The fp32 forward on TF32 tensor cores: [out] or, with ``save``, [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF] (all
    fp32) -- the layout of ``dispatch.swa_dit_block_fwd``. q [N, S, C] fp32 contiguous, mod [B S, 6C] fp32, cos / sin [B S, D/2] fp32,
    seqused [N] int32, the five weights fp32. q may be bf16 (round 3 part 6, master training: read in place); ``out16`` the output in bf16 (None: q's dtype). The saves stay
    fp32."""
    N, S, _ = q.shape
    _sm.need_s128(S)
    glob = half_window != HW  # global attention (supported_tf32_global): the bf16 global attention between the TF32 row stages
    A = N // B
    with torch.cuda.device(q.device):
        K = kernels_tf32(_index(q))
        # the weight forms: one wprep launch on a cache miss (every replay inside a capture, unless the weights are static), the
        # head of the PDL chain -- qkvg / ffn read them after their griddepcontrol.wait
        W, WO, WAB, WD = _forms(_tf32_block_forms, wqkv, wg, wo, wu, wd, save=save)
        run, (Qh, Kh, Vh, G, X, PQ, PK) = K.qkvg.bind(q, mod, cos, sin, W, A, B, save=save)
        run()
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        if glob:  # global attention: the bf16 global attention (FlashAttention-4, or PyTorch's FlashAttention-2) on the
            # TF32 stages' fp32 heads rounded to bf16 -- the attention precision of the fp32 block's Triton path and of the H100 fp32 global block
            # (the three casts and the zeroed O in one launch: GlobalGlue.cast)
            # the native forward (every row of O written: no zero fill) for inference, and for training where the native backward serves
            # (sm100.NATIVE_TRAIN_FWD, as sm100.block_fwd), else FA4's (O zeroed by the cast launch)
            nat = not save or (_sm.NATIVE_TRAIN_FWD and _sm.native_global_train(Qh))
            if not save and F32IO_GLOBAL and N * S <= F32IO_MAX_ROWS and _sm._native_global(Qh):
                # inference (round 3 part 4): the fp32 q heads read by the attention kernel itself (rounded to bf16 in shared memory) and O
                # written fp32 (the bf16 values) -- no q cast, no O16.float() (K | V: one cast launch); the same bits as the cast route
                O = torch.empty(N * S, C, device=q.device, dtype=torch.float32)
                K16 = _sm.kernels(_index(q))
                if _sm.F32IO_MODE == 2:  # K | V rounded once by the cast launch (in the kernel they would be once per item)
                    b16 = global_glue(_index(q)).cast((Kh, Vh), 0)
                    Kh, Vh = (b16[i].view(N, H, S, D) for i in range(2))
                run, _, _ = K16.gattn32.bind(Qh, Kh, Vh, seqused, O=O)
                run()
                run, (out, *_rest) = K.ffn.bind(q.reshape(N * S, C), G, O, mod, WO, WAB, WD, A, B, save=False, out16=out16)
                run()
                return [out.view(N, S, C)]
            if save and nat and F32IO_TRAIN and F32IO_GLOBAL and N * S <= F32IO_MAX_ROWS and _sm._native_global(Qh) and _sm.F32IO_MODE == 2:
                # training (round 3 part 16): the same F32IO build with the native backward's lse save (negated log2) -- the q cast and
                # the O16.float() copy (~3.5 us a block at L 2048) gone; the saves stay the fp32 heads (the backward casts them); same bits
                O = torch.empty(N * S, C, device=q.device, dtype=torch.float32)
                b16 = global_glue(_index(q)).cast((Kh, Vh), 0)
                run, _, _ = _sm.kernels(_index(q)).gattn32.bind(
                    Qh, *(b16[i].view(N, H, S, D) for i in range(2)), seqused, O=O, LSE=lse, log2=True
                )
                run()
                run, (out, q1, att, y, ffn) = K.ffn.bind(q.reshape(N * S, C), G, O, mod, WO, WAB, WD, A, B, save=save, out16=out16)
                run()
                return [out.view(N, S, C), Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, att, y, ffn]
            b16 = global_glue(_index(q)).cast((Qh, Kh, Vh), 0 if nat else 1)
            O16 = torch.empty(N * S, C, device=q.device, dtype=torch.bfloat16) if nat else b16[3].view(N * S, C)
            _sm.attn_global_fwd(*(b16[i].view(N, H, S, D) for i in range(3)), seqused, O16, lse, native=nat, log2=save and nat)
            O = O16.float()
        else:
            O = torch.empty(N * S, C, device=q.device, dtype=torch.float32)
            run, _, _ = K.attn.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
            run()
        run, (out, q1, att, y, ffn) = K.ffn.bind(q.reshape(N * S, C), G, O, mod, WO, WAB, WD, A, B, save=save, out16=out16)
        run()
    if not save:
        return [out.view(N, S, C)]
    return [out.view(N, S, C), Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, att, y, ffn]


def mod_fwd_tf32(c: torch.Tensor, wmod: torch.Tensor, infer: bool = False) -> torch.Tensor:
    """silu(c) Wmod^T [R, 6C] fp32 on TF32 tensor cores; c [R, C] fp32 contiguous, Wmod [6C, C] fp32. ``infer``: an inference call (the
    rounded Wmod cached as an inference form, :func:`_forms`)."""
    with torch.cuda.device(c.device):
        K = kernels_tf32(_index(c))
        run, out = K.mod.bind(c, _forms(_tf32_wmod, wmod, save=not infer))
        run()
    return out


#: round 3 part 7 (development switch): fp32 inference of one augment with shift_a / scale_a made inside the qkvg stage (:func:`block_fwd_tf32_c`)
MODF = True


def modf_ok(q: torch.Tensor, B: int, half_window: int) -> bool:
    """:func:`block_fwd_tf32_c` serves this fp32 inference call: one augment (N = B), window attention, the qkvg stage's 4 ntile items and
    the modulation kernel's CTAs on the SMs at once (the modulation runs beside the qkvg stage, both resident)."""
    N, S, _ = q.shape
    if not (MODF and PDL and q.dtype == torch.float32 and N == B and half_window == HW and S % 128 == 0):
        return False
    return _modf_groups(N * S) is not None


def _modf_groups(R: int) -> int | None:
    """The modulation kernel's channel groups beside the MODF qkvg stage: 6 (one 128-channel block per CTA) when its CTAs fit the SMs
    next to the qkvg stage's 4 ntile CTAs, else None. Measured (B200 2026-11-17, `_tools/modf.py`, graph replay, L2 flushed, block
    alone): 1024 rows 49.2 -> 47.1 us; 2048 rows (3 groups) 49.1 -> 49.2, 3072 rows (1 group) 49.2 -> 53.2 -- with fewer groups the
    modulation ends after the attention and the FFN waits for it."""
    nsm, ntile = num_sms(), R // 128
    return 6 if 10 * ntile <= nsm else None


def block_fwd_tf32_c(q, c, wmod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window=HW):
    """fp32 inference of one augment from c (:func:`modf_ok`; B200 2026-11-17, round 3 part 7): the modulation kernel (LATE_TRIGGER build)
    and the qkvg stage (MODF build: shift_a / scale_a from c inside) start together, so the chain is qkvg -> attention -> FFN (was mod ->
    qkvg -> attention -> FFN); the FFN reads the modulation's other columns. The same bits as ``mod_fwd_tf32`` + :func:`block_fwd_tf32`."""
    N, S, _ = q.shape
    with torch.cuda.device(q.device):
        K = kernels_tf32(_index(q))
        if K.mod_lt is None:
            K.mod_lt = _load("mod_fwd_tf32_lt")
        # every weight form before the modulation launch (a cache miss launches its kernels ahead of the pair)
        wr = _forms(_tf32_wmod, wmod, save=False)
        W, WO, WAB, WD = _forms(_tf32_block_forms, wqkv, wg, wo, wu, wd, save=False)
        run_m, mod = K.mod.bind(c, wr, k=K.mod_lt, ng=_modf_groups(c.shape[0]))
        run_m()
        run, (Qh, Kh, Vh, G, _X, _PQ, _PK) = K.qkvg.bind(q, mod, cos, sin, W, 1, B, cw=(c, wr))
        run()
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        O = torch.empty(N * S, C, device=q.device, dtype=torch.float32)
        run, _, _ = K.attn.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
        run()
        run, (out, *_rest) = K.ffn.bind(q.reshape(N * S, C), G, O, mod, WO, WAB, WD, 1, B, save=False)
        run()
    return out.view(N, S, C)
