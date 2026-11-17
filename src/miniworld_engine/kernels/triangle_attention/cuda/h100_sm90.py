"""H100 (sm_90a) TriangleAttention, starting node, d_pair = 128, 4 heads x 32, bf16.

Kernel bodies are raw-PTX wgmma / TMA hand-CUDA in ``h100_sources/`` (no CUTLASS / CuTe), on the kernel boundaries of the B200
path (``b200_triattn.py``): front (LN + projections + bias) | attention core | tail (gate + out-projection + residual), and the
matching backward.
"""

import functools
import weakref
from pathlib import Path
from typing import Any

import torch

from miniworld_engine.kernels._nvcc import (
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)

_SRC = Path(__file__).with_name("h100_sources")

C = 128  # d_pair
H = 4  # heads
D = C // H  # head dim (the kernels take 32 only)
TOK = 128  # query / key tile; L must be a multiple of it


def _flags():
    return [
        *host_flags(),
        "-std=c++17",
        "-O3",
        *gencodes("90a"),
        "--use_fast_math",
        "-DNDEBUG",
        f"-I{_SRC}",
        "-Xptxas=-v",
    ]


@functools.lru_cache(maxsize=None)
def _core(rows: int = 2, defs: tuple[str, ...] = ()):
    """The attention core; `defs` = extra -D schedule knobs (experiments), part of the build name."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_core_r{rows}{tag}_v12",
        sources=[str(_SRC / "triattn_sm90.cu")],
        extra_cuda_cflags=[*_flags(), f"-DTA_R={rows}", *(f"-D{d}" for d in defs)],
        verbose=True,
    )


@functools.lru_cache(maxsize=None)
def _core_h16(defs: tuple[str, ...] = ()):
    """The inference core on fp16 operands (h100_sources/triattn_h16_sm90.cu): q / k / v / sbias fp16, out bf16."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_core_h16{tag}_v3",
        sources=[str(_SRC / "triattn_h16_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


# ---- opt-in FP8 PV route (inference, head width 32 / 16): P and V on the e4m3 tensor-core path (triattn_h16_sm90.cu, -DTA_F8PV=1) ----
#: False: the fp16 core (default; bit-identical to before).  True: the fp16 core's call sites run the FP8 variant: V is quantized to e4m3 per
#: (batch, head) and stored transposed [B, N, H, 32, S] with the keys of each group of 32 permuted (:func:`vt_perm`); P is formed as e4m3 bytes
#: directly from the logits (piecewise-linear 2^x, no MUFU) and the row sums come from the quantized P.  q, k, the bias and the score GEMM stay fp16.
#: The LSE and anything else the variant does not take run the fp16 core (heads of 16 channels are served: V^T rows 16 - 31 are TMA zero fill).
FP8_PV = False


#: True: V on two e4m3 planes (hi, and the rounding residual x16 rounded again): V^T [B, N, H, 2, 32, S], PV on m64n72k32; the V rounding error falls
#: from ~2^-4 to ~2^-8 of V (cost: one wider PV).  Read at the call, not cached with the core.
FP8_PV_HL = False


def _core_f8pv():
    return _core_h16((("TA_F8PV=1", "TA_F8HL=1") if FP8_PV_HL else ("TA_F8PV=1",)))


#: opt-in, d_pair 128 inference only (needs FP8_PV; FP8_PV_HL picks the hi + lo planes, else the hi plane alone): the front writes V as the core's e4m3 planes of V^T itself (``tri_front_vt``), with
#: the static per-head scale ``FP8_VSCALE`` (fp32 [4], powers of two; see :func:`vscale_from_amax`); the standalone ``vt_quant`` pass is gone.  Values past 448 x scale
#: saturate and set a device flag: :func:`take_overflow` reads (and clears) it -- the caller re-scales and redoes the forward when it is set.
FP8_PV_FUSED = False
FP8_VSCALE: Any = None
_OVF: dict = {}


#: the smallest static V scale (as the TriMul planes' ``_h100_fp8.SCALE_FLOOR``).  A head whose calibration absmax is ~0 (all zero, or the 1e-30 clamp of the
#: dynamic pass's scale) got 2^-99 .. 2^-107 before, which saturates the e4m3 cast on any later V of that head; such a head now takes scale 1 (V up to 448 fits,
#: a later saturation still sets the overflow flag).
VSCALE_FLOOR = 2.0**-30


def vscale_from_amax(amax, headroom: float = 2.0):
    """Per-head power-of-two V scale from the heads' V absmax (fp32 [H]): the smallest 2^k >= headroom x amax / 448; 1 where that would be at or below
    ``VSCALE_FLOOR`` (a head that was ~0 at calibration), so no scale is ever below the floor."""
    t = amax.float() * headroom / 448.0
    s = torch.exp2(torch.ceil(torch.log2(t.clamp_min(VSCALE_FLOOR))))
    return torch.where(t > VSCALE_FLOOR, s, torch.ones_like(s))


def take_overflow() -> bool:
    """True when a fused-front forward saturated a V value since the last call (clears the flags; synchronizes)."""
    hit = False
    for f in _OVF.values():
        hit |= bool(f.item())
        f.zero_()
    return hit


#: True: the scale of each weight pack is calibrated on its first fused call (eager; one extra front launch): the per-head power of two above 4 x the V absmax of
#: that call; :func:`rescale_all` raises every calibrated scale (after :func:`take_overflow` reported a saturation, then redo the forward).
FP8_VSCALE_AUTO = False
_VS_AUTO: dict = {}
#: derived, per static-scale key: 1 / scale fp32 [H] (the planes' pass / front takes it) and the scale repeated per sample [B * H] (the core takes it).  Kept in memory that
#: :func:`rescale_all` updates in place (a captured graph follows), so a call launches no torch glue kernel for them.
_VINV: dict = {}
_VSB: dict = {}


def _vinv_of(key, vscale):
    t = _VINV.get(key)
    if t is None:
        t = _VINV[key] = (1.0 / vscale).contiguous()
    return t


def _vsb_of(key, vscale, batch):
    t = _VSB.get((key, batch))
    if t is None:
        t = _VSB[key, batch] = vscale.repeat(batch).contiguous()
    return t


def _drop_derived(key) -> None:
    _VINV.pop(key, None)
    for k2 in [k for k in _VSB if k[0] == key]:
        del _VSB[k2]


def _forget(key) -> None:
    _VS_AUTO.pop(key, None)
    _drop_derived(key)


def _forget_on_free(t, key) -> None:
    """The static scales are keyed by a tensor's address (the fused front's packed weights, the V weight of an unfused site): drop the entry when that tensor
    is freed, so another module whose tensor later sits at the same address calibrates its own scale instead of reading a stale one."""
    if t is not None:
        weakref.finalize(t, _forget, key)


def rescale_all(factor: float = 4.0) -> None:
    for key in _VS_AUTO:
        _VS_AUTO[key].mul_(
            factor
        )  # in place: a captured graph reads the scale from this memory (the fused front takes 1 / scale inside the capture), so a replay follows
    for key, t in _VINV.items():
        t.copy_(1.0 / _VS_AUTO[key])
    for (key, batch), t in _VSB.items():
        t.copy_(_VS_AUTO[key].repeat(batch))


#: the shortest S at which the fused d_pair-128 route (the front writes the V^T planes) replaces the fp16 core (hi plane; x 2 with hi + lo, as the other gates).
#: Measured on 2.5.12 (module, CUDA-graph replay, same process): L128 +10 % (cool and sustained, control within 1 %), L256 -6 %; 0 = no gate.
FP8_PV_FUSED_MIN_S = 256


def _fused_vt(dev) -> bool:
    return FP8_PV and FP8_PV_FUSED and (FP8_VSCALE is not None or FP8_VSCALE_AUTO) and INFERENCE_FP16 and INFERENCE_TAIL_GATE


@functools.lru_cache(maxsize=None)
def vt_perm(device):
    """src[p] = the key (inside a group of 32) whose e4m3 probability sits at position p of the PV wgmma's k32 A fragment: lane quad t4 holds the
    accumulator columns 2 t4 + {0, 1} of the 8-column blocks 4 s .. 4 s + 3, which the k32 fragment reads as positions 4 t4 + {0 .. 3} (blocks
    0 / 1) and 16 + 4 t4 + {0 .. 3} (blocks 2 / 3)."""
    p = torch.arange(32)
    half, t4, rem = p // 16, (p % 16) // 4, p % 4
    return (8 * (2 * half + rem // 2) + 2 * t4 + rem % 2).to(device)


def quantize_v_torch(v):
    """Reference of :func:`quantize_v` in torch (several passes: slow).  v [B, N, S, H, 32] fp16 (rows at any common stride) -> (vt [B, N, H, 32, S] float8_e4m3fn with permuted keys, scale fp32 [B, H]):
    v = vt * scale, scale = amax over (N, S, channels) of the head / 448."""
    B, N, S, Hh, Dh = v.shape
    amax = v.abs().amax(dim=(1, 2, 4)).float()
    scale = (amax / 448.0).clamp_min(1e-30)
    x = v.float() / scale[:, None, None, :, None]
    planes = [x.to(torch.float8_e4m3fn)]
    if FP8_PV_HL:
        planes.append(((x - planes[0].float()) * 16.0).to(torch.float8_e4m3fn))
    out = []
    for vt in planes:  # [B, N, S, H, 32]
        vt = vt.reshape(B, N, S // 32, 32, Hh, Dh)[:, :, :, vt_perm(v.device)]  # keys permuted inside each group of 32
        out.append(vt.permute(0, 1, 4, 5, 2, 3).reshape(B, N, Hh, Dh, S))
    vt = torch.stack(out, dim=3) if FP8_PV_HL else out[0]
    return vt.contiguous(), scale.contiguous()


#: None: V is scaled per (batch, head) by its amax (one extra read of v); a float: that amax for every head (no pass; v beyond it saturates at e4m3 max)
FP8_PV_AMAX = None


def quantize_v(v):
    """V^T e4m3 + scale by the CUDA pass (`vt_quant`) when v is [B, N, S, H<=8, 32] fp16 with S % 128 == 0, else the torch reference."""
    B, N, S, Hh, Dh = v.shape
    if (
        S % 128 == 0
        and Hh <= 16
        and Dh in (16, 32)
        and v.stride(4) == 1
        and v.stride(3) == Dh
        and v.stride(2) % 8 == 0
        and v.stride(1) == S * v.stride(2)
        and v.stride(0) == N * v.stride(1)
    ):
        return tuple(_core_f8pv().vt_quant(v, 0.0 if FP8_PV_AMAX is None else float(FP8_PV_AMAX)))
    return quantize_v_torch(v)


#: head channels -> the shortest S at which the standalone-V^T FP8 core (hi plane; x 2 for hi + lo) replaces the fp16 core (below it the V^T pass costs more than the core gains: measured per config);
#: empty = no gate (direct use of ``FP8_PV``).  Set by :func:`enable_fp8_pv_for`.
FP8_PV_MIN_S: dict = {}
#: with ``FP8_PV_FUSED``: also serve the call sites that have no fused front (d_pair 64 / 256 / 384 and the like) by the standalone V^T pass, gated by ``FP8_PV_MIN_S``
FP8_PV_UNFUSED = False


#: the call site of the next standalone-V^T core call (the weight pointer, set by the integration): with ``FP8_VSCALE_AUTO`` its V scale is calibrated once and
#: then static (one pass over V, no amax pass), with the same overflow flag / :func:`take_overflow` / :func:`rescale_all` contract as the fused front
_SITE: Any = None
#: the tensor whose address ``_SITE`` is (set with it by the integration): its release drops the site's static scale (:func:`_forget_on_free`)
_SITE_T: Any = None


def quantize_v_static(v, site):
    """:func:`quantize_v` with a per-site static scale: the first call of a site (eager) takes the dynamic scales and keeps, per head, the power of two above
    4 x their absmax; the later calls run the single quantize pass with it (saturation sets the :func:`take_overflow` flag)."""
    key = ("vq", site, v.shape[3], v.shape[4])
    vscale = _VS_AUTO.get(key)
    if vscale is None:
        vt, vs = quantize_v(v)
        _VS_AUTO[key] = vscale_from_amax(vs.amax(dim=0) * 448.0, 4.0)
        _forget_on_free(_SITE_T if _SITE_T is not None and _SITE_T.data_ptr() == site else None, key)
        _drop_derived(key)
        return vt, vs
    if (
        v.size(2) % 128 == 0
        and v.size(3) <= 16
        and v.size(4) in (16, 32)
        and v.stride(4) == 1
        and v.stride(3) == v.size(4)
        and v.stride(2) % 8 == 0
        and v.stride(1) == v.size(2) * v.stride(2)
        and v.stride(0) == v.size(1) * v.stride(1)
    ):
        ovf = _OVF.setdefault(v.device, torch.zeros(1, dtype=torch.int32, device=v.device))
        return tuple(_core_f8pv().vt_quant(v, 0.0, _vinv_of(key, vscale), ovf))
    return quantize_v(v)


#: the width-generic fronts that make the V^T planes themselves (``front_fused3_sm90.cu`` with ``MW_F3_VTW``: d_pair 64, the bf16 / Y16 one-kernel fronts) instead of the standalone
#: ``vt_quant`` pass; the planes are bit for bit the pass's (same fp16-rounded V, same static scale, same cvt).  False: the standalone pass.
FP8_FRONT_VT = True


#: (d_pair, heads, channels) -> the shortest S at which the call takes the V^T planes from the front (hi plane; x 2 with hi + lo): the front's planes cost ~nothing (the pass is gone),
#: so the gate is lower than ``FP8_PV_MIN_S`` of the standalone pass.  Measured per config (module time against the 2.5.9 default, interleaved, same-cubin control): see docs/fp8.md.
#: A config not listed uses the standalone pass's gate.  Set by :func:`enable_fp8_pv_for`; empty = no gate.
FP8_PV_FRONT_MIN_S: dict = {}
FP8_PV_UNFUSED_FRONT_MIN_S = {(64, 4, 32): 256, (64, 2, 32): 256, (64, 4, 16): 256, (256, 8, 32): 768}
CALIBRATE = "calibrate"


def front_vt_args(device, length: int, heads: int, channels: int, batch: int = 1, dpair: int = 0):
    """None (the call takes the standalone pass / the fp16 core), ``CALIBRATE`` (the call site is eligible but its static scale is not taken yet: the caller runs the front as
    it is and takes the scale with :func:`quantize_v_static`, eager), or (1 / per-head scale fp32 [H], overflow flag, planes, the scale per sample [batch * H]): the front makes the
    e4m3 V^T planes itself.  Eligible: the FP8 PV route is on for the unfused sites, the site key (``_SITE``) is set, and ``length`` is past the front's S gate."""
    if not (FP8_PV and FP8_FRONT_VT and (not FP8_PV_FUSED or FP8_PV_UNFUSED) and FP8_VSCALE_AUTO and _SITE is not None and INFERENCE_FP16):
        return None
    gate = FP8_PV_FRONT_MIN_S.get((dpair, heads, channels), FP8_PV_MIN_S.get(channels, 0))
    if channels not in (D, 16) or heads > 16 or length < gate * (2 if FP8_PV_HL else 1):
        return None
    key = ("vq", _SITE, heads, channels)
    vscale = _VS_AUTO.get(key)
    if vscale is None:
        return CALIBRATE
    ovf = _OVF.setdefault(device, torch.zeros(1, dtype=torch.int32, device=device))
    return _vinv_of(key, vscale), ovf, 2 if FP8_PV_HL else 1, _vsb_of(key, vscale, batch)


class _F8Core:
    """Stands in for the loaded fp16-core extension while ``FP8_PV`` is on (same ``triattn_fwd_h16`` signature)."""

    @staticmethod
    def triattn_fwd_h16(q, k, v, sbias, scale, want_lse, gate=None, want_out=True, key_tiles=None, key_mask=None, bias_transposed=False):
        if q.size(4) not in (D, 16) or want_lse or q.size(3) > 16 or q.size(2) < FP8_PV_MIN_S.get(q.size(4), 0) * (2 if FP8_PV_HL else 1):
            return _core_h16().triattn_fwd_h16(q, k, v, sbias, scale, want_lse, gate, want_out, key_tiles, key_mask, bias_transposed)
        vt, vs = quantize_v_static(v, _SITE) if FP8_VSCALE_AUTO and _SITE is not None else quantize_v(v)
        return _core_f8pv().triattn_fwd_h16(q, k, k, sbias, scale, False, gate, want_out, key_tiles, key_mask, bias_transposed, vt, vs)


def core_h16() -> Any:
    """The loaded fp16 core (default), or its FP8-PV stand-in when ``FP8_PV`` is set (not with ``FP8_PV_FUSED``: then only the fused d_pair-128 front
    feeds the FP8 core -- the other call sites would need the standalone ``vt_quant`` pass, which costs more than the core gains there)."""
    return _F8Core if FP8_PV and (not FP8_PV_FUSED or FP8_PV_UNFUSED) else _core_h16()


#: Models whose end-level accuracy was checked with the FP8 PV route (FoldForge, 200 steps, trunk seeds 0-3: same-seed RMSD inside the bf16 trunk-seed spread),
#: with the call sites checked: "fused" = the d_pair-128 front that writes V^T itself, "unfused" = the other fronts through the standalone V^T pass (gated by S).
#: NOT a general route: Boltz-2 9c90 is bistable and flips under V rounding, so "boltz2" is deliberately absent.  Extend only after an end-level check.
FP8_PV_WHITELIST: dict = {"protenix1": frozenset({"fused"}), "opendde": frozenset({"unfused"})}

#: the S gate of the unfused sites that :func:`enable_fp8_pv_for` sets (head channels -> shortest S, hi plane; measured 2026-11-15, docs/fp8.md)
FP8_PV_UNFUSED_MIN_S = {32: 1024, 16: 512}


def enable_fp8_pv_for(model: str, *, hl: bool = False) -> bool:
    """Per-model opt-in of the FP8 PV route (inference): sets ``FP8_PV`` / ``FP8_VSCALE_AUTO`` and ``FP8_PV_FUSED`` (the d_pair-128 front writes V^T; and ``FP8_PV_HL``
    when ``hl``), plus ``FP8_PV_UNFUSED`` with the S gate when ``model`` is whitelisted for the unfused sites; returns True.  Any other model gets the default fp16 core
    (returns False, nothing changed).  The caller must read :func:`take_overflow` after a forward and, when set, call :func:`rescale_all` and redo it.  Undo with
    :func:`disable_fp8_pv`."""
    global FP8_PV, FP8_PV_FUSED, FP8_VSCALE_AUTO, FP8_PV_HL, FP8_PV_UNFUSED, FP8_PV_MIN_S, FP8_PV_FRONT_MIN_S
    routes = FP8_PV_WHITELIST.get(model)
    if routes is None:
        return False
    FP8_PV, FP8_PV_FUSED, FP8_VSCALE_AUTO, FP8_PV_HL = True, "fused" in routes, True, bool(hl)
    FP8_PV_UNFUSED = "unfused" in routes
    FP8_PV_MIN_S = dict(FP8_PV_UNFUSED_MIN_S) if FP8_PV_UNFUSED else {}
    FP8_PV_FRONT_MIN_S = dict(FP8_PV_UNFUSED_FRONT_MIN_S) if FP8_PV_UNFUSED else {}
    return True


def disable_fp8_pv() -> None:
    global FP8_PV, FP8_PV_FUSED, FP8_VSCALE_AUTO, FP8_PV_HL, FP8_PV_UNFUSED, FP8_PV_MIN_S, FP8_PV_FRONT_MIN_S
    FP8_PV = FP8_PV_FUSED = FP8_VSCALE_AUTO = FP8_PV_HL = FP8_PV_UNFUSED = False
    FP8_PV_MIN_S = {}
    FP8_PV_FRONT_MIN_S = {}


# inference(): the front writes q / k / v and the bias in fp16 for the fp16 core (faster, and closer to fp32: see triattn_h16_sm90.cu)
INFERENCE_FP16 = True
#: inference(): the tail projects the gate from x itself (g = LN(x) . Wg^T in fp32, ``tri_tail(wg=...)``) and the front skips g -- a
#: [R, 128] bf16 store and its load fewer (2026-11-13)
INFERENCE_TAIL_GATE = True


def supports(length: int) -> bool:
    """Lengths the kernels take: whole 128-token tiles."""
    return length > 0 and length % TOK == 0


def attention(q, k, v, bias, scale=D**-0.5, want_lse=False, rows=2, defs=()):
    """softmax(scale q.k + bias) v on [B, N, S, H, 32] tensors, bias [B, H, S, S] -> (out, lse, flags)."""
    return _core(rows, tuple(defs)).triattn_fwd(q, k, v, scaled_bias(bias, scale), scale, want_lse)


def key_tiles(mask, length, tile=TOK):
    """The fp16 core's ``key_tiles``: int32 [B], the ``tile``-key tiles (128; the h100_dh core: 64) of each batch element up to its last
    attended key (at least 1), or None without a mask.  The tiles past them hold keys masked in every row (-inf in the bias): the core
    skips them (same bits, fewer tiles; the requirement inputs pad the last L // 8 keys, the kit's core skips them too)."""
    if mask is None:
        return None
    pos = torch.arange(1, length + 1, device=mask.device, dtype=torch.int32)
    last = (mask.reshape(-1, length).to(torch.int32) * pos).amax(-1)  # 1 + the last attended key, 0 when none
    return ((last + (tile - 1)) // tile).clamp_min(1).to(torch.int32)


#: the fp16 core counts the live key tiles from the key mask itself up to this many batch elements (KM_MAXB of triattn_h16_sm90.cu)
KEY_MASK_MAXB = 64


def key_args(mask, length):
    """The fp16 core's key-tile arguments: ``key_mask`` (the bool [B, L] mask as it is: the kernel counts each element's live 128-key tiles,
    no glue kernels -- seven small launches of ``key_tiles`` were a fifth of a d_pair 64 L128 call) when it fits the kernel's table, else
    ``key_tiles``; nothing without a mask."""
    if mask is None:
        return {}
    m = mask.reshape(-1, length)
    if m.dtype == torch.bool and m.is_contiguous() and m.shape[0] <= KEY_MASK_MAXB and m.storage_offset() % 16 == 0:
        return {"key_mask": m}
    return {"key_tiles": key_tiles(mask, length)}


def scaled_bias(bias, scale=D**-0.5):
    """The core's bias operand: bf16(bias / scale) [B, H, S, S] (the module's front writes it directly)."""
    return (bias.float() / scale).to(torch.bfloat16).contiguous()


@functools.lru_cache(maxsize=None)
def _mod(defs: tuple[str, ...] = ()):
    """The module kernels around the core (front, tail, backward); `defs` = extra -D knobs (experiments)."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_mod{tag}_v11",
        sources=[str(_SRC / "triattn_mod_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


def pack_front(wq, wk, wv, wg, wb):
    """[Wq; Wk; Wv; Wg] -> bf16 [512, 128]; Wb -> bf16 [16, 128] (zero rows below the 4 heads)."""
    w4 = torch.cat([wq, wk, wv, wg]).to(torch.bfloat16).contiguous()
    wb16 = wb.new_zeros(16, wb.shape[1], dtype=torch.bfloat16)
    wb16[: wb.shape[0]] = wb.to(torch.bfloat16)
    return w4, wb16


def flip_bias(sbias, mask):
    """The ending node on AlphaFold 3's bias orientation, from the front's output: the transpose of the bias the front projected
    from the transposed pair, with the key mask applied afterwards (to the transposed axis).  ``sbias`` is [B, H, L, L] bf16 made
    with no mask; masked keys are -inf, as the front writes them."""
    out = sbias.transpose(-1, -2)
    if mask is not None:
        out = out.masked_fill(~mask[:, None, None, :], float("-inf"))
    return out.contiguous()


#: d_pair -> heads (x 32) of the wide front (``tri_front_wide``, inference): the fused d_pair-128 front's design at a width whose four
#: projections no longer fit one CTA.  Training at these widths runs the width-generic kernels (``h100_sm90_generic``).
WIDE = {256: 8}
#: the wide front writes q | k | v | g as column blocks of one [M, 4 C] buffer (the width-generic GEMM's layout) instead of four tensors
WIDE_PACKED = False


def wide(d_pair: int, d_hidden: int, n_head: int) -> bool:
    """(d_pair, d_hidden, heads) served by the wide fused inference path."""
    return WIDE.get(d_pair) == n_head and d_hidden == d_pair


def inference(
    pair, lnw, lnb, w4, wb16, wo16, mask=None, eps=1e-5, trans=False, out=None, flip=False, gate_bias=None, out_bias=None, fold=None
):
    """pair + TriangleAttention(pair) for the starting node, no grad: front (LN + projections + bias) -> core -> tail (gate +
    out-projection + residual).  w4, wb16 = pack_front(...); wo16 = Wo in bf16; `out`: a contiguous tensor like pair to write into.  ``flip``: the ending node (``trans``) takes its
    bias from the pair as it is (:func:`flip_bias`). ``gate_bias`` [128] / ``out_bias`` [C]: biases of the gate / out-projection
    (RoseTTAFold3), added to the front's gate and to the result.  At a ``WIDE`` width (w4, wb16, wo16, fold = ``_pack_params(..., lnw, lnb)``,
    the wide pack) the call runs :func:`inference_wide`."""
    if pair.shape[-1] != C:
        return inference_wide(pair, fold, w4, wb16, wo16, mask, eps, trans, out, flip, gate_bias, out_bias)
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m = _mod()
    empty = torch.empty(0, device=pair.device)
    fused_vt = _fused_vt(pair.device) and L >= FP8_PV_FUSED_MIN_S * (2 if FP8_PV_HL else 1)
    if fused_vt:
        ovf = _OVF.setdefault(pair.device, torch.zeros(1, dtype=torch.int32, device=pair.device))
        vscale = FP8_VSCALE if FP8_VSCALE is not None else _VS_AUTO.get(w4.data_ptr())
        if vscale is None:  # calibration: the V absmax per head of this call
            v0 = m.tri_front(x, lnw, lnb, eps, w4, wb16, empty.bool(), B, L, D**-0.5, trans, True, False)[2]
            vscale = _VS_AUTO[w4.data_ptr()] = vscale_from_amax(v0.view(-1, H, D).abs().amax(dim=(0, 2)), 4.0)
            _forget_on_free(w4, w4.data_ptr())
        vscale = vscale.to(pair.device, torch.float32)
        q, k, v, g, sbias, vt = m.tri_front_vt(
            x,
            lnw,
            lnb,
            eps,
            w4,
            wb16,
            mask if mask is not None and not flip else empty.bool(),
            B,
            L,
            D**-0.5,
            trans,
            (1.0 / vscale).contiguous(),
            ovf,
            2 if FP8_PV_HL else 1,
        )
    else:
        q, k, v, g, sbias = m.tri_front(
            x,
            lnw,
            lnb,
            eps,
            w4,
            wb16,
            mask if mask is not None and not flip else empty.bool(),
            B,
            L,
            D**-0.5,
            trans,
            INFERENCE_FP16,
            not INFERENCE_TAIL_GATE,
        )
    if flip:
        sbias = flip_bias(sbias, mask)
    shp = (B, L, L, H, D)
    if fused_vt:
        o = _core_f8pv().triattn_fwd_h16(
            q.view(shp), k.view(shp), k.view(shp), sbias, D**-0.5, False, None, True, vt=vt, vscale=vscale.repeat(B), **key_args(mask, L)
        )[0]
    elif INFERENCE_FP16:  # the dead trailing key tiles skipped
        o = core_h16().triattn_fwd_h16(q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, False, **key_args(mask, L))[0]
    else:
        o = _core().triattn_fwd(q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, False)[0]
    # the biases in fp32 inside the tail, before its one rounding: added to the bf16 gate / sum afterwards they were a second
    # rounding each, the out bias at the residual's scale (RoseTTAFold3's trunk drifted from its reference by it)
    fp32 = lambda t: None if t is None else t.float().contiguous()  # noqa: E731
    if INFERENCE_TAIL_GATE:  # w4 = [Wq; Wk; Wv; Wg]: rows 3C.. are Wg (contiguous)
        gate = dict(wg=w4[3 * C :], lnw=lnw, lnb=lnb, eps=eps)
        return m.tri_tail(
            empty.bfloat16(), o.view(-1, C), x, wo16, empty.bfloat16(), L, trans, out, bo=fp32(out_bias), gb=fp32(gate_bias), **gate
        ).view(B, L, L, C)
    return m.tri_tail(g, o.view(-1, C), x, wo16, empty.bfloat16(), L, trans, out, bo=fp32(out_bias), gb=fp32(gate_bias)).view(B, L, L, C)


@torch.autocast("cuda", enabled=False)
def inference_wide(pair, fold, wpk, wb8, wo16, mask=None, eps=1e-5, trans=False, out=None, flip=False, gate_bias=None, out_bias=None):
    """``inference`` at a ``WIDE`` width (d_pair 256, 8 heads x 32): the wide front (LN + q | k | v | g + head-major bias in one kernel,
    ``tri_front_wide``) -> the core with the gate in its epilogue (u = sigmoid(g) o O, as the width-generic path) -> x + u Wo^T [+ bo]
    (cublasLt, the residual in its epilogue: the generic path's out-projection).  wpk, wb8, wo16, fold = ``_pack_params(..., lnw, lnb)``:
    the LayerNorm is folded into the pack (W diag(gamma) and per-output constants), so the front's GEMM operand is the pair as loaded."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90_generic as gen

    B, L, _, cw = pair.shape
    nh = cw // D
    M = B * L * L
    x = pair.contiguous().view(M, cw)
    empty = torch.empty(0, device=pair.device)
    gb = None if gate_bias is None else gate_bias.float().contiguous()
    # opt-in FP8 PV route: the group-2 CTA of the wide front writes the e4m3 V^T hi plane itself (static scale; the hi + lo planes stay on the standalone pass)
    vtargs = front_vt_args(pair.device, L, nh, D, B, cw) if INFERENCE_FP16 and nh == 8 and L % 128 == 0 else None
    calib = vtargs == CALIBRATE  # the site's static V scale is not taken yet (the front runs as it is; the scale comes from this call's V)
    if calib:
        vtargs = None
    if vtargs is not None and vtargs[2] != 1:
        vtargs = None
    res = _mod().tri_front_wide(
        x,
        fold,
        eps,
        wpk,
        wb8,
        mask if mask is not None and not flip else empty.bool(),
        B,
        L,
        nh,
        D**-0.5,
        trans,
        INFERENCE_FP16,
        gb,
        WIDE_PACKED,
        *((vtargs[0], vtargs[1]) if vtargs is not None else ()),
    )
    q, k, v, g, sbias = res[:5]
    vt, vs = (res[5], vtargs[3]) if vtargs is not None else (None, None)
    if calib:
        vt, vs = quantize_v_static(v.view(B, L, L, nh, D), _SITE)
    if flip:
        sbias = flip_bias(sbias, mask)
    shp = (B, L, L, nh, D)
    if vt is not None:  # the FP8 core on the front's V^T plane
        u = _core_f8pv().triattn_fwd_h16(
            q.view(shp),
            k.view(shp),
            k.view(shp),
            sbias,
            D**-0.5,
            False,
            g.view(shp),
            False,
            **key_args(mask, L),
            vt=vt,
            vscale=vs,
        )[3]
    elif INFERENCE_FP16:  # the dead trailing key tiles skipped
        u = core_h16().triattn_fwd_h16(
            q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, False, g.view(shp), False, **key_args(mask, L)
        )[3]
    else:
        u = _core().triattn_fwd(q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, False, g.view(shp), False)[3]
    bo = None if out_bias is None else out_bias.to(torch.bfloat16).contiguous()
    return gen._lt().lt_mm(u.view(M, cw), wo16, x, L, trans, out, bo).view(B, L, L, cw)


@functools.lru_cache(maxsize=None)
def _bwd(defs: tuple[str, ...] = ()):
    """The core backward kernels (delta, key side, query side); `defs` = extra -D knobs (experiments)."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_bwd{tag}_v12",
        sources=[str(_SRC / "triattn_bwd_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


def _pack_params(wq, wk, wv, wg, wb, wo, lnw=None, lnb=None):
    """(w4, wb16, wo16) in one launch when the six parameters share a dtype (fp32 master or bf16); the torch pack otherwise.
    At a ``WIDE`` width: (wpk, wb8, wo16, fold) of ``tri_pack_wide`` with the LayerNorm (``lnw`` / ``lnb``) folded in (parameters of
    mixed dtypes are cast to bf16 first)."""
    if wq.shape[1] != C:
        ws = (wq, wk, wv, wg, wb, wo)
        if len({t.dtype for t in ws}) != 1 or wq.dtype not in (torch.float32, torch.bfloat16):
            ws = tuple(t.to(torch.bfloat16) for t in ws)
        return _mod().tri_pack_wide(*(t.contiguous() for t in ws), lnw.float().contiguous(), lnb.float().contiguous())
    if (
        len({t.dtype for t in (wq, wk, wv, wg, wb, wo)}) == 1
        and wq.dtype in (torch.float32, torch.bfloat16)
        and all(t.is_contiguous() for t in (wq, wk, wv, wg, wb, wo))
    ):
        return _mod().tri_pack(wq, wk, wv, wg, wb, wo)
    return _pack_params_torch(wq, wk, wv, wg, wb, wo)


def _pack_params_torch(wq, wk, wv, wg, wb, wo):
    """``_pack_params`` by torch statements only (no extension launch): the fallback, and the form a derivation on FakeTensors records."""
    w4, wb16 = pack_front(wq, wk, wv, wg, wb)
    return w4, wb16, wo.to(torch.bfloat16).contiguous()


def train_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need=True, trans=False):
    """The module forward with the tensors its backward reads: -> (out [B, L, L, 128], (q, k, v, g, sbias, o, lse)); o / lse are empty
    unless `need`.  Every returned tensor is a fresh allocation (none aliases an argument), so it can be an output of a custom op."""
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m = _mod()
    w4, wb16, wo16 = _pack_params(wq, wk, wv, wg, wb, wo)
    lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
    empty = torch.empty(0, device=pair.device)
    q, k, v, g, sbias = m.tri_front(
        x,
        lw,
        lb,
        eps,
        w4,
        wb16,
        mask if mask is not None else empty.bool(),
        B,
        L,
        D**-0.5,
        trans,
    )
    shp = (B, L, L, H, D)
    o, lse, _, _ = _core().triattn_fwd(q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, need)
    drop = drop if drop is not None else empty.bfloat16()
    out = m.tri_tail(g, o.view(-1, C), x, wo16, drop, L, trans)
    return out.view(B, L, L, C), (q, k, v, g, sbias, o, lse)


def train_backward(dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, eps, drop, saved, trans=False):
    """Gradients (pair, lnw, lnb, wq, wk, wv, wg, wb, wo) of `train_forward`, each in its input's dtype."""
    q, k, v, g, sbias, o, lse = saved
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m, bw = _mod(), _bwd()
    w4, _, wo16 = _pack_params(wq, wk, wv, wg, wb, wo)
    lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
    empty = torch.empty(0, device=pair.device)
    drop = drop if drop is not None else empty.bfloat16()
    dbias = torch.zeros(B, H, L, L, device=x.device, dtype=torch.float32)  # the query-side kernel adds its partials in
    dy = dy.contiguous().view(-1, C)
    do, dg, delta, dpair, dwo_part = m.tri_gate_bwd(dy, g, o.view(-1, C), wo16, drop, B, L, trans)
    shp = (B, L, L, H, D)
    q5, k5, v5, do5 = q.view(shp), k.view(shp), v.view(shp), do.view(shp)
    dk, dv = bw.triattn_bwd_kv(q5, k5, v5, sbias, do5, lse, delta, D**-0.5)
    dq = bwd_q(bw, q5, k5, v5, sbias, do5, lse, delta, D**-0.5, dbias)
    dq, dk, dv = dq.view(-1, C), dk.view(-1, C), dv.view(-1, C)
    m.tri_head_bwd(dq, dk, dv, dg, dbias, x, w4, wb, lw, eps, dpair, B, L, trans)
    # fp32 outputs whenever any parameter is fp32 (master weights, or the bf16 model's fp32 LayerNorm affine): the wrapper's `.to` then
    # rounds the bf16 ones exactly as the kernel's own bf16 store would, and gamma / beta keep their fp32 value
    fp32 = wq.dtype == torch.float32 or lnw.dtype == torch.float32
    dwq, dwk, dwv, dwg, dgamma, dbeta, dwb, dwo = m.tri_wgrad(dq, dk, dv, dg, dbias, x, eps, B, L, w4, wb, lw, lb, dwo_part, fp32, trans)
    # separate buffers from the kernel (no slices: the outputs of a custom op may not alias each other); .to is a no-op in the common case
    return (
        dpair.view(B, L, L, C),
        dgamma.to(lnw.dtype),
        dbeta.to(lnb.dtype),
        dwq.to(wq.dtype),
        dwk.to(wk.dtype),
        dwv.to(wv.dtype),
        dwg.to(wg.dtype),
        dwb.to(wb.dtype),
        dwo.to(wo.dtype),
    )


class TriAttnFused(torch.autograd.Function):
    """The whole module (residual included): pair [B, L, L, 128] -> pair + drop o TriangleAttention(pair), starting node.
    Parameters may be bf16 or fp32 (master weights); the gradients come back in their dtype.
    drop: the module's broadcast dropout scale as [B, L, 128] bf16 (index = the row's last position), or None.
    (The module path goes through the custom ops of ``kernels.integrations.triattn_h100_sm90``, which call `train_forward` / `train_backward`.)"""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop):
        need = any(ctx.needs_input_grad)
        out, saved = train_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need)
        if need:
            ctx.save_for_backward(
                pair,
                lnw,
                lnb,
                wq,
                wk,
                wv,
                wg,
                wb,
                wo,
                drop if drop is not None else pair.new_empty(0),
                *saved,
            )
            ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved = ctx.saved_tensors
        grads = train_backward(
            dy,
            pair,
            lnw,
            lnb,
            wq,
            wk,
            wv,
            wg,
            wb,
            wo,
            ctx.eps,
            drop if drop.numel() else None,
            saved,
        )
        return (*grads, None, None, None)


def triangle_attention(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None):
    """pair + drop o TriangleAttention(pair) for the starting node; the caller transposes for the ending node."""
    return TriAttnFused.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop)


#: Query-side backward launches: tasks = (b, h, 128-query tile) x slabs, each a whole tile's pair rows (its dbias partials
#: reduce-added in order).  A launch of many samples was slower per sample than launches of a few (B48: 1.07-1.55x over H 4-16 and
#: L128-768, 2026-10-20): the tasks after the first wave start out of step with the other query tiles of their (b, h), which then
#: read the same K / V rows from DRAM at different times.  So B samples run in launches of ``g`` samples whose tasks make one wave of
#: ~96 CTAs (one sample, any g >= 1: the single launch as before).
BWD_Q_WAVE = 96


def bwd_q(bw, q, k, v, sbias, do, lse, delta, scale, dbias, dq=None):
    """dq of the core's query side (dbias partials added into ``dbias``), the B samples in launches of ``g`` samples (see BWD_Q_WAVE)."""
    B, _, S, H, _ = q.shape
    g = max(1, BWD_Q_WAVE // (H * (S // TOK)))
    if g >= B:
        return bw.triattn_bwd_q(q, k, v, sbias, do, lse, delta, scale, dbias, dq)
    if dq is None:
        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    for i in range(0, B, g):
        s = slice(i, min(B, i + g))
        bw.triattn_bwd_q(q[s], k[s], v[s], sbias[s], do[s], lse[s], delta[s], scale, dbias[s], dq[s])
    return dq
