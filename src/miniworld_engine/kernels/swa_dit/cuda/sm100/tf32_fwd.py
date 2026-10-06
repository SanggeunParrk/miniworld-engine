"""sm_100a (B200) forward of the fused SWA atom DiT block in fp32: every stage on hand-written tcgen05 / TMEM / TMA kernels with TF32
tensor-core MMAs (``tcgen05.mma kind::tf32``, fp32 accumulation), every elementwise step in fp32. No Triton, no PyTorch kernels
besides the cached weight rounding.

    modulation  silu(c) Wmod^T                                         mod_fwd_tf32.cu   (swa_mod_fwd_tf32_sm100)
    qkvg        RMS-adaLN, q | k | v | gate projections, head RMS, RoPE qkvg_fwd_tf32.cu  (swa_qkvg_fwd_tf32_sm100)
    attention   window attention |i - j| <= 64, fp32 softmax, lse      attn_fwd_tf32.cu  (swa_attn_fwd_tf32_sm100)
    ffn         gated out-projection, residual, RMS-adaLN, SwiGLU,      ffn_fwd_tf32.cu   (swa_ffn_fwd_tf32_sm100)
                residual

The same equations as the Triton fp32 path (``triton/forward_fp32.py``) with two differences of precision, both toward fp32: the window
attention runs on TF32 Q / K / V / P (the Triton path rounds them to bf16, as FlashAttention-4 does), and every GEMM is single-pass
TF32 with operands ROUNDED to the nearest TF32 (the kernels round their activation operands with ``cvt.rna``; :func:`_round_tf32`
rounds the weights once per weight version) -- the tensor core alone would truncate the low 13 mantissa bits, a bias toward zero that
does not average out over K (that truncation is why the Triton path needs "tf32x3" for its FFN GEMMs).

Saved for the backward (``save``), all fp32: Qh, Kh, Vh head-major [N, 4, S, 32] (after head RMS + RoPE, TF32-valued: exactly the
attention's operands), G [N S, C] (raw gate, pre-sigmoid), O [N S, C] (attention output, pre-gate), lse [N, 4, S] (natural log of the
sum of exp of the scaled scores; 0 on rows without a valid key), q1, X (the TF32-rounded qkvg MMA operand), PQ, PK (raw projections,
pre head RMS), Att (raw out-projection, pre-gate_a), Y (fp32, unrounded), FF (raw down-projection) [N S, C] -- the meanings of the
bf16 sm_100a saves (``sm100.block_fwd``).

Served (:func:`supported_tf32`): B200, fp32, C = 128 with 4 heads of 32, SwiGLU hidden 256, half window 64 (not global), fp32 eps, the
atom count a multiple of 128 (else the Triton fp32 path serves the call). The cubins are built on first use by the newest nvcc that knows
sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached under ``MINIWORLD_ENGINE_JIT_ROOT`` (``swa_dit_sm100_tf32/``);
the launches go through the CUDA driver on torch's current stream (programmatic dependent launch between the stages), CUDA-graph
capturable. Page: ``docs/gpus/b200/swa_atom_dit/swa_atom_dit.md`` ("fp32 (TF32) path").
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import device_constant
from miniworld_engine.kernels.swa_dit.cuda import sm100 as _sm

C, H, D, NHID, HW = _sm.C, _sm.H, _sm.D, _sm.NHID, _sm.HW
EPS = _sm.EPS
PDL = True                                  # programmatic dependent launch between the forward kernels
_dir = Path(__file__).parent

#: cubin name -> (source stem, extra nvcc flags)
CUBINS = {
    "mod_fwd_tf32": ("mod_fwd_tf32", ()),
    "qkvg_fwd_tf32": ("qkvg_fwd_tf32", ()),
    "attn_fwd_tf32": ("attn_fwd_tf32", ()),
    "ffn_fwd_tf32": ("ffn_fwd_tf32", ()),
}
#: dynamic shared memory per kernel (the sources' SMEM / SMEM_BYTES)
SMEM = {"mod_fwd_tf32": 8 * 16384 + 64, "qkvg_fwd_tf32": 8 * 16384 + 12 * 8192 + 512,
        "attn_fwd_tf32": 2 * 81920 + 2048 + 256, "ffn_fwd_tf32": 8 * 16384 + 12 * 8192 + 2048 + 512}
#: kernel function per cubin
FUNCS = {"mod_fwd_tf32": "swa_mod_fwd_tf32_sm100", "qkvg_fwd_tf32": "swa_qkvg_fwd_tf32_sm100",
         "attn_fwd_tf32": "swa_attn_fwd_tf32_sm100", "ffn_fwd_tf32": "swa_ffn_fwd_tf32_sm100"}


# --------------------------------------------------------------------------------------------------- build / load
@functools.lru_cache(maxsize=None)
def cubin(name: str) -> str:
    """Path of the cubin ``name`` (see CUBINS), built on first use; rebuilt only when a source, a flag or the toolchain changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    stem, extra = CUBINS[name]
    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}", *extra)
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "swa_dit_sm100_tf32" / f"{name}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(_dir / f"{stem}.cu"), "-o", str(tmp)], capture_output=True, text=True,
                             timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {stem}.cu {' '.join(extra)}:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


def _load(name: str):
    """The kernel of cubin ``name``, loaded in the current device's context."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.Kernel(cubin(name), FUNCS[name], SMEM[name], pdl=PDL)


def _f32map(t, dims, strides, box, swizzle=128):
    return _sm._map(t, dims, strides, box, swizzle=swizzle, dtype="f32")


def tiling(A: int) -> tuple[int, int]:
    """(SP augments, AT atoms) of a <= 128-row tile: SP = min(A, 16), AT = 128 // SP (the modulation rows are read from global memory,
    so any AT works: A = 1 takes 128 atoms per tile)."""
    SP = min(A, 16)
    return SP, 128 // SP


def _rows_map(t, A, B, S, SP, AT):
    """[A, B, S, 128] fp32 rows -> 4-D TMA map, box (32 channels, AT atoms, 1, SP augments), 128-B swizzle (one k-block per load)."""
    return _f32map(t, [C, S, B, A], [C * 4, S * C * 4, B * S * C * 4], [32, AT, 1, SP])


# --------------------------------------------------------------------------------------------------- kernels (host side)
class ModFwdTf32:
    """silu(c) Wmod^T: c [R, C] fp32 contiguous, Wmod [6C, C] fp32 (TF32-rounded) -> [R, 6C] fp32."""

    def __init__(self):
        self.k = _load("mod_fwd_tf32")

    def bind(self, c, wmod, out=None):
        R = c.shape[0]
        out = torch.empty(R, 6 * C, device=c.device, dtype=torch.float32) if out is None else out
        maps = (_f32map(c, [C, R], C * 4, [32, 128]), _f32map(wmod, [C, 6 * C], C * 4, [32, 128]),
                _f32map(out, [6 * C, R], 6 * C * 4, [32, 128]))

        def run():
            self.k(((R + 127) // 128, 6, 1), (128, 1, 1), *maps)
        run.keep = maps
        return run, out


class QkvgFwdTf32:
    """q [N, S, C] fp32 (N = A B), mod [B S, 6C] fp32, cos / sin [B S, D/2] fp32, W = [Wqkv; Wg] [4C, C] fp32 (TF32-rounded) ->
    Qh, Kh, Vh [N, H, S, D] fp32 (TF32-valued), G [N S, C] fp32 and, with save, X, PQ, PK [N S, C] fp32."""

    def __init__(self):
        self.k = _load("qkvg_fwd_tf32")

    def bind(self, q, mod, cos, sin, W, A, B, save=False):
        N, S, _ = q.shape
        SP, AT = tiling(A)
        dev = q.device
        f32 = torch.float32
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=f32) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=f32)
        X, PQ, PK = ((torch.empty(N * S, C, device=dev, dtype=f32) for _ in range(3)) if save else (G, G, G))
        maps = (_rows_map(q, A, B, S, SP, AT), _f32map(W, [C, 4 * C], C * 4, [32, 64]))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        NG = 1                                                             # projection groups: spread small problems over the SMs
        while NG < 4 and ntile * NG * 2 <= _sm.nsm():
            NG *= 2
        grid = (min(_sm.nsm(), ntile * NG), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), int(NG),
                   float(EPS), float(EPS), int(save), mod, cos, sin, Qh, Kh, Vh, G, X, PQ, PK)
        run.keep = (maps, W)
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
        maps = (_f32map(Qh.view(rows, D), [D, rows], D * 4, [D, 128]),
                _f32map(Kh.view(rows, D), [D, rows], D * 4, [D, 256]),
                _f32map(Vh.view(rows, D), [D, rows], D * 4, [D, 256], swizzle="128a32"))   # MN-major PV operand
        items = N * (S // 128) * H
        grid = (min(_sm.nsm(), items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, O, LSE, int(S), int(N), float(D ** -0.5))
        run.keep = maps
        return run, O, LSE


class FfnFwdTf32:
    """q, G, O [N S, C] fp32, mod [B S, 6C] fp32, Wo [C, C], Wab [2 NHID, C] (rows [Wu[32 j ..]; Wu[NHID + 32 j ..]] per 32-unit
    chunk j), Wd [C, NHID] (all TF32-rounded) -> out and, with save, q1, att, y, ffn [N S, C] fp32."""

    def __init__(self):
        self.k = _load("ffn_fwd_tf32")

    def bind(self, q, g, o, mod, wo, wab, wd, A, B, save=False):
        M = q.numel() // C
        S = M // (A * B)
        SP, AT = tiling(A)
        dev = q.device
        out = torch.empty(M, C, device=dev, dtype=torch.float32)
        Q1, Att, Y, Ff = ((torch.empty(M, C, device=dev, dtype=torch.float32) for _ in range(4)) if save else (out, out, out, out))
        maps = (_rows_map(g, A, B, S, SP, AT), _rows_map(o, A, B, S, SP, AT), _rows_map(q, A, B, S, SP, AT),
                _f32map(wo, [C, C], C * 4, [32, 64]), _f32map(wab, [C, 2 * NHID], C * 4, [32, 64]),
                _f32map(wd, [NHID, C], NHID * 4, [32, 64]))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(_sm.nsm(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS),
                   int(save), mod, out, Q1, Att, Y, Ff)
        run.keep = maps
        return run, (out, Q1, Att, Y, Ff)


class _Kernels:
    """Every TF32 forward kernel, loaded on one device."""

    def __init__(self) -> None:
        self.mod, self.qkvg, self.attn, self.ffn = ModFwdTf32(), QkvgFwdTf32(), AttnFwdTf32(), FfnFwdTf32()

    def stats(self) -> dict[str, tuple[int, int]]:
        """kernel -> (registers per thread, local memory bytes per thread: > 0 means spills)."""
        return {name: (k.k.regs, k.k.lmem) for name, k in
                (("mod_fwd_tf32", self.mod), ("qkvg_fwd_tf32", self.qkvg), ("attn_fwd_tf32", self.attn), ("ffn_fwd_tf32", self.ffn))}


@functools.lru_cache(maxsize=16)
def kernels_tf32(index: int) -> _Kernels:
    """Build and load every TF32 forward kernel on device ``index`` (raises on a toolchain or driver failure)."""
    with torch.cuda.device(index):
        return _Kernels()


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
    return (q.is_cuda and q.dtype == torch.float32 and q.dim() == 3 and q.shape[-1] == C and q.shape[1] > 0 and q.shape[1] % 128 == 0
            and nhid == NHID and half_window == HW and eps == EPS and is_b200(q.device))


# --------------------------------------------------------------------------------------------------- weight forms (cached per version)
def _round_tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 holding the nearest TF32 value, ties away from zero (what ``cvt.rna.tf32.f32`` gives): the weights' MMA operand."""
    i = t.contiguous().view(torch.int32)
    return ((i + 0x1000) & -0x2000).view(torch.float32)


def _tf32_w_qkvg(wqkv, wg):
    return _round_tf32(torch.cat([wqkv, wg]))


def _tf32_w(w):
    return _round_tf32(w)


def _tf32_wab(wu):
    """rows per 32-unit hidden chunk j: [Wu[32 j .. 32 j + 31]; Wu[NHID + 32 j ..]] (dispatch._pack_ffn's wab), TF32-rounded."""
    nh = wu.shape[0] // 2
    return _round_tf32(wu.reshape(2, nh // 32, 32, wu.shape[1]).permute(1, 0, 2, 3).reshape(2 * nh, wu.shape[1]))


# --------------------------------------------------------------------------------------------------- the block
def block_fwd_tf32(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window=HW):
    """The fp32 forward on TF32 tensor cores: [out] or, with ``save``, [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF] (all
    fp32) -- the layout of ``dispatch.swa_dit_block_fwd``. q [N, S, C] fp32 contiguous, mod [B S, 6C] fp32, cos / sin [B S, D/2] fp32,
    seqused [N] int32, the five weights fp32."""
    N, S, _ = q.shape
    _sm.need_s128(S)
    assert half_window == HW, "the TF32 path serves the window (half window 64) only"
    A = N // B
    with torch.cuda.device(q.device):
        K = kernels_tf32(_index(q))
        # the weight forms first: their (cache-miss) torch kernels must finish before the PDL chain starts
        W = _sm._cached(_tf32_w_qkvg, wqkv, wg)
        WO = _sm._cached(_tf32_w, wo)
        WAB = _sm._cached(_tf32_wab, wu)
        WD = _sm._cached(_tf32_w, wd)
        run, (Qh, Kh, Vh, G, X, PQ, PK) = K.qkvg.bind(q, mod, cos, sin, W, A, B, save=save)
        run()
        O = torch.empty(N * S, C, device=q.device, dtype=torch.float32)
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        run, _, _ = K.attn.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
        run()
        run, (out, q1, att, y, ffn) = K.ffn.bind(q.reshape(N * S, C), G, O, mod, WO, WAB, WD, A, B, save=save)
        run()
    if not save:
        return [out.view(N, S, C)]
    return [out.view(N, S, C), Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, att, y, ffn]


def mod_fwd_tf32(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """silu(c) Wmod^T [R, 6C] fp32 on TF32 tensor cores; c [R, C] fp32 contiguous, Wmod [6C, C] fp32."""
    with torch.cuda.device(c.device):
        K = kernels_tf32(_index(c))
        run, out = K.mod.bind(c, _sm._cached(_tf32_w, wmod))
        run()
    return out
