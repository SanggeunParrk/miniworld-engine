"""fp32 (kind::tf32) GEMMs of the token DiT's few-row fp32 inference on a CTA pair (``tdt_pair32_sm100.cu``, sm_100a only; round 3
part 9, docs/b200/token-dit.md): the 640-row GEMMs of a five-sample L 128 step with their row pass as the epilogue.

    mm(a, w, bias, rnac)            out = a w^T + bias, columns < rnac rounded to TF32 (the q | k | v of the TF32 attention core)
    swiglu(x, wpk, h, up)           h = silu(a) b [u] of [a | b | u] = x W^T, W packed by ``pack_swiglu`` ([a | b] never reaches HBM)
    resgate_out(h, ws, x, gl, P)    out = x + sigmoid(gl[row % P]) (h ws^T)       (the block's last residual)

The weights are expected TF32-rounded (``round_tf32``, once per static pack): the tensor cores read the TF32 bits of each operand."""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path

import torch

_dir = str(Path(__file__).parent)
_STP = [None]  # (timing builds, -DSTAMP=1: the stamp buffer address)


class _Epi(ctypes.Structure):
    _fields_ = [
        ("out", ctypes.c_void_p),
        ("ldo", ctypes.c_long),
        ("bias", ctypes.c_void_p),
        ("rnac", ctypes.c_int),
        ("x", ctypes.c_void_p),
        ("ldx", ctypes.c_long),
        ("gl", ctypes.c_void_p),
        ("ldg", ctypes.c_long),
        ("P", ctypes.c_int),
        ("stp", ctypes.c_void_p),
    ]


def _smem(hr: int, nh: int, kd: int, kbx: int, maxstg: int = 8) -> int:
    ns = kd // 32 // kbx
    ast, wst = kbx * hr * 128, kbx * nh * 128
    stg = min(ns, min((232448 - 2048) // (ast + wst), maxstg))
    return 1024 + stg * (ast + wst) + 256


@functools.lru_cache(maxsize=None)
def _kernel(device_index: int, hr: int, nh: int, kd: int, kbx: int, epi: int, ng: int):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    defs = (f"HR={hr}", f"NH={nh}", f"KD={kd}", f"KBX={kbx}", f"EPI={epi}", f"NG={ng}")
    defs += tuple(filter(None, os.environ.get("MINIWORLD_PAIR32_DEFS", "").split(",")))  # (timing builds: NOLOAD=1, NOMMA=1, MAXSTG=n)
    return sm100._sm100_kernel(
        "tdt_pair32_sm100", "tdt_pair32", device_index, pdl=True, src_dir=_dir, defs=defs, cluster=2, smem=_smem(hr, nh, kd, kbx)
    )


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _map(t: torch.Tensor, rows: int, box_rows: int, kbx: int):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    K = t.shape[1]
    assert t.dtype is torch.float32 and t.stride(1) == 1 and (t.stride(0) * 4) % 16 == 0 and t.data_ptr() % 16 == 0
    return sm100._tm(t, [32, rows, K // 32], [t.stride(0) * 4, 128], [32, box_rows, kbx], dtype="f32")


def _ptr(t):
    return None if t is None else t.data_ptr()


def _env(name: str, default: int) -> int:
    return int(os.environ.get(f"MINIWORLD_PAIR32_{name}", str(default)))


def _launch(a, w, epi, e, cfg):
    hr, nh, kbx, ng = cfg
    M, K = a.shape
    N = w.shape[0]
    assert N % (2 * nh) == 0 and K % (32 * kbx) == 0, (N, K, cfg)
    npn = N // (2 * nh)
    tiles = -(-M // (2 * hr))
    k = _kernel(_index(a), hr, nh, K, kbx, epi, ng)
    k((tiles * npn * 2, 1, 1), (128 + 128 * ng, 1, 1), _map(a, M, hr, kbx), _map(w, N, nh, kbx), M, npn, e)


#: (HR, NH, KBX, NG) per GEMM role, from the sweep (``_r3/m/p9g.py``)
CFG = {
    "plain": (128, 128, 2, 1),  # q | k | v | g, 640 x 3072 x 768 (in the step, A/B: 128 x 128 and 128 x 64 level, 64 x 128 slower)
    "wo": (64, 32, 4, 1),  # 640 x 768 x 768: 5.94 (cuBLAS 6.75)
    "swiglu_up": (128, 96, 2, 1),  # 640 x 4608 x 768 + SwiGLU: 15.56 (cuBLAS + swiglu_rows 16.38)
    "swiglu": (64, 64, 2, 1),
    "resgate": (64, 64, 2, 1),
}
for _r in CFG:  # (sweeps: MINIWORLD_PAIR32_CFG_<role>=HR,NH,KBX,NG)
    if os.environ.get(f"MINIWORLD_PAIR32_CFG_{_r}"):
        CFG[_r] = tuple(int(v) for v in os.environ[f"MINIWORLD_PAIR32_CFG_{_r}"].split(","))


def defs_of(role: str, kd: int = 768) -> tuple[str, ...]:
    """the -D flags of a role's build (``_kernel``'s, for the prebuild plan)"""
    hr, nh, kbx, ng = CFG[role]
    epi = {"plain": 0, "wo": 0, "swiglu_up": 1, "swiglu": 2, "resgate": 3}[role]
    return (f"HR={hr}", f"NH={nh}", f"KD={kd}", f"KBX={kbx}", f"EPI={epi}", f"NG={ng}")


def prebuild_calls(d: int = 768, up: bool = False):
    """the cubins ``token_inference`` loads on the pair path of a d-wide fp32 block (q | k | v | g, the out projection, the up-gate
    expand)"""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    roles = (
        ("plain", "swiglu_up") if up else ("plain",)
    )  # (the out projection stays on cuBLAS by default: MINIWORLD_TOKEN_B200_PAIR32 bit 2)
    return [functools.partial(sm100.cubin, "tdt_pair32_sm100", defs_of(r, d), src_dir=_dir) for r in roles]


def mm(a, w, bias=None, out=None, rnac=0, cfg=None):
    M, N = a.shape[0], w.shape[0]
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.float32)
    e = _Epi(_ptr(out), out.stride(0), _ptr(bias), int(rnac), None, 0, None, 0, 1, _STP[0])
    _launch(a, w, 0, e, cfg or CFG["plain"])
    return out


def swiglu(x, wpk, h, up=True, cfg=None):
    """h = silu(a) b [u] (rounded to TF32), wpk from ``pack_swiglu`` with the same NH."""
    e = _Epi(_ptr(h), h.stride(0), None, 0, None, 0, None, 0, 1, _STP[0])
    _launch(x, wpk, 1 if up else 2, e, cfg or CFG["swiglu_up" if up else "swiglu"])
    return h


def resgate_out(h, ws, x, gl, P, out, cfg=None):
    """out = x + sigmoid(gl[row % P]) (h ws^T); x [M, D] fp32, gl [P, D] rows (any row stride)."""
    e = _Epi(_ptr(out), out.stride(0), None, 0, _ptr(x), x.stride(0), _ptr(gl), gl.stride(0), int(P), _STP[0])
    _launch(h, ws, 3, e, cfg or CFG["resgate"])
    return out


def round_tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 values rounded to TF32 (nearest, ties away: cvt.rna), as a new tensor."""
    i = t.contiguous().view(torch.int32)
    r = (i + 0x1000) & ~0x1FFF
    keep = (i & 0x7F800000) == 0x7F800000  # inf / nan unchanged
    return torch.where(keep, i, r).view(torch.float32)


def pack_swiglu(ws, nh, up=True):
    """[Wa; Wb (; Wu)] ([nW H, K], H hidden units) regrouped into weight halves of NH rows, each [Wa | Wb (| Wu)] of U = NH / nW
    consecutive units: the layout the SwiGLU epilogue reads. TF32-rounded."""
    nw = 3 if up else 2
    H = ws.shape[0] // nw
    U = nh // nw
    assert nh % nw == 0 and H % U == 0
    parts = ws.view(nw, H // U, U, ws.shape[1]).transpose(0, 1).reshape(nw * H, ws.shape[1])
    return round_tf32(parts)


__all__ = ["CFG", "mm", "pack_swiglu", "resgate_out", "round_tf32", "swiglu"]
