"""CUDA passes of AttentionPairBias (the Pairformer single track) on B200 (``apb_rows.cu``): the pair -> bias projection
with its LayerNorm (mma.sync), its backward, and the row passes of the single track (LayerNorm, sigmoid gate, their
backwards). Its attention cores are this family's sm_100a kernels (``../sm100``). Built on first use (``load_extension``),
never at import; ``integrations/attention_pair_bias_b200.py`` drives them.

The sm_100 build also carries the B200 host path (``-DAPB_SM100_HOST``): the torch ops ``apb_b200::train_fwd`` / ``train_bwd`` /
``infer`` run a whole op's launch sequence in C++ (``host_ops`` loads them, registers their fake implementations and the
resolver through which they fetch the attention cubins on first use)."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent
RW = 4                    # rows per block of the row kernels (a warp each)
DP = 128                  # d_pair
# (heads, d_single) -> (real head dim, head width in memory: 16 x 24 is padded to 32)
GEOMETRY = {(8, 384): (48, 48), (12, 384): (32, 32), (16, 384): (24, 32), (24, 384): (16, 16), (16, 512): (32, 32)}


def width(heads: int, d: int = 384) -> int:
    """Row width of q / k / v / g / O and their gradients: d_single, or 512 for 16 x 24 (padded to 32)."""
    return heads * GEOMETRY[(heads, d)][1]


def acc_n(heads: int, d: int = 384, dp: int = DP) -> int:
    """The backward's fp32 accumulator: dlnw [d] | dlnb [d] | dbq [W] | dWf [heads, dp] | per-head dbias sums [heads] (dp: d_pair,
    128, or 256 with 16 heads: Chai-1)."""
    return 2 * d + width(heads, d) + heads * dp + heads


def ext(shadow: bool = False):
    """The row / pair extension. ``shadow`` (H100 only) also builds the fp32 pair-weight oracle (``apb_h100_exact_dwb``), which
    needs CUTLASS 4.2 headers (``CUTLASS_PATH``); the default build has no such dependency."""
    arch = "90a" if torch.cuda.get_device_capability() == (9, 0) else "100"
    return _ext(arch, shadow and arch == "90a")


@functools.lru_cache(maxsize=None)
def _ext(arch: str, shadow: bool = False):
    from ...._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    includes, shadow_flags = [], []
    if shadow:
        root = os.environ.get("CUTLASS_PATH")
        if not root or not (Path(root) / "include/cutlass/cutlass.h").is_file():
            raise RuntimeError("H100 APB requires CUTLASS_PATH pointing to CUTLASS 4.2 headers")
        includes = [str(Path(root) / "include")]
        shadow_flags = ["-DAPB_H100_FP32_SHADOW=1"]
    if arch == "100":
        shadow_flags = ["-DAPB_SM100_HOST=1"]
    return load_extension(
        name=f"apb_rows_cuda_sm{arch}" + ("_shadow" if shadow else ""),
        sources=[str(_dir / "apb_rows.cu")],
        extra_include_paths=includes,
        extra_ldflags=["-lcublas", "-lcublasLt"],
        extra_cuda_cflags=[*host_flags(), *shadow_flags, "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes(arch, ptx=() if arch == "90a" else ("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def available() -> bool:
    ext()
    return True


def _resolve_core(key: str) -> tuple[int, int]:
    """The C++ host path's cubin lookup: key "<kind>/<device>/<heads>/<d_single>/<variant>" -> (CUfunction handle, dynamic smem).
    Kinds fwd2 / inf (variant 1: the key-split build), dkv (variant 1: the query-split build), dqb (-DDQPART)."""
    from ..sm100 import _apb_defs, _sm100_kernel

    kind, dev, heads, d, variant = key.split("/")
    stem, qpair = {"fwd2": ("attn_fwd2", True), "inf": ("attn_inf", True), "dkv": ("attn_dkv", False), "dqb": ("attn_dqb", False)}[kind]
    defs = _apb_defs(int(heads), qpair, int(d), variant == "1" and qpair) + (("DQPART=1",) if kind == "dqb" else ())
    if kind == "dkv" and variant == "1":
        defs += ("QSPLIT=1",)
    k = _sm100_kernel(stem, f"augattn_{stem.removeprefix('attn_')}_sm100", int(dev), defs=defs)
    try:
        handle = int(k.func)
    except TypeError:
        handle = int(k.func.getPtr())
    return handle, int(k.smem)


def _fake_train_fwd(single, pair, mask, params, eps_s, eps_p, ns, gate_bias=0.0):
    S, L, D = single.shape
    H, dp = params[9].shape
    W, M, bf, f32 = width(H, D), S * L, torch.bfloat16, torch.float32
    e = single.new_empty
    return [e(single.shape), e((M, D), dtype=bf), e((M, 2), dtype=f32), e((M, 4 * W), dtype=bf), e((S * H, L, L), dtype=bf),
            e((M, W), dtype=f32), e((S * H, L), dtype=f32), e((M, W), dtype=bf), e((H, dp), dtype=bf), e((4 * W, D), dtype=bf),
            e((D, W), dtype=bf), e((H, dp), dtype=bf)]


def _fake_train_bwd(single, pair, mask, params, saved, dy, eps_p, nq):
    return [torch.empty_like(single), torch.empty_like(pair), *(single.new_empty(t.shape, dtype=torch.float32) for t in params)]


def _fake_infer(single, pair, mask, packs, eps_s, eps_p, ns):
    return single.new_empty(single.shape)


_HOST_READY = False


def host_ops() -> bool:
    """Load the sm_100 build with the host path, register the fakes of its ops and the cubin resolver (once per process)."""
    global _HOST_READY
    if not _HOST_READY:
        e = _ext("100")
        e.set_resolver(_resolve_core)
        for name, fake in (("train_fwd", _fake_train_fwd), ("train_bwd", _fake_train_bwd), ("infer", _fake_infer)):
            torch.library.register_fake(f"apb_b200::{name}", fake)
        _HOST_READY = True
    return True


def pair_bias(pair2d: torch.Tensor, wf: torch.Tensor, mask: torch.Tensor | None, L: int, eps: float, neg: float) -> torch.Tensor:
    """bias [S H, L, L] bf16 = wf . LN(pair) (no LN affine: fold its weight into ``wf`` [H, 128] bf16, H = 8, 12, 16 or 24), ``neg``
    on the keys where ``mask`` [S L] is False. pair2d [S L L, 128] bf16 contiguous (S samples), L a multiple of 16."""
    out = torch.empty(pair2d.shape[0] // L * wf.shape[0] // L, L, L, device=pair2d.device, dtype=torch.bfloat16)
    ext().pair_bias_fwd(pair2d, wf, mask, out, int(L), float(eps), float(neg))
    return out


def pair_bias_bwd(pair2d: torch.Tensor, dbias: torch.Tensor, wf: torch.Tensor, L: int, eps: float, acc: torch.Tensor | None = None,
                  d: int = 384):
    """(dpair [L L, d_pair] bf16, dWf [H, d_pair] fp32, per-head dbias sums [H]) of ``pair_bias`` (without the mask: masked keys
    carry zero dbias); d_pair 128, or 256 with 16 heads (read off ``wf``). dWf and the sums are added into ``acc`` (``acc_n(H, d,
    d_pair)`` floats, zeroed by the caller) when given; the returned dWf / sums are views of it."""
    heads, dp = wf.shape
    if acc is None:
        acc = torch.zeros(acc_n(heads, d, dp), device=pair2d.device)
    dz = torch.empty_like(pair2d)
    ext().pair_bias_bwd(pair2d, dbias.contiguous(), wf, dz, acc, int(L), float(eps), int(d))
    if acc.ndim == 2:
        return dz, None, None  # Hopper block partials are reduced by finalize.
    o = 2 * d + width(heads, d)
    return dz, acc[o:o + heads * dp].view(heads, dp), acc[o + heads * dp:]


def pair_bias_stats(pair2d: torch.Tensor, wf: torch.Tensor, mask: torch.Tensor | None, L: int, eps: float, neg: float):
    """``pair_bias`` that also returns the LayerNorm statistics of every pair row, fp32 [L L, 2] (mean, rstd) -- the backward of the d_pair 256 stack reads
    them instead of recomputing; ``wf`` [8, 256] runs the 256-wide ring (8 heads), [H, 128] the 128-wide one."""
    out = torch.empty(pair2d.shape[0] // L * wf.shape[0] // L, L, L, device=pair2d.device, dtype=torch.bfloat16)
    st = torch.empty(pair2d.shape[0], 2, device=pair2d.device, dtype=torch.float32)
    ext().pair_bias_fwd_stats(pair2d, wf, mask, out, st, int(L), float(eps), float(neg))
    return out, st


__all__ = ["DP", "GEOMETRY", "RW", "acc_n", "available", "ext", "host_ops", "pair_bias", "pair_bias_bwd", "pair_bias_stats", "width"]
