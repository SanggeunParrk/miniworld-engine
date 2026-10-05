"""A100 (sm_80) hand-CUDA MSA pair-weighted averaging, forward and backward (bf16; 8 heads of 8 / 16 / 32 channels, d_msa 64 / 128, d_pair 128 / 256 / 384).

    out = msa + dropout( Wo ( sigmoid(LN(msa) Wg^T) * ( softmax_j( LN(pair) Wb^T ) . (LN(msa) Wv^T) ) ) )

The contraction over the keys, o[h] = w[h] v[h] with w [8, L, L] and v [8, L, S C] (head-major, so that it is a plain batched GEMM: M = N-runs of S C), is cuBLAS; the
kernels (``sm80/``) are everything around it:

  pair_fwd   LN(pair), the 8-head projection, key mask, softmax over the keys -> w                     (one CTA per query row)
  ln_v       LN(msa) . Wv -> v head-major, the LayerNorm statistics                                    (persistent: 128 MSA rows of one token per tile)
  gate_out   sigmoid gate, output projection, row dropout, residual -> out                               (persistent: query row x 128 MSA rows per tile)

``integrations/pwa_sm80.py`` drives them from ``MSAPairWeightedAveraging.forward``; the extension is built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

HEADS = 8
D_MSA = (64, 128)
D_PAIR = (128, 256, 384)
D_HIDDEN = (8, 16, 32)
MAX_LEN = 1024                             # the pair kernel keeps a row of logits per head in shared memory


@functools.lru_cache(maxsize=1)
def ext():
    """The extension (built on first use into ``TORCH_EXTENSIONS_DIR``)."""
    ensure_cuda_home()
    return load_extension(
        name="pwa_sm80", sources=[str(_dir / "ops.cu")], extra_include_paths=[str(_dir)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-lineinfo", *gencodes("80"), "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False)


def supported(d_msa: int, d_pair: int, n_head: int, d_hidden: int) -> bool:
    """(d_msa, d_pair, heads, per-head width) the kernels are instantiated for. d_msa 128 with 32-wide heads would need a 128 x 256 weight-gradient accumulator per CTA: not built."""
    return n_head == HEADS and d_msa in D_MSA and d_pair in D_PAIR and d_hidden in D_HIDDEN and d_msa * d_hidden <= 16 * 128 and not (d_msa == 128 and d_hidden == 32)


def _none(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.new_empty((0,), dtype=dtype)


def pair_fwd(z: torch.Tensor, key_mask: torch.Tensor | None, lnw: torch.Tensor, lnb: torch.Tensor, wb: torch.Tensor, eps: float) -> torch.Tensor:
    """z [L, L, DZ] bf16, key_mask [L] uint8 (or None), lnw / lnb fp32 [DZ], wb [8, DZ] bf16 -> w [8, L, L] bf16 (softmax over the last axis)."""
    length = z.shape[0]
    w = z.new_empty((HEADS, length, length))
    ext().pair_fwd(z, _none(z, torch.uint8) if key_mask is None else key_mask, lnw, lnb, wb, w, eps)
    return w


SPLIT_ENV = "MINIWORLD_PWA_SM80_SPLIT"


def pick_split(depth: int) -> int:
    """The number of chunks the MSA rows are split in for the cuBLAS contractions (head-major tensors [8 ns, L, S C / ns]: one batch per chunk).  o = w v and dv = w^T dO run 15-20 %
    faster as 8 ns short batches than as 8 long ones, and the weight-gradient product dw = dO v^T (K = S C) 20-30 % (its fp32 partials are summed by ``pair_bwd``): 4 when S is a
    multiple of 512, 2 when it is a multiple of 256, else 1 (8 is not better: the partials of dw double).  ``MINIWORLD_PWA_SM80_SPLIT=<n>`` forces n (1: off; 8 allowed)."""
    forced = os.environ.get(SPLIT_ENV)
    if forced is not None:
        n = int(forced)
        return n if n in (1, 2, 4, 8) and depth % (128 * n) == 0 else 1
    for n in (4, 2):
        if depth % (128 * n) == 0:
            return n
    return 1


def ln_v(m: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wv: torch.Tensor, eps: float, save_stats: bool, ns: int = 1):
    """m [S, L, D] bf16, wv [8 C, D] bf16 -> (v [8 ns, L, S C / ns] bf16 (ns = 1: [8, L, S C]), stats [S, L, 2] fp32 or an empty tensor)."""
    depth, length, _ = m.shape
    c = wv.shape[0] // HEADS
    v = m.new_empty((HEADS * ns, length, depth // ns * c))
    stats = m.new_empty((depth, length, 2), dtype=torch.float32) if save_stats else _none(m, torch.float32)
    ext().ln_v(m, lnw, lnb, wv, v, stats, eps)
    return v, stats


def gate_out(m: torch.Tensor, o: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, keep: torch.Tensor | None, eps: float,
             dscale: float) -> torch.Tensor:
    """m [S, L, D], o [8 ns, L, S C / ns] (= w v; ns = 1: [8, L, S C]), wg [8 C, D], wo [D, 8 C], keep [L, D] bf16 0 / 1 (or None) -> m + dropout(Wo (sigmoid(LN(m) Wg^T) o)) [S, L, D] bf16."""
    out = torch.empty_like(m)
    ext().gate_out(m, o, lnw, lnb, wg, wo, _none(m, torch.bfloat16) if keep is None else keep, out, eps, dscale)
    return out


# ---------------------------------------------------------------------------------------------------------------- backward
def reduce_rows(part: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The rows of an fp32 [R, W] partial buffer summed in order (W a multiple of 4) -> [W] in ``dtype`` (fp32 or bf16): no atomics, bit-reproducible."""
    out = part.new_empty((part.shape[1],), dtype=dtype)
    ext().reduce_rows(part, out)
    return out


def glue(m: torch.Tensor, dres: torch.Tensor, o: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, keep: torch.Tensor | None,
         eps: float, dscale: float):
    """m, dres [S, L, D] (dres: the output gradient, dropout not yet applied), o [8 ns, L, S C / ns] (the saved contraction), wg [8 C, D], wo [D, 8 C], keep [L, D] bf16 (or None) ->
    (do, dgp [8 ns, L, S C / ns] bf16, dWo [D, 8 C], dWg [8 C, D] fp32): the cotangent of o (dv = w^T do, dw = do v^T are the caller's bmm), the gate's pre-activation gradient and
    the two weight gradients of the gate / output projections (fp32 partials of the CTAs summed in a fixed order)."""
    depth, length, d = m.shape
    hc = wg.shape[0]
    gch = min(hc, 64)
    ng = hc // gch
    nbx = ext().bwd_grids(depth, length, d, hc // HEADS)[0]
    do = torch.empty_like(o)
    dgp = torch.empty_like(o)
    part = m.new_empty((nbx, ng, 2 * d * gch), dtype=torch.float32)
    ext().glue(m, dres, o, lnw, lnb, wg, wo, _none(m, torch.bfloat16) if keep is None else keep, do, dgp, part, eps, dscale)
    total = reduce_rows(part.view(nbx, -1), torch.float32).view(ng, 2 * d * gch)
    dwo = total[:, :d * gch].reshape(ng, d, gch).permute(1, 0, 2).reshape(d, hc)
    dwg = total[:, d * gch:].reshape(hc, d)
    return do, dgp, dwo, dwg


def dgv_bwd(m: torch.Tensor, dres: torch.Tensor, dgp: torch.Tensor, dv: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wg: torch.Tensor, wv: torch.Tensor, eps: float):
    """dgp, dv [8 ns, L, S C / ns] -> (dm bf16 [S, L, D] = LN_backward(dgp Wg + dv Wv) + dres, dWv [8 C, D], dgamma [D], dbeta [D]) (fp32 but dm)."""
    depth, length, d = m.shape
    hc = wg.shape[0]
    nb = ext().bwd_grids(depth, length, d, hc // HEADS)[1]
    dm = torch.empty_like(m)
    part = m.new_empty((nb, hc * d + 2 * d), dtype=torch.float32)
    ext().dgv_bwd(m, dres, dgp, dv, lnw, lnb, wg, wv, dm, part, eps)
    total = reduce_rows(part, torch.float32)
    return dm, total[:hc * d].view(hc, d), total[hc * d:hc * d + d], total[hc * d + d:]


def pair_bwd(z: torch.Tensor, w: torch.Tensor, dw: torch.Tensor, key_mask: torch.Tensor | None, lnw: torch.Tensor, lnb: torch.Tensor, wb: torch.Tensor, eps: float,
             variant: int | None = None):
    """z [L, L, DZ], w [8, L, L] bf16 (the saved softmax), dw [8 ns, L, L] fp32 (the cotangent of w: ns partial products per head, summed in order), key mask [L] uint8 (or None) ->
    (dz bf16 [L, L, DZ], dWb [8, DZ], dgamma, dbeta [DZ]) (fp32 but dz). dbeta is zero up to rounding: a softmax cannot see a shift shared by all keys."""
    length, _, dz_ = z.shape
    dz = torch.empty_like(z)
    pm = z.new_empty((length, HEADS * dz_), dtype=torch.float32)
    ps = z.new_empty((length, HEADS), dtype=torch.float32)
    if variant is None:
        variant = int(os.environ.get("MINIWORLD_PWA_SM80_PB", "0"))         # schedule experiments (CTAs per SM); 0 is the shipped one
    ext().pair_bwd(z, w, dw, _none(z, torch.uint8) if key_mask is None else key_mask, lnw, lnb, wb, dz, pm, ps, eps, variant)
    big_m = reduce_rows(pm, torch.float32).view(HEADS, dz_)
    s = reduce_rows(ps, torch.float32)
    wbf = wb.float()
    return dz, lnw[None, :] * big_m + lnb[None, :] * s[:, None], (wbf * big_m).sum(0), (wbf * s[:, None]).sum(0)
