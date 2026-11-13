"""The RTX 5090's (sm_120a) batched GEMMs of the MSA pair-weighted averaging (``sm80/gemm_sm120.cu``: CUTLASS 2.x sm_80 tensor-op GEMMs within
the card's 99 KB of shared memory).  cuBLAS's heuristic picks a 32 x 32 WMMA kernel there for the key contraction at L 128..640 (half the
throughput it reaches from L 768 on); these GEMMs serve every L on the 5090 instead.  The A100 / H100 paths never load this extension.

  contract     o[b] = w[b / ns] v[b]        (transpose: w[b / ns]^T v[b])   w [8, L, L], v [8 ns, L, K] bf16 -> bf16
  weight_grad  dw[b] = d[b] v[b]^T                                         d, v [B, L, K] bf16 -> fp32 [B, L, L]
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent / "sm80"

#: tile configuration per product (``MWE_CFGS`` in the source): (TM, TN, TK, warp M, warp N, stages)
CFGS = {
    0: (128, 128, 32, 64, 64, 4),
    1: (128, 256, 32, 64, 64, 3),
    2: (256, 128, 32, 64, 64, 3),
    3: (128, 128, 64, 64, 64, 3),
    4: (64, 128, 64, 32, 64, 4),
    5: (64, 64, 64, 32, 32, 6),
}


@functools.lru_cache(maxsize=1)
def ext():
    from miniworld_engine.kernels._nvcc import cutlass_include, ensure_cuda_home, host_flags, load_extension
    from miniworld_engine.kernels._sm120_builds import SM120

    ensure_cuda_home()
    return load_extension(
        name="pwa_gemm_sm120",
        sources=[str(_dir / "gemm_sm120.cu")],
        extra_include_paths=[str(_dir), str(cutlass_include())],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", SM120, "--expt-relaxed-constexpr", "-DMWE_SM120=1"],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def contract_cfg(length: int, k: int, batches: int) -> int:
    """The tile of o = w v (M = L, N = K, K = L; measured on the 5090 over L 128..2048, K 2048..32768): 128 x 128 for the short rows (N <= 3072, or
    L <= 384 with N <= 6144), 128 x 256 past them (2-5 % ahead there)."""
    return 0 if k <= 3072 or (length <= 384 and k <= 6144) else 1


#: dw = d v^T (M = N = L, K = S C / ns: few output tiles, a long K) on the 5090, by L: (ns > 1 batches, one batch per head). The tile that keeps
#: the most SMs busy at that L (one CTA per SM for every tile here); -1 is cuBLAS (its split-K wins for L 128 with one batch per head).
_DW = {128: (5, -1), 256: (0, 5), 384: (3, 4), 512: (5, 0), 640: (3, 5), 768: (3, 0), 1024: (3, 5)}


def weight_grad_cfg(length: int, k: int, batches: int) -> int:
    """The tile of dw (``_DW``; L between the measured ones: the next measured L's, 128 x 128 past 1024); -1: cuBLAS."""
    for top in sorted(_DW):
        if length <= top:
            return _DW[top][0 if batches > 8 else 1]
    return 3


def contract(w: torch.Tensor, v: torch.Tensor, transpose: bool = False, cfg: int | None = None) -> torch.Tensor:
    """w [8, L, L], v [8 ns, L, K] bf16 -> [8 ns, L, K] bf16: w[b / ns] v[b] (``transpose``: w[b / ns]^T v[b])."""
    out = torch.empty_like(v)
    ext().contract(w, v, out, transpose, contract_cfg(v.shape[1], v.shape[2], v.shape[0]) if cfg is None else cfg)
    return out


def weight_grad(d: torch.Tensor, v: torch.Tensor, cfg: int | None = None) -> torch.Tensor:
    """d, v [B, L, K] bf16 -> fp32 [B, L, L] = d[b] v[b]^T (cuBLAS where ``weight_grad_cfg`` says -1)."""
    cfg = weight_grad_cfg(d.shape[1], d.shape[2], d.shape[0]) if cfg is None else cfg
    if cfg < 0:
        return torch.bmm(d, v.transpose(1, 2), out_dtype=torch.float32)
    out = torch.empty((d.shape[0], d.shape[1], d.shape[1]), dtype=torch.float32, device=d.device)
    ext().weight_grad(d, v, out, cfg)
    return out


def mm(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    cfg: int = 0,
    c: torch.Tensor | None = None,
    live: torch.Tensor | None = None,
    ch: int = 1,
) -> torch.Tensor:
    """op(a) @ op(b) (+ c, fp32 sum rounded once), row-major bf16 -> bf16 [M, N]; with a 3-D operand a batch [B, M, N] (a 2-D operand shared)."""
    m = a.shape[-1] if trans_a else a.shape[-2]
    n = b.shape[-2] if trans_b else b.shape[-1]
    batch = a.shape[0] if a.dim() == 3 else b.shape[0] if b.dim() == 3 else None
    out = a.new_empty((m, n) if batch is None else (batch, m, n))
    empty = a.new_empty((0,))
    ext().mm(a, b, empty if c is None else c, out, trans_a, trans_b, cfg, empty.view(torch.uint8) if live is None else live, ch)
    return out


def vg_head_major(y: torch.Tensor, wvg: torch.Tensor, depth: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The BIG width's [v | g] = y [Wv; Wg]^T (y [S L, 256], wvg [512, 256]) -> (v head-major [8, L, S 32], g [S L, 256]): one GEMM whose
    epilogue stores v head-major (no to_hm pass)."""
    rows = depth * length
    flat = y.new_empty((2 * rows * 256,))
    ext().vg_head_major(y, wvg, flat, depth, length)
    return flat[: rows * 256].view(8, length, depth * 32), flat[rows * 256 :].view(rows, 256)


def chai_wo_grads(dwp: torch.Tensor, dbo: torch.Tensor, wo: torch.Tensor, gam: torch.Tensor, bet: torch.Tensor):
    """The Chai-1 OuterProductMean's out-projection gradients in one kernel (W' = W gamma, b' = b + W beta): (dWo, dgamma, dbeta) fp32 from
    dW' [CZ, P] / dbo [CZ] fp32, W [CZ, P], gamma / beta [P]."""
    dwo = torch.empty_like(dwp)
    dgam = dwp.new_empty((dwp.shape[1],))
    dbet = dwp.new_empty((dwp.shape[1],))
    ext().chai_wo_grads(dwp, dbo, wo.contiguous(), gam.float().contiguous(), bet.float().contiguous(), dwo, dgam, dbet)
    return dwo, dgam, dbet


def live_tokens(mask: torch.Tensor, depth: int, length: int) -> torch.Tensor:
    """[S N] bool / uint8 MSA mask -> [N] uint8: 1 where some MSA row keeps the token (one small kernel)."""
    live = torch.empty((length,), dtype=torch.uint8, device=mask.device)
    ext().live_tokens(mask.reshape(depth, length).contiguous(), live)
    return live


def colsum2(part: torch.Tensor, split: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Column sums of fp32 [R, C] partials as two separate tensors (columns [0, split) and [split, C)), one launch."""
    out_a = part.new_empty((split,))
    flat = part.reshape(part.shape[0], -1)
    out_b = part.new_empty((flat.shape[1] - split,))
    ext().colsum2(flat, out_a, out_b)
    return out_a, out_b


def wgrad_parts(a: torch.Tensor, b: torch.Tensor, cfg: int, splits: int) -> torch.Tensor:
    """The split-K fp32 partials [splits, M, N] of a^T b (``wgrad`` without the sum)."""
    part = torch.empty((splits, a.shape[1], b.shape[1]), dtype=torch.float32, device=a.device)
    ext().wgrad_splitk(a, b, part, cfg)
    return part
