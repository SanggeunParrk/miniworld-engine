"""Chai-1 OuterProductMean (``groups = 8``, d_hidden 64, product LayerNorm) on A100 (sm_80): ``sm80/chai_sm80.cu``, the passes of the H100 ``chai.cu``
on mma.sync / cp.async, with the same entry points and contracts as ``cuda.chai_proj`` / ``chai_epilogue`` / ``chai_bwd`` / ``chai_projbwd`` (the
driver ``outer_product_mean/chai.py`` picks the build by the card)."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent / "sm80"


@functools.lru_cache(maxsize=None)
def ext():
    """``chai_sm80.cu``, sm_80. Built on first use (or prebuilt: ``kernels/integrations`` plans it for the grouped OPM rows)."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    from foldjet_engine.kernels import _msa_card as _card

    ensure_cuda_home()
    return load_extension(
        name=_card.name("opm_chai_sm80"),
        sources=[str(_dir / "chai_sm80.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=_card.cflags([*host_flags(), "-O3", "-std=c++17", *gencodes("80")]),
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def chai_epilogue(o, wfold, bias, residual=None, save=False, eps=0.1):
    if _sm120():  # the RTX 5090 build takes the column sums of W' (cached while the folded weights are the same tensor)
        from foldjet_engine.kernels import _capture

        (colsum,) = _capture.param_forms(
            "chai_colsum_sm120", [wfold], lambda: (wfold.float().sum(1).contiguous(),), capture_reads_eager=not save
        )
        return ext().chai_epilogue(o, wfold, bias, residual, bool(save), float(eps), colsum)
    return ext().chai_epilogue(o, wfold, bias, residual, bool(save), float(eps))


def chai_proj(x, mask, gamma, beta, w, eps=1e-5, save=False):
    return ext().chai_proj(x, mask, gamma, beta, w, float(eps), bool(save))


def chai_bwd(dy, y, stats, n):
    return ext().chai_bwd(dy, y, stats, int(n))


def chai_projbwd(x, mask, da, db, gamma, beta, stats, wlr):
    return ext().chai_projbwd(x, mask, da, db, gamma, beta, stats, wlr)


def _per_group(s: int, n: int) -> bool:
    """A100: whether the Chai-1 products run as eight per-group skip-kernel launches (``ogemm_sm80.outer``) instead of cuBLAS's bmm / the batched launch. cuBLAS's
    bmm falls to a slow kernel when the MSA is deep and the pair long (A-B-A-B, L2 flushed, bmm vs per-group ms: S 1024 L 1280 8.2 vs 8.5, L 1536 12.3 vs 12.1,
    L 2048 28.2 vs 21.2; S 2048 L 640 3.9 vs 4.0, L 768 5.9 vs 5.75, L 1024 13.0 vs 10.1; S 4096 L 512 4.8 vs 5.1, L 640 8.4 vs 7.8, L 768 13.4 vs 11.3; S 8192 L 768 27.1 vs
    22.4; S <= 512 cuBLAS leads to L 2048)."""
    return n >= (1536 if s < 2048 else 768 if s < 4096 else 640) and s >= 1024


def chai_products(a, b, mask, s, n):
    """O [8, N K, N K] = a_g^T b_g per group (a, b [8, S N, 8]): with a mask the CUTLASS product that skips the tokens masked in every MSA row
    (``ogemm_sm80``; their blocks are exactly 0), one launch per group, from L 384; else the cuBLAS batched GEMM."""
    import torch

    if (
        _sm120() and mask is not None and not (s >= 1024 and n <= 256)
    ):  # S >= 1024 to N 256: cuBLAS (``_products_cfg``) leads the live-token launch (probe, L2 flushed, S 1024: N 128 110 vs 119 us, N 256 391 vs 383-405 us)
        # the RTX 5090: one batched launch that writes the dead-token tiles as zeros (256 x 128 tiles, 64 x 128 at N 128)
        live = _gemm_sm120().live_tokens(mask, s, n)  # (10-15 % ahead of cuBLAS / the per-group masked product at L 128..512)
        return _gemm_sm120().mm(a.view(8, s, n * 8), b.view(8, s, n * 8), True, False, 2 if n >= 256 else 4, live=live, ch=8)
    if mask is None or n < 384:  # short lengths: the eight launches cost more than the skipped blocks save (measured L128 1.4x)
        cfg = _products_cfg(s, n) if _sm120() else -1  # the RTX 5090: the CUTLASS batched GEMM where it beats cuBLAS (-1: cuBLAS)
        if cfg >= 0:
            return _gemm_sm120().mm(a.view(8, s, n * 8), b.view(8, s, n * 8), True, False, cfg)
        return torch.bmm(a.view(8, s, n * 8).transpose(1, 2), b.view(8, s, n * 8))
    from . import ogemm_sm80

    mk = mask.view(s, n)
    if not _sm120() and _per_group(s, n):  # the eight per-group CUTLASS launches (the previous source's path), live or padded
        live = ogemm_sm80.live_tokens(mk.view(torch.uint8) if mk.dtype is torch.bool else mk.to(torch.uint8))
        o = a.new_empty((8, n * 8, n * 8))
        for g in range(8):
            ogemm_sm80.outer(a[g].view(s, n * 8), b[g].view(s, n * 8), live, 8, o[g])
        return o
    if ogemm_sm80.all_live(
        mk.view(torch.uint8) if mk.dtype is torch.bool else mk.to(torch.uint8)
    ):  # nothing to skip: the cuBLAS batched GEMM
        return torch.bmm(a.view(8, s, n * 8).transpose(1, 2), b.view(8, s, n * 8))
    live = ogemm_sm80.live_tokens(mk.view(torch.uint8) if mk.dtype is torch.bool else mk.to(torch.uint8))
    return ogemm_sm80.outer_batched(
        a.view(8, s, n * 8), b.view(8, s, n * 8), live, 8
    )  # one launch for the eight groups (8 launches ran 3 waves each, the last one 0.67 full)


def chai_bwd_products(a, b, do, s, n):
    """(dA, dB) [8, S, N K] = (b_g do_g^T, a_g do_g) per group (a, b [8, S N, 8], do [8, N K, N K]): cuBLAS batched GEMMs on the A100, the
    CUTLASS batched GEMM on the RTX 5090."""
    import torch

    a3, b3 = a.view(8, s, n * 8), b.view(8, s, n * 8)
    cfg = _bwd_cfg(s, n) if _sm120() else -1
    if cfg >= 0:  # the RTX 5090: the CUTLASS batched GEMM where it beats cuBLAS
        g = _gemm_sm120()
        do = do.contiguous()
        return g.mm(b3, do, False, True, cfg), g.mm(a3, do, False, False, cfg)
    return torch.bmm(b3, do.transpose(1, 2)), torch.bmm(a3, do)


def _sm120() -> bool:
    from foldjet_engine.kernels import _msa_card

    return _msa_card.sm120()


def _gemm_sm120():
    from foldjet_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120

    return gemm_sm120


def _products_cfg(s: int, n: int) -> int:
    """The RTX 5090's tile of O = a_g^T b_g (``gemm_sm120.CFGS``; M = N = 8 N, K = S) where the masked product does not serve, measured at S 384 /
    1024: cuBLAS (-1) at S >= 1024 to N 256 (its 64 x 64 kernel leads there), 64 x 128 x 64 at N 128, 128 x 128 below N 384, 256 x 128 past it."""
    if s >= 1024 and n <= 256:
        return -1
    return 4 if n <= 128 else 0 if n < 384 else 2


def _bwd_cfg(s: int, n: int) -> int:
    """The RTX 5090's tile of dA = b_g do_g^T / dB = a_g do_g (M = S, N = K = 8 N): cuBLAS (-1) at S >= 1024 to N 256, 64 x 128 x 64 to N 384,
    128 x 128 x 64 past it."""
    if s >= 1024 and n <= 256:
        return -1
    return 4 if n <= 384 else 3
