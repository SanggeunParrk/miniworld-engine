"""Row kernels of the hoisted pair bias (``kernels/integrations/bias_only_dit_hoist.py``, B200), bf16 and fp32, pair width 128 or 256.

  forward    LN0(pair): the token DiT rows' ``layernorm128_rows`` (one warp per row, no affine; the inference hoist's kernel), its
             ``layernorm_rows`` (the same kernel template) for 256 channels
  backward   ``bias_only_dit_hoist_rows.cu`` ln0_bwd_rows: d pair from d LN0 (fp32) and the pair rows, the row statistics
             recomputed with the forward's arithmetic, and the dW' GEMM's operand made from the same fp32 LN0 (bf16: the two-term
             split [hi | lo], 2 C columns; fp32: LN0 itself)

Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
#: switches of the exact-fp32 fusions (A-B tests in one process; all on in a release): "logits" saves the fp32 logits (no re-forming
#: in the backward), "bwd" the one-pass hoist backward of a block, "fwd" the one-pass logits of a block
#: devices of the tensor-core logits / hoist backward (TF32 mma.sync, split fp32 operands): H100 and B200 (mma.sync runs on sm_100)
_TC_CAPS = ((9, 0), (10, 0))
OPT = {
    "logits": True,
    "bwd": True,
    "fwd": True,
    "rows": True,
    "bwd_rg": 2,
    "eval_tc": True,
    "sm100_logits": True,
    "train_tc": True,
    "bwd_tc": True,
    "bwd_mb": 0,
    "tc_mb": 0,
}


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_hoist_rows_cuda",
        sources=[str(_dir / "bias_only_dit_hoist_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("80", "90", "100", ptx=("100",)),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def ln0_rows(x: torch.Tensor, out: torch.Tensor, eps: float = 1e-5) -> None:
    """out = LN(x) without affine; x [R, C] (bf16 / fp32 rows, C = 128 or 256), out contiguous [R, C] (bf16 / fp32)."""
    from foldjet_engine.kernels.conditioned_transition import cuda as rows

    if x.shape[-1] == 128:
        rows._ext().layernorm128_rows(x, out, float(eps), False)
    else:  # the same row kernel template (layernorm_rows_kernel<C>) at 256 channels
        rows._ext().layernorm_rows_cuda(x, out, float(eps))


def ln0_fp32(x: torch.Tensor, eps: float = 1e-5, stats: bool = False):
    """LN(x) without affine in fp32 [R, C] from x bf16 / fp32 [R, C] (C = 128 or 256), for the EXACT fp32 pair-bias logits (user rule
    2026-11-19). ``stats``: also (mean, rstd) as float [R, 2], what the pair_bias backward kernels read. Returns (y, pst or None)."""
    y = torch.empty(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
    pst = torch.empty(x.shape[0], 2, device=x.device, dtype=torch.float32) if stats else x.new_empty(0, dtype=torch.float32)
    ext().ln0_fwd_rows_cuda(x, y, pst, float(eps))
    return y, (pst if stats else None)


@contextlib.contextmanager
def exact_fp32_gemms():
    """cuBLAS fp32 GEMMs on the FP32 (FMA) pipe, no TF32: the pair-bias logits (values 1e3..2e5, key spread ~500) need fp32 products
    (a TF32 / bf16 split leaves 1e-3..1 absolute error: the attention picks other keys)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def exact_logits(wf32: torch.Tensor, ln32: torch.Tensor) -> torch.Tensor:
    """W' [N, C] fp32 x LN0 [R, C] fp32 -> logits [N, R] fp32 (head-major) on the FMA pipe."""
    with exact_fp32_gemms(), torch.autocast("cuda", enabled=False):
        return torch.mm(wf32, ln32.t())


def ln0_bwd_rows(x: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> None:
    """dx = the LayerNorm-without-affine backward of dy at x: rstd (dy - mean(dy) - y mean(dy y)), and ``y`` = LN0(x) as the dW'
    GEMM's operand: bf16 [R, 2 C] = [bf16(y) | bf16(y - bf16(y))], or fp32 [R, C]. x / dx bf16 or fp32, dy fp32 [R, C] (C = 128 or
    256); all contiguous."""
    ext().ln0_bwd_rows_cuda(x, dy, dx, y, float(eps))


def hoist_bwd_fused(x: torch.Tensor, dball: torch.Tensor, wf32: torch.Tensor, pst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact fp32 backward of one block's hoisted pair bias in two passes over the rows (``dln_bwd_kernel``, ``dwf_kernel``): x [R, 128]
    bf16 / fp32, dball [16, R] fp32, W' [16, 128] fp32 -> (d pair [R, 128] in x's dtype, dW' [16, 128] fp32). d LN0 and Y stay on chip;
    the per-warp dW' partials are summed in a fixed order."""
    R = x.shape[0]
    nw = min(2112, -(-(R // 32) // 8) * 8)  # 264 blocks of 8 warps (2 an SM on 132 SMs: whole waves), fewer for small R; partials <= 17 MB
    part = torch.empty(nw, 16, 128, device=x.device, dtype=torch.float32)
    dx = torch.empty_like(x)
    ext().hoist_bwd_fused_cuda(x, dball, wf32, pst, dx, part, nw, OPT["bwd_rg"])
    return dx, part.sum(0)


def dbias_rows(logits: torch.Tensor, dp: torch.Tensor, out: torch.Tensor, mask: torch.Tensor | None, pin: bool = False) -> bool:
    """out [H L, L] fp32 <- P o (dp - D) (``dbias_rows_kernel``; out may be dp): P = softmax(logits) re-formed in fp32 (key mask folded in), D = sum_j P dp the exact
    fp32 row sum; one read of the logits (``pin``: of P itself) and dP, one write. False (nothing done) where L is not 128 .. 1024 in steps of 128."""
    L = dp.shape[-1]
    if not OPT["rows"] or L % 128 or L > 1024 or not dp.is_contiguous() or not out.is_contiguous() or not logits.is_contiguous():
        return False
    m = mask.reshape(L).to(torch.bool).contiguous() if mask is not None else torch.empty(0, device=dp.device, dtype=torch.bool)
    ext().dbias_rows_cuda(logits.view(-1, L), dp.view(-1, L), out.view(-1, L), m, pin)
    return True


def permute_dv(do: torch.Tensor, v: torch.Tensor, samples: int, heads: int, length: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Head-major copies [H, L, A dh] of do and v (bf16 [A L, H dh], v a column slice of v | g): the operands of dP[h] = dO[h] V[h]^T (K = (a, d)) in one
    kernel (the two torch permute copies took 1.7 x its time). None where it does not serve (``OPT["rows"]`` off, dh not a multiple of 8, > 128 chunks a row)."""
    dh = do.shape[1] // heads
    if not OPT["rows"] or dh % 8 or heads * (dh // 8) > 128 or do.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        return None
    a, b = (torch.empty(heads, length, samples * dh, device=do.device, dtype=torch.bfloat16) for _ in range(2))
    ext().permute_dv_cuda(do, v, a, b, samples, heads, length)
    return a, b


def pair_logits(p2: torch.Tensor, wf32: torch.Tensor, stats: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The EXACT fp32 logits [N, R] (head-major) of the pair rows p2 [R, C] and the gamma-folded fp32 weights wf32 [N, C], and (``stats``) the row
    statistics float [R, 2]: LN0 in fp32, every product on the FMA pipe. One fused pass (``logits_fused_kernel``: no LN0 in memory) for 16 heads of
    128 channels, else ln0_fp32 + the cuBLAS FMA GEMM."""
    R = p2.shape[0]
    # 16 heads, 128 / 256 channels: the one-pass tensor-core logits (+ the row statistics)
    if OPT["train_tc"] and logits_tc_ok(p2, wf32):
        lg = torch.empty(16, R, device=p2.device, dtype=torch.float32)
        pst = torch.empty(R, 2, device=p2.device, dtype=torch.float32) if stats else None
        logits_tc(p2, wf32, lg, pst)
        return lg, pst
    if OPT["fwd"] and tuple(wf32.shape) == (16, 128) and p2.shape[1] == 128 and R % 32 == 0 and p2.is_contiguous() and wf32.is_contiguous():
        lg = torch.empty(16, R, device=p2.device, dtype=torch.float32)
        pst = torch.empty(R, 2, device=p2.device, dtype=torch.float32) if stats else p2.new_empty(0, dtype=torch.float32)
        ext().logits_fused_cuda(p2, wf32, lg, pst, 1e-5)
        return lg, (pst if stats else None)
    ln32, pst = ln0_fp32(p2, 1e-5, stats=stats)
    return exact_logits(wf32, ln32), pst


def logits_tc(p2: torch.Tensor, wf32: torch.Tensor, out: torch.Tensor, pst: torch.Tensor | None = None) -> None:
    """out [16, R] fp32 <- one block's 16-head logits of the pair rows p2 [R, 128 | 256] (bf16 / fp32) and W' wf32 [16, C] fp32 in ONE pass
    (``logits_tc_k``): LN0 in fp32 on chip, the products on TF32 tensor cores with the split fp32 operands (all 4 hi / lo products, a fresh
    partial per k-step added in round-to-nearest fp32), the
    inference hoist's logits (max-abs error 0.4x the FMA GEMM's against fp64: tests/gpu/test_bias_only_fp32_logits.py)."""
    if logits_sm100_ok(p2, wf32):
        logits_sm100(p2, wf32, out, pst)
        return
    from foldjet_engine.kernels.bias_only_dit.cuda import tf32

    # launch bounds (blocks of 8 warps an SM): 2 at 128 channels; 1 at 256 (register file: 2 spilled 360 bytes, 190 us against 295 us at L 768)
    mb = OPT["tc_mb"] or (1 if p2.shape[1] == 256 else 2)
    tf32.rows32().logits_tc_cuda(p2, wf32, out, 1e-5, mb, out.new_empty(0) if pst is None else pst)


_SM100_RUNS = None


def logits_sm100_ok(p2: torch.Tensor, wf32: torch.Tensor) -> bool:
    """``logits_sm100`` (the tcgen05 kernel ``bo_logits_sm100.cu``) serves bf16 pair rows of width 128 / 256, R a multiple of 128, on a B200."""
    return (
        OPT["sm100_logits"]
        and p2.dtype is torch.bfloat16
        and p2.dim() == 2
        and p2.shape[1] in (128, 256)
        and p2.shape[0] % 128 == 0
        and p2.is_contiguous()
        and wf32.dtype is torch.float32
        and tuple(wf32.shape) == (16, p2.shape[1])
        and wf32.is_contiguous()
        and torch.cuda.get_device_capability(p2.device) == (10, 0)
    )


def logits_sm100(p2: torch.Tensor, wf32: torch.Tensor, out: torch.Tensor, pst: torch.Tensor | None = None) -> None:
    """out [16, R] fp32 <- one block's 16-head logits of the bf16 pair rows p2 [R, 128 | 256] and W' wf32 [16, C] fp32 (and the row statistics (mean, rstd)
    float [R, 2] in ``pst``) in one pass on a tcgen05 kernel: a row whose |mean| <= std / 2 as rs (sum W' x - mean sum W') on the bf16 tensor core with W' in
    three bf16 pieces (x exact, every product exact in fp32, the pieces and 64-column slabs added in fp32), other rows from their exact fp32 LN0 values on the
    FMA pipe: the error class of the exact fp32 FMA GEMM either way (tests/gpu/test_bias_only_fp32_logits.py)."""
    global _SM100_RUNS
    from foldjet_engine.kernels.augmented_attention.cuda import sm100
    from foldjet_engine.kernels.bias_only_dit.cuda import PTR, _descriptors, _Launch, _Runs

    if _SM100_RUNS is None:
        _SM100_RUNS = _Runs()
    R, cp = p2.shape
    dev = p2.device.index if p2.device.index is not None else torch.cuda.current_device()

    def bind():
        k = sm100._sm100_kernel("bo_logits_sm100", "bo_logits_sm100", dev, src_dir=str(_dir), defs=(f"CP={cp}",), smem=232448)
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        m = sm100._tm(p2, [cp, R], cp * 2, [64, 128])
        return _Launch(k, (min(nsm, R // 128), 1, 1), (384, 1, 1), *_descriptors(m), PTR, PTR, PTR, R, 0 if pst is None else 1)

    key = (dev, p2.data_ptr(), R, cp, pst is not None)
    _SM100_RUNS.bind(key, bind)(wf32, out, out if pst is None else pst)


def logits_tc_ok(p2: torch.Tensor, wf32: torch.Tensor) -> bool:
    """``logits_tc`` serves a 16-head block on an H100 / B200 with contiguous bf16 / fp32 pair rows of width 128 or 256, R a multiple of 16."""
    return (
        OPT["eval_tc"]
        and tuple(wf32.shape) == (16, p2.shape[1])
        and p2.shape[1] in (128, 256)
        and p2.shape[0] % 16 == 0
        and p2.is_contiguous()
        and p2.dtype in (torch.bfloat16, torch.float32)
        and wf32.is_contiguous()
        and torch.cuda.get_device_capability(p2.device) in _TC_CAPS
    )


def hoist_bwd_tc_ok(x: torch.Tensor, dball: torch.Tensor, wf32: torch.Tensor) -> bool:
    """``hoist_bwd_tc`` serves one block's 16 heads on an H100 / B200: x [R, 128 | 256] bf16 / fp32 contiguous (R a multiple of 16), dbias [16, R] fp32."""
    return (
        OPT["bwd_tc"]
        and tuple(dball.shape[:1]) == (16,)
        and tuple(wf32.shape) == (16, x.shape[1])
        and x.shape[1] in (128, 256)
        and x.shape[0] % 16 == 0
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float32)
        and dball.dtype is torch.float32
        and dball.is_contiguous()
        and wf32.is_contiguous()
        and torch.cuda.get_device_capability(x.device) in _TC_CAPS
    )


def hoist_bwd_tc(x: torch.Tensor, dball: torch.Tensor, wf32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact-class backward of one block's hoisted pair bias in ONE pass over the rows (``hoist_bwd_tc_k``: d LN0 = dbias^T W', the LN0 backward, dW' = dbias Y on TF32 mma,
    four-term split operands, fp32 partials in round-to-nearest): -> (d pair [R, C] in x's dtype, dW' [16, C] fp32); the per-warp dW' partials are summed in a fixed order."""
    from foldjet_engine.kernels.bias_only_dit.cuda import tf32

    R, cp = x.shape
    cap = min(3200, -(-(R // 16) // 4) * 8)
    part = torch.empty(cap, 16, cp // 2, device=x.device, dtype=torch.float32)
    dx = torch.empty_like(x)
    # launch bounds (blocks of 8 warps an SM): 1 at 256 channels (2: 208 .. 304 bytes spilled, 745 us against 548 us at L 768), 2 at 128
    w = tf32.rows32().hoist_bwd_tc_cuda(x, dball, wf32, dx, part, 1e-5, OPT["bwd_mb"] or (1 if cp == 256 else 2))
    halves = part[:w].view(w // 2, 2, 16, cp // 2).sum(0)  # [half, 16, cp / 2]: the pairs' partials of each channel half, in a fixed order
    return dx, halves.permute(1, 0, 2).reshape(16, cp)


def operand_cols(dtype: torch.dtype, width: int = 128) -> int:
    """Columns of ``ln0_bwd_rows``'s y for a path's dtype and the pair width C: 2 C (bf16, the two-term split) or C (fp32)."""
    return 2 * width if dtype is torch.bfloat16 else width


def kernel_attrs() -> list[tuple[str, int, int]]:
    """(name, registers, local-memory bytes) of every kernel of ``bias_only_dit_hoist_rows.cu``."""
    return [tuple(a) for a in ext().func_attrs()]


__all__ = [
    "dbias_rows",
    "logits_sm100",
    "logits_sm100_ok",
    "logits_tc",
    "logits_tc_ok",
    "pair_logits",
    "exact_fp32_gemms",
    "exact_logits",
    "ext",
    "hoist_bwd_fused",
    "hoist_bwd_tc",
    "hoist_bwd_tc_ok",
    "kernel_attrs",
    "ln0_bwd_rows",
    "ln0_fp32",
    "ln0_rows",
    "operand_cols",
]
