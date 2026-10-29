"""Row kernels of the hoisted pair bias (``integrations/bias_only_dit_hoist.py``), bf16 and fp32, pair width 128 or 256.

  forward    LN0(pair): the token DiT rows' ``layernorm128_rows`` (one warp per row, no affine; the inference hoist's kernel), its
             ``layernorm_rows`` (the same kernel template) for 256 channels
  backward   ``bias_only_dit_hoist_rows.cu`` ln0_bwd_rows: d pair from d LN0 (fp32) and the pair rows, the row statistics
             recomputed with the forward's arithmetic, and the dW' GEMM's operand made from the same fp32 LN0 (bf16: the two-term
             split [hi | lo], 2 C columns; fp32: LN0 itself)

The one-pass kernels (``bo_hoist_pass.cu``) take both widths; the multi-block bf16 backward (``bo_hoist_bwd.cu``) 128 only.
B200 bf16 (``sm100_fwd`` / ``sm100_bwd``, ``bo_hoist_sm100.cu``): the whole hoist in one pass over the pair each way -- the projection on
tcgen05 from the raw pair rows with the LayerNorm in the epilogue, and d LN0, the LN0 backward and dW' in one kernel (up to 128 / 64 weight
rows). The cubins are built on first use into ``settings.engine_jit_root``.

Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_hoist_rows_cuda",
        sources=[str(_dir / "bias_only_dit_hoist_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def ln0_rows(x: torch.Tensor, out: torch.Tensor, eps: float = 1e-5) -> None:
    """out = LN(x) without affine; x [R, C] (bf16 / fp32 rows, C = 128 or 256), out contiguous [R, C] (bf16 / fp32)."""
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    if x.shape[-1] == 128:
        rows._ext().layernorm128_rows(x, out, float(eps), False)
    else:  # the same row kernel template (layernorm_rows_kernel<C>) at 256 channels
        rows._ext().layernorm_rows_cuda(x, out, float(eps))


def ln0_bwd_rows(x: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> None:
    """dx = the LayerNorm-without-affine backward of dy at x: rstd (dy - mean(dy) - y mean(dy y)), and ``y`` = LN0(x) as the dW'
    GEMM's operand: bf16 [R, 2 C] = [bf16(y) | bf16(y - bf16(y))], or fp32 [R, C]. x / dx bf16 or fp32, dy fp32 [R, C] (C = 128 or
    256); all contiguous."""
    ext().ln0_bwd_rows_cuda(x, dy, dx, y, float(eps))


@functools.lru_cache(maxsize=None)
def bwd_ext():
    """``bo_hoist_bwd.cu`` (sm_90a): the hoisted pair bias's backward in one kernel. Built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_hoist_bwd_sm90a",
        sources=[str(_dir / "bo_hoist_bwd.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__", *gencodes("90a")],
        extra_cflags=["-std=c++17"], verbose=False)


@functools.lru_cache(maxsize=None)
def pass_ext():
    """``bo_hoist_pass.cu`` (sm_90a): the forward (LayerNorm + projection) and the one-block backward in one pass over the pair each. Built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_hoist_pass_sm90a",
        sources=[str(_dir / "bo_hoist_pass.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__", *gencodes("90a")],
        extra_cflags=["-std=c++17"], verbose=False)


#: Most rows of the folded weights (blocks x heads) the one-pass forward takes
PASS_FWD_MAX_ROWS = 64


def _sm90(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (9, 0)


#: Pair widths of the one-pass kernels (``bo_hoist_pass.cu``) and of the row kernels
PAIR_WIDTHS = (128, 256)


def pass_fwd_served(pair2d: torch.Tensor, wf: torch.Tensor) -> bool:
    """The one-pass forward serves bf16 / fp32 pair rows [R, 128 or 256] (R a multiple of 16) and at most ``PASS_FWD_MAX_ROWS`` weight rows (a
    multiple of 16) in the pair's dtype, on sm_90."""
    return (
        pair2d.dtype in (torch.bfloat16, torch.float32)
        and wf.dtype is pair2d.dtype
        and _sm90(pair2d)
        and pair2d.shape[-1] in PAIR_WIDTHS
        and wf.shape[-1] == pair2d.shape[-1]
        and pair2d.shape[0] % 16 == 0
        and wf.shape[0] % 16 == 0
        and 16 <= wf.shape[0] <= PASS_FWD_MAX_ROWS
    )


def pass_fwd(pair2d: torch.Tensor, wf: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """bias_all^T [N, R] = W' LN0(pair)^T in the pair's dtype, in one pass over the pair (bf16: LN0 rounded to bf16 for the projection, fp32
    accumulation; fp32: TF32 operands)."""
    out = torch.empty(wf.shape[0], pair2d.shape[0], device=pair2d.device, dtype=pair2d.dtype)
    pass_ext().hoist_fwd(pair2d, wf, out, float(eps))
    return out


def pass_bwd_served(pair2d: torch.Tensor, wf: torch.Tensor) -> bool:
    """The one-pass backward serves one block of 16 head rows (N = 16), bf16 or fp32 (TF32), pair rows of 128 or 256, R a multiple of 16, on
    sm_90."""
    return (
        pair2d.dtype in (torch.bfloat16, torch.float32)
        and wf.dtype is pair2d.dtype
        and _sm90(pair2d)
        and pair2d.shape[-1] in PAIR_WIDTHS
        and wf.shape[-1] == pair2d.shape[-1]
        and pair2d.shape[0] % 16 == 0
        and wf.shape[0] == 16
    )


def pass_bwd(pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float = 1e-5):
    """(d pair [R, C], dW' [16, C] fp32) for N = 16: d LN0 = dbias^T W', the LayerNorm backward and dW' = dbias LN0 in one pass over the pair;
    the per-CTA partials of dW' are summed here in a fixed order."""
    e = pass_ext()
    dpair = torch.empty_like(pair2d)
    part = torch.empty(e.hoist_pass_grid(), 16, pair2d.shape[-1], device=pair2d.device, dtype=torch.float32)
    e.hoist_bwd1(pair2d, wf, dball, dpair, part, float(eps))
    return dpair, part.sum(0)


#: Most rows of dbias_all (blocks x heads) the fused backward takes
FUSED_MAX_ROWS = 192


def fused_bwd_served(pair2d: torch.Tensor, wf: torch.Tensor) -> bool:
    """The fused backward serves bf16 pair rows [R, 128] with R a multiple of 64, at most ``FUSED_MAX_ROWS`` weight rows, on sm_90 (a
    256-wide pair takes the two GEMMs around ``ln0_bwd_rows``: its [3, 8, 4] dW' accumulators per lane would double)."""
    return (
        pair2d.dtype is torch.bfloat16
        and pair2d.is_cuda
        and torch.cuda.get_device_capability(pair2d.device) == (9, 0)
        and pair2d.shape[-1] == 128
        and pair2d.shape[0] % 64 == 0
        and wf.shape[0] <= FUSED_MAX_ROWS
    )


def hoist_bwd_fused(pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float = 1e-5):
    """(d pair [R, 128] bf16, dW' [N, 128] fp32) from pair rows [R, 128], W'_all [N, 128] and dbias_all [N, R] (all bf16): d LN0 = dbias_all^T
    W', the LayerNorm backward and dW' = dbias_all LN0 (hi + lo) in one pass over the pair, the forward's LN0 arithmetic. The per-CTA fp32
    partials of dW' are summed here, in a fixed order."""
    R, N = pair2d.shape[0], wf.shape[0]
    npt = (N + 15) // 16
    grid = min(num_sms(pair2d.device), R // 64)
    dpair = torch.empty_like(pair2d)
    part = torch.empty(grid, npt * 16, 128, device=pair2d.device, dtype=torch.float32)
    bwd_ext().hoist_bwd(pair2d, wf, dball, dpair, part, float(eps))
    return dpair, part.sum(0)[:N]


#: Most weight rows (blocks x heads) the sm_100a one-pass hoist takes (``bo_hoist_sm100.cu``; padded to a multiple of 16): the forward
#: 128, the backward 64 (its shared memory)
SM100_MAX_ROWS = 128
SM100_BWD_MAX_ROWS = 64
_SMEM_MAX = 232448


def _np16(n: int) -> int:
    return (n + 15) // 16 * 16


def _sm100_smem(np_: int, bwd: bool) -> int:
    """Dynamic shared memory of ``bo_hoist_sm100.cu`` for ``NP`` = np_ (mirrors its constants)."""
    xt, wt, dbt = 128 * 128 * 2, np_ * 128 * 2, np_ * 128 * 2
    o_w = (wt + 1023) // 1024 * 1024
    if bwd:
        stage = xt + dbt
        o_st = o_w + 4 * dbt + 8 * 2 * 32 * 64  # W' | d' [2 groups][hi, lo] | dx staging
        nst = min(6, (_SMEM_MAX - 1024 - o_st - 512) // stage)
        return 1024 + o_st + nst * stage + 256
    nst = min(8, (_SMEM_MAX - 1024 - o_w - 1024) // xt)
    return 1024 + o_w + nst * xt + np_ * 4 + 256


def sm100_served(pair2d: torch.Tensor, wf: torch.Tensor, bwd: bool = False) -> bool:
    """The sm_100a one-pass hoist serves bf16 pair rows [R, 128] (R a positive multiple of 128) and at most ``SM100_MAX_ROWS`` bf16
    weight rows (``bwd``: the backward, ``SM100_BWD_MAX_ROWS``), on sm_100 (``settings.bias_only_dit_hoist_sm100``)."""
    from miniworld_engine import settings

    return (
        settings.current().bias_only_dit_hoist_sm100
        and pair2d.dtype is torch.bfloat16
        and wf.dtype is torch.bfloat16
        and pair2d.is_cuda
        and torch.cuda.get_device_capability(pair2d.device) == (10, 0)
        and pair2d.dim() == 2
        and pair2d.shape[1] == 128
        and pair2d.is_contiguous()
        and pair2d.shape[0] > 0
        and pair2d.shape[0] % 128 == 0
        and wf.is_contiguous()
        and 1 <= wf.shape[0] <= (SM100_BWD_MAX_ROWS if bwd else SM100_MAX_ROWS)
    )


def _eps_bits(eps: float) -> int:
    import struct

    return struct.unpack("<i", struct.pack("<f", float(eps)))[0]


class _Hoist100:
    """Bound launches of ``bo_hoist_sm100.cu`` for one device: the forward and the backward, cached by buffer addresses."""

    def __init__(self, device_index: int):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        from . import _Runs

        self.device_index, self._tm = device_index, sm100._tm
        self.nsm = num_sms(device_index)
        self.runs = _Runs()

    def _kernel(self, np_: int, bwd: bool):
        from . import sm100_kernel

        return sm100_kernel(
            "bo_hoist_sm100",
            "bo_hoist_bwd_sm100" if bwd else "bo_hoist_fwd_sm100",
            self.device_index,
            src_dir=str(_dir),
            defs=(f"NP={np_}",),
            smem=_sm100_smem(np_, bwd),
        )

    def _maps(self, pair2d, wf, np_):
        tm = self._tm
        R, N = pair2d.shape[0], wf.shape[0]
        return (
            tm(pair2d, [128, R], 256, [64, 128]),
            tm(wf, [128, N], 256, [64, np_]),  # rows N .. NP - 1: TMA zero fill
        )

    def fwd(self, pair2d, wf, out, eps):
        from . import PTR, _descriptors, _Launch

        R, N = pair2d.shape[0], wf.shape[0]
        np_ = _np16(N)
        key = ("f", pair2d.data_ptr(), wf.data_ptr(), out.data_ptr(), R, N, eps)

        def make():
            mx, mw = self._maps(pair2d, wf, np_)
            grid = (min(self.nsm, R // 128), 1, 1)
            return _Launch(self._kernel(np_, False), grid, (320, 1, 1), *_descriptors(mx, mw), PTR, PTR, R, N, _eps_bits(eps))

        self.runs.bind(key, make)(out, wf)

    def bwd(self, pair2d, wf, dball, dpair, part, eps):
        from . import PTR, _descriptors, _Launch

        R, N = pair2d.shape[0], wf.shape[0]
        np_ = _np16(N)
        key = ("b", pair2d.data_ptr(), wf.data_ptr(), dball.data_ptr(), dpair.data_ptr(), part.data_ptr(), R, N, eps)

        def make():
            mx, mw = self._maps(pair2d, wf, np_)
            mdb = self._tm(dball, [R, N], R * 2, [64, np_])  # rows N .. NP - 1: TMA zero fill
            mdx = self._tm(dpair, [128, R], 256, [32, 32], swizzle=64)  # the staged d pair stores
            return _Launch(self._kernel(np_, True), (part.shape[0], 1, 1), (320, 1, 1), *_descriptors(mx, mdb, mw, mdx), PTR, R,
                           _eps_bits(eps))

        self.runs.bind(key, make)(part)

    def grid(self, R: int) -> int:
        return min(self.nsm, R // 128)


@functools.lru_cache(maxsize=None)
def _hoist100(device_index: int) -> _Hoist100:
    return _Hoist100(device_index)


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def sm100_fwd(pair2d: torch.Tensor, wf: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """bias_all^T [N, R] = W' LN0(pair)^T (bf16) in one pass over the pair: the projection on the raw rows (tcgen05), the LayerNorm
    applied in the epilogue as rstd (x W'^T - mean s), s = the row sums of W'. Call ``sm100_served`` first."""
    out = torch.empty(wf.shape[0], pair2d.shape[0], device=pair2d.device, dtype=pair2d.dtype)
    _hoist100(_index(pair2d)).fwd(pair2d, wf, out, float(eps))
    return out


def sm100_bwd(pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float = 1e-5):
    """(d pair [R, 128] bf16, dW' [N, 128] fp32) from dbias_all [N, R] bf16 in one pass over the pair: d LN0 = dbias_all^T W' on tcgen05,
    the LayerNorm backward, and S = (dbias_all diag(rstd)) pair (the scaled d bias as two bf16 terms against the raw pair rows, fp32
    accumulation); dW' = S - its mean over the channels (every LN0 row sums to zero). The per-CTA fp32 partials of S are summed here in a
    fixed order. Call ``sm100_served(..., bwd=True)`` first."""
    R, N = pair2d.shape[0], wf.shape[0]
    h = _hoist100(_index(pair2d))
    dball = dball.contiguous()
    dpair = torch.empty_like(pair2d)
    part = torch.empty(h.grid(R), _np16(N), 128, device=pair2d.device, dtype=torch.float32)
    h.bwd(pair2d, wf, dball, dpair, part, float(eps))
    S = part.sum(0)[:N]
    return dpair, S - S.mean(1, keepdim=True)


def operand_cols(dtype: torch.dtype, width: int = 128) -> int:
    """Columns of ``ln0_bwd_rows``'s y for a path's dtype and the pair width C: 2 C (bf16, the two-term split) or C (fp32)."""
    return 2 * width if dtype is torch.bfloat16 else width


def kernel_attrs() -> list[tuple[str, int, int]]:
    """(name, registers, local-memory bytes) of every kernel of ``bias_only_dit_hoist_rows.cu``."""
    return [tuple(a) for a in ext().func_attrs()]


__all__ = ["PAIR_WIDTHS", "PASS_FWD_MAX_ROWS", "FUSED_MAX_ROWS", "SM100_BWD_MAX_ROWS", "SM100_MAX_ROWS", "sm100_bwd", "sm100_fwd", "sm100_served", "pass_bwd", "pass_bwd_served", "pass_ext", "pass_fwd", "pass_fwd_served", "bwd_ext", "ext", "fused_bwd_served", "hoist_bwd_fused", "kernel_attrs", "ln0_bwd_rows", "ln0_rows", "operand_cols"]
