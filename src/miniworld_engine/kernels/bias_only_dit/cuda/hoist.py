"""Row kernels of the hoisted pair bias (``integrations/bias_only_dit_hoist.py``), bf16 and fp32, pair width 128 or 256.

  forward    LN0(pair): the token DiT rows' ``layernorm128_rows`` (one warp per row, no affine; the inference hoist's kernel), its
             ``layernorm_rows`` (the same kernel template) for 256 channels
  backward   ``bias_only_dit_hoist_rows.cu`` ln0_bwd_rows: d pair from d LN0 (fp32) and the pair rows, the row statistics
             recomputed with the forward's arithmetic, and the dW' GEMM's operand made from the same fp32 LN0 (bf16: the two-term
             split [hi | lo], 2 C columns; fp32: LN0 itself)

The one-pass kernels (``bo_hoist_pass.cu``) take both widths; the multi-block bf16 backward (``bo_hoist_bwd.cu``) 128 only.
B200 bf16 and fp32 (``sm100_fwd`` / ``sm100_bwd``, ``bo_hoist_sm100.cu``, pair width 128): the whole hoist in one pass over the pair each way per range
of weight rows -- the projection on tcgen05 (bf16: from the raw pair rows with the LayerNorm in the epilogue; fp32: kind::tf32 on the
tile normalised in shared memory), and d LN0, the LN0 backward and dW' in one kernel (bf16 up to 64 weight rows; more rows and fp32 over
weight-row chunks, ``bwd_plan``). The cubins are built on first use into ``settings.engine_jit_root``.

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


#: Weight rows (blocks x heads) per launch of the sm_100a one-pass hoist forward (``bo_hoist_sm100.cu``, W' resident in shared memory
#: beside the pair stages): bf16 384 (96 KB, four 32 KB stages), fp32 128 (64 KB, two 64 KB stages); more rows run as several
#: launches over row ranges (each reads the pair once). One launch takes NCH chunks of NP <= 128 rows (four TMEM accumulators of NP).
SM100_FWD_LAUNCH_ROWS = {torch.bfloat16: 384, torch.float32: 128}
#: Most weight rows of the one-chunk bf16 backward (W' resident; its shared memory); more rows, and fp32, take the chunked backward
#: (``-DBWD2``: chunks of ``SM100_BWD_CHUNK`` rows streamed with each tile, S for at most 256 rows per launch)
SM100_BWD_MAX_ROWS = 64
SM100_BWD_CHUNK = {torch.bfloat16: 64, torch.float32: 32}
_BWD2_SMEM = 1024 + 229376 + 512
#: Most weight rows the sm_100a hoist SERVES (``sm100_served``; the kernels themselves take any N). Past this the LayerNorm rows +
#: cuBLAS path (T7) is faster: the tiled kernels run their weight-row chunks one after another per tile (the backward's d' / d per
#: chunk, the forward's per-row stores). Measured (B200, hoist op alone, CUDA graphs, ms, sm100 / T7 at R = 1.77 M | 7.08 M rows):
#: bf16 N128 fwd 0.227 / 0.384 | 0.858 / 1.529, bwd 0.520 / 0.981 | 2.001 / 3.852; N256 fwd 0.415 / 0.524 | 1.597 / 1.874, bwd 1.023 /
#: 1.002 | 4.102 / 4.066; N384 bwd 1.761 / 1.172 | 7.259 / 4.737. fp32 N128 fwd 0.395 / 0.589 | 1.545 / 2.364, bwd 0.882 / 1.088 |
#: 3.507 / 4.289; N256 bwd 1.448 / 1.438 | 5.926 / 5.717.
SM100_MAX_ROWS = {torch.bfloat16: 192, torch.float32: 128}
_SMEM_MAX = 232448


def _np16(n: int) -> int:
    return (n + 15) // 16 * 16


def fwd_plan(N: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """(rows per launch, NCH, NP) of the forward for N weight rows: up to ``SM100_FWD_LAUNCH_ROWS`` rows per launch in NCH chunks of NP
    (a multiple of 16, at most 128) -- N <= 128 is one launch of one chunk (NP = N rounded up to 16)."""
    if N <= 128:
        return N, 1, _np16(N)
    cap = SM100_FWD_LAUNCH_ROWS[dtype]
    nl = -(-N // cap)
    per = -(-N // nl)
    nch = -(-per // 128)
    return per, nch, _np16(-(-per // nch))


def _sm100_smem(np_: int, bwd: bool, nch: int = 1, tf32: bool = False) -> int:
    """Dynamic shared memory of ``bo_hoist_sm100.cu`` for ``NP`` = np_, ``NCH`` = nch, -DTF32 (mirrors its constants)."""
    if bwd:
        xt, wt, dbt = 128 * 128 * 2, np_ * 128 * 2, np_ * 128 * 2
        o_w = (wt + 1023) // 1024 * 1024
        stage = xt + dbt
        o_st = o_w + 4 * dbt + 8 * 2 * 32 * 64  # W' | d' [2 groups][hi, lo] | dx staging
        nst = min(6, (_SMEM_MAX - 1024 - o_st - 512) // stage)
        return 1024 + o_st + nst * stage + 256
    es = 4 if tf32 else 2
    xt, wt, sn = 128 * 128 * es, np_ * 128 * es, 0 if tf32 else nch * np_ * 4
    o_st = (nch * wt + 1023) // 1024 * 1024
    nst = min(8, (_SMEM_MAX - 1024 - o_st - sn - 512) // xt)
    return 1024 + o_st + nst * xt + sn + 512


def sm100_served(pair2d: torch.Tensor, wf: torch.Tensor, bwd: bool = False) -> bool:
    """The sm_100a one-pass hoist serves pair rows [R, 128] (R a positive multiple of 128) with weight rows W' [N, 128] of the pair's
    dtype, on sm_100 (``settings.bias_only_dit_hoist_sm100``), bf16 or fp32 (TF32), up to ``SM100_MAX_ROWS`` weight rows (where the
    kernels beat the LayerNorm rows + cuBLAS path; ``sm100_fwd`` / ``sm100_bwd`` take any N): the forward over row ranges per launch
    (``fwd_plan``), the backward (``bwd``) in one launch up to ``SM100_BWD_MAX_ROWS`` bf16 rows, else over weight-row chunks
    (``bwd_plan``)."""
    from miniworld_engine import settings

    dt = pair2d.dtype
    if dt in SM100_MAX_ROWS and wf.shape[0] > SM100_MAX_ROWS[dt]:
        return False  # the LayerNorm rows + cuBLAS path is faster there (see SM100_MAX_ROWS)
    if dt is torch.float32 and settings.current().bias_only_dit_pair_exact:
        return False  # the exact (three-product split fp32) pair bias was asked for: the kernels that have it
    return (
        settings.current().bias_only_dit_hoist_sm100
        and dt in (torch.bfloat16, torch.float32)
        and wf.dtype is dt
        and pair2d.is_cuda
        and torch.cuda.get_device_capability(pair2d.device) == (10, 0)
        and pair2d.dim() == 2
        and pair2d.shape[1] == 128
        and pair2d.is_contiguous()
        and pair2d.shape[0] > 0
        and pair2d.shape[0] % 128 == 0
        and wf.is_contiguous()
        and wf.dim() == 2
        and wf.shape[1] == 128
        and wf.shape[0] >= 1
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

    def _kernel(self, np_: int, bwd: bool, nch: int = 1, tf32: bool = False):
        from . import sm100_kernel

        defs = (f"NP={np_}",) + ((f"NCH={nch}",) if nch > 1 else ()) + (("TF32",) if tf32 else ())
        return sm100_kernel(
            "bo_hoist_sm100",
            "bo_hoist_bwd_sm100" if bwd else "bo_hoist_fwd_sm100",
            self.device_index,
            src_dir=str(_dir),
            defs=defs,
            smem=_sm100_smem(np_, bwd, nch, tf32),
        )

    def _maps(self, pair2d, wf, np_):
        tm = self._tm
        R, N = pair2d.shape[0], wf.shape[0]
        if pair2d.dtype is torch.float32:  # 128-byte boxes of 32 fp32 channels
            return (
                tm(pair2d, [128, R], 512, [32, 128], dtype="f32"),
                tm(wf, [128, N], 512, [32, np_], dtype="f32"),  # rows N .. NP - 1: TMA zero fill
            )
        return (
            tm(pair2d, [128, R], 256, [64, 128]),
            tm(wf, [128, N], 256, [64, np_]),  # rows N .. NP - 1: TMA zero fill
        )

    def fwd(self, pair2d, wf, out, eps):
        """out [N, R] = W' LN0(pair)^T: one launch per range of ``fwd_plan`` rows (wf / out row slices: contiguous)."""
        from . import PTR, _descriptors, _Launch

        R, N = pair2d.shape[0], wf.shape[0]
        tf32 = pair2d.dtype is torch.float32
        per, nch, np_ = fwd_plan(N, pair2d.dtype)
        for n0 in range(0, N, per):
            w, o = wf[n0 : n0 + per], out[n0 : n0 + per]
            n = w.shape[0]
            key = ("f", pair2d.data_ptr(), w.data_ptr(), o.data_ptr(), R, n, eps, nch, np_, tf32)

            def make(w=w, n=n):
                mx, mw = self._maps(pair2d, w, np_)
                grid = (min(self.nsm, R // 128), 1, 1)
                return _Launch(self._kernel(np_, False, nch, tf32), grid, (320, 1, 1), *_descriptors(mx, mw), PTR, PTR, R, n,
                               _eps_bits(eps))

            self.runs.bind(key, make)(o, w)

    def bwd2(self, pair2d, wf, dball, dpair, parts, eps):
        """The chunked backward (``-DBWD2``): one launch per ``bwd_plan`` entry, its S partials into ``parts[k]``."""
        from . import PTR, _descriptors, _Launch, sm100_kernel

        R, N = pair2d.shape[0], wf.shape[0]
        f32 = pair2d.dtype is torch.float32
        npc = SM100_BWD_CHUNK[pair2d.dtype]
        plan = bwd_plan(N, pair2d.dtype)
        for (c0, c1, s0, s1, dx), part in zip(plan, parts, strict=True):
            key = ("b2", pair2d.data_ptr(), wf.data_ptr(), dball.data_ptr(), dpair.data_ptr(), part.data_ptr(), R, N, eps, c0, s0)

            def make(c0=c0, c1=c1, s0=s0, s1=s1, dx=dx, part=part):
                tm = self._tm
                if f32:
                    a32 = dict(swizzle="128a32", dtype="f32")
                    maps = (
                        tm(pair2d, [128, R], 512, [32, 128], **a32),
                        tm(dball, [R, N], R * 4, [32, npc], **a32),  # rows past N: TMA zero fill
                        tm(wf, [128, N], 512, [32, npc], **a32),
                        tm(dpair, [128, R], 512, [32, 32], **a32),  # dx stored from the pair tile
                    )
                else:
                    maps = (
                        tm(pair2d, [128, R], 256, [64, 128]),
                        tm(dball, [R, N], R * 2, [64, npc]),
                        tm(wf, [128, N], 256, [64, npc]),
                        tm(dpair, [128, R], 256, [32, 32], swizzle=64),
                    )
                k = sm100_kernel(
                    "bo_hoist_sm100",
                    "bo_hoist_bwd2_sm100",
                    self.device_index,
                    src_dir=str(_dir),
                    defs=(f"NP={npc}", "BWD2", *(("TF32",) if f32 else ())),
                    smem=_BWD2_SMEM,
                )
                return _Launch(k, (part.shape[0], 1, 1), (320, 1, 1), *_descriptors(*maps), PTR, R, c0, c1, s0, s1, int(dx),
                               _eps_bits(eps))

            self.runs.bind(key, make)(part)

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
    """bias_all^T [N, R] = W' LN0(pair)^T in the pair's dtype, one pass over the pair per ``fwd_plan`` row range. bf16: the projection
    on the raw rows (tcgen05 kind::f16), the LayerNorm applied in the epilogue as rstd (x W'^T - mean s), s = the row sums of W'. fp32:
    the rows normalised in shared memory (LN0 rounded to tf32), then the kind::tf32 projection. Call ``sm100_served`` first."""
    out = torch.empty(wf.shape[0], pair2d.shape[0], device=pair2d.device, dtype=pair2d.dtype)
    _hoist100(_index(pair2d)).fwd(pair2d, wf, out, float(eps))
    return out


def bwd_plan(N: int, dtype: torch.dtype) -> list[tuple[int, int, int, int, bool]]:
    """The chunked backward's launches for N weight rows: (c0, c1, s0, s1, dx) over chunks of ``SM100_BWD_CHUNK`` rows -- the first
    takes every chunk for d LN0 and d pair and S for the first 256 rows, the others S for 256 rows more each (the pair read again)."""
    npc = SM100_BWD_CHUNK[dtype]
    nch, sc = -(-N // npc), 256 // npc
    plan = [(0, nch, 0, min(nch, sc), True)]
    plan += [(s, min(nch, s + sc), s, min(nch, s + sc), False) for s in range(sc, nch, sc)]
    return plan


def sm100_bwd(pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float = 1e-5):
    """(d pair [R, 128] bf16, dW' [N, 128] fp32) from dbias_all [N, R] bf16 in one pass over the pair: d LN0 = dbias_all^T W' on tcgen05,
    the LayerNorm backward, and S = (dbias_all diag(rstd)) pair (the scaled d bias as two bf16 terms against the raw pair rows, fp32
    accumulation); dW' = S - its mean over the channels (every LN0 row sums to zero). The per-CTA fp32 partials of S are summed here in a
    fixed order. Call ``sm100_served(..., bwd=True)`` first."""
    R, N = pair2d.shape[0], wf.shape[0]
    h = _hoist100(_index(pair2d))
    dball = dball.contiguous()
    dpair = torch.empty_like(pair2d)
    if pair2d.dtype is torch.float32 or N > SM100_BWD_MAX_ROWS:  # the chunked backward (-DBWD2)
        npc = SM100_BWD_CHUNK[pair2d.dtype]
        parts = [torch.empty(h.grid(R), (s1 - s0) * npc, 128, device=pair2d.device, dtype=torch.float32)
                 for (_, _, s0, s1, _) in bwd_plan(N, pair2d.dtype)]
        h.bwd2(pair2d, wf, dball, dpair, parts, float(eps))
        S = torch.cat([p.sum(0) for p in parts])[:N]
        if pair2d.dtype is torch.float32:
            return dpair, S  # S = dbias xhat: dW' itself
        return dpair, S - S.mean(1, keepdim=True)
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


__all__ = ["PAIR_WIDTHS", "PASS_FWD_MAX_ROWS", "FUSED_MAX_ROWS", "SM100_BWD_MAX_ROWS", "SM100_FWD_LAUNCH_ROWS", "SM100_MAX_ROWS", "SM100_BWD_CHUNK", "bwd_plan", "fwd_plan", "sm100_bwd", "sm100_fwd", "sm100_served", "pass_bwd", "pass_bwd_served", "pass_ext", "pass_fwd", "pass_fwd_served", "bwd_ext", "ext", "fused_bwd_served", "hoist_bwd_fused", "kernel_attrs", "ln0_bwd_rows", "ln0_rows", "operand_cols"]
