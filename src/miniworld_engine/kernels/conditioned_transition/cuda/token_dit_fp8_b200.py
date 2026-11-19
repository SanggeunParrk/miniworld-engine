"""EXPERIMENTAL B200 (sm_100a) e4m3 placements of the bf16 token DiT inference block (opt-in, default OFF; docs/b200/fp8.md).

The H100 recipe of ``token_dit_fp8.py`` unchanged (dynamic per-row activation scale made by the quantising AdaLN row pass, static
per-output-channel weight scale, no calibration / guard), with the product on tcgen05 ``kind::f8f6f4``: ``gemm_swiglu_f8_sm100.cu`` (the 2-CTA
expand + SwiGLU of ``gemm_swiglu2_sm100.cu`` with e4m3 operands and the scales applied in its epilogue). The row passes are the H100 file
``token_dit_fp8_rows.cu`` (arch-neutral) built for sm_100a. ``token_dit_fp8.ext()`` returns :func:`ext` on a B200 so the call sites of the
block are the H100 ones."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent
#: extra -D flags of the expand cubin (SIG_TANH (default): tanh.approx sigmoid in the epilogue -- below the e4m3 operand error, -7 % kernel time)
DEFS = tuple(d for d in os.environ.get("MINIWORLD_TOKEN_DIT_FP8_B200_DEFS", "SIG_TANH").split() if d)
#: CTAs (2 per tile pair) of the expand grid; None = every SM. MINIWORLD_TOKEN_DIT_FP8_B200_CTAS (experiments: the parallel block runs the transition beside the attention core)
GRID_CTAS = int(os.environ["MINIWORLD_TOKEN_DIT_FP8_B200_CTAS"]) if os.environ.get("MINIWORLD_TOKEN_DIT_FP8_B200_CTAS") else None


@functools.lru_cache(maxsize=None)
def _rows_ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="token_dit_fp8_rows_sm100",
        sources=[str(_dir / "token_dit_fp8_bind_sm100.cu"), str(_dir / "token_dit_fp8_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("100", ptx=("100",)),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


class SwigluF8:
    """``gemm_swiglu_f8_sm100.cu``: h = bf16(silu(a) b), [a | b] = (sx x8)(Wab8 s)^T; 2-CTA persistent (tile pair, 128-unit chunk) items."""

    def __init__(self, device_index: int, K: int, H: int):
        from miniworld_engine.kernels._device import num_sms

        from ...augmented_attention.cuda import sm100

        assert K % 128 == 0 and H % K == 0 and H % 128 == 0
        self._tm = sm100._tm
        self.K, self.H = K, H
        self.k = sm100._sm100_kernel(
            "gemm_swiglu_f8_sm100",
            "gemm_swiglu_f8_sm100",
            device_index,
            src_dir=str(_dir),
            defs=(f"DIM={K}", f"HMUL={H // K}", "ITEM_SCHED", *DEFS),
            cluster=2,
            smem=5 * 32768 + 32768 + 512,
        )
        self.nsm = num_sms(device_index)
        self.runs: dict = {}

    def _bind(self, x8, w8, out, sx, sw, nsm):
        M, K = x8.shape
        H = self.H
        assert K == self.K and w8.shape == (2 * H, K) and out.shape == (M, H) and M % 128 == 0
        assert x8.is_contiguous() and w8.is_contiguous() and out.is_contiguous() and sx.is_contiguous() and sw.is_contiguous()
        wa, wb = w8[:H], w8[H:]
        tm = self._tm
        maps = (
            tm(x8.view(torch.uint8), [K, M], K, [128, 64], dtype="u8"),
            tm(wa.view(torch.uint8), [K, H], K, [128, 128], dtype="u8"),
            tm(wb.view(torch.uint8), [K, H], K, [128, 128], dtype="u8"),
            tm(out, [H, M], H * 2, [64, 64]),
        )
        swa, swb = sw[:H], sw[H:]
        grid = (min(self.nsm, GRID_CTAS if GRID_CTAS is not None else nsm) & ~1, 1, 1)
        tiles = M // 128

        def run():
            self.k(grid, (512, 1, 1), *maps, sx, swa, swb, int(tiles))

        run.keep = (maps, x8, w8, out, sx, sw)  # ty: ignore[unresolved-attribute]
        return run

    def __call__(self, x8, w8, out, sx, sw, nsm=1 << 30):
        key = (x8.data_ptr(), w8.data_ptr(), out.data_ptr(), sx.data_ptr(), tuple(x8.shape), nsm)
        run = self.runs.get(key)
        if run is None:
            if len(self.runs) > 256:
                self.runs.clear()
            run = self.runs[key] = self._bind(x8, w8, out, sx, sw, nsm)
        run()
        return out




class SwigluF8Up(SwigluF8):
    """``gemm_swiglu_f8_up_sm100.cu``: h = bf16(silu(a) b u), W8 [3 H, K] = the H100 ``_bf16_wab64`` pack ([Wa 64 | Wb 64 | Wu 64] per chunk), 64-unit items."""

    def __init__(self, device_index: int, K: int, H: int):
        from miniworld_engine.kernels._device import num_sms

        from ...augmented_attention.cuda import sm100

        assert K % 128 == 0 and H % K == 0 and H % 64 == 0
        self._tm = sm100._tm
        self.K, self.H = K, H
        self.k = sm100._sm100_kernel(
            "gemm_swiglu_f8_up_sm100",
            "gemm_swiglu_f8_up_sm100",
            device_index,
            src_dir=str(_dir),
            defs=(f"DIM={K}", f"HMUL={H // K}", *DEFS),
            cluster=2,
            smem=5 * 28672 + 32768 + 512,
        )
        self.nsm = num_sms(device_index)
        self.runs: dict = {}

    def _bind(self, x8, w8, out, sx, sw, nsm):
        M, K = x8.shape
        H = self.H
        assert K == self.K and w8.shape == (3 * H, K) and out.shape == (M, H) and M % 128 == 0
        assert x8.is_contiguous() and w8.is_contiguous() and out.is_contiguous() and sx.is_contiguous() and sw.is_contiguous()
        tm = self._tm
        maps = (
            tm(x8.view(torch.uint8), [K, M], K, [128, 64], dtype="u8"),
            tm(w8.view(torch.uint8), [K, 3 * H], K, [128, 96], dtype="u8"),
            tm(out, [H, M], H * 2, [64, 64]),
        )
        grid = (min(self.nsm, GRID_CTAS if GRID_CTAS is not None else nsm) & ~1, 1, 1)
        tiles = M // 128

        def run():
            self.k(grid, (512, 1, 1), *maps, sx, sw, int(tiles))

        run.keep = (maps, x8, w8, out, sx, sw)  # ty: ignore[unresolved-attribute]
        return run


_OPS: dict = {}


class _Ext:
    """The call surface ``token_dit_fp8.ext()`` has on H100 (the row passes and ``gemm_swiglu_f8``), B200 implementation."""

    def __getattr__(self, name):
        return getattr(_rows_ext(), name)

    @staticmethod
    def gemm_swiglu_f8(x8, a8, h, sx, sa, nsm, ht, pdl, up=False):
        idx = x8.device.index if x8.device.index is not None else torch.cuda.current_device()
        key = (idx, x8.shape[1], h.shape[1], up)
        op = _OPS.get(key)
        if op is None:
            op = _OPS[key] = (SwigluF8Up if up else SwigluF8)(idx, x8.shape[1], h.shape[1])
        return op(x8, a8, h, sx.view(-1), sa.view(-1), int(nsm))


@functools.lru_cache(maxsize=None)
def ext():
    return _Ext()


def prebuild_items(plain: bool = True, up: bool = False):
    """Prebuild plan entries (``_prebuild --arch sm100``): the rows module and the cubin of the width-768 expand (``up``: Boltz-2's up-gate expand)."""
    yield _rows_ext
    from ...augmented_attention.cuda import sm100

    if plain:
        yield functools.partial(sm100.cubin, "gemm_swiglu_f8_sm100", ("DIM=768", "HMUL=2", "ITEM_SCHED", *DEFS), src_dir=str(_dir))
    if up:
        yield functools.partial(sm100.cubin, "gemm_swiglu_f8_up_sm100", ("DIM=768", "HMUL=2", *DEFS), src_dir=str(_dir))
