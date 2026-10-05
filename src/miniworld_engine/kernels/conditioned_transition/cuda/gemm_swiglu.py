"""h = bf16(silu(a) * b), [a | b] = X [Wa; Wb]^T, on sm_100a: the token DiT transition's expand GEMM with the SwiGLU in its
epilogue, so the [M, 2H] pre-activation never reaches HBM (``gemm_swiglu2_sm100.cu``, 2-CTA persistent). Launched through
the sm_100a driver of ``kernels/augmented_attention/cuda/sm100``. ``gemm_swiglu_sm100.cu`` (1-CTA, interleaved W) is the
first version, kept for the record: 540 TFLOPS, ingress-bound."""

from __future__ import annotations

import os
from pathlib import Path

import torch

_dir = str(Path(__file__).parent)


def interleave(wa: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
    """[H, K] gate and up weights -> [2H, K] with rows a0, b0, a1, b1, ..."""
    return torch.stack([wa, wb], 1).reshape(-1, wa.shape[1]).contiguous()


#: Below this many rows the (tile pair, chunk) items leave most SM pairs idle in the last round (L384, S = 5: 96 items on 74
#: pairs, 14.0 us against cuBLAS + the SwiGLU row pass's 13.7); at M = 3840 (L768) it is 18.4 against 22.3 us.
#: MINIWORLD_TOKEN_DIT_GEMM_SWIGLU=1 forces it, =0 turns it off.
MIN_ROWS = 3840


def supported(x: torch.Tensor, w_ab: torch.Tensor) -> bool:
    """bf16 on sm_100, K a multiple of 64, H a multiple of 128 and of K, and M >= MIN_ROWS unless forced."""
    mode = os.environ.get("MINIWORLD_TOKEN_DIT_GEMM_SWIGLU", "auto")
    if mode == "0" or (mode == "auto" and x.shape[0] < MIN_ROWS):
        return False
    if not (x.is_cuda and x.dtype is torch.bfloat16 and w_ab.dtype is torch.bfloat16):
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    H = w_ab.shape[0] // 2
    return (torch.cuda.get_device_capability(idx) == (10, 0) and x.shape[1] % 64 == 0 and H % 128 == 0
            and H % x.shape[1] == 0 and w_ab.shape[1] == x.shape[1])


class GemmSwiglu:
    """``gemm_swiglu2_sm100.cu``: 2-CTA clusters, persistent over (tile pair, 128-unit chunk) items, h only -- or, with
    ``save_ab`` (the training forward), h and the bf16 pre-activation [a | b] for the backward. W is taken as ``[Wa; Wb]``
    ([2H, K], gate rows first) -- the leader CTA streams the Wa half, its peer the Wb half. Bound launches are cached per
    (buffer, weight, output): the TMA descriptors carry the pointers."""

    def __init__(self, device_index: int, K: int = 768, H: int = 1536, save_ab: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        assert K % 64 == 0 and H % K == 0 and H % 128 == 0
        self._tm = sm100._tm
        self.K, self.H, self.save_ab = K, H, save_ab
        # ring stages x 32 KB, the h staging (32 KB), with save_ab the a | b staging (64 KB, one ring stage fewer), barriers
        self.SMEM = (4 * 32768 + 32768 + 65536 + 512) if save_ab else (5 * 32768 + 32768 + 512)
        defs = (f"DIM={K}", f"HMUL={H // K}", "ITEM_SCHED", *(("SAVE_AB",) if save_ab else ()))
        self.k = sm100._sm100_kernel("gemm_swiglu2_sm100", "gemm_swiglu2_sm100", device_index, src_dir=_dir,
                               defs=defs, cluster=2, smem=self.SMEM)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs: dict = {}

    def _bind(self, x, w_ab, out, ab=None):
        M, K = x.shape
        H = self.H
        assert K == self.K and w_ab.shape == (2 * H, K) and out.shape == (M, H)
        assert x.is_contiguous() and w_ab.is_contiguous() and out.is_contiguous()
        wa, wb = w_ab[:H], w_ab[H:]
        maps = (self._tm(x, [K, M], K * 2, [64, 64]), self._tm(wa, [K, H], K * 2, [64, 128]),
                self._tm(wb, [K, H], K * 2, [64, 128]), self._tm(out, [H, M], H * 2, [64, 64]))
        if self.save_ab:
            assert ab is not None and ab.shape == (M, 2 * H) and ab.is_contiguous()
            maps = (*maps, self._tm(ab[:, :H], [H, M], 2 * H * 2, [64, 64]), self._tm(ab[:, H:], [H, M], 2 * H * 2, [64, 64]), 1)
        tiles = (M + 127) // 128
        grid = (self.nsm & ~1, 1, 1)             # ITEM_SCHED: (tile pair, chunk) items dealt over every pair of SMs

        def run():
            self.k(grid, (512, 1, 1), *maps, int(tiles))
        run.keep = (maps, x, w_ab, out, ab)
        return run

    def __call__(self, x, w_ab, out, ab=None):
        """out = silu(a) * b; with save_ab also ab = [a | b] ([M, 2H] bf16)."""
        if self.save_ab:                 # training: fresh (saved) buffers every call; a cached binding would pin them
            self._bind(x, w_ab, out, ab)()
            return out
        key = (x.data_ptr(), w_ab.data_ptr(), out.data_ptr(), tuple(x.shape))
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(x, w_ab, out, ab)
        run()
        return out


#: sm_90a (``gemm_swiglu_sm90.cu``) against cuBLAS + the SwiGLU row pass, H100, do_bench: M = 1920 24.1 vs 27.8 us, 3840 33.3
#: vs 47.1, 36864 302 vs 383 -- it wins at every row count, so no threshold beyond its tile.
MIN_ROWS_SM90 = 128


def supported_sm90(x: torch.Tensor, w_ab: torch.Tensor) -> bool:
    """bf16 or fp32 (TF32) on sm_90, M a multiple of 128 (>= MIN_ROWS_SM90 unless forced), K a multiple of 64 / 32, H of 128."""
    mode = os.environ.get("MINIWORLD_TOKEN_DIT_GEMM_SWIGLU", "auto")
    if mode == "0" or (mode == "auto" and x.shape[0] < MIN_ROWS_SM90):
        return False
    if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and w_ab.dtype is x.dtype):
        return False
    # fp32 (TF32) is opt-in: its TMA-fed operands are truncated to TF32 where cuBLAS rounds them -- 1.6e-3 against 4.2e-4 for
    # cuBLAS + the row pass -- and it wins only at M = 3840 (70.9 vs 85.5 us; 47.6 vs 45.9 at 1920, 759 vs 725 at 36864)
    if x.dtype is torch.float32 and os.environ.get("MINIWORLD_TOKEN_DIT_GEMM_SWIGLU_FP32", "0") != "1":
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    H = w_ab.shape[0] // 2
    kq = 64 if x.dtype is torch.bfloat16 else 32
    return (torch.cuda.get_device_capability(idx) == (9, 0) and x.shape[0] % 128 == 0 and x.shape[1] % kq == 0
            and H % 128 == 0 and w_ab.shape[1] == x.shape[1])


class GemmSwigluSm90:
    """``gemm_swiglu_sm90.cu``: one sm_90a GEMM, 128 x 256 tiles whose B rows are 128 of Wa then the same 128 of Wb, the
    SwiGLU in the epilogue. Takes W as ``[Wa; Wb]`` and packs it per tile once per weight tensor."""

    def __init__(self, device_index: int):
        from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
        ensure_cuda_home()
        k = Path(_dir).parent.parent
        self.ext = load_extension(
            name="gemm_swiglu_sm90", sources=[str(Path(_dir) / "gemm_swiglu_sm90.cu")],
            extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a"), "--expt-relaxed-constexpr",
                               f"-I{k / 'transition' / 'cuda' / 'anthropic_v5'}", f"-I{k / 'transition' / 'cuda' / 'wide' / 'kernels'}",
                               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
            extra_cflags=["-std=c++17"], verbose=False)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.packed: dict = {}

    def __call__(self, x, w_ab, out):
        key = (w_ab.data_ptr(), w_ab._version)
        wp = self.packed.get(key)
        if wp is None:
            H, K = w_ab.shape[0] // 2, w_ab.shape[1]
            wp = torch.stack([w_ab[:H].view(H // 128, 128, K), w_ab[H:].view(H // 128, 128, K)], 1).reshape(2 * H, K).contiguous()
            if not torch.cuda.is_current_stream_capturing():
                self.packed[key] = wp
        self.ext.gemm_swiglu(x, wp, out, self.nsm)
        return out


__all__ = ["GemmSwiglu", "GemmSwigluSm90", "interleave", "supported", "supported_sm90"]
