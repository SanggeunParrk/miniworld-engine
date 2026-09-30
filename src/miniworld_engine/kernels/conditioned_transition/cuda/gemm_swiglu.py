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


__all__ = ["GemmSwiglu", "interleave", "supported"]
