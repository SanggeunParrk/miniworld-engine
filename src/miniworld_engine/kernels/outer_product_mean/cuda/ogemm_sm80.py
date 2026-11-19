"""The OuterProductMean's grouped outer product O = A^T B on A100 with the dead tokens (masked in every MSA row) skipped: ``sm80/ogemm_cutlass.cu``
(a CUTLASS 2.x sm_80 GEMM; a threadblock whose row or column tokens are all dead writes zeros). Exact: those O blocks are 0 in the dense product."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent / "sm80"


@functools.lru_cache(maxsize=1)
def ext():
    from ..._nvcc import cutlass_include, ensure_cuda_home, host_flags, load_extension
    from miniworld_engine.kernels import _msa_card as _card

    ensure_cuda_home()
    return load_extension(
        name=_card.name("opm_ogemm_sm80"),
        sources=[str(_dir / "ogemm_cutlass.cu")],
        extra_include_paths=[str(_dir), str(cutlass_include())],
        extra_cuda_cflags=_card.cflags(
            [*host_flags(), "-O3", "-std=c++17", "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr"]
        ),
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def live_tokens(mask_u8: torch.Tensor) -> torch.Tensor:
    """[S, L] uint8 mask -> [L] uint8: 1 where some MSA row keeps the token."""
    from miniworld_engine.kernels import _msa_card

    if _msa_card.sm120():
        if mask_u8.is_cuda and mask_u8.dim() == 2 and mask_u8.element_size() == 1:
            # RTX 5090: the one-launch column reduction (``gemm_sm120.live_tokens``: 3.6 us against the torch amax's 8.4 us + a copy at S 1024, L 128); 0 / 1 masks: the same bytes
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120

            return gemm_sm120.live_tokens(mask_u8, mask_u8.shape[0], mask_u8.shape[1])
        return mask_u8.amax(0).contiguous()
    if mask_u8.shape[0] > 2048:  # (the one-block-column kernel is latency bound at a deep MSA)
        return mask_u8.amax(0).contiguous()
    return ext().live_tokens(mask_u8.contiguous())  # A100: one small kernel (torch's amax over the MSA rows costs 13-21 us at S <= 2048, this 8-10 us)


def outer(a: torch.Tensor, b: torch.Tensor, live: torch.Tensor, ch: int = 32, out: torch.Tensor | None = None) -> torch.Tensor:
    """a, b [S, ch L] bf16 -> O [ch L, ch L] bf16 = a^T b (the dead tokens' blocks written as zeros without the product); ``ch`` 32 (OPM) or 8 (a
    Chai-1 group); ``out`` a contiguous buffer to write into."""
    m = a.shape[1]
    out = a.new_empty((m, m)) if out is None else out
    ext().ogemm_skip(a, b, live, out, int(ch))
    return out


def outer_batched(a: torch.Tensor, b: torch.Tensor, live: torch.Tensor, ch: int = 8) -> torch.Tensor:
    """a, b [G, S, ch L] bf16 -> O [G, ch L, ch L] bf16 = a_g^T b_g (the dead tokens' blocks zeros), ONE launch for the G products (A100, not the RTX 5090)."""
    g, _, m = a.shape
    out = a.new_empty((g, m, m))
    ext().ogemm_skip_batched(a, b, live, out, int(ch))
    return out


_PROBES: dict = {}


def all_live(mask_u8: torch.Tensor) -> bool:
    """Whether no token is masked in every MSA row (the dense product is then the whole answer and cuBLAS runs it 3-25 % faster than the skip kernel).

    A host-side choice of the product only (both are exact): read once per mask tensor (address, version, shape) outside a graph capture -- one small
    reduction and a sync -- and remembered; during a capture the remembered answer is used (a mask not seen before: the skip kernel). A graph is
    captured after its eager warm-up calls, which have probed the same static mask. ``MINIWORLD_OPM_PROBE=0`` always takes the skip kernel."""
    import os

    if os.environ.get("MINIWORLD_OPM_PROBE", "1") == "0":
        return False
    key = (mask_u8.data_ptr(), mask_u8._version, tuple(mask_u8.shape), mask_u8.device.index)
    hit = _PROBES.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return False
    res = bool((mask_u8 != 0).all().item())
    if len(_PROBES) > 256:
        _PROBES.clear()
    _PROBES[key] = res
    return res
