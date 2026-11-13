"""Which card the MSA families' sm_80 sources (OuterProductMean, MSAPairWeightedAveraging) are compiled for.

The A100 (sm_80) builds are unchanged: the same extension names and flags. On a GeForce RTX 5090 (capability 12.0) the same sources are compiled
for ``sm_120a`` into separate extensions (names suffixed ``_sm120``) with ``-DMWE_SM120=1``, which the shared sources use behind compile-time
guards for what differs on that card (99 KB of dynamic shared memory per block instead of 163 KB, no A100-searched cuBLASLt algorithms).

The card is the current CUDA device's, or the one ``planning_for`` names while the prebuild plans a target (no device there)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

_SM80_LITERAL = "-gencode=arch=compute_80,code=sm_80"
_SM120_LITERAL = "-gencode=arch=compute_120a,code=sm_120a"
DEFINE = "-DMWE_SM120=1"

_FORCED: list[str] = []


@contextlib.contextmanager
def planning_for(arch: str) -> Iterator[None]:
    """``arch`` ("sm80" / "sm120") is the card the loaders called inside build for."""
    _FORCED.append(arch)
    try:
        yield
    finally:
        _FORCED.pop()


def sm120() -> bool:
    """The builds are the RTX 5090's (sm_120a)."""
    if _FORCED:
        return _FORCED[-1] == "sm120"
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def name(base: str) -> str:
    """The extension name: ``base`` on the A100, ``base + "_sm120"`` on the RTX 5090."""
    return base + "_sm120" if sm120() else base


def cflags(flags: list[str]) -> list[str]:
    """``flags`` unchanged on the A100; on the RTX 5090 the sm_80 gencode (literal or ``_nvcc.gencodes("80")``) becomes sm_120a, plus ``DEFINE``."""
    if not sm120():
        return list(flags)
    from miniworld_engine.kernels._codegraph import GENCODE_MARKER

    out = []
    for f in flags:
        if f == _SM80_LITERAL:
            out.append(_SM120_LITERAL)
        elif f == f"{GENCODE_MARKER}80":
            out.append(f"{GENCODE_MARKER}120a")
        else:
            out.append(f)
    if out == list(flags):
        raise ValueError("no sm_80 gencode in the flags to retarget for sm_120a")
    return [*out, DEFINE]


def device_ok(t: torch.Tensor) -> bool:
    """A CUDA tensor on an A100 (8.0) or an RTX 5090 (12.0): the cards these builds serve."""
    return t.is_cuda and torch.cuda.get_device_capability(t.device) in ((8, 0), (12, 0))


def prebuild_sm120(rows_opm: bool = True, rows_pwa: bool = True, rows_chai: bool = True, rows_wide: bool = True):
    """The RTX 5090 builds the OuterProductMean / MSAPairWeightedAveraging ``cuda`` paths reach (for ``kernels/_prebuild.py``'s sm120 target;
    ``kernels/integrations.prebuild`` yields these for the sm120 arch): each loader called under ``planning_for("sm120")``."""
    import functools

    def under(fn):
        def call():
            with planning_for("sm120"):
                return fn()

        functools.update_wrapper(call, fn)
        return call

    from miniworld_engine.kernels.outer_product_mean.cuda import chai_sm80, mask_count_ballot_sm80, mask_epilogue_sm80, ogemm_sm80
    from miniworld_engine.kernels.outer_product_mean.cuda import sm80 as opm
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80 as pwa
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120, wide_sm80

    loaders = []
    if rows_opm:
        loaders += [opm.ext, ogemm_sm80.ext, mask_count_ballot_sm80._ext, mask_epilogue_sm80._ext, gemm_sm120.ext]
    if rows_chai:
        loaders += [chai_sm80.ext, ogemm_sm80.ext, gemm_sm120.ext]
    if rows_pwa:
        loaders += [pwa.ext, gemm_sm120.ext]  # (the cuBLAS broadcast_bmm is not reached on the 5090: gemm_sm120 serves the contraction)
    if rows_wide:
        from miniworld_engine.kernels.integrations import pwa_wide_sm80
        from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90_generic import _lt

        loaders += [wide_sm80.ext, pwa_wide_sm80._glue_sm120, _lt, gemm_sm120.ext]
    seen = set()
    for fn in loaders:
        if fn not in seen:
            seen.add(fn)
            yield under(fn)
