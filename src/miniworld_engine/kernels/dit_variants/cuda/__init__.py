"""CUDA kernels of the ATOM DiT block variants, built on first use (``load_extension``), never at import.

H100: ``edge.cu`` (the slid key window's five edge windows, forward and backward) for sm_90a, read by the one atom DiT implementation
(``kernels/integrations/local_dit_h100``); every atom DiT block runs there. A100: ``train.py`` (autograd Functions over ``train_rows.cu``,
``edge.cu`` and the mma.sync window attention) on the sm_80 builds of ``sm80.py``. The token blocks run the one token DiT algorithm
(``kernels/integrations/token_dit``)."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def edge_ext():
    """The atom stacks' slid-window edge kernels (``edge.cu``): sm_90a, built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="dit_variants_edge_sm90a",
        sources=[str(_dir / "edge.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("90a"),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def prebuild(target, rows):
    """Every build of the DiT-variant family (kernels/_prebuild.py) the requirements rows ask for (the builds take no model dims). H100:
    ``edge_ext`` (the slid edge windows of local_dit_h100). Which build serves which card is this family's knowledge."""
    if target.arch == "sm80":
        # A100: the atom blocks' training path (every atom call, inference included: cuda/sm80.py); the A100 TokenDiTBoltz2 attention is
        # augmented_attention's
        if rows:
            import functools as ft

            from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

            from . import sm80

            yield sm80.train_ext
            yield sm80.edge_ext
            yield ft.partial(sm80.cubin, "lattn90")
            yield ft.partial(sm80.cubin, "lattn90_tf32")
            yield ft.partial(sm80_atom_local.cubin, "lbias")
        return
    if rows:
        yield edge_ext
