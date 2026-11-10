"""Private opaque SAVEfalse FFN geometry; no module dispatch change."""

import functools, pathlib
import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels.swa_dit.cuda import sm80


@functools.lru_cache(maxsize=1)
def extension():
    ensure_cuda_home()
    d = pathlib.Path(__file__).parent
    return load_extension(
        name="swa_a100_inference_rowgrid_ffn_sm80",
        sources=[str(d / "inference_rowgrid_ffn_sm80_a100.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{d}",
            f"-I{pathlib.Path(sm80.__file__).parent}",
            "-Xcompiler=-fno-gnu-unique",
            "-Xptxas=-v",
        ],
        extra_cflags=["-std=c++17", "-O3", "-fno-gnu-unique"],
        verbose=True,
    )


def _ffn_fake(qi, g, o, mod, wo, wu, wd, S, B, eps, cfg):
    """Same shape and dtype as ``qi`` (the block output rows), freshly allocated."""
    return torch.empty_like(qi)


@opaque(fake=_ffn_fake, name="swa_dit_inference_rowgrid_ffn_sm80")
def ffn(
    qi: torch.Tensor,
    g: torch.Tensor,
    o: torch.Tensor,
    mod: torch.Tensor,
    wo: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    S: int,
    B: int,
    eps: float,
    cfg: int,
) -> torch.Tensor:
    """SAVE=false FFN stage (row-grid geometry ``cfg``) of the A100 inference block; returns the block output rows like ``qi``."""
    return extension().compute(qi, g, o, mod, wo, wu, wd, S, B, eps, cfg)
