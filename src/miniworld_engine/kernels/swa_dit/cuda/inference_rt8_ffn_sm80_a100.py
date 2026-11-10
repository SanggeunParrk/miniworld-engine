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
        name="swa_a100_inference_rt8_ffn_sm80",
        sources=[str(d / "inference_rt8_ffn_sm80_a100.cu")],
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


@opaque(fake=_ffn_fake, name="swa_dit_inference_rt8_ffn_sm80")
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
    """SAVE=false FFN stage (row-tile geometry ``cfg``) of the A100 inference block; returns the block output rows like ``qi``."""
    return extension().compute(qi, g, o, mod, wo, wu, wd, S, B, eps, cfg)


# Public owners remain exact; original official MINIWORLD/TRITON enum is not an engine-backend override.
from miniworld_engine.kernels.swa_dit.cuda import inference_sm80_a100 as baseline

_GEOMETRY_KERNELS = baseline._GEOMETRY_KERNELS
qkvg_extension = baseline.qkvg_extension
_SELECT = {(1, 2048): 3, (5, 1024): 3}


def forward(
    x: torch.Tensor,
    c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqused: torch.Tensor,
    wmod: torch.Tensor,
    wq: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
) -> torch.Tensor:
    a, s, _ = x.shape
    b = c.shape[0]
    eps = float(torch.finfo(torch.float32).eps)
    qkv_kind, qkv_cfg, ffn_kind = _GEOMETRY_KERNELS[a, s]
    mod = sm80.mod_fwd(c.reshape(-1, 128).contiguous(), wmod, save=False)[0]
    qf = x.reshape(-1, 128)
    cos2, sin2 = (
        cos[:b].reshape(-1, 16).contiguous(),
        sin[:b].reshape(-1, 16).contiguous(),
    )
    if qkv_kind == "original":
        qh, kh, vh, g, *_ = sm80.qkvg_fwd(
            qf, mod, cos2, sin2, wq, wg, b, s, eps, eps, False
        )
    elif qkv_kind == "singlehead":
        qh, kh, vh, g = qkvg_extension().singlehead(
            qf, mod, cos2, sin2, wq, wg, s, b, eps, eps, False, qkv_cfg
        )
    else:
        qh, kh, vh, g = qkvg_extension().headgroup(
            qf, mod, cos2, sin2, wq, wg, s, b, eps, eps, False, qkv_cfg
        )
    o, _ = sm80.attn_fwd(
        qh, kh, vh, seqused, save_lse=False
    )
    out = ffn(qf, g, o, mod, wo, wu, wd, s, b, eps, _SELECT[a, s])
    return out.view(a, s, 128)
