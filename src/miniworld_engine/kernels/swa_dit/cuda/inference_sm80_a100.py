"""A100 BF inference paths selected by physical geometry and layout."""

import functools
from pathlib import Path
import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)
from miniworld_engine.kernels.swa_dit.cuda import sm80

_GEOMETRY_KERNELS = {
    (1, 1024): ("singlehead", 0, "tuned"),
    (1, 2048): ("singlehead", 0, "tuned"),
    (1, 4096): ("singlehead", 0, "tuned"),
    (5, 1024): ("headgroup", 1, "tuned"),
    (5, 2048): ("headgroup", 1, "original"),
    (5, 4096): ("original", 1, "tuned"),
}


@functools.lru_cache(maxsize=1)
def qkvg_extension():
    ensure_cuda_home()
    d = Path(__file__).parent
    return load_extension(
        name="swa_a100_inference_qkvg_v2_sm80",
        sources=[str(d / "inference_qkvg_sm80_a100.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{d}",
            "-Xcompiler=-fno-gnu-unique",
            "-Xptxas=-v",
        ],
        extra_cflags=["-std=c++17", "-O3", "-fno-gnu-unique"],
        verbose=True,
    )


@functools.lru_cache(maxsize=1)
def ffn_extension():
    ensure_cuda_home()
    d = Path(__file__).parent
    public = Path(sm80.__file__).parent
    return load_extension(
        name="swa_a100_inference_ffn_v2_sm80",
        sources=[str(d / "inference_ffn_sm80_a100.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{d}",
            f"-I{public}",
            "-Xcompiler=-fno-gnu-unique",
            "-Xptxas=-v",
        ],
        extra_cflags=["-std=c++17", "-O3", "-fno-gnu-unique"],
        verbose=True,
    )


def _block_fake(x, c, cos, sin, seqused, wmod, wq, wg, wo, wu, wd):
    """Same shape and dtype as ``x`` ([A, S, 128] block output), freshly allocated."""
    return torch.empty_like(x)


@opaque(fake=_block_fake, name="swa_dit_inference_block_sm80")
def block(
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
    """A100 SAVE=false inference block (modulation, qkvg, attention, FFN) selected by the physical geometry."""
    a, s, _ = x.shape
    b = c.shape[0]
    eps = float(torch.finfo(torch.float32).eps)
    qkv_kind, qkv_cfg, ffn_kind = _GEOMETRY_KERNELS[(a, s)]
    mod = sm80.mod_fwd(c.reshape(-1, 128).contiguous(), wmod, save=False)[0]
    qf = x.reshape(-1, 128)
    cos2, sin2 = (
        cos[:b].reshape(-1, 16).contiguous(),
        sin[:b].reshape(-1, 16).contiguous(),
    )
    if qkv_kind == "original":
        qh, kh, vh, g, *_ = sm80.qkvg_fwd(qf, mod, cos2, sin2, wq, wg, b, s, eps, eps, False)
    elif qkv_kind == "singlehead":
        qh, kh, vh, g = qkvg_extension().singlehead(qf, mod, cos2, sin2, wq, wg, s, b, eps, eps, False, qkv_cfg)
    else:
        qh, kh, vh, g = qkvg_extension().headgroup(qf, mod, cos2, sin2, wq, wg, s, b, eps, eps, False, qkv_cfg)
    o, _ = sm80.attn_fwd(qh, kh, vh, seqused, save_lse=False)
    if ffn_kind == "original":
        out, *_ = sm80.ffn_fwd(qf, g, o, mod, wo, wu, wd, b, s, eps, False)
    else:
        out = ffn_extension().compute(qf, g, o, mod, wo, wu, wd, s, b, eps, 0)
    return out.view(a, s, 128)


def eligible(x, c, ap, weights, n_head, half_window):
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 3
        or x.shape[2] != 128
        or (x.shape[0], x.shape[1]) not in _GEOMETRY_KERNELS
        or c.shape != x.shape
        or c.dtype != torch.bfloat16
        or n_head != 4
        or half_window != 64
        or len(ap) != 6
    ):
        return False
    if any(t.device != x.device or t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x, c, *weights)):
        return False
    expected = ((768, 128), (384, 128), (128, 128), (128, 128), (512, 128), (128, 256))
    if len(weights) != 6 or any(tuple(w.shape) != shape for w, shape in zip(weights, expected)):
        return False
    a, n = x.shape[:2]
    if any(t.device != x.device or t.dtype != torch.float32 or not t.is_contiguous() or tuple(t.shape) != (a, n, 16) for t in ap[:2]):
        return False
    if (
        ap[2].dtype != torch.int32
        or ap[2].device != x.device
        or tuple(ap[2].shape) != (a,)
        or not ap[2].is_contiguous()
        or ap[3].dtype != torch.int32
        or ap[3].device != x.device
        or tuple(ap[3].shape) != (a + 1,)
        or not ap[3].is_contiguous()
        or ap[4] != n
        or ap[5].dtype != torch.bool
        or ap[5].device != x.device
        or tuple(ap[5].shape) != (a, n)
        or not ap[5].is_contiguous()
    ):
        return False
    from miniworld_engine.kernels.swa_dit.dispatch import _sm80

    return _sm80(x, 256, half_window, float(torch.finfo(torch.float32).eps)) is not None
