"""A100 (sm_80) hand-CUDA kernels of the ProteinMPNN hidden-message reduction (``reduced[g] = sum_k mask[g, k] gelu(gelu(P[g, k]) W^T + b) / scale``).

``cuda/sm80/msg_fwd_sm80.cuh``: the forward and the no-grad inference in ONE kernel (GELU, the 128 x 128 projection on the tensor cores, bias, GELU, mask, neighbour sum; nothing edge-sized is written);
the backward is ONE kernel as well, ``msg_bwd_fused_sm80.cuh`` (dP, dW and db; the older two-step form ``msg_bwd_sm80.cuh`` + a cuBLAS GEMM stays reachable).  All run the GELUs through the cheap exact form of
``mpnn_common.cuh``.  Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

NEIGHBORS = 48
WIDTH = 128


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_MPNN_MSG_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_MPNN_MSG_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"mpnn_message_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=bool(extra),
    )


def forward(p: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor, scale: int) -> torch.Tensor:
    """``reduced`` [groups, 128] fp32 from P [groups * 48, 128] bf16 (contiguous), weight [128, 128] / bias [128] bf16, mask [groups * 48] fp32."""
    return _ext().msg_fwd(p, weight, bias, mask, float(scale))


#: groups per backward launch of the split variant: the two temporaries of the weight-gradient GEMM (a = bf16(gelu(P)) and dproj) are chunk-sized, not edge-tensor-sized (the Triton path chunks at 262144 rows)
CHUNK_GROUPS = 8192
FUSED_TILES_PER_STAGE = 8     # msg_bwd_fused_sm80.cuh: MsgBwdFCfg::NW


def backward(p: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor, grad_reduced: torch.Tensor,
             scale: int, fused: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dP [rows, 128] bf16, dW [128, 128] fp32, db [128] fp32)`` of :func:`forward`, from P [groups * 48, 128] and ``grad_reduced`` [groups, 128] fp32.

    One kernel (``msg_bwd_fused_sm80.cuh``): it replays the forward (nothing was saved), writes dP, and accumulates dW = dproj^T a on the tensor cores in registers (per-CTA slices of fp32, summed in CTA order by a
    small kernel: no atomics, bit-reproducible); db is the fixed-order sum of the CTAs' partial rows.  ``fused=False`` (or MINIWORLD_MPNN_MSG_SM80_BWD=split) selects the older two-step form for A/B runs: the
    kernel of ``msg_bwd_sm80.cuh`` writes dP, a and dproj and dW is one cuBLAS GEMM per chunk of rows."""
    if fused is None:
        fused = os.environ.get("MINIWORLD_MPNN_MSG_SM80_BWD", "fused") != "split"
    if fused:
        return _backward_fused(p, weight, bias, mask, grad_reduced, scale)
    return _backward_split(p, weight, bias, mask, grad_reduced, scale)


def _backward_fused(p, weight, bias, mask, grad_reduced, scale):
    groups = p.shape[0] // NEIGHBORS
    ext = _ext()
    num_sms = torch.cuda.get_device_properties(p.device).multi_processor_count
    ctas = min(num_sms, -(-groups * 3 // FUSED_TILES_PER_STAGE))
    dp = torch.empty_like(p)
    dw_part = torch.empty((ctas, WIDTH * WIDTH), device=p.device, dtype=torch.float32)
    db_part = torch.empty((ctas, WIDTH), device=p.device, dtype=torch.float32)
    ext.msg_bwd_fused(p, weight, bias, mask, grad_reduced, dp, dw_part, db_part, float(scale))
    return dp, ext.dw_reduce(dw_part, ctas), db_part.sum(dim=0)


def _backward_split(p, weight, bias, mask, grad_reduced, scale):
    groups = p.shape[0] // NEIGHBORS
    ext = _ext()
    num_sms = torch.cuda.get_device_properties(p.device).multi_processor_count
    dp = torch.empty_like(p)
    chunk = min(groups, CHUNK_GROUPS)
    nchunks = -(-groups // chunk)
    a_buf = p.new_empty((chunk * NEIGHBORS, WIDTH))
    d_buf = p.new_empty((chunk * NEIGHBORS, WIDTH))
    db_part = torch.zeros((nchunks, num_sms, WIDTH), device=p.device, dtype=torch.float32)
    dw = None
    for c in range(nchunks):
        g0, g1 = c * chunk, min(groups, (c + 1) * chunk)
        r0, r1 = g0 * NEIGHBORS, g1 * NEIGHBORS
        n = r1 - r0
        ext.msg_bwd(p[r0:r1], weight, bias, mask[r0:r1], grad_reduced[g0:g1], dp[r0:r1], a_buf[:n], d_buf[:n], db_part[c], float(scale))
        part = torch.mm(d_buf[:n].t(), a_buf[:n], out_dtype=torch.float32)
        dw = part if dw is None else dw.add_(part)
    return dp, dw, db_part.sum(dim=(0, 1))


def gelu_test(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The device ``gelu`` and ``gelu'`` of ``mpnn_common.cuh`` on an fp32 tensor (tests)."""
    g, d = _ext().gelu_test(x.contiguous())
    return g, d
