"""A100 (sm_80) hand-CUDA kernels of the fused ProteinMPNN encoder node message

    reduced[g] = sum_k mask[g, k] gelu(gelu(q[g] + E[g, k] W1e^T + nb[idx[g, k]]) W2^T + b2) / scale          (bf16 roundings as the Triton kernel)

``cuda/sm80/node_fwd_sm80.cuh``: the forward and the no-grad inference in ONE kernel per group (two tensor-core GEMMs with the neighbour gather, both GELUs and the masked neighbour sum between and after
them; nothing edge-sized is written).  The backward is ``node_bwd_sm80.cuh``.  Built on first use (``load_extension``), never at import; the two weights are passed as contiguous bf16 [128, 128] (the edge block
of the packed W1 is a strided slice: ``integrations/mpnn_msg_sm80.py`` copies it, and casts fp32 parameters, which are 32 KB each).
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"
_common = Path(__file__).resolve().parents[2] / "mpnn_message" / "cuda" / "sm80"

WIDTH = 128
MAX_NEIGHBORS = 128


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_MPNN_NODE_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_MPNN_NODE_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"mpnn_node_message_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", f"-I{_common}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=bool(extra),
    )


#: rows per backward launch: the two temporaries of dW2 (dh and act) are chunk-sized, not edge-tensor-sized
CHUNK_ROWS = 393216


def backward(edge: torch.Tensor, query: torch.Tensor, nbt: torch.Tensor, idx: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, mask: torch.Tensor, grad_reduced: torch.Tensor,
             neighbors: int, scale: int) -> list[torch.Tensor]:
    """``[dedge [rows, 128] bf16, dquery [groups, 128] bf16, dnb [NN, 128] bf16, dW1e [128, 128] fp32, dW2 [128, 128] fp32, db2 [128] fp32]`` of :func:`forward` (operands as there, ``grad_reduced``
    [groups, 128] fp32).  The kernel replays the forward and writes dedge, dquery, dpre, dh and act; the weight gradients are cuBLAS GEMMs over the rows of each chunk (dW1e = dpre^T E, dW2 = dh^T act:
    fp32 outputs, accumulated across chunks in fp32), the neighbour gradient a deterministic segmented sum of the dpre rows over the edges sorted (stably) by the node they point at, db2 the fixed-order sum of
    the CTAs' partial rows."""
    ext = _ext()
    groups = query.shape[0]
    rows = groups * neighbors
    nn = nbt.shape[0]
    num_sms = torch.cuda.get_device_properties(edge.device).multi_processor_count
    dedge = torch.empty_like(edge)
    dpre = torch.empty_like(edge)
    dquery = torch.empty_like(query)
    chunk = max(1, min(groups, CHUNK_ROWS // neighbors))
    nchunks = -(-groups // chunk)
    dh_buf = edge.new_empty((chunk * neighbors, WIDTH))
    act_buf = edge.new_empty((chunk * neighbors, WIDTH))
    db_part = torch.zeros((nchunks, num_sms, WIDTH), device=edge.device, dtype=torch.float32)
    dw1 = dw2 = None
    for c in range(nchunks):
        g0, g1 = c * chunk, min(groups, (c + 1) * chunk)
        r0, r1 = g0 * neighbors, g1 * neighbors
        n = r1 - r0
        ext.node_bwd(edge[r0:r1], query[g0:g1], nbt, idx[r0:r1], w1, w2, b2, mask[r0:r1], grad_reduced[g0:g1], dedge[r0:r1], dpre[r0:r1], dh_buf[:n], act_buf[:n], dquery[g0:g1], db_part[c],
                     neighbors, float(scale))
        p1 = torch.mm(dpre[r0:r1].t(), edge[r0:r1], out_dtype=torch.float32)
        p2 = torch.mm(dh_buf[:n].t(), act_buf[:n], out_dtype=torch.float32)
        dw1 = p1 if dw1 is None else dw1.add_(p1)
        dw2 = p2 if dw2 is None else dw2.add_(p2)
    # the neighbour gradient: edges sorted stably by destination (ascending edge id inside a node: a fixed summation order), segment starts from the histogram (an integer scatter_add: exact, no sync)
    perm = torch.argsort(idx.to(torch.int32), stable=True)          # int32 keys: half the radix passes of int64 (the table has far fewer than 2^31 rows)
    counts = torch.zeros(nn, device=idx.device, dtype=torch.int64).scatter_add_(0, idx, torch.ones_like(idx))
    off = torch.nn.functional.pad(torch.cumsum(counts, 0), (1, 0))
    dnb = ext.nb_reduce(dpre, perm, off, nn)
    return [dedge, dquery, dnb, dw1, dw2, db_part.sum(dim=(0, 1))]


def forward(edge: torch.Tensor, query: torch.Tensor, nbt: torch.Tensor, idx: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, mask: torch.Tensor, neighbors: int,
            scale: int) -> torch.Tensor:
    """``reduced`` [groups, 128] fp32 from edge [groups * K, 128], query [groups, 128], the neighbour table nbt [NN, 128] (all bf16), idx [groups * K] int64 (rows of the table), the contiguous bf16
    weights w1 (the edge block of W1) / w2 [128, 128], b2 [128] and mask [groups * K] (fp32 or bf16)."""
    return _ext().node_fwd(edge, query, nbt, idx, w1, w2, b2, mask, neighbors, float(scale))
