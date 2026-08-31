"""Drivers for the ``mpnn_edge_tail`` family, and the shape the three mpnn families share.

One module, not three, because the three families share one shape: a graph of `N` nodes with `k`
nearest neighbours each, at a channel width of 128. `_graph` builds it once and every driver below
takes its slice, the same way `drivers/fused_ln_mask.py` borrows `layernorm_linear`'s.

The LENGTH a unit drives is the NODE count, never the row count. An edge launch iterates `N * k`
rows -- 98,304 at the shipped crop of 2,048 -- and that is what `both_key` buckets, but it is not a
number anything in the model configures. `MINIWORLD_DRIVER_SIDE=edge` is how the builder says the
length it passed means nodes.

Several drivers here reach their kernel through a launcher that runs two or three of them. That is
deliberate and not a shortcut: `_launch_forward` IS the only way `_project_hidden` is ever launched,
so a driver that reached it any other way would be measuring a launch production does not make.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, DRIVER_LENGTH, dev

#: ProteinMPNN's hidden/node/edge dimension. 128 in every shipped configuration, and the kernels
#: read it as a `_WIDTH` constant rather than an argument.
_WIDTH = 128
#: k nearest neighbours. 48 is the source default and the only value the kernels are built for.
_NEIGHBORS = 48
#: Nodes, when nothing says otherwise. 2,048 is the shipped training crop (`CROP=2048`).
_NODES = 2048
#: Relative-position buckets. 66 is the shipped table height (`2 * 32 + 2`), and the reduction's
#: whole point is that the destination count is tiny against the 6.29M rows feeding it.
_BUCKETS = 66
#: The embedding's channel count, which is NOT 128: the relative-position table is 16 wide.
_BUCKET_WIDTH = 16


def _nodes() -> int:
    return DRIVER_LENGTH or _NODES


def _graph(nodes: int | None = None, *, grad: bool = False):
    """The tensors the edge tail and the node message both take, on the current device.

    Batch 1 on purpose: the shipped training launcher concatenates its eight examples along the
    length axis and hands the model `[1, sum(L_i), ...]`, so a physical batch is not a shape this
    model presents. Sweeping N is sweeping exactly what production varies.
    """
    n = nodes or _nodes()
    d, k = dev(), _NEIGHBORS
    act = lambda *shape: torch.randn(*shape, device=d, dtype=BF16, requires_grad=grad)
    par = lambda *shape: torch.randn(*shape, device=d, dtype=BF16, requires_grad=grad)
    return {
        "edge_states": act(1, n, k, _WIDTH),
        "query_projection": act(1, n, _WIDTH),
        "neighbor_projection": act(1, n, _WIDTH),
        "flat_neighbor_indices": torch.randint(0, n, (1, n, k), device=d, dtype=torch.long),
        "edge_weight": par(_WIDTH, _WIDTH),
        "hidden_weight": par(_WIDTH, _WIDTH),
        "hidden_bias": par(_WIDTH),
        "output_weight": par(_WIDTH, _WIDTH),
        "output_bias": par(_WIDTH),
        "norm_weight": par(_WIDTH),
        "norm_bias": par(_WIDTH),
        "seed": torch.randint(0, 2**31, (1,), device=d, dtype=torch.int64),
    }


def _edge_tail(backend: str, *, backward: bool) -> None:
    """Run one encoder edge tail, forward or forward+backward, on one of the two policies."""
    from miniworld_engine.kernels.mpnn_edge_tail.interface import edge_tail_update

    tensors = _graph(grad=backward)
    out = edge_tail_update(**tensors, eps=1e-5, dropout_probability=0.1, backend=backend)
    if backward:
        out.sum().backward()


def mpnn_edge_tail_fwd_gemm_gather_saveact_triton() -> None:
    _edge_tail("triton_compute", backward=False)


def mpnn_edge_tail_fwd_gemm_b2b_saveact_triton() -> None:
    _edge_tail("triton_compute", backward=False)


def mpnn_edge_tail_fwd_gemm_layernorm_saveact_triton() -> None:
    _edge_tail("triton_compute", backward=False)


def mpnn_edge_tail_bwd_layernorm_saveact_triton() -> None:
    _edge_tail("triton_compute", backward=True)


def mpnn_edge_tail_bwd_dx_saveact_triton() -> None:
    _edge_tail("triton_compute", backward=True)


def mpnn_edge_tail_bwd_dx_gather_saveact_triton() -> None:
    _edge_tail("triton_compute", backward=True)


def mpnn_edge_tail_fwd_gemm_recompute_triton() -> None:
    _edge_tail("triton", backward=False)


def mpnn_edge_tail_fwd_gemm_layernorm_recompute_triton() -> None:
    _edge_tail("triton", backward=False)


def mpnn_edge_tail_bwd_recompute_triton() -> None:
    _edge_tail("triton", backward=True)


def mpnn_edge_tail_bwd_dx_recompute_triton() -> None:
    _edge_tail("triton", backward=True)
