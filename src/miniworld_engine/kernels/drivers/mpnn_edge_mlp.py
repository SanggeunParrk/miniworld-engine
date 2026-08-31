"""Drivers for the ``mpnn_edge_mlp`` family.

The shape comes from `drivers/mpnn_edge_tail.py` -- see `DRIVER_SHAPE_OWNERS`. Two backends, and
they are two different kernels rather than two settings: `triton_memory` fuses both projections into one
launch and `triton_compute` runs them as two, which is why the family has two registry rows.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers.mpnn_edge_tail import _graph


def _edge_mlp(backend: str) -> None:
    from miniworld_engine.kernels.mpnn_edge_mlp.interface import edge_mlp_update

    t = _graph()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        edge_mlp_update(
            t["edge_states"], t["hidden_weight"], t["hidden_bias"],
            t["output_weight"], t["output_bias"], backend=backend,
        )


def mpnn_edge_mlp_fwd_gemm_b2b_triton() -> None:
    _edge_mlp("triton_memory")


def mpnn_edge_mlp_fwd_gemm_triton() -> None:
    _edge_mlp("triton_compute")
