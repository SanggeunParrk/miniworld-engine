"""Accuracy checks for the ``mpnn_edge_tail`` family.

The inputs come from this family's DRIVER (`drivers.mpnn_edge_tail._graph`), never from a builder
of this module's own: a reference built on different values answers a different question. The two
sibling mpnn checker modules take the same shape from the same place, which is why that builder
lives in the driver rather than here -- `checks/` modules are peers and none imports another.

Dropout is off in all of these. It is a real part of the forward and `_project_output` compiles
differently with it on, but its decision is drawn from a seed inside the kernel and the reference
cannot reproduce that draw -- `edge_tail_update_pytorch` takes an explicit `keep_mask` for exactly
that reason, and no mask the checker can build is the one the kernel drew. `p=0` is a shape the
model runs (inference, and the shipped `dropout=0.0` configurations), so this compares the
arithmetic the kernels do rather than a coin flip neither side can agree on. The BACKWARD checkers
below are where the dropout-carrying kernels earn their band, because backward reads the mask the
forward stored rather than drawing a new one.
"""
from __future__ import annotations

from miniworld_engine.kernels.checks import _EPS, _grads
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _graph


def _edge_tail_pair(backend: str):
    """(kernel output, fp32 reference) for one encoder edge tail at `backend`'s policy."""
    from miniworld_engine.kernels.mpnn_edge_tail.interface import edge_tail_update
    from miniworld_engine.kernels.mpnn_edge_tail.reference import (
        edge_tail_update_pytorch,
    )

    t = _graph()
    out = edge_tail_update(**t, eps=_EPS, dropout_probability=0.0, backend=backend)
    ref = edge_tail_update_pytorch(
        *(v.float() if v.is_floating_point() else v for v in (
            t["edge_states"], t["query_projection"], t["neighbor_projection"],
            t["flat_neighbor_indices"], t["edge_weight"], t["hidden_weight"],
            t["hidden_bias"], t["output_weight"], t["output_bias"],
            t["norm_weight"], t["norm_bias"])),
        None, _EPS, 0.0,
    )
    return out, ref


def _edge_tail_grads(backend: str):
    """Every gradient of one encoder edge tail against fp32 autograd on the same values."""
    from miniworld_engine.kernels.mpnn_edge_tail.interface import edge_tail_update
    from miniworld_engine.kernels.mpnn_edge_tail.reference import (
        edge_tail_update_pytorch,
    )

    t = _graph(grad=True)
    names = ("edge_states", "query_projection", "neighbor_projection", "edge_weight",
             "hidden_weight", "hidden_bias", "output_weight", "output_bias",
             "norm_weight", "norm_bias")
    leaves = [t[n] for n in names]
    index, seed = t["flat_neighbor_indices"], t["seed"]

    def kernel(*args):
        vals = dict(zip(names, args, strict=True))
        return edge_tail_update(
            flat_neighbor_indices=index, seed=seed, eps=_EPS, dropout_probability=0.0,
            backend=backend, **vals)

    def reference(*args):
        vals = dict(zip(names, args, strict=True))
        return edge_tail_update_pytorch(
            vals["edge_states"], vals["query_projection"], vals["neighbor_projection"],
            index, vals["edge_weight"], vals["hidden_weight"], vals["hidden_bias"],
            vals["output_weight"], vals["output_bias"], vals["norm_weight"],
            vals["norm_bias"], None, _EPS, 0.0)

    return _grads(kernel, leaves, reference, names)


def mpnn_edge_tail_fwd_gemm_gather_saveact_triton():
    """_project_edge: the W1 GEMM with the query broadcast and the neighbour gather folded in."""
    return _edge_tail_pair("triton_compute")


def mpnn_edge_tail_fwd_gemm_b2b_saveact_triton():
    """_project_hidden: the second projection, reached only through the same launcher."""
    return _edge_tail_pair("triton_compute")


def mpnn_edge_tail_fwd_gemm_layernorm_saveact_triton():
    """_project_output: the third projection with dropout, the residual and LayerNorm after it."""
    return _edge_tail_pair("triton_compute")


def mpnn_edge_tail_bwd_layernorm_saveact_triton():
    """_norm_backward, through every gradient the tail produces."""
    return _edge_tail_grads("triton_compute")


def mpnn_edge_tail_bwd_dx_saveact_triton():
    """_project_backward, which runs twice per launch -- once per inner projection."""
    return _edge_tail_grads("triton_compute")


def mpnn_edge_tail_bwd_dx_gather_saveact_triton():
    """_edge_backward: the last dX GEMM, the residual add and the scattered query gradient."""
    return _edge_tail_grads("triton_compute")


def mpnn_edge_tail_fwd_gemm_recompute_triton():
    """_edge_tail_project_kernel: the same forward, saving no activation."""
    return _edge_tail_pair("triton")


def mpnn_edge_tail_fwd_gemm_layernorm_recompute_triton():
    """_edge_tail_norm_kernel: the norm stage of the recompute policy's forward."""
    return _edge_tail_pair("triton")


def mpnn_edge_tail_bwd_recompute_triton():
    """_edge_tail_replay_kernel: the forward replayed inside backward, checked by its gradients."""
    return _edge_tail_grads("triton")


def mpnn_edge_tail_bwd_dx_recompute_triton():
    """_edge_tail_dx_kernel: the dX pass of the recompute policy."""
    return _edge_tail_grads("triton")
