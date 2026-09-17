"""Check all three saved/output tensors, including nonuniform dropout rows."""

import torch

from miniworld_engine.kernels.drivers.trimul_output import operands


def output_f567_train():
    from miniworld_engine.kernels.trimul_inproj.triton.output_fused import (
        output_f567_train as launch,
    )

    norm, x, wp, wg, residual, ds, length = args = operands()
    y, proj, gate = launch(*args)
    p = norm.float() @ wp.float().t()
    g = torch.sigmoid(x.float() @ wg.float())
    rows = torch.arange(norm.shape[0], device=norm.device) % length
    expected = residual.float() + p * g * ds.float()[rows]
    return {"y": (y, expected), "proj": (proj, p), "gate": (gate, g)}
