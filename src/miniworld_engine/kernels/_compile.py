"""How a kernel entry point is exposed to ``torch.compile``. One switch, one wrapper.

A Triton or CuTeDSL launcher cannot be traced by Dynamo, so every entry point has to declare what
Dynamo should do with it. There are exactly two answers and they are interchangeable -- the kernel
runs the same and returns the same numbers either way:

``disable``    graph break: Dynamo stops, runs the launcher eagerly, resumes after. Default.
``custom_op``  the launcher is registered as an opaque ``torch.library`` op, so it stays in the
               compiled graph and the surrounding ops keep fusing across it.

This is a backend choice like ``layernorm_bwd_path`` or ``trimul_impl``, not a per-kernel property,
so it is one ``settings`` field applied uniformly rather than a decision hardcoded per file.

``disable`` is the default because it is what this repo measures faster: the ``custom_op`` variant
pays for saving activations as graph outputs (docs/benchmarking-cautions.md), and nothing here
drives the kernels through ``torch.compile`` today -- the benchmarks capture CUDA graphs manually,
which removes the launch overhead a graph break would otherwise cost.

The mode is read at import time because registration has to happen at import; changing it after the
kernel modules are loaded has no effect.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings


def opaque(fake=None, *, name: str | None = None):
    """Mark a kernel entry point as untraceable by Dynamo, per ``settings.compile_wrap``.

    ``fake(*args, **kwargs)`` returns output tensors of the right shape/dtype without doing the
    work. It is required only by ``custom_op`` mode -- Dynamo needs it to keep shape inference going
    across the opaque node -- and a missing one fails loudly there instead of silently downgrading
    to a graph break, which would make the mode a no-op that still looks enabled.
    """

    def decorate(fn):
        if settings.current().compile_wrap == "disable":
            return torch.compiler.disable(fn)
        if fake is None:
            raise ValueError(
                f"compile_wrap='custom_op' needs a fake implementation for {fn.__name__}; "
                f"pass opaque(fake=...) or run with compile_wrap='disable'")
        op = torch.library.custom_op(f"miniworld_engine::{name or fn.__name__}",
                                     fn, mutates_args=())
        op.register_fake(fake)
        return op

    return decorate
