"""How a kernel entry point is exposed to ``torch.compile``. One switch, one wrapper.

A Triton or CuTeDSL launcher cannot be traced by Dynamo, so every entry point has to declare what
Dynamo should do with it. There are exactly two answers and they are interchangeable -- the kernel
runs the same and returns the same numbers either way:

``disable``    graph break: Dynamo stops, runs the launcher eagerly, resumes after. Default.
``custom_op``  the launcher is registered as an opaque ``torch.library`` op, so it stays in the
               compiled graph and the surrounding ops keep fusing across it.

This is a backend choice like ``layernorm_bwd_path`` or ``trimul_impl``, not a per-kernel property,
so it is one ``settings`` field applied uniformly rather than a decision hardcoded per file.

``custom_op`` is the default. ``disable`` held that spot only because it was the only mode that
could LOAD: 47 of the 58 entry points wrapped an ``autograd.Function`` method, which ``custom_op``
cannot register, so choosing it raised on import. Measured on an A6000 (Pairformer x4, L=384),
once every launch has a fake:

    training, torch.compile, no CUDA graph   164.4 ms -> 156.0 ms
    training, mode="reduce-overhead"         166.0 ms -> 155.1 ms   (was slower than eager)
    training, compile + captured graph       CRASHED  -> 153.8 ms

The last two are the point. A graph break does not merely cost fusion: inductor's cudagraph-trees
bail on one, so ``reduce-overhead`` silently degraded into something slower than eager, and a
manual capture over a compiled module died with ``cudaErrorStreamCaptureInvalidated`` because part
of the region dropped back to eager mid-capture. ``disable`` stays available for A/B, and as the
escape hatch if a fake is ever wrong -- it needs none.

The mode is read at import time because registration has to happen at import; changing it after the
kernel modules are loaded has no effect.


WHAT GOES IN AN OP
------------------
``custom_op`` has a contract, and it is narrower than "any function that launches a kernel":

* every returned tensor must be FRESHLY ALLOCATED -- returning an input, or a view/reshape of one,
  is a contract violation that ``torch.library.opcheck`` reports and that silently breaks
  functionalization;
* arguments and returns must be schema types (``Tensor``, ``Tensor | None``, ``int``, ``float``,
  ``bool``, ``str``, and lists/tuples of those). A ``jaxtyping`` annotation like
  ``Float[torch.Tensor, "... d"]`` is NOT one, and neither is an ``autograd`` ``ctx``;
* a ``fake`` must produce the same output STRUCTURE for the same non-tensor arguments, because it
  is what Dynamo traces -- it may branch on ``save_xn``, never on what the GPU turns out to be.

So the unit that gets wrapped is the LAUNCH, not the ``autograd.Function`` method that surrounds
it. ``forward(ctx, ...)`` cannot be an op (it takes ``ctx``, and it returns reshapes of its own
inputs), so each kernel splits in two: a plain launch function, wrapped here, and the
``autograd.Function`` that owns ``ctx`` and calls it. Dynamo traces THROUGH the
``autograd.Function`` -- input reshapes, autocast casts and ``save_for_backward`` all stay in the
graph -- and stops only at the launch. That is strictly more graph than putting ``@opaque`` on
``forward`` ever gave, and it is why the split is worth the churn rather than a formality.

A THIRD KIND: DEVICE CONSTANTS
------------------------------
Not every break comes from a launch. The per-GPU dispatch lookups (``gpu_key``, the calibrated
gate/layernorm choices) run INSIDE the forward and call ``torch.cuda.get_device_name`` /
``get_device_capability``, which return strings and tuples -- "torch.* op returned non-Tensor", a
graph break with no kernel anywhere near it. Those were the LAST breaks left in a pairformer block
after every launcher had a fake.

``device_constant`` marks them: Dynamo calls the function once while tracing and bakes the answer
in. That is sound because the answer is a property of the card and of a committed calibration
file, neither of which changes inside a run -- and it is the reason ``functools.lru_cache`` is not
enough, since Dynamo ignores the cache wrapper and re-traces the body underneath it.

WHAT IS *NOT* AN OP, AND WHY THAT IS FINE
-----------------------------------------
Two shapes cannot be registered at all: a bound method (an ``nn.Module`` is not a schema type)
and a wrapper whose body calls ``SomeFunction.apply`` (an op is opaque to AUTOGRAD as well as to
Dynamo, so registering one would return a tensor with no ``grad_fn`` -- forward numbers still
correct, training silently not learning through it).

Both were briefly marked as permanent graph breaks. That was wrong, and the fix is the point of
this module: such a function does not need to BE an op once everything it CALLS is one. Dynamo
traces straight through ``TriangleMultiplication._forward_triton`` and through ``bidir_forward``,
because every launch underneath them is opaque and nothing else in them is untraceable. Removing
those six method-level and four wrapper-level breaks is what took a pairformer block from 26
breaks to 0.

So there is no escape hatch here, deliberately: if something still breaks, the answer is to find
the launch underneath it and give THAT a fake.

``register_autograd`` is deliberately NOT used. It only lets ``setup_context`` save the op's inputs
and outputs, so every intermediate the backward needs (LN stats, the normalized activation) would
have to become a forward RETURN. Keeping ``autograd.Function`` keeps ``save_for_backward`` free to
save intermediates, and keeps the eager path byte-identical to what shipped.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings

#: Ops already registered in this process, by qualified name. ``torch.library.custom_op`` raises on
#: a duplicate name, and a module can legitimately be imported twice (a test that reconfigures
#: ``compile_wrap`` and reloads, ``importlib.reload``, a re-entrant plugin import). Re-registering
#: the same launcher is a no-op in intent, so hand back the op that is already there rather than
#: making import order decide whether the library loads at all.
_REGISTERED: dict[str, object] = {}


def _op_name(fn, name: str | None) -> str:
    """The ``miniworld_engine::`` op name for ``fn``.

    Defaults to the qualified name, not ``__name__``: the launchers live inside per-kernel modules
    and share short names (``_fwd``, ``_bwd``, ``forward``), and the op namespace is flat and
    global. ``TransitionFused._fwd`` -> ``TransitionFused__fwd`` is unique by construction, so no
    site has to invent -- or, worse, forget to invent -- a unique string.
    """
    if name is not None:
        return name
    qual = getattr(fn, "__qualname__", None) or fn.__name__
    return qual.replace(".", "__").replace("<", "").replace(">", "")


def opaque(fake=None, *, name: str | None = None, mutates_args=()):
    """Mark a kernel launch as untraceable by Dynamo, per ``settings.compile_wrap``.

    ``fake(*args, **kwargs)`` returns output tensors of the right shape/dtype/device without doing
    the work. It is required only by ``custom_op`` mode -- Dynamo needs it to keep shape inference
    going across the opaque node -- and a missing one fails loudly there instead of silently
    downgrading to a graph break, which would make the mode a no-op that still looks enabled.

    ``name`` overrides the op name (default: the decorated function's qualified name).
    ``mutates_args`` names arguments the launch writes into, for launchers that fill a caller-owned
    output buffer instead of allocating one.
    """

    def decorate(fn):
        if settings.current().compile_wrap == "disable":
            return torch.compiler.disable(fn)
        if fake is None:
            raise ValueError(
                f"compile_wrap='custom_op' needs a fake implementation for "
                f"{getattr(fn, '__qualname__', fn.__name__)}; pass opaque(fake=...) or run with "
                f"compile_wrap='disable'")
        qualname = f"miniworld_engine::{_op_name(fn, name)}"
        cached = _REGISTERED.get(qualname)
        if cached is not None:
            return cached
        op = torch.library.custom_op(qualname, fn, mutates_args=mutates_args)
        op.register_fake(fake)
        _REGISTERED[qualname] = op
        return op

    return decorate


def device_constant(fn):
    """A per-GPU lookup that is a CONSTANT to the compiled graph.

    Dynamo evaluates ``fn`` once at trace time and inlines the result, so a device-property query
    or a calibrated-choice lookup stops splitting the graph. Use it only for values that cannot
    change within a run: the card's identity, its compute capability, a choice read from the
    committed dispatch cache. A recompile re-evaluates it, so a value that legitimately changes
    between runs is still picked up -- but one that changes DURING a run would be missed, which is
    exactly why this is not the default for anything shape-dependent.
    """
    return torch._dynamo.assume_constant_result(fn)
