"""Side streams inside a CUDA-graph capture (B200): independent parts of a step as parallel branches of the captured graph.

A replay runs the graph's branches concurrently; a step captured on one stream is a chain, every kernel after the one before it. The
bias-only token DiT has two parts that read nothing of each other until the attention core -- the pair part (pair LayerNorm + bias
projection + softmax, and its backward) and the conditioning / single part -- so a capturing step forks the pair part onto a stream of
its own and joins it where its result is first read (2026-11-17, docs/b200/bias-only.md).

Rules of use (the allocator keys its free blocks by stream, so a block freed and reused on ONE stream is ordered by that stream):

* ``fork`` orders the side stream after everything launched so far on the current stream;
* a tensor a side kernel reads or writes stays referenced until ``join`` (a block freed earlier could be handed to a later kernel of the
  current stream that is not ordered after the side kernel);
* nothing on the current stream reads a side kernel's output before ``join``.

Eager calls (no capture) and the other cards never fork: ``fork`` returns None and ``on(None)`` / ``join(.., None)`` do nothing.
"""

from __future__ import annotations

import contextlib

import torch


def fork(device):
    """A new stream ordered after the current one's work so far when that stream is capturing a CUDA graph on a B200, else None."""
    if torch.cuda.get_device_capability(device) != (10, 0) or not torch.cuda.is_current_stream_capturing():
        return None
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    return s


def on(stream):
    """The context that launches on ``stream`` (None: the current stream, unchanged)."""
    return contextlib.nullcontext() if stream is None else torch.cuda.stream(stream)


def join(device, stream) -> None:
    """The current stream waits for ``stream`` (None: nothing)."""
    if stream is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
