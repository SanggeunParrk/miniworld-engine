"""Stream bookkeeping of the sm_80-family kernels (A100, RTX 5090) that fork a side stream."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _skips_in_capture(index: int) -> bool:
    """The card is not an H100: only the sm_80-family cards skip ``record_stream`` during a capture (the H100 keeps it, as shipped)."""
    return torch.cuda.get_device_capability(index) != (9, 0)


def record_stream(t: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """``t.record_stream(stream)``, except on the sm_80-family cards (A100, RTX 5090) while a CUDA graph is being captured: the capture pool frees nothing
    a side stream "still uses" until the capture ends, so every recorded temporary stayed live (a 48-block Protenix step graph needed 9+ GB more than
    eager and ran out of memory on 32 GB). A capture is fork/join ordered, and the pool never hands a block to anything outside the graph. The H100
    path always records (its capture-replay tests, ``test_token_dit_cond_pipe_capture``, depend on it)."""
    if t.is_cuda and torch.cuda.is_current_stream_capturing() and _skips_in_capture(t.device.index if t.device.index is not None else torch.cuda.current_device()):
        return
    t.record_stream(stream)
