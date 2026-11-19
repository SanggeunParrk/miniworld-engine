"""Stream bookkeeping of the sm_80-family kernels (A100, RTX 5090) that fork a side stream."""

from __future__ import annotations

import torch


def record_stream(t: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """``t.record_stream(stream)``, except while a CUDA graph is being captured: the capture pool frees nothing a side stream "still uses" until the
    capture ends, so every recorded temporary stayed live (a 48-block Protenix step graph needed 9+ GB more than eager and ran out of memory on
    32 GB). A capture is fork/join ordered, and the pool never hands a block to anything outside the graph."""
    if not torch.cuda.is_current_stream_capturing():
        t.record_stream(stream)
