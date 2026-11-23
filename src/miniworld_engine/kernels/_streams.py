"""Stream bookkeeping of the sm_80-family kernels (A100, RTX 5090) that fork a side stream."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _skips_in_capture(index: int) -> bool:
    """Only the sm_80-family cards (A100 sm_80, RTX 5090 sm_120) skip ``record_stream`` during a capture; the H100 and the B200 keep it.
    (Until 2026-11-23 the test was "not an H100", which took the B200 in too: its captured token DiT steps then raced in ``_cond_gemm``,
    whose side-stream GEMMs are not fork/join ordered -- AF3 5I28 n = 4 bf16 pLDDT 43.8 against 88.2 eager.)"""
    return torch.cuda.get_device_capability(index)[0] in (8, 12)


def record_stream(t: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """``t.record_stream(stream)``, except on the sm_80-family cards (A100, RTX 5090) while a CUDA graph is being captured: the capture pool frees nothing
    a side stream "still uses" until the capture ends, so every recorded temporary stayed live (a 48-block Protenix step graph needed 9+ GB more than
    eager and ran out of memory on 32 GB). The skip is safe only where the side stream is fork/join ordered around every use (joined back before the
    tensor is freed). ``conditioned_transition.cuda._cond_gemm`` is not: its next side-stream GEMM forks from an EARLIER event of the caller's stream (a
    block's start), so an unrecorded block is handed to it while the current block's row passes still read it -- a race in every replay (open on the
    sm_80 family). The H100 and the B200 always record (their capture-replay tests, ``test_token_dit_cond_pipe_capture``, depend on it)."""
    if t.is_cuda and torch.cuda.is_current_stream_capturing() and _skips_in_capture(t.device.index if t.device.index is not None else torch.cuda.current_device()):
        return
    t.record_stream(stream)
