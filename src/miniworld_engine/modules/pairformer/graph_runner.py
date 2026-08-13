"""Bucketed CUDA-graph runner for variable-length Pairformer INFERENCE.

Our sm100 cute kernels are fastest under CUDA-graph replay (graph capture amortizes
the per-launch host cost). Training uses a fixed crop so one captured graph suffices,
but inference sees variable sequence length L, and a single graph is tied to one shape.

This wrapper does what AlphaFold3 does for the same problem — **compilation buckets +
padding** — with CUDA graphs instead of XLA executables:

  * capture ONE graph per length bucket (lazily, on first use), with static input /
    mask / output buffers at fixed addresses;
  * for a real length L, pick the smallest bucket ``Lb >= L``, copy the input into the
    top-left ``[:, :L, :L]`` of the static buffer, set the residue mask to valid on
    ``[:, :L]`` and padded on ``[:, L:]``, replay, and return ``out[:, :L, :L]``.

Correctness of the pad relies on the pair-mask being folded into LayerNorm (row_scale)
on the fast free path (see TriangleMultiplication / bidirectional ``_forward_cute_free``):
padded rows are zeroed, so padded residues never contribute to a valid position's
trimul k-contraction or triangle-attention keys. Padded output positions are garbage
and are sliced off.

Buckets share a single CUDA-graph memory pool to cap peak memory. B=1, bf16, eval/no_grad.
Requires the graph-capturability workarounds (transition b2b off on sm100, non-fused
triangle-attention gate) — apply_workarounds() sets them.
"""

from __future__ import annotations


import torch
import torch.nn as nn


def apply_workarounds() -> None:
    """Make every sub-module CUDA-graph-capturable (same as the bench runner).

    Both of these used to reach around the engine: one wrote an environment variable that another
    module would later read, the other rebound ``gate_use_fused`` on the module object. They are
    settings now, so a caller can see what capture changed -- and change it back.
    """
    import dataclasses

    from miniworld_engine import settings

    active = settings.current()
    defaults = {f.name: f.default for f in dataclasses.fields(settings.Settings)}
    changes = {}
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
        # setdefault semantics: an explicit choice by the caller still wins.
        if active.transition_cuda_b2b == defaults["transition_cuda_b2b"]:
            changes["transition_cuda_b2b"] = False
    if active.pin_gate_backend is None:
        changes["pin_gate_backend"] = "split"
    if changes:
        settings.configure(**changes)


class BucketedPairformer(nn.Module):
    """Variable-L inference wrapper around a :class:`Pairformer` (or any
    ``forward(pair[B,L,L,d], mask[B,L]|None) -> [B,L,L,d]`` module) using a
    per-bucket CUDA-graph cache with pad-to-bucket + mask.

    Parameters
    ----------
    model : nn.Module
        The pairformer to wrap. Put it in eval() bf16 on CUDA before wrapping.
    buckets : sequence[int]
        Ascending length buckets. An input L is padded up to the smallest bucket >= L.
    warmup : int
        Warmup iterations per bucket (compiles the per-shape kernels before capture).
    """

    def __init__(self, model: nn.Module, buckets=(256, 384, 512, 768, 1024), warmup: int = 8):
        super().__init__()
        apply_workarounds()
        self.model = model.eval()
        self.buckets = sorted(int(b) for b in buckets)
        self.warmup = warmup
        self._cache: dict[int, dict] = {}
        p = next(model.parameters())
        self._device, self._dtype = p.device, p.dtype
        # infer d_pair from the first block's config if present, else lazily at first call
        self._d = getattr(getattr(model, "config", None), "d_pair", None)

    def _capture(self, Lb: int, B: int, d: int) -> dict:
        dev, dt = self._device, self._dtype
        pair_in = torch.zeros(B, Lb, Lb, d, device=dev, dtype=dt)
        mask_in = torch.zeros(B, Lb, device=dev, dtype=torch.bool)
        # capture-time buffer values are irrelevant (a graph replays ops, reading the
        # static buffers' values at replay); the per-call code fills them before replay.

        # warmup on a side stream (builds/compiles the per-shape kernels)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(self.warmup):
                self.model(pair_in, mask_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Each bucket gets its OWN graph memory pool. Sharing one pool across buckets
        # that are replayed independently (not as an ordered pipeline) corrupts the
        # larger buckets — the pool reuses memory assuming graphs don't hold live state
        # across each other. Independent pools cost more memory but are correct.
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_out = self.model(pair_in, mask_in)
        cap = {"graph": graph, "pair_in": pair_in, "mask_in": mask_in, "out": static_out}
        self._cache[Lb] = cap
        return cap

    @torch.no_grad()
    def forward(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, L2, d = pair.shape
        assert B == 1 and L == L2, "B=1, square L"
        Lb = next((b for b in self.buckets if b >= L), None)
        if Lb is None:
            raise ValueError(f"L={L} exceeds largest bucket {self.buckets[-1]}")
        cap = self._cache.get(Lb) or self._capture(Lb, B, d)

        # pad input into the static buffer's top-left; mask valid on [:L], padded after.
        cap["pair_in"].zero_()
        cap["pair_in"][:, :L, :L].copy_(pair)
        cap["mask_in"].zero_()
        cap["mask_in"][:, :L] = True if mask is None else mask
        cap["graph"].replay()
        return cap["out"][:, :L, :L].clone()
