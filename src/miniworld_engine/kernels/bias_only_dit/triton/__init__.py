"""Triton backend of the bias-only token DiT: the portable kernels every card other than H100 / B200 runs.

The block has no Triton kernel of its own: its portable parts are the bias-only attention kernels (the pair LayerNorm + bias projection and the
training core), which ``modules/bias_only_dit`` names through this namespace. They live in ``kernels/bias_only_attention/triton`` and are
re-exported here unchanged, so the tuned caches, the registry rows and the record gate keep their identities.
"""

from __future__ import annotations

from miniworld_engine.kernels.bias_only_attention.triton.core_t import bias_only_core_t
from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import pair_bias_logits, supported

__all__ = ["bias_only_core_t", "pair_bias_logits", "supported"]
