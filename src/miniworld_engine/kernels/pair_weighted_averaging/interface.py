"""Public entry point for the pair-weighted-averaging family.

MSAPairWeightedAveraging averages the MSA's value projection over the keys of each token with softmax weights taken from the
pair representation, gates it and projects it back: msa + dropout(Wo (sigmoid(g) * softmax(LN(pair) Wb) (LN(msa) Wv))). The
Triton path keeps the two contractions over keys in cuBLAS and fuses every LayerNorm, projection, gate and reduction around them.
"""
from __future__ import annotations

from miniworld_engine.kernels.pair_weighted_averaging.triton.main import (
    refusal,
    triton_pair_weighted_averaging,
)

__all__ = ["refusal", "triton_pair_weighted_averaging"]
