"""Public entry point for the outer-product-mean family.

OuterProductMean turns an MSA stack into a pair update: LN -> left/right projections -> mask -> the outer product averaged over
the MSA rows -> the c_hidden^2 -> c_z projection. The Triton path keeps the O(L^2 S) outer product in cuBLAS and fuses the rest
into four kernels (two forward, three backward) so no [L, L, c_hidden^2] tensor is ever materialised in the pair layout.
"""
from __future__ import annotations

from miniworld_engine.kernels.outer_product_mean.triton.main import (
    refusal,
    triton_outer_product_mean,
)

__all__ = ["refusal", "triton_outer_product_mean"]
