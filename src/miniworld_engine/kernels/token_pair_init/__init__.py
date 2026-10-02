"""Fused token-pair initialisation of the input feature embedder (see ``interface``)."""
from miniworld_engine.kernels.token_pair_init.interface import refusal, token_pair_init
from miniworld_engine.kernels.token_pair_init.reference import token_pair_init_reference

__all__ = ["refusal", "token_pair_init", "token_pair_init_reference"]
