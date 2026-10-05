"""Hand-written A100 (sm_80) CUDA kernels of the ProteinMPNN encoder edge side (edge tail, edge MLP, edge LayerNorm and edge dropout share one extension).

``sm80.py`` builds the extension on first use (``load_extension``, never at import) and exposes it; ``integrations/mpnn_edge_sm80.py`` decides when the family
interfaces use it (sm_80, ``MINIWORLD_MPNN_EDGE_SM80`` not 0, engine backend not forced to Triton) and owns the autograd boundary.
"""
