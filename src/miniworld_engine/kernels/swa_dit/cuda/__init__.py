"""Hand-CUDA sm_90a bf16-wgmma kernels of the fused SWA atom DiT block, JIT-built by ``loader.py``.

``gen/gen_wgmma_bf16.py`` generates ``gen/wgmma_bf16.inc`` (the wgmma wrappers the three ``.cu`` files include); it is
kept beside its output and is not imported or shipped (``gen/`` is not a package and ``*.py`` is not package data)."""
