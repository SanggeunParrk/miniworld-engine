"""Hand-written CUDA of the pair-weighted-averaging family: the A100 (sm_80) kernels (``sm80.py``, sources in ``sm80/``).

The H100 / B200 paths of this op live in ``integrations/pwa_train.py`` and ``integrations/csrc``; the portable Triton path is ``../triton``."""
