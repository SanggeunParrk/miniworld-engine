"""E5: registry of the external TRIANGLE-MULTIPLICATION backends (one line per backend; the coordinator merges it into externals.py).

Each adapter lives in its own ``ext_<name>.py`` and defines ``PROBE`` (the import it needs), ``supports(self, pair, mask, dropout_p)`` (None or the
reason it does not serve the call) and ``forward(self, pair, mask, dropout_p)`` (the module's whole update incl. residual). ``install(names)``
attaches them to ``TriangleMultiplication`` exactly as ``externals.install`` does for the attention classes (nothing under src/ changes). A name
here is the measurer backend name; ``run_ext_trimul.py`` routes ``--ext <name>`` to this registry.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REGISTRY = {
    "fast_trimul": "ext_fast_trimul",  # tiagomonteiro0715/fast_trimul 3.0.6 (Apache-2.0): CuTe DSL fp16 fused update
    "gpumode": "ext_gpumode_trimul",  # arseniivanov/trimul (MIT): GPU MODE TriMul competition Triton kernels (A100 / H100 / B200 files)
    "geonn": "ext_geonn_trimul",  # ROCm/geometric-nn-ops 1.0.0 (Apache-2.0): Triton forward, cuEquivariance signature
}
AVAILABLE = tuple(REGISTRY)
_HERE = Path(__file__).resolve().parent


def add_paths(third_party_dir: str) -> None:
    """Make the fetched upstream tree importable: $PAPER_EXT_PATH (default ~/paper-ext), then the repo's third_party/<dir>/upstream."""
    ext = os.environ.get("PAPER_EXT_PATH") or os.path.expanduser("~/paper-ext")
    for p in (ext, str(_HERE.parents[2] / "third_party" / third_party_dir / "upstream")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)  # after site-packages: the engine environment wins for every package it has


def library_refusal(self, pair, mask) -> str | None:
    """What every adapter needs: [B, L, L, d] pair, a bool [B, L] residue mask, one LayerNorm epsilon."""
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return "mask must be a bool [B, L] residue mask"
    if self.ln_pair.eps != self.ln_out.eps:
        return "the library op takes one LayerNorm epsilon for LN_in and LN_out"
    if not torch.cuda.is_available():
        return "no CUDA device"
    return None


def import_error(module: str, probe: str) -> str | None:
    try:
        importlib.import_module(probe)
    except Exception as exc:  # recorded as the reason the backend does not serve the case
        return f"{probe}: {type(exc).__name__}: {str(exc)[:240]} ({module}; see third_party/ and E5_other_kernels/README.md)"
    return None


def install(names) -> None:
    import miniworld_engine.modules.triangle as tri

    for name in names:
        if name not in REGISTRY:
            raise ValueError(f"unknown external trimul backend {name!r}; choose from {AVAILABLE}")
        mod = importlib.import_module(REGISTRY[name])
        cls = tri.TriangleMultiplication
        if name not in cls.backends:
            cls.backends = (*cls.backends, name)

        def supports(self, pair, mask=None, dropout_p=None, _m=mod):
            why = library_refusal(self, pair, mask) or import_error(_m.__name__, _m.PROBE)
            return why or _m.supports(self, pair, mask, dropout_p)

        def fwd(self, pair, mask=None, dropout_p=None, _m=mod):
            return _m.forward(self, pair, mask, dropout_p)

        setattr(cls, f"_supports_{name}", supports)
        setattr(cls, f"_forward_{name}", fwd)


import torch
