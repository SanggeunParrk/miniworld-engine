"""The portable Triton family uses no Hopper-only data movement: no TMA tensor descriptor in any Triton file.

H100 is covered by hand CUDA, so the Triton kernels exist for the other cards and may not need ``tl.make_tensor_descriptor`` /
``TensorDescriptor`` (a TMA feature of sm_90 and newer). This test reads the sources (no GPU) of every ``kernels/*/triton/`` tree and of the
shared Triton helpers, and fails on a use of either name or on an import of a ``_tma`` module.
"""
from __future__ import annotations

from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
KERNELS = REPO / "src" / "miniworld_engine" / "kernels"
SHARED = ("_compile.py", "_tiles.py", "_dot_precision.py", "_tf32.py", "_cast.py", "_capture.py", "_fastmath.py")
FORBIDDEN = ("make_tensor_descriptor", "TensorDescriptor", "experimental_descriptor", "_tma import", "import _tma")


def _code(path: Path) -> list[tuple[int, str]]:
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if code.strip():
            out.append((n, code))
    return out


def test_no_tma_descriptor_in_a_triton_file():
    files = sorted(KERNELS.glob("*/triton/**/*.py")) + [KERNELS / s for s in SHARED if (KERNELS / s).is_file()]
    assert files
    bad = [f"{f.relative_to(KERNELS)}:{n}: {code.strip()}" for f in files for n, code in _code(f) if any(w in code for w in FORBIDDEN)]
    assert not bad, "TMA in the portable Triton family:\n  " + "\n  ".join(bad)
