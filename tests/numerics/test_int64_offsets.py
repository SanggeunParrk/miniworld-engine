"""Static guard: Triton kernels whose flat offset scales with the logical
sequence length ``M = B*L*L`` must promote the M-index to int64, or they issue
illegal memory accesses once ``M*stride`` exceeds 2**31 at large L.

This locks in the int64 hardening (so a refactor can't silently drop it) and
documents the one deliberate exception — the transition triton family, which is
on the B200 hot path and is left int32 because (a) it is provably int32-safe at
the shapes in use (L<=1024, d<=512 keeps M*stride < 2**31) and (b) adding int64
there risks the "no perf regression" bar. Revisit only with a benchmark.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "miniworld_engine" / "kernels"

# Files whose M-index must stay promoted. The check is a REGEX over the tile-axis name, not a
# literal: pinning ``tl.arange(0, BLOCK_M)`` made this test fail the moment fcd3c7a renamed the
# axis to BLOCK_M1, even though the ``.to(tl.int64)`` it exists to protect was untouched. A
# literal that names the axis fails on a rename and passes on a file that merely spells the axis
# the old way without promoting it -- wrong in both directions.
_HARDENED = [
    ("trimul_inproj/triton/front.py", r"\.to\(tl\.int64\)"),
    ("trimul_inproj/triton/back_fused.py", r"\.to\(tl\.int64\)"),
    ("trimul_inproj/triton/gate_elem.py", r"\.to\(tl\.int64\)"),
    ("tm1/triton/main.py", r"tl\.arange\(0, BLOCK_M\w*\)\.to\(tl\.int64\)"),
    ("tm2/triton/main.py", r"tl\.arange\(0, BLOCK_M\w*\)\.to\(tl\.int64\)"),
]

# Known int32-offset kernels intentionally left as-is (hot path, int32-safe at
# current shapes). Documented so the guard is a conscious allowlist, not silence.
_KNOWN_INT32_HOT = [
    "transition/triton/fused.py",
]


@pytest.mark.parametrize(("rel", "pattern"), _HARDENED)
def test_m_index_is_int64(rel: str, pattern: str):
    text = (_SRC / rel).read_text()
    assert re.search(pattern, text), (
        f"{rel} lost its int64 M-index promotion (no match for {pattern!r}); large-L offsets "
        f"will overflow int32. See tests/test_int64_offsets.py."
    )


def test_known_int32_hot_files_exist():
    """Sanity: the documented exceptions still exist (rename -> update the note)."""
    for rel in _KNOWN_INT32_HOT:
        assert (_SRC / rel).exists(), f"documented int32 hot file moved: {rel}"
