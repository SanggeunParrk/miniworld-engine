"""Entry point of the ext_attn E5 stage: ``run_measure.py`` with the backends of ``ext_attn_registry`` (and those of externals.py) installable by ``--ext``.

python E5_other_kernels/run_ext_attn.py E5_other_kernels/cfg_ext_attn.yaml --kind speed --out records/<card>/ext_attn.r1.jsonl --ext gnops of3tri ...
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import ext_attn_registry
import externals

_orig = externals.install


def _install(names):
    new = [n for n in names if n in ext_attn_registry.REGISTRY]
    ext_attn_registry.install(new)
    old = [n for n in names if n not in new]
    if old:
        _orig(old)


externals.install = _install

from paper_common import run_measure

if __name__ == "__main__":
    sys.exit(run_measure.main())
