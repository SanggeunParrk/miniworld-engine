"""Launcher: ``paper_common/run_measure.py`` with ``--ext`` names of ext_trimul_registry.py served by it (the others by externals.py).

    python E5_other_kernels/run_ext_trimul.py <run_measure.py arguments>     (e.g. ... cfg_trimul.yaml --kind speed --out records/h100/ext_trimul.r1.jsonl --ext fast_trimul)

run_measure.py imports ``externals`` and calls ``externals.install(args.ext)``; this wrapper routes the trimul names to the registry first. It
replaces nothing (until the coordinator merges the registry into externals.py).
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ext_trimul_registry as reg
import externals

_attn_install = externals.install


def install(names):
    names = list(names)
    mine = [n for n in names if n in reg.REGISTRY]
    reg.install(mine)
    rest = [n for n in names if n not in reg.REGISTRY]
    if rest:
        _attn_install(rest)


externals.install = install
sys.argv = [str(HERE.parent / "paper_common" / "run_measure.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
