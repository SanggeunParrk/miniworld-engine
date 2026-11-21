"""Registry of the E5 pair-bias-attention external backends (one line per adapter file ``ext_<name>.py``; the coordinator merges this
into externals.py). ``install(names)`` imports the adapter's file and attaches the backend (nothing under src/ changes)."""

from __future__ import annotations

import importlib
import os
import sys

REGISTRY = {
    "gnops": "ext_gnops",
    "of3tri": "ext_of3tri",
    "megafold": "ext_megafold",
    "bionemo_ir": "ext_bionemo_ir",
    "turbopf": "ext_turbopf",
    "ligo_pwa": "ext_ligo_pwa",
    "megafold_tr": "ext_megafold_tr",
}


def install(names) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for name in names:
        if name not in REGISTRY:
            raise ValueError(f"unknown ext_attn backend {name!r}; choose from {sorted(REGISTRY)}")
        importlib.import_module(REGISTRY[name]).install()
