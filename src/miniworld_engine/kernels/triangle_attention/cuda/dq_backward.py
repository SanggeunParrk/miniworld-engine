"""Prebuilt native H100 dQ; called inside the grouped-backward opaque boundary."""
from __future__ import annotations
import functools
import importlib.util
import json
from pathlib import Path
import sys
import torch

_ROOT = Path(__file__).resolve().parent
_EXT = None

@functools.lru_cache(maxsize=1)
def available():
    try:
        data = json.loads((_ROOT / 'dq_manifest.json').read_text())
        return (data['torch'] == str(torch.__version__)
                and data['python_abi'] == sys.implementation.cache_tag
                and (_ROOT / data['binary']).is_file())
    except (OSError, ValueError, KeyError):
        return False

def extension():
    global _EXT
    if _EXT is None:
        data = json.loads((_ROOT / 'dq_manifest.json').read_text())
        spec = importlib.util.spec_from_file_location(data['module_name'], _ROOT / data['binary'])
        _EXT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXT)
    return _EXT

def backward(q, k, v, b, m, delta, dy):
    return extension().backward(q, k, v, b, m, delta, dy)
