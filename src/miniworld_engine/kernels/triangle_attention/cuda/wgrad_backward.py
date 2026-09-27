"""Four wide projection weight gradients sharing TMA-loaded LayerNorm output."""
from __future__ import annotations
import functools
import importlib.util
import json
import sys
from pathlib import Path
import torch
from miniworld_engine.kernels._compile import device_constant, opaque

_ROOT = Path(__file__).resolve().parent
_EXT = None
_SPLITS = {384 * 384: 2304, 768 * 768: 8960, 1024 * 1024: 15936}

@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data = json.loads((_ROOT / 'wgrad_manifest.json').read_text())
        return (data['python_abi'] == sys.implementation.cache_tag
                and data['torch'] == str(torch.__version__)
                and (_ROOT / data['binary']).is_file())
    except (OSError, ValueError, KeyError):
        return False

@device_constant
def _available(device):
    return torch.cuda.get_device_capability(device) == (9, 0) and _artifact_ready()

def can_use(dy, z):
    return (z.is_cuda and z.dtype == torch.bfloat16 and z.ndim == 2
            and z.shape[0] in _SPLITS and z.shape[1] == 128 and z.is_contiguous()
            and len(dy) == 4 and all(d.device == z.device and d.dtype == z.dtype
                and d.shape == z.shape and d.is_contiguous() for d in dy)
            and _available(z.device))

def extension():
    global _EXT
    if _EXT is None:
        data = json.loads((_ROOT / 'wgrad_manifest.json').read_text())
        spec = importlib.util.spec_from_file_location(data['module_name'], _ROOT / data['binary'])
        _EXT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXT)
    return _EXT

def _fake(dy, z, split):
    return torch.empty((4, 128, 128), device=z.device, dtype=z.dtype)

@opaque(fake=_fake, name='triangle_four_weight_grad_cuda')
def _backward(dy: list[torch.Tensor], z: torch.Tensor, split: int) -> torch.Tensor:
    return extension().backward(dy, z, split)

def backward(dy, z):
    return _backward(dy, z, _SPLITS[z.shape[0]]).unbind(0)
