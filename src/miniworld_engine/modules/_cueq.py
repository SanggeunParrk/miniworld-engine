"""Loader of NVIDIA cuEquivariance (the ``"cueq"`` backend of the shared modules): ``cuequivariance_torch`` + ``cuequivariance_ops_torch``.

The adapters call the public ops ``cuequivariance_torch.triangle_multiplicative_update`` / ``triangle_attention``. Both ops run their own
PyTorch formula for some calls (short sequences, head widths their kernels do not take): the modules' ``_supports_cueq`` refuse exactly those
calls, reading the library's own thresholds (:func:`ops`), so a ``"cueq"`` forward is always a cuEquivariance kernel.

:func:`unavailable` says why the library cannot be used (``None`` when it can): not installed, another major / minor version than the one
these adapters were written against, or not an H100 (the device the adapters are verified on).
"""

from __future__ import annotations

import importlib

#: The cuEquivariance release the adapters are written and verified against (major, minor).
VERSION = (0, 12)
_STATE: dict = {}


def _load() -> str | None:
    if "why" in _STATE:
        return _STATE["why"]
    why = None
    try:
        cet = importlib.import_module("cuequivariance_torch")
        importlib.import_module("cuequivariance_ops_torch")
        version = tuple(int(p) for p in str(getattr(cet, "__version__", "0.0")).split(".")[:2])
        if version != VERSION:
            why = f"cuequivariance_torch {cet.__version__} is installed, the adapters are written for {VERSION[0]}.{VERSION[1]}"
    except ImportError as exc:
        why = f"cuEquivariance is not importable: {exc}"
    _STATE["why"] = why
    return why


def unavailable(device=None) -> str | None:
    """Why cuEquivariance cannot serve a call on ``device`` here, or ``None``."""
    why = _load()
    if why is not None:
        return why
    import torch

    if device is not None and getattr(device, "type", device) != "cuda":
        return "the cuEquivariance kernels run on CUDA tensors only"
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.cuda.get_device_capability(device) != (9, 0):
        return "the cueq backend is verified on the H100 (sm_90) only"
    return None


def torch_api():
    """``cuequivariance_torch`` (the public ops)."""
    why = _load()
    if why is not None:
        raise RuntimeError(why)
    return importlib.import_module("cuequivariance_torch")


def ops(name: str):
    """``cuequivariance_ops_torch.<name>``: the op modules whose routing constants the ``_supports_cueq`` checks read."""
    why = _load()
    if why is not None:
        raise RuntimeError(why)
    if name not in ("triangle_attention", "triangle_multiplicative_update"):
        raise ValueError(f"no cuEquivariance op module {name!r} is used by the engine")
    return importlib.import_module(f"cuequivariance_ops_torch.{name}")
