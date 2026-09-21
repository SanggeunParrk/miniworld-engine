"""Serve TriangleMultiplication from Anthropic's native TriMul payload (inference only).

The payload is the `trimul_native` package of Anthropic's `uplifting-biomolecular-modeling` release, optionally with this repo's
`experiments/trimul_k1k3_inference` overlay; it is NOT vendored here. It is named at runtime by `TRIMUL_NATIVE_BUILD_DIR`, which points at
a payload's `build/` directory (its `python/`, `csrc/` and `testvectors/` sit beside it — `experiments/trimul_k1k3_inference/build_payload.py`
assembles one). Setting that variable is the opt-in: with it set, a module built with `implementation="miniworld"` uses the payload for the
forward passes it can serve and its own kernels for everything else; `implementation="anthropic"` demands it and refuses rather than reroute.

What it serves (anything else falls back to the engine's own backends):

  * sm_90 (the payload's units are `sm_90a`), bf16 pair, no autograd and no live dropout scale — the release calls are forward-only.
  * a unit for the module's (c_z, c_hidden) in the loaded payload: `tmn90_z128_h128` for one direction, `tmn90_z128_h256` for the
    bidirectional module (outgoing + incoming share one input LayerNorm and one 2*c_hidden output LayerNorm, so it is ONE unit at twice
    the hidden width, not two unidirectional calls, which would normalise each half separately and compute a different function).

One direction goes through `trimul_native.face.serve`, which applies the release's own manifest and test-vector gate. The bidirectional
composition has no face entry, so it is composed here from the package's own primitives — K1 over the doubled hidden width, the two
half-channel contractions (outgoing NT, incoming TN), K3 with the residual fused — after the same `face.check()`.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch

ENV = "TRIMUL_NATIVE_BUILD_DIR"
_LOADED: dict[str, Any] = {}
_CHECKED: set[int] = set()


class PayloadUnavailable(RuntimeError):
    """The payload named by the environment cannot serve this call."""


def payload_dir() -> str | None:
    """The payload `build/` directory named by the environment, or None."""
    return os.environ.get(ENV) or None


def _load() -> dict[str, Any]:
    """Import `trimul_native` from the payload beside `$TRIMUL_NATIVE_BUILD_DIR`, once per process."""
    build = payload_dir()
    if not build:
        raise PayloadUnavailable(f"{ENV} is not set (it must name a payload's build/ directory)")
    py = Path(build).resolve().parent / "python"
    if _LOADED.get("python") == str(py):
        return _LOADED
    if not (py / "trimul_native" / "face.py").is_file():
        raise PayloadUnavailable(f"{py} has no trimul_native package (a payload keeps python/ beside build/)")
    already = sys.modules.get("trimul_native")
    if already is not None and Path(already.__file__).resolve().parent != py / "trimul_native":
        raise PayloadUnavailable(f"trimul_native is already imported from {already.__file__}, not from {py}")
    if _LOADED:
        raise PayloadUnavailable("changing the payload in a live process is not supported")
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    _LOADED.update(python=str(py), face=importlib.import_module("trimul_native.face"),
                   ops=importlib.import_module("trimul_native.ops"))
    return _LOADED


def _checked(device: torch.device) -> dict[str, Any]:
    p = _load()
    idx = 0 if device.index is None else device.index
    if idx not in _CHECKED:
        p["face"].check(device=idx, gate=False)      # the release's manifest / test-vector gate
        _CHECKED.add(idx)
    return p


def wanted(implementation) -> bool:
    """Should this module even ask the payload?  ``anthropic`` always (and refuses if it cannot serve); ``miniworld`` only when a payload
    is named and the engine is not pinned to Triton; every other option is the caller's explicit choice and is left alone."""
    from miniworld_engine import settings
    from miniworld_engine.modules.exceptions import ImplementationType
    if implementation == ImplementationType.ANTHROPIC:
        return True
    if implementation != ImplementationType.MINIWORLD:
        return False
    return bool(payload_dir()) and settings.current().engine_backend != "triton"


def refusal(pair: torch.Tensor, c_z: int, c_hidden: int, *, grad: bool, dropout: bool) -> str | None:
    """None if the payload named by the environment can run this forward, else why it cannot. Never raises."""
    try:
        if not payload_dir():
            return f"{ENV} is not set"
        if grad:
            return "the release calls are forward-only (grad is enabled)"
        if dropout:
            return "a live row-dropout scale needs the training path"
        if pair.dtype != torch.bfloat16:
            return f"the payload units are bf16, got {pair.dtype}"
        if not pair.is_cuda:
            return "the pair is not on a CUDA device"
        if pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]:
            return f"one square pair plane per call, got {tuple(pair.shape)}"
        if torch.cuda.get_device_capability(pair.device) != (9, 0):
            return "the payload units are sm_90a"
        p = _load()
        if not p["ops"].kernels().has_unit(c_z, c_hidden):
            return f"the payload has no unit for (c_z={c_z}, c_hidden={c_hidden})"
        return None
    except Exception as exc:                          # an unusable payload is a fallback, not a crash
        return f"{type(exc).__name__}: {exc}"


def serves(pair: torch.Tensor, c_z: int, c_hidden: int, *, grad: bool, dropout: bool) -> bool:
    """Can the payload run this forward? The caller falls back to its own backends when not."""
    return refusal(pair, c_z, c_hidden, grad=grad, dropout=dropout) is None


def require(pair: torch.Tensor, c_z: int, c_hidden: int, *, grad: bool, dropout: bool) -> None:
    """An explicit ``implementation="anthropic"`` refuses with the reason instead of rerouting."""
    why = refusal(pair, c_z, c_hidden, grad=grad, dropout=dropout)
    if why is not None:
        raise PayloadUnavailable("the anthropic TriMul payload cannot serve this call: " + why)


def _weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {"ln_in_w": module.ln_pair.weight, "ln_in_b": module.ln_pair.bias,
            "w_ag": module.to_left_gate.weight, "w_ap": module.to_left.weight,
            "w_bg": module.to_right_gate.weight, "w_bp": module.to_right.weight,
            "ln_out_w": module.ln_out.weight, "ln_out_b": module.ln_out.bias,
            "w_o": module.to_out.weight, "w_og": module.to_gate.weight}


def _signature(tensors) -> tuple:
    # Replacement, load_state_dict/copy_, device and dtype changes all have to invalidate the packed copy.
    return tuple((id(t), t._version, t.device, t.dtype, tuple(t.shape)) for t in tensors)


def _prepared(module: torch.nn.Module) -> tuple[dict, dict]:
    w = _weights(module)
    key = (_signature(w.values()), module.ln_pair.eps)
    if getattr(module, "_native_key", None) != key:
        module._native_weights = {k: t.detach().contiguous() for k, t in w.items()}
        module._native_cache = {}
        module._native_key = key
    return module._native_weights, module._native_cache


def _pair_mask(pair: torch.Tensor, mask: torch.Tensor | None, ops) -> torch.Tensor | None:
    """The pair mask in an element type THIS payload's K1 can read.

    A payload built with the templated mask declares the types it instantiates (`ops.MASK_NATIVE_DTYPES`) and takes the module's own
    bool mask as-is; the upstream package has only the fp32 kernel and reads whatever buffer it is handed AS fp32 — handing it a bool
    tensor is a four-times-too-long read, which at L384 returned wrong numbers and at L768 was an illegal access. So: bool when the
    payload says it can, fp32 otherwise.
    """
    if mask is None:
        return None
    m = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).reshape(pair.shape[1], pair.shape[2])
    native = getattr(ops, "MASK_NATIVE_DTYPES", ())
    return m.contiguous() if torch.bool in native else m.to(torch.float32).contiguous()


def update_unidirectional(module: torch.nn.Module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """`pair + trimul(pair)` for one direction, residual fused in K3, through the payload's served face."""
    if module.ln_pair.eps != module.ln_out.eps:
        raise PayloadUnavailable("the native TriMul normalises input and output with one epsilon")
    p = _checked(pair.device)
    weights, cache = _prepared(module)
    out = p["face"].serve(pair, _pair_mask(pair, mask, p["ops"]), direction="outgoing" if module.outgoing else "incoming",
                          weights=weights, residual=True, cache=cache, eps=module.ln_pair.eps)
    module.native_selection = {k: v for k, v in cache.items() if isinstance(k, tuple) and k and k[0] == "_sel"}
    return out


def update_bidirectional(module: torch.nn.Module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """`pair + bidir_trimul(pair)` as one unit at c_hidden = 2 * d_hidden: K1 (natural layout) -> the two half-channel
    contractions -> K3 over both halves with the residual fused."""
    if module.ln_pair.eps != module.ln_out.eps:
        raise PayloadUnavailable("the native TriMul normalises input and output with one epsilon")
    p = _checked(pair.device)
    ops = p["ops"]
    weights, cache = _prepared(module)
    z3 = pair[0] if pair.shape[0] == 1 else pair.reshape(pair.shape[1], pair.shape[2], pair.shape[3])
    z3 = z3.contiguous()
    packed = ops._packed(weights, cache, z3.device)      # memoised by weight address: a fresh pack per call would be captured into the graph
    ops._check(z3, packed)
    ch, h = packed["ch"], module.d_hidden
    Np = ops.ceil16(z3.shape[0])
    ab = ops.planes(z3, _pair_mask(pair, mask, ops), packed, transpose=False, lnm=2, Np=Np, cache=cache, eps=module.ln_pair.eps)
    tri = ops._buf(cache, ("tri", Np, ch), (ch, Np, Np), torch.bfloat16, z3.device)
    a, b = ab[:ch], ab[ch:]
    torch.bmm(a[:h], b[:h].transpose(1, 2), out=tri[:h])                 # outgoing half: sum_k a[i,k] b[j,k]
    torch.bmm(a[h:].transpose(1, 2), b[h:], out=tri[h:])                 # incoming half: sum_k a[k,i] b[k,j]
    out = ops.epilogue(tri, z3, packed, residual=True, lnm=1, cache=cache, eps=module.ln_pair.eps)
    module.native_selection = {"unit": f"z{packed['cz']}_h{ch}", "composed": "K1 + 2 contractions + K3"}
    return out.unsqueeze(0)
