"""Per-GPU dispatch for the bias-only triangle-attention backends.

Three crossovers, all measured on H100 / d_pair=128 (see benchmark/bias_only.md):

  * ``KERNEL_MIN_L`` (384): below this L the triton kernels' launch/dispatch cost
    loses to the plain torch path -> fall back to torch.
  * gate backend by ``d_hidden``: DH<=128 -> ``fused_gate_out`` (gate-mul folded into
    the to_out GEMM); DH>=256 -> the SPLIT (one-pass sigmoid*mul + cuBLAS to_out),
    because the fused tl.dot's wide tile degrades on SM90. This crossover is the real
    perf cliff and is GPU-dependent, so it is *calibrated and cached per GPU*.
  * ``INFER_CONCAT_MAX_DH`` (256): above this the inference LN+proj concat
    (``layernorm_linear``) regresses (concat GEMM too wide) -> standard path.

Only ever chooses among *correct* backends, so a stale/corrupt cache yields at worst
a slower (never wrong) path; any error falls back to the static H100 heuristic.

Env:
  MINIWORLD_BIASONLY_AUTOTUNE = auto (default) | off | force
      off   -> never calibrate; always the static H100 thresholds
      force -> calibrate even on H100 (sm90), ignoring the static fast-path
  MINIWORLD_KERNELS_CACHE_DIR -> cache root (shared with the other kernels)
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch

# Reuse the shared per-GPU key (name + compute capability + triton version) and
# M-bucketing from the layernorm dispatch cache so all kernels key GPUs identically.
from miniworld_kernels.kernels.layernorm.dispatch_cache import gpu_key, mbucket

# ---- static H100 thresholds (defaults / fallback) --------------------------------
KERNEL_MIN_L = 384
# Fused gate+to_out vs split, keyed on the OUTPUT width n_out (= d_pair = to_out's
# out_features). fused wins for n_out <= 128 (incl. the bidirectional case where the
# contraction d_hidden=2h is large but the output stays d_pair=128); split wins for
# wider outputs where the fused tl.dot tile degrades on SM90.
GATE_FUSED_MAX_DH = 128
INFER_CONCAT_MAX_DH = 256

_SUBDIR = "bias_only_dispatch"


def autotune_mode() -> str:
    return (os.environ.get("MINIWORLD_BIASONLY_AUTOTUNE") or "auto").strip().lower()


def _cache_dir() -> Path:
    base = os.environ.get("MINIWORLD_KERNELS_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "miniworld_kernels")
    return Path(base) / _SUBDIR


def _file(idx: int) -> Path:
    return _cache_dir() / f"{gpu_key(idx)}.json"


@functools.lru_cache(maxsize=8)
def _load(idx: int) -> dict:
    try:
        return json.loads(_file(idx).read_text())
    except (OSError, ValueError):
        return {}


def _store(idx: int, key: str, choice: str, times_ms: dict[str, float]) -> None:
    data = _load(idx)
    data[key] = {"choice": choice, "ms": {k: round(v, 6) for k, v in times_ms.items()}}
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        fp = _file(idx)
        tmp = fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(fp)
    except OSError:
        pass  # read-only fs -> keep in-memory choice only
    _load.cache_clear()
    _load(idx).update(data)


def _is_sm90(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx)[0] == 9


def use_kernels(L: int) -> bool:
    """Engage the triton kernels (else torch fallback). Launch-overhead crossover."""
    return L >= KERNEL_MIN_L


def use_infer_concat(d_hidden: int) -> bool:
    """Use the inference LN+proj concat fusion (layernorm_linear)."""
    return d_hidden <= INFER_CONCAT_MAX_DH


def _calibrate_gate(d_hidden: int, n_out: int, M: int, device: torch.device,
                    dtype: torch.dtype) -> tuple[str, dict[str, float]]:
    """Time fused vs split gate+to_out (forward) on dummy tensors; return the winner."""
    import triton

    from .triton.gate_out import fused_gate_out, sigmoid_gate_fused

    g = torch.randn(M, d_hidden, device=device, dtype=dtype)
    o = torch.randn(M, d_hidden, device=device, dtype=dtype)
    wo = torch.randn(n_out, d_hidden, device=device, dtype=dtype)

    def split():
        return torch.nn.functional.linear(sigmoid_gate_fused(g, o), wo)

    def fused():
        return fused_gate_out(g, o, wo)

    def t(fn):
        return triton.testing.do_bench(fn, warmup=10, rep=50, quantiles=[0.5])[0]

    tf, ts = t(fused), t(split)
    return ("fused" if tf <= ts else "split"), {"fused": tf, "split": ts}


def gate_use_fused(d_hidden: int, n_out: int, M: int, device: torch.device,
                   dtype: torch.dtype) -> bool:
    """True -> fused_gate_out; False -> split. Static H100 by DH; calibrated+cached
    per GPU on other arches (or when forced)."""
    mode = autotune_mode()
    # The fused tl.dot's tile is [BLOCK_M, OUTPUT_N] (N = n_out = d_pair); the
    # contraction d_hidden is just looped over (BLOCK_K). So fused-vs-split is decided
    # by the OUTPUT width n_out, NOT by d_hidden. (bench_back_designs conflated them
    # because there d_hidden == n_out; the bidir case d_hidden=2h, n_out=d_pair shows
    # fused still wins at n_out=128 despite d_hidden=256.)
    static = n_out <= GATE_FUSED_MAX_DH
    if mode == "off":
        return static
    if mode != "force" and _is_sm90(device):
        return static  # H100: trust the measured static threshold

    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = f"gate|{d_hidden}|{n_out}|{mbucket(M)}"
    cached = _load(idx).get(key)
    if cached is not None:
        return cached["choice"] == "fused"
    try:
        choice, times = _calibrate_gate(d_hidden, n_out, M, device, dtype)
        _store(idx, key, choice, times)
        return choice == "fused"
    except Exception:  # noqa: BLE001 -- any failure -> safe static fallback
        return static
