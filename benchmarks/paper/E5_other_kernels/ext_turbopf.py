"""E5 external backend ``turbopf``: TurboPairFormer triangle attention (arXiv 2610.05854, Lambda / Stevens / OpenFold; ``pip`` turbopairformer 0.1.3.post1, Apache-2.0).

Hopper only (sm_90a), BF16 only, head width <= 64, forward + backward (deterministic single-reconstruction backward). The PyPI wheel is cp314 + torch 2.10
only, so its prebuilt extensions do not load in the engine environment (py3.12, torch 2.13); the wheel's Python and CUDA SOURCES are used (fetch.sh keeps them,
drops the ``_ext/*.so``) and the four extensions are built here with the library's own development JIT (``TURBOPAIRFORMER_ALLOW_JIT=1``, nvcc, sm_90a)
into ``$PAPER_EXT_ATTN/torch_ext`` (``third_party/turbopairformer/build_local.sh`` prebuilds them). A100 / RTX 5090: ``unsupported`` (Hopper-only).

``turbo_attention(q, k, v, res_mask, pair_bias)``: q / k / v ``[B, O, H, N, d]`` (q raw, scale inside), ``res_mask`` additive ``[B, O, 1, 1, N]`` (0 keep, -inf
masked), ``pair_bias`` ``[B, 1, H, N, N]`` -> ``[B, O, N, H, d]``.
"""

from __future__ import annotations

import os
import sys

import ext_attn_common as common
import torch

NAME = "turbopf"


def _mod():
    root = common.upstream("turbopairformer")
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("TURBOPAIRFORMER_ALLOW_JIT", "1")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(common.ext_root(), "torch_ext"))
    import turbopairformer

    return turbopairformer


def _probe():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise ImportError(
            f"Hopper (sm_90) only; this device is capability {torch.cuda.get_device_capability() if torch.cuda.is_available() else None}"
        )
    _ = _mod().turbo_attention  # lazy attribute: the import check


def evo(q, k, v, keep, bias):
    tq, tk, tv = (t.transpose(2, 3).contiguous() for t in (q, k, v))
    mask = None
    if keep is not None:
        mask = torch.zeros(keep.shape, dtype=torch.float32, device=q.device).masked_fill_(~keep, float("-inf"))
    return _mod().turbo_attention(tq, tk, tv, res_mask=mask, pair_bias=bias)


def reason(dtype, head_dim, length, grad):
    if dtype is not torch.bfloat16:
        return f"dtype {dtype}: TurboPairFormer takes bf16 only"
    if head_dim > 64:
        return f"head width {head_dim} > 64"
    return None


def install() -> None:
    common.attach(NAME, evo, reason, _probe, apb=False)
