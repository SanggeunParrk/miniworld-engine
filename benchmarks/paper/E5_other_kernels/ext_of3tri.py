"""E5 external backend ``of3tri``: OpenFold3's own Triton Evoformer-attention kernel (Apache-2.0, forward + backward).

Upstream: github.com/aqlaboratory/openfold-3, ``openfold3/core/kernels/triton/evoformer.py`` (the kernel the OpenFold3 docs / TurboPairFormer paper call
the Triton backend, ``use_triton_triangle_kernels``; AMD-authored). Loaded as one source file, so no OpenFold3 dependency is imported. Call:
``TritonEvoformer(q, k, v, res_mask, pair_bias, has_pair_bias)`` with q / k / v ``[B, N, Q, H, d]``, ``res_mask`` an ADDITIVE ``[B, N, 1, 1, K]`` mask
in the q dtype (0 / -1e9 here), ``pair_bias`` ``[B, 1, H, Q, K]``; bf16 / fp16 / fp32. Serves triangle attention and AttentionPairBias.
"""

from __future__ import annotations

import ext_attn_common as common
import torch

NAME = "of3tri"
_MASK_FILL = -1e9


def _mod():
    import os

    d = os.path.join(common.upstream("openfold3"), "openfold3", "core", "kernels", "triton")
    common.load_file("of3tri_dot_precision", os.path.join(d, "dot_precision.py"))
    # evoformer.py imports its sibling as a relative module; register the sibling under the name it asks for
    import sys
    import types

    pkg = "openfold3.core.kernels.triton"
    for p in ("openfold3", "openfold3.core", "openfold3.core.kernels", pkg):
        sys.modules.setdefault(p, types.ModuleType(p))
        sys.modules[p].__path__ = getattr(sys.modules[p], "__path__", [])
    sys.modules[f"{pkg}.dot_precision"] = sys.modules["of3tri_dot_precision"]
    sys.modules[pkg].dot_precision = sys.modules["of3tri_dot_precision"]
    mod = common.load_file(f"{pkg}.evoformer", os.path.join(d, "evoformer.py"))
    if mod.TritonEvoformer is None:
        raise ImportError("OpenFold3 Triton evoformer: triton unavailable")
    return mod


def evo(q, k, v, keep, bias):
    b, n = q.shape[:2]
    if keep is None:
        res = torch.zeros(b, n, 1, 1, q.shape[2], dtype=q.dtype, device=q.device)
    else:
        res = torch.zeros(keep.shape, dtype=q.dtype, device=q.device).masked_fill_(~keep, _MASK_FILL)
    return _mod().TritonEvoformer(q, k, v, res, bias.to(q.dtype).contiguous(), True)


def reason(dtype, head_dim, length, grad):
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {dtype}: bf16 / fp16 / fp32 only"
    return None


def install() -> None:
    common.attach(NAME, evo, reason, _mod)
