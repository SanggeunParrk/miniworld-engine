"""Serve the MSA module's two heavy operators -- OuterProductMean and MSAPairWeightedAveraging -- from
Anthropic's `opt_core` MSA cells and this engine's own fused OPM epilogue (inference only).

The upstream half is the `common/opt_core` runtime of Anthropic's `uplifting-biomolecular-modeling`
release; it is NOT vendored here. It is named at runtime by `OPT_CORE_DIR`, which points at that
directory. Setting the variable is the opt-in: with it set, a module built with
`implementation="miniworld"` uses these paths where they fit; `implementation="anthropic"` demands
them and refuses with a reason rather than reroute.

What each path is, and who wrote which kernel:

  outer_product_mean   NOT the upstream fused cell. Its one-kernel form drives the outer-product GEMM
                       at ~26 % of bf16 peak, which pays at the upstream shapes (N=705, S=4724: a
                       4.8 TFLOP GEMM) but loses at ours (N=384, S=1024: 309 GFLOP), where cuBLAS
                       reaches ~76 % on the same product. So this path keeps upstream's fused Triton
                       PROLOGUE (LayerNorm + both projections + mask, written straight into the GEMM
                       layouts), hands the outer product to cuBLAS in its GROUPED layout
                       O[(i,c),(j,e)] -- the one layout that is a matmul, so no transpose exists to
                       be materialised -- and finishes with THIS repo's `csrc/opm_epilogue.cu`, which
                       does the [i,j,c,e] -> [(i,j),(c,e)] permute, the / mask-count division, the
                       bf16 cast and the c_hidden^2 -> c_z projection (+bias) in one pass.
                       Measured H100, L=384 / S=1024, against this engine's own path: 2.17 -> 0.88 ms.

  pair_weighted_avg    upstream's `msa_pwa` cell as it ships (config `g_fo4p`: fused LN -> v|gate
                       prologue, then the K=N contraction with the gate and the output projection in
                       one epilogue), with ONE substitution: the pair LayerNorm is this engine's
                       fused LayerNorm instead of the stock torch call the cell makes, which is worth
                       1.18x on its own (NCU: 193 us -> 24 us at our shape). 1.52 -> 0.69 ms.

Both are forward-only. There is no backward for either, so a grad-enabled call is refused, never
silently rerouted; training keeps the engine's own path until the backward kernels land.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch

ENV = "OPT_CORE_DIR"
PWA_CFG = "g_fo4p"                     # the config that wins at our shape; the shipped default too
_LOADED: dict[str, Any] = {}
_EXT: dict[str, Any] = {}

C_HIDDEN, C_Z, C_M, HEADS = 32, 128, 64, 8          # the shapes the kernels are built for
_TILE_I, _TILE_J = 4, 16                            # the epilogue's CTA tile, so N must divide by both


class PayloadUnavailable(RuntimeError):
    """The payload named by the environment cannot serve this call."""


def payload_dir() -> str | None:
    """The `common/opt_core` directory named by the environment, or None."""
    return os.environ.get(ENV) or None


def _load() -> dict[str, Any]:
    """Import the opt_core MSA cells from `$OPT_CORE_DIR`, once per process."""
    root = payload_dir()
    if not root:
        raise PayloadUnavailable(f"{ENV} is not set (it must name the release's common/opt_core directory)")
    root = str(Path(root).resolve())
    if _LOADED.get("root") == root:
        return _LOADED
    if not (Path(root) / "opt_core" / "ops" / "msa_opm" / "__init__.py").is_file():
        raise PayloadUnavailable(f"{root} has no opt_core/ops/msa_opm (point {ENV} at common/opt_core)")
    already = sys.modules.get("opt_core")
    if already is not None and Path(already.__file__).resolve().parent.parent != Path(root):
        raise PayloadUnavailable(f"opt_core is already imported from {already.__file__}, not from {root}")
    if _LOADED:
        raise PayloadUnavailable("changing the payload in a live process is not supported")
    if root not in sys.path:
        sys.path.insert(0, root)
    _LOADED.update(root=root,
                   opm=importlib.import_module("opt_core.ops.msa_opm"),
                   pwa=importlib.import_module("opt_core.ops.msa_pwa"))
    # the cell picks its tile config from this variable; set it once, not per call
    os.environ.setdefault("FPF_PWA_CFG", PWA_CFG)
    return _LOADED


def _ext():
    """Build (once) this repo's fused OPM epilogue."""
    if "mod" in _EXT:
        return _EXT["mod"]
    from torch.utils.cpp_extension import load
    src = Path(__file__).with_name("csrc") / "opm_epilogue.cu"
    build = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit")) / "opm_epilogue"
    build.mkdir(parents=True, exist_ok=True)
    _EXT["mod"] = load(name="miniworld_opm_epilogue", sources=[str(src)], build_directory=str(build),
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], extra_cflags=["-O3"])
    return _EXT["mod"]


def wanted(implementation) -> bool:
    """Should this module even ask?  ``anthropic`` always (and refuses if it cannot serve); ``miniworld`` only when a
    payload is named and the engine is not pinned to Triton; every other option is the caller's explicit choice."""
    from miniworld_engine import settings
    from miniworld_engine.modules.exceptions import ImplementationType
    if implementation == ImplementationType.ANTHROPIC:
        return True
    if implementation != ImplementationType.MINIWORLD:
        return False
    return bool(payload_dir()) and settings.current().engine_backend != "triton"


def _common_refusal(x: torch.Tensor, *, grad: bool) -> str | None:
    if not payload_dir():
        return f"{ENV} is not set"
    if grad:
        return "these are forward-only paths (grad is enabled); there is no backward yet"
    if x.dtype != torch.bfloat16:
        return f"the kernels are bf16, got {x.dtype}"
    if not x.is_cuda:
        return "the input is not on a CUDA device"
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        return "the epilogue is built for sm_90a"
    if x.shape[0] != 1:
        return f"one MSA stack per call, got batch {x.shape[0]}"
    return None


def opm_refusal(msa: torch.Tensor, d_hidden: int, d_pair: int, *, grad: bool, interchain: bool) -> str | None:
    """None if this path can run this OuterProductMean forward, else why it cannot. Never raises."""
    try:
        # what the caller asked for comes first: a shape or a flag this path cannot serve is the same answer on
        # any machine, and saying so beats reporting whichever machine the caller happened to be on.
        if interchain:
            return "mask_interchain is applied after the projection and is not fused here"
        if (d_hidden, d_pair) != (C_HIDDEN, C_Z):
            return f"the epilogue is built for (d_hidden={C_HIDDEN}, d_pair={C_Z}), got ({d_hidden}, {d_pair})"
        n = msa.shape[2]
        if n % _TILE_I or n % _TILE_J:
            return f"the epilogue tiles {_TILE_I}x{_TILE_J} tokens, so N must divide by both, got {n}"
        why = _common_refusal(msa, grad=grad)
        if why is not None:
            return why
        _load()
        return None
    except Exception as exc:                          # an unusable payload is a fallback, not a crash
        return f"{type(exc).__name__}: {exc}"


def pwa_refusal(msa: torch.Tensor, d_msa: int, d_pair: int, n_head: int, d_hidden: int,
                *, grad: bool, dropout: bool) -> str | None:
    """None if this path can run this MSAPairWeightedAveraging forward, else why it cannot. Never raises."""
    try:
        # see opm_refusal: the request's own shape and mode are answered before the machine's.
        if (d_msa, d_pair, n_head, d_hidden) != (C_M, C_Z, HEADS, C_HIDDEN):
            return (f"the cell serves (d_msa={C_M}, d_pair={C_Z}, n_head={HEADS}, d_hidden={C_HIDDEN}), "
                    f"got ({d_msa}, {d_pair}, {n_head}, {d_hidden})")
        if dropout:
            return "a live row-dropout scale needs the training path"
        why = _common_refusal(msa, grad=grad)
        if why is not None:
            return why
        _load()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def serves_opm(*a, **kw) -> bool:
    return opm_refusal(*a, **kw) is None


def serves_pwa(*a, **kw) -> bool:
    return pwa_refusal(*a, **kw) is None


def require_opm(*a, **kw) -> None:
    """An explicit ``implementation="anthropic"`` refuses with the reason instead of rerouting."""
    why = opm_refusal(*a, **kw)
    if why is not None:
        raise PayloadUnavailable("the anthropic OPM path cannot serve this call: " + why)


def require_pwa(*a, **kw) -> None:
    why = pwa_refusal(*a, **kw)
    if why is not None:
        raise PayloadUnavailable("the anthropic PWA path cannot serve this call: " + why)


def _swizzle_b(wot: torch.Tensor) -> torch.Tensor:
    """[K, c_z] -> [K/16][c_z/8][32][4] in mma.m16n8k16 B-fragment order, so the epilogue reads a whole
    fragment with one 8-byte per-thread load. b0 = k (lane%4)*2 + {0,1}, b1 = that + 8, n = lane/4."""
    k, n = wot.shape
    kk = torch.arange(k // 16, device=wot.device)[:, None, None]
    nn = torch.arange(n // 8, device=wot.device)[None, :, None]
    lane = torch.arange(32, device=wot.device)[None, None, :]
    idx_n = nn * 8 + lane // 4
    k_lo = (lane % 4) * 2
    return torch.stack([wot[kk * 16 + k_lo + off, idx_n] for off in (0, 1, 8, 9)], dim=-1).contiguous()


def _opm_pack(module) -> dict[str, Any]:
    """The weights this path needs, in the layouts the kernels want. Built once per module (inference weights)."""
    pack = getattr(module, "_anthropic_msa_pack", None)
    if pack is not None:
        return pack
    pack = {"wa_t": module.to_left.weight.detach().to(torch.bfloat16).t().contiguous(),
            "wb_t": module.to_right.weight.detach().to(torch.bfloat16).t().contiguous(),
            "bf": _swizzle_b(module.to_out.weight.detach().t().contiguous().to(torch.bfloat16)),
            "bias": module.to_out.bias.detach().to(torch.bfloat16).to(torch.float32).contiguous()}
    module._anthropic_msa_pack = pack
    return pack


def outer_product_mean(module, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """OuterProductMean(msa, mask) WITHOUT the residual -- the module adds its own."""
    p = _load()["opm"]
    pack = _opm_pack(module)
    s, n = msa.shape[1], msa.shape[2]
    mask16 = mask.to(torch.bfloat16)
    # upstream's fused prologue, asked for the unblocked A layout (BI=BJ=1), which puts A2 row = i*c_hidden + c
    a2, bt, _, _, _ = p.fused_prologue(msa[0], mask16[0], module.ln_msa, pack["wa_t"], pack["wb_t"], None, None,
                                       1, 1, 64, C_HIDDEN)
    o = torch.matmul(a2, bt.t())                                   # the grouped outer product, cuBLAS NT
    mf = mask[0].to(torch.float32)
    norm = (mf.t() @ mf).clamp_(min=1).contiguous()                # the module's own fp32 mask count
    return _ext().opm_epilogue(o, norm, pack["bf"], pack["bias"], n)


def pair_weighted_averaging(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """MSAPairWeightedAveraging(msa, pair, mask) WITHOUT the residual -- the module adds its own."""
    p = _load()["pwa"]
    n = msa.shape[2]
    if mask is None:
        mask = torch.ones(msa.shape[0], n, dtype=torch.bool, device=msa.device)
    pair_mask = mask[:, None, :].expand(-1, n, -1).to(torch.bfloat16)   # the cell masks the key axis j
    return p.forward_masked(_PwaView(module), msa, pair, pair_mask, chunk_heads=False)


class _PwaView:
    """The cell reads a stock module's attribute names; this is the same weights under those names. `norm_z` is
    deliberately this engine's fused LayerNorm rather than the stock torch call the cell would otherwise make."""

    __slots__ = ("norm_m", "proj_m", "proj_g", "norm_z", "proj_z", "proj_o", "inf", "num_heads", "c_h")

    def __init__(self, module):
        self.norm_m = module.ln_msa
        self.proj_m = module.to_value
        self.proj_g = module.to_gate
        self.norm_z = module.ln_pair
        self.proj_z = module.to_bias
        self.proj_o = module.to_out
        # the module masks the key axis by filling the bias with finfo.min; the cell subtracts `inf`. Any value
        # large enough to zero the softmax term is the same function, and 1e9 is exact in bf16's exponent range.
        self.inf = 1e9
        self.num_heads = module.n_head
        self.c_h = module.to_value.weight.shape[0] // module.n_head
