"""E5: every external candidate kernel as a backend of the measurer, installed at run time by ONE entry point (``install(names)``; ``run_measure.py --ext <name>``).

Nothing under src/ is edited: ``install`` adds the backend name to ``backends`` of the module classes (a class attribute tuple) and attaches
``_supports_<name>`` / ``_forward_<name>`` to them, exactly the shape the measurers read (``BackendModule.backends``, ``unsupported`` -> ``_supports_<name>``,
``_forward_<name>``). A case a library does not serve is recorded ``unsupported`` with the reason, never re-routed.

Families (``AVAILABLE`` lists all names; ``CLASS_OF`` says which operator class each serves):
  inline (this file)          ds4sci, trifast: triangle-attention cores (the module's own ``_residual`` with the library's attention CORE).
  triangle multiplication     fast_trimul, gpumode, geonn: adapters ``ext_<name>_trimul.py`` define ``PROBE``, ``supports(self, pair, mask, dropout_p)`` and
                              ``forward(self, pair, mask, dropout_p)`` (the whole update incl. residual); attached to ``TriangleMultiplication``.
  Pairformer operators        gnops, of3tri, megafold, bionemo_ir, turbopf, ligo_pwa, megafold_tr: adapters ``ext_<name>.py`` with ``install()``
                              (TriAttn / AttentionPairBias / OPM / PWA / Transition; see ``ext_attn_common.py``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
INLINE = ("ds4sci", "trifast")
#: triangle-multiplication adapters (measurer backend name -> adapter module)
TRIMUL_REGISTRY = {
    "fast_trimul": "ext_fast_trimul",  # tiagomonteiro0715/fast_trimul 3.0.6 (Apache-2.0): CuTe DSL fp16 fused update
    "gpumode": "ext_gpumode_trimul",  # arseniivanov/trimul (MIT): GPU MODE TriMul competition Triton kernels (A100 / H100 / B200 files)
    "geonn": "ext_geonn_trimul",  # ROCm/geometric-nn-ops 1.0.0 (Apache-2.0): Triton forward, cuEquivariance signature
}
#: Pairformer-operator adapters (each module's ``install()`` attaches its backend to the classes it serves)
ATTN_REGISTRY = {
    "gnops": "ext_gnops",
    "of3tri": "ext_of3tri",
    "megafold": "ext_megafold",
    "bionemo_ir": "ext_bionemo_ir",
    "turbopf": "ext_turbopf",
    "ligo_pwa": "ext_ligo_pwa",
    "megafold_tr": "ext_megafold_tr",
}
AVAILABLE = (*INLINE, *TRIMUL_REGISTRY, *ATTN_REGISTRY)
#: operator classes each backend serves (the adapters refuse what a library does not cover, with a reason)
CLASS_OF = {
    "ds4sci": ("TriAttn",),
    "trifast": ("TriAttn",),
    "fast_trimul": ("TriMul",),
    "gpumode": ("TriMul",),
    "geonn": ("TriMul",),
    "gnops": ("TriAttn", "AttentionPairBias"),
    "of3tri": ("TriAttn", "AttentionPairBias"),
    "megafold": ("TriAttn", "AttentionPairBias"),
    "bionemo_ir": ("TriAttn", "OPM", "PWA", "Transition"),
    "turbopf": ("TriAttn",),
    "ligo_pwa": ("PWA",),
    "megafold_tr": ("Transition",),
}


# ------------------------------------------------------------------------------------------------------------------------------- cores
def _ds4sci_core(q, k, v, bias, mask):
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    b, n, _h, s, _d = q.shape
    dt = q.dtype
    # The kernel applies the d^-0.5 softmax scale itself (checked against a float64 reference: pre-scaling q, as the first E5 run did, scales twice -> relative error 0.19, gradients 0.9)
    qq, kk, vv = ((t.transpose(2, 3)).contiguous() for t in (q, k, v))  # [B, N, Q, H, d]
    if mask is None:
        mbias = torch.zeros(b, n, 1, 1, s, dtype=dt, device=q.device)
    else:
        mbias = ((~mask.bool()).to(dt) * -1e9)[:, None, None, None, :].expand(b, n, 1, 1, s).contiguous()
    out = DS4Sci_EvoformerAttention(qq, kk, vv, [mbias, bias.to(dt)])
    return out.transpose(2, 3)


def _trifast_core(q, k, v, bias, mask):
    """trifast.triangle_attention (latkins/trifast 0.1.13): q / k / v ``[B, H, N, N, d]``, bias ``[B, H, N, N]``, mask ``[B, N, N]`` bool where True =
    masked (trifast/equiv.py: ``neg_inf * mask``); the softmax scale ``d^-0.5`` is inside. Our core layout is ``[B, N, H, Q, d]``."""
    from trifast import triangle_attention

    b, n, _h, s, _d = q.shape
    tq, tk, tv = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    tb = bias[:, 0].to(q.dtype).contiguous()
    tm = (
        torch.zeros(b, n, s, dtype=torch.bool, device=q.device) if mask is None else (~mask.bool())[:, None, :].expand(b, n, s).contiguous()
    )
    return triangle_attention(tq, tk, tv, tb, tm).transpose(1, 2)


# --------------------------------------------------------------------------------------------------------------------------- adapters
_IMPORT_PROBES = {"deepspeed": "deepspeed.ops.deepspeed4science", "trifast": "trifast"}


def _import_error(probe: str) -> str | None:
    """Why the library cannot be imported (not installed, a missing dependency, a failed import), or None."""
    try:
        importlib.import_module(_IMPORT_PROBES.get(probe, probe))
    except Exception as exc:  # recorded as the reason the backend does not serve the case
        return f"{probe}: {type(exc).__name__}: {str(exc)[:200]} (see E5_other_kernels/README.md)"
    return None


def _supports(name: str, probe: str):
    def supports(self, pair, mask=None):
        why = _import_error(probe)
        if why is not None:
            return why
        if name == "ds4sci" and pair.dtype not in (torch.bfloat16, torch.float16):
            return f"DS4Sci_EvoformerAttention takes bf16 / fp16, got {pair.dtype}"
        if not torch.cuda.is_available():
            return "no CUDA device"
        return self._core_refusal(pair, mask)

    return supports


def _forward(core):
    def fwd(self, pair, mask=None):
        from miniworld_engine.modules.primitives import master_autocast

        with master_autocast(pair, self):
            return self._residual(pair, mask, core)

    return fwd


def add_paths(third_party_dir: str) -> None:
    """Make a fetched upstream tree importable: ``$PAPER_EXT_PATH`` (default ``~/paper-ext``), ``$PAPER_EXT_ATTN/<dir>`` (default ``~/paper-ext-attn``),
    then the repo's ``third_party/<dir>/upstream``."""
    ext = os.environ.get("PAPER_EXT_PATH") or os.path.expanduser("~/paper-ext")
    attn = os.environ.get("PAPER_EXT_ATTN") or os.path.expanduser("~/paper-ext-attn")
    for p in (ext, os.path.join(attn, third_party_dir), str(_HERE.parents[2] / "third_party" / third_party_dir / "upstream")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)  # after site-packages: the engine environment wins for every package it has


def library_refusal(self, pair, mask) -> str | None:
    """What every TriMul adapter needs: [B, L, L, d] pair, a bool [B, L] residue mask, one LayerNorm epsilon."""
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return "mask must be a bool [B, L] residue mask"
    if self.ln_pair.eps != self.ln_out.eps:
        return "the library op takes one LayerNorm epsilon for LN_in and LN_out"
    if not torch.cuda.is_available():
        return "no CUDA device"
    return None


def import_error(module: str, probe: str) -> str | None:
    try:
        importlib.import_module(probe)
    except Exception as exc:  # recorded as the reason the backend does not serve the case
        return f"{probe}: {type(exc).__name__}: {str(exc)[:240]} ({module}; see third_party/ and E5_other_kernels/README.md)"
    return None


def _install_trimul(names) -> None:
    import miniworld_engine.modules.triangle as tri

    for name in names:
        mod = importlib.import_module(TRIMUL_REGISTRY[name])
        cls = tri.TriangleMultiplication
        if name not in cls.backends:
            cls.backends = (*cls.backends, name)

        def supports(self, pair, mask=None, dropout_p=None, _m=mod):
            why = library_refusal(self, pair, mask) or import_error(_m.__name__, _m.PROBE)
            return why or _m.supports(self, pair, mask, dropout_p)

        def fwd(self, pair, mask=None, dropout_p=None, _m=mod):
            return _m.forward(self, pair, mask, dropout_p)

        setattr(cls, f"_supports_{name}", supports)
        setattr(cls, f"_forward_{name}", fwd)


def _install_attn(names) -> None:
    for name in names:
        importlib.import_module(ATTN_REGISTRY[name]).install()


def _install_inline(names) -> None:
    import miniworld_engine.modules.triangle as tri

    ext = os.environ.get("PAPER_EXT_PATH") or os.path.expanduser("~/paper-ext")
    if os.path.isdir(ext) and ext not in sys.path:
        sys.path.append(ext)  # after site-packages: the engine env wins for every package it has
    table = {"ds4sci": ("deepspeed", _ds4sci_core), "trifast": ("trifast", _trifast_core)}
    for name in names:
        probe, core = table[name]
        for cls in _classes(tri):
            if name not in cls.backends:
                cls.backends = (*cls.backends, name)
            setattr(cls, f"_supports_{name}", _supports(name, probe))
            setattr(cls, f"_forward_{name}", _forward(core))


def install(names) -> None:
    """Install the named external backends (any of ``AVAILABLE``) on the module classes they serve."""
    names = list(names)
    unknown = [n for n in names if n not in AVAILABLE]
    if unknown:
        raise ValueError(f"unknown external backend(s) {unknown}; choose from {AVAILABLE}")
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))  # the adapters ext_<name>.py and ext_attn_common.py sit next to this file
    _install_inline([n for n in names if n in INLINE])
    _install_trimul([n for n in names if n in TRIMUL_REGISTRY])
    _install_attn([n for n in names if n in ATTN_REGISTRY])


def _classes(tri):
    out, todo = [], [tri._TriangleAttentionBase]
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(c.__subclasses__())
    return [c for c in out if "cueq" in c.backends]
