"""Shared plumbing of the E5 ``ext_<name>.py`` adapters of the pair-bias attention libraries (triangle attention, AttentionPairBias).

A library adapter hands this module an *evo core*: ``core(q, k, v, keep, bias) -> out`` with q / k / v ``[B, N, Q, H, d]`` (``N`` rows of
attention: the triangle-attention rows, or 1 for AttentionPairBias), key mask ``keep`` ``[B, N, 1, 1, K]`` bool (True = attend) or None,
bias ``[B, 1, H, Q, K]`` -> ``[B, N, Q, H, d]`` (the layout of the Evoformer-attention kernels of DeepSpeed4Science, OpenFold3, MegaFold and
geometric-nn-ops), and a ``reason(dtype, head_dim, length, grad) -> str | None`` (the library's own range). :func:`attach` then adds the
backend to the triangle-attention classes (the module's own LayerNorm, projections, gate, out-projection and residual, via ``_residual``) and
to ``AttentionPairBias`` (the statements of its ``_forward_triton``); nothing under ``src/`` changes. A case the library does not serve is
``unsupported`` with the reason (a missing package, dtype, head width, mask), never re-routed.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch


def ext_root() -> str:
    """Where the adapters' upstream trees are installed on this card (``fetch.sh`` of each third_party/<name>/): ``PAPER_EXT_ATTN``."""
    return os.environ.get("PAPER_EXT_ATTN") or os.path.expanduser("~/paper-ext-attn")


def upstream(name: str) -> str:
    return os.path.join(ext_root(), name)


def find_upstream(name: str, package: str) -> str:
    """The root that contains ``package``: ``$PAPER_EXT_ATTN/<name>``, ``$PAPER_EXT_PATH`` (the TriMul adapters' install dir), then the repo's
    ``third_party/<name>/upstream`` (``geometric-nn-ops`` is one entry for the ``geonn`` and ``gnops`` adapters); the first existing, else the first."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        upstream(name),
        os.environ.get("PAPER_EXT_PATH") or os.path.expanduser("~/paper-ext"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "third_party", name, "upstream"),
    ]
    return next((r for r in roots if os.path.isdir(os.path.join(r, package))), roots[0])


def load_file(modname: str, path: str):
    """Import one source file as its own module (no package ``__init__`` runs), registered as ``modname``."""
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None or not os.path.isfile(path):
        raise ImportError(f"{path} is not installed (run third_party/<name>/fetch.sh)")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    return mod


def _grad_wanted(module, x) -> bool:
    return torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in module.parameters()))


def _tri_core(evo):
    def core(q, k, v, bias, mask):
        """The cuEquivariance-signature core of the triangle-attention classes (q / k / v ``[B, N, H, Q, d]``, bias ``[B, 1, H, Q, K]``, key mask
        ``[B, K]``) on an evo core."""
        b, n = q.shape[:2]
        keep = None if mask is None else mask.bool()[:, None, None, None, :].expand(b, n, 1, 1, mask.shape[-1]).contiguous()
        out = evo(q.transpose(2, 3).contiguous(), k.transpose(2, 3).contiguous(), v.transpose(2, 3).contiguous(), keep, bias.to(q.dtype))
        return out.transpose(2, 3)

    return core


def _tri_classes():
    import miniworld_engine.modules.triangle as tri

    out, todo = [], [tri._TriangleAttentionBase]
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(c.__subclasses__())
    return [c for c in out if "cueq" in c.backends]


def attach(name: str, evo, reason, probe, *, triangle: bool = True, apb: bool = True) -> None:
    """Attach backend ``name`` (evo core ``evo``, range ``reason``, import check ``probe`` raising on failure)."""
    from miniworld_engine.modules import attention as attn
    from miniworld_engine.modules.primitives import master_autocast

    def import_error():
        try:
            probe()
        except Exception as exc:
            return f"{name}: {type(exc).__name__}: {str(exc)[:200]} (see third_party/ README, fetch.sh)"
        return None

    core = _tri_core(evo)

    def tri_supports(self, pair, mask=None):
        why = import_error()
        if why is not None:
            return why
        if not torch.cuda.is_available():
            return "no CUDA device"
        why = self._core_refusal(pair, mask)
        if why is not None:
            return why
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else pair.dtype
        return reason(dt, self._head_dim(), pair.shape[1], _grad_wanted(self, pair))

    def tri_forward(self, pair, mask=None):
        with master_autocast(pair, self):
            return self._residual(pair, mask, core)

    if triangle:
        for cls in _tri_classes():
            if name not in cls.backends:
                cls.backends = (*cls.backends, name)
            setattr(cls, f"_supports_{name}", tri_supports)
            setattr(cls, f"_forward_{name}", tri_forward)

    def apb_supports(self, single, pair, mask=None, extra_pair_bias=None):
        why = import_error()
        if why is not None:
            return why
        if single.dim() != 3:
            return "single must be [B, L, d]"
        why = self._call_refusal(single, pair, mask) or self._extra_reason(single, extra_pair_bias)
        if why is not None:
            return why
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else single.dtype
        return reason(dt, self._head_dim(), single.shape[1], _grad_wanted(self, single))

    def apb_forward(self, single, pair, mask=None, extra_pair_bias=None):
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return single
        with master_autocast(single, self):
            b, n = single.shape[:2]
            h, d = self.n_head, self._head_dim()
            s = self.ln_single(single)
            q, k, v = (t.view(b, 1, n, h, d) for t in (self.to_query(s), self.to_key(s), self.to_value(s)))
            if self.use_qk_norm:
                q, k = self.norm_query(q), self.norm_key(k)
            bias = self.to_bias(self.ln_pair(pair))  # [B, L, L, H]
            if extra_pair_bias is not None:
                bias = bias + attn._extra_planes(extra_pair_bias, b, h, n).to(bias.dtype).permute(0, 2, 3, 1)
            keep = None if mask is None else mask.bool()[:, None, None, None, :].contiguous()
            out = evo(q, k.to(q.dtype), v.to(q.dtype), keep, bias.permute(0, 3, 1, 2)[:, None].contiguous().to(q.dtype))
            gate = self.to_gate(s)
            if self.gate_bias:
                gate = gate + self.gate_bias
            out = attn.sigmoid_gate(gate, out.reshape(b, n, h * d).to(gate.dtype))
            return single + self.to_out(out)

    if apb:
        cls = attn.AttentionPairBias
        if name not in cls.backends:
            cls.backends = (*cls.backends, name)
        setattr(cls, f"_supports_{name}", apb_supports)
        setattr(cls, f"_forward_{name}", apb_forward)
