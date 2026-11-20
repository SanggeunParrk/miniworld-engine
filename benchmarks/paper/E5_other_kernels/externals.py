"""E5: other public triangle-attention / triangle-multiplication kernels as external backends of the measurer, installed at run time.

Nothing under src/ is edited: ``install(names)`` adds the backend name to ``backends`` of the triangle-attention classes (a class attribute
tuple) and attaches ``_supports_<name>`` / ``_forward_<name>`` to them, exactly the shape the measurers read (``BackendModule.backends``,
``unsupported`` -> ``_supports_<name>``, ``_forward_<name>``). The forward is the module's own ``_residual`` with the library's attention CORE
(the cuEquivariance-signature core: q / k / v ``[B, N, H, Q, d]``, bias ``[B, 1, H, Q, K]``, key mask ``[B, K]``), so LayerNorm, projections,
gate, out-projection and residual are the module's own statements, as for ``cueq`` / ``cuda``: only the attention core differs.

Backends (a missing package or an unsupported shape is recorded as ``unsupported`` with the reason, never re-routed):
  ds4sci   DeepSpeed4Science ``DS4Sci_EvoformerAttention`` (deepspeed.ops.deepspeed4science): bf16 / fp16, the library's own head-width limits.
  trifast  ``trifast.triangle_attention`` (pip install trifast): Triton kernels, fp32 / bf16 / fp16.
Triangle multiplication: no published fused kernel besides cuEquivariance (``cueq``, already a backend) and the Anthropic kits was found;
``tm_candidates()`` lists what was checked (README).
"""

from __future__ import annotations

import importlib
import importlib.util

import torch

AVAILABLE = ("ds4sci", "trifast")


# ------------------------------------------------------------------------------------------------------------------------------- cores
def _ds4sci_core(q, k, v, bias, mask):
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    b, n, _h, s, d = q.shape
    dt = q.dtype
    qq, kk, vv = ((t.transpose(2, 3)).contiguous() for t in (q * d**-0.5, k, v))  # [B, N, Q, H, d]; the library has no internal scale
    if mask is None:
        mbias = torch.zeros(b, n, 1, 1, s, dtype=dt, device=q.device)
    else:
        mbias = ((~mask.bool()).to(dt) * -1e9)[:, None, None, None, :].expand(b, n, 1, 1, s).contiguous()
    out = DS4Sci_EvoformerAttention(qq, kk, vv, [mbias, bias.to(dt)])
    return out.transpose(2, 3)


def _trifast_layout():
    """The layout trifast's own test-tensor generator uses: shapes of (q, k, v, bias, mask), probed at run time (the README states none)."""
    from trifast.utils import gen_tensors

    got = gen_tensors(8, 32, 2, use_mask=True, device="cuda", dtype=torch.bfloat16, std=1.0)
    return [tuple(t.shape) for t in got[:5]]


def _trifast_core(q, k, v, bias, mask):
    from trifast import triangle_attention

    b = q.shape[0]
    # the tensor layout is not documented: probed once from trifast.utils.gen_tensors, the call is made in that layout
    shapes = _trifast_layout()
    q5 = tuple(shapes[0])
    if len(q5) == 4:  # [N, H, S, d] or [S, ...]: no batch dim -> one sample per call
        if b != 1:
            raise RuntimeError(f"trifast takes no batch dimension (layout {shapes}); B={b}")
        tq, tk, tv = (t[0] for t in (q, k, v))
        tb = bias[0, 0]
        tm = mask[0] if mask is not None else None
        out = triangle_attention(tq, tk, tv, tb, tm)
        return out[None]
    return triangle_attention(q, k, v, bias.float(), mask)


# --------------------------------------------------------------------------------------------------------------------------- adapters
def _supports(name: str, probe: str):
    def supports(self, pair, mask=None):
        if importlib.util.find_spec(probe) is None:
            return f"{probe} is not installed (see E5_other_kernels/README.md)"
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


def install(names) -> None:
    """Add the named external backends to every single-direction triangle-attention class that has the cueq core signature."""
    import miniworld_engine.modules.triangle as tri

    table = {"ds4sci": ("deepspeed", _ds4sci_core), "trifast": ("trifast", _trifast_core)}
    for name in names:
        if name not in table:
            raise ValueError(f"unknown external backend {name!r}; choose from {AVAILABLE}")
        probe, core = table[name]
        for cls in _classes(tri):
            if name not in cls.backends:
                cls.backends = (*cls.backends, name)
            setattr(cls, f"_supports_{name}", _supports(name, probe))
            setattr(cls, f"_forward_{name}", _forward(core))


def _classes(tri):
    out, todo = [], [tri._TriangleAttentionBase]
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(c.__subclasses__())
    return [c for c in out if "cueq" in c.backends]


def tm_candidates() -> list[str]:
    return [
        "cuequivariance_torch.triangle_multiplicative_update (already the `cueq` backend)",
        "Anthropic kits (already the `anthropic` backend)",
        "DeepSpeed4Science: attention only (no triangle multiplication kernel)",
        "trifast: attention only",
    ]
