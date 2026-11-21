"""E5 external backend ``bionemo_ir``: NVIDIA BioNeMo Inference Runtime (``pip install bionemo-ir``, Apache-2.0 wheel; the CuTeDSL kernel SOURCE is not published,
the wheel ships the Python callables and precompiled CUBINs). Inference only.

Triangle attention: ``HeuristicTriangleAttention`` (SM80 / SM90 CuTeDSL CUBIN; SM90 bf16 d = 32 large shapes the Claude-kit kernel when its launcher is in the wheel;
cuEquivariance / SDPA fallbacks inside the library, recorded as the route): q / k / v ``[B, I, J, H, d]``, ``biases = [actual_s_kv [B, I] int32 (left-aligned
key mask), pair_bias [B, H, J, J]]``; fp16 / bf16, forward only (a gradient call is ``unsupported``).
The ``bionemo_ir`` package ``__init__`` (model registry, ray, ...) is NOT run: the submodules are imported under a stub package.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import ext_attn_common as common
import torch

NAME = "bionemo_ir"
ROUTES: dict = {}  # (h, d, length, dtype) -> the delegate the library chose (CuTeDSL / ClaudeKit / CUEQUIV / SDPA)


def stub_package() -> None:
    """Make ``bionemo_ir.*`` importable without running the package ``__init__``."""
    if "bionemo_ir" not in sys.modules:
        stub = types.ModuleType("bionemo_ir")
        stub.__path__ = [os.path.join(common.upstream("bionemo-ir"), "bionemo_ir")]
        sys.modules["bionemo_ir"] = stub


def _import():
    stub_package()
    return importlib.import_module("bionemo_ir._torch.attention_backend.triangle_attention.heuristic")


_TRI: dict = {}


def _backend(h, d):
    if (h, d) not in _TRI:
        _TRI[(h, d)] = _import().HeuristicTriangleAttention(0, h, d)
    return _TRI[(h, d)]


def evo(q, k, v, keep, bias):
    b, n, s, h, d = q.shape
    lengths = torch.full((b, n), s, dtype=torch.int32, device=q.device) if keep is None else keep[:, :, 0, 0, :].sum(-1, dtype=torch.int32)
    be = _backend(h, d)
    out = be.forward(q, k, v, biases=[lengths, bias[:, 0].contiguous()])
    ROUTES[(h, d, s, str(q.dtype))] = be.last_backend_name
    return out


def reason(dtype, head_dim, length, grad):
    if grad:
        return "BioIR kernels are inference only (no backward)"
    if dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {dtype}: the CuTeDSL kernels take fp16 / bf16"
    if ((head_dim + 31) // 32) * 32 not in (32, 64, 128, 256):
        return f"head width {head_dim}: padded to 32 / 64 / 128 / 256 only"
    return None


def install() -> None:
    common.attach(NAME, evo, reason, _import, apb=False)
    _attach_ops()


# ------------------------------------------------------------------------------------------------------- the other operators (inference only)
# Each op takes the module's own prologue (LayerNorm, projections, masks, softmax) done in PyTorch and runs the library's fused contraction / epilogue kernel;
# the library silently falls back to its own PyTorch formula outside its shipped envelope, so the envelope is checked here and a call outside it is `unsupported`.
def _ops(name, sub="ops"):
    stub_package()
    return importlib.import_module(f"bionemo_ir._torch.custom_ops.{name}.{sub}")


def _grad(module, x):
    return torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in module.parameters()))


def _cast(w, dt):
    return None if w is None else w.to(dt)


def _opm_supports(self, msa, mask=None, token_asym_id=None, residual=None):
    try:
        ops = _ops("outer_product_mean")
    except Exception as exc:
        return f"bionemo_ir: {type(exc).__name__}: {str(exc)[:200]}"
    if _grad(self, msa):
        return "BioIR kernels are inference only (no backward)"
    if self.groups != 1 or self.mask_interchain:
        return "groups / mask_interchain (Chai-1 forms): the op is the standard outer product mean"
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else msa.dtype
    if dt not in (torch.bfloat16, torch.float16):
        return f"dtype {dt}: bf16 / fp16"
    d_hidden, d_pair = self.to_left.weight.shape[0], self.to_out.weight.shape[0]
    if (d_hidden, d_hidden) != (ops._KERNEL_C, ops._KERNEL_D) or d_pair not in ops._SUPPORTED_CZ:
        return f"d_hidden {d_hidden}, d_pair {d_pair}: the op ships d_hidden 32 with d_pair in {ops._SUPPORTED_CZ}"
    sm = ops.get_sm_version()
    if dt not in ops._CUTE_NATIVE.get(sm, ()) and not (sm >= ops._TRITON_MIN_SM and dt in ops._TRITON_DTYPES):
        return f"sm_{sm} dtype {dt}: no CuTe / Triton kernel (the library would run its PyTorch formula)"
    return None


def _opm_forward(self, msa, mask=None, token_asym_id=None, residual=None):
    from miniworld_engine.modules.primitives import master_autocast

    if self.dead and residual is not None and not (self.training or torch.is_grad_enabled()):
        return residual
    with master_autocast(msa, self):
        ops = _ops("outer_product_mean")
        m = self.ln_msa(msa)
        left, right = self.to_left(m), self.to_right(m)
        if mask is None:
            mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
        left, right = left * mask[..., None], right * mask[..., None]
        mf = mask.to(torch.float32)
        norm = torch.bmm(mf.transpose(1, 2), mf)  # [B, I, J] pair counts (fp32)
        norm = norm + self.count_offset if self.count_offset is not None else norm.clamp(min=1)
        op = ops.get_outer_product_mean_op(left.dtype, left.shape[-1], right.shape[-1], self.to_out.weight.shape[0])
        pair = op(left, right, norm, _cast(self.to_out.weight, left.dtype), _cast(self.to_out.bias, left.dtype), self.normalize_before_proj)
        return residual + pair if residual is not None else pair


def _pwa_supports(self, msa, pair, mask=None):
    try:
        ops = _ops("pair_weighted_averaging")
    except Exception as exc:
        return f"bionemo_ir: {type(exc).__name__}: {str(exc)[:200]}"
    if _grad(self, msa) or (self.training and self.drop_msa.p_drop > 0):
        return "BioIR kernels are inference only (no backward)"
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else msa.dtype
    if dt not in (torch.bfloat16, torch.float16):
        return f"dtype {dt}: bf16 / fp16"
    d_msa, _d_pair, h, d = self._widths()
    cfg = _ops("pair_weighted_averaging", "_config")
    if not cfg.is_supported_dims(h, d, d_msa):
        return f"{h} heads x {d}, d_msa {d_msa}: the op ships H = {ops._KERNEL_H} with (D, c_m) in {cfg._SUPPORTED_DIMS}"
    sm = ops.get_sm_version()
    if sm not in ops._SUPPORTED_SM:
        return f"sm_{sm}: not an SM the op serves"
    if not cfg.is_profitable_shape(
        h, d, d_msa, max(msa.shape[2], pair.shape[1]), sm_version=sm, dtype_str="bf16" if dt is torch.bfloat16 else "fp16"
    ):
        return f"L {msa.shape[2]}: past the op's fused token limit (the library falls back to its chunked PyTorch path)"
    return None


def _pwa_forward(self, msa, pair, mask=None):
    import torch.nn.functional as F
    from miniworld_engine.modules.primitives import master_autocast

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return msa
    with master_autocast(msa, self):
        ops = _ops("pair_weighted_averaging")
        m = self.ln_msa(msa)
        value, gate = self.to_value(m), self.to_gate(m)
        bias = self.to_bias(self.ln_pair(pair)).permute(0, 3, 1, 2)  # [B, H, I, J]
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
        w = F.softmax(bias, dim=-1)
        op = ops.get_pair_weighted_averaging_op(value.dtype, value.shape[-1] // self.n_head, msa.shape[-1], self.n_head)
        if w.shape[-1] % 8:  # the op wants the key axis padded to a multiple of 8 (zero weights)
            w = F.pad(w, (0, 8 - w.shape[-1] % 8))
        out = op(w.contiguous(), value, gate, _cast(self.to_out.weight, value.dtype))
        return msa + self.drop_msa(out)


def _tr_supports(self, x, *args, **kwargs):
    try:
        ops = _ops("transition_mlp")
    except Exception as exc:
        return f"bionemo_ir: {type(exc).__name__}: {str(exc)[:200]}"
    if _grad(self, x):
        return "BioIR kernels are inference only (no backward)"
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if dt is not torch.bfloat16:
        return f"dtype {dt}: the transition kernel ships bf16"
    w, hidden = self.d_hidden, self.d_hidden * self.n
    op = ops.get_transition_mlp_op(dt, width=w, hidden=hidden, activation="silu_gate", has_bias=False, has_mask=False, has_residual=True)
    if op is None:
        return f"width {w}, hidden {hidden}: no shipped silu_gate (no bias / no mask / residual) variant on this SM"
    return None


_W1: dict = {}


def _tr_forward(self, x):
    from miniworld_engine.modules.primitives import master_autocast

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return x
    with master_autocast(x, self):
        ops = _ops("transition_mlp")
        h = self.ln_in(x)
        key = (id(self), x.dtype, self.expand_a.weight._version, self.expand_b.weight._version, self.squeeze.weight._version)
        if key not in _W1:  # value rows (expand_b) then gate rows (expand_a): the op's silu_gate layout
            _W1[key] = (
                torch.cat([self.expand_b.weight, self.expand_a.weight], 0).to(x.dtype).contiguous(),
                self.squeeze.weight.to(x.dtype).contiguous(),
            )
        w1, w2 = _W1[key]
        op = ops.get_transition_mlp_op(
            x.dtype,
            width=self.d_hidden,
            hidden=self.d_hidden * self.n,
            activation="silu_gate",
            has_bias=False,
            has_mask=False,
            has_residual=True,
        )
        return op(h, w1, None, w2, None, x, None)


def _attach_ops() -> None:
    from miniworld_engine.modules import msa, transition

    for cls, sup, fwd in (
        (msa.OuterProductMean, _opm_supports, _opm_forward),
        (msa.MSAPairWeightedAveraging, _pwa_supports, _pwa_forward),
        (transition.Transition, _tr_supports, _tr_forward),
    ):
        if NAME not in cls.backends:
            cls.backends = (*cls.backends, NAME)
        setattr(cls, f"_supports_{NAME}", sup)
        setattr(cls, f"_forward_{NAME}", fwd)
