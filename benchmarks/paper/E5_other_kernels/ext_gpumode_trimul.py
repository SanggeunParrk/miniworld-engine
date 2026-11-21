"""E5 adapter: the GPU MODE TriMul competition kernels (arseniivanov/trimul, MIT): ``triton_a100`` / ``triton_h100`` / ``triton_b200``.

The task (AF3 outgoing TriMul, forward only) is a slightly different function from the engine's update, so the adapter maps it exactly:

  task:    out = W_o( LN_out(t) * sigmoid(W_og x) )         [output gate in the HIDDEN width, before W_o]; no residual, mask [B, L, L]
  engine:  out = sigmoid(W_g x) * W_o( LN_out(t) )          [output gate in the pair width, after W_o];  residual added by the module

  * the kernel's output-gate weight W_og is set to zero (sigmoid(0) = 1/2) and W_o is doubled: the kernel then returns exactly W_o(LN_out(t)),
    at the cost of the output-gate GEMM it computes anyway (the competition's work);
  * the engine's gate ``sigmoid(W_g LN_in(x)) * out + x`` (and the LN_in it needs) is done by the adapter in the activation dtype: an extra
    LayerNorm + d x d GEMM + elementwise pass that a native implementation would fuse. This overhead is part of the measured time and is
    stated in the report; the numbers are an upper bound for the kernel in this module slot.
  * the mask: the engine multiplies a_ik by (m_i & m_k) and b_jk by (m_j & m_k); the task multiplies a_ik by mask_ik and b_jk by mask_jk. With
    the task mask set to the pair mask m_i & m_j both are the same function (verified in the accuracy check).
Kernel facts: fp16 tensor-core math, fp32 input cast inside, outgoing direction only (incoming = unsupported), forward only (training =
unsupported), the module file per GPU (sm80 a100, sm90 h100, sm100 b200; the 5090 / sm120 uses the h100 file, untuned). For L <= 512
(a100: < 512, b200: < 800) the competition entry is a torch.compile'd PyTorch function, for longer rows the Triton kernels: the entry is
measured as submitted.
"""

from __future__ import annotations

import importlib
import weakref

import torch
from ext_trimul_registry import add_paths

add_paths("gpumode_trimul")
PROBE = "gpumode_trimul.triton_h100"
_FILE = {(8, 0): "triton_a100", (9, 0): "triton_h100", (10, 0): "triton_b200"}
_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # engine module -> (versions, weights)


def _entry():
    cap = torch.cuda.get_device_capability()
    name = _FILE.get(cap, "triton_h100")  # sm120 and anything else: the h100 file, untuned for it
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    try:
        return importlib.import_module(f"gpumode_trimul.{name}")
    finally:  # the kernel files set these globals at import; the engine's measurement keeps its own
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags


def supports(self, pair, mask, dropout_p):
    if not self.outgoing:
        return "the competition task is the outgoing update only"
    if pair.dtype not in (torch.bfloat16, torch.float32, torch.float16):
        return f"dtype {pair.dtype}"
    if torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters())):
        return "training: forward-only kernels (no backward)"
    if self.ln_pair.eps != 1e-5:
        return "the kernels hard-code the LayerNorm epsilon 1e-5"
    return None


def _weights(self):
    srcs = (self.ln_pair.weight, self.ln_pair.bias, self.ln_out.weight, self.ln_out.bias, self.to_left.weight, self.to_right.weight,
            self.to_left_gate.weight, self.to_right_gate.weight, self.to_out.weight)  # fmt: skip
    ver = tuple((p.data_ptr(), p._version) for p in srcs)
    ent = _CACHE.get(self)
    if ent is None or ent[0] != ver:
        f = lambda t: t.detach().to(torch.float32).contiguous()
        w = {
            "norm.weight": f(self.ln_pair.weight), "norm.bias": f(self.ln_pair.bias),
            "left_proj.weight": f(self.to_left.weight), "right_proj.weight": f(self.to_right.weight),
            "left_gate.weight": f(self.to_left_gate.weight), "right_gate.weight": f(self.to_right_gate.weight),
            "out_gate.weight": torch.zeros_like(f(self.to_left.weight)),  # sigmoid(0) = 1/2 ...
            "to_out_norm.weight": f(self.ln_out.weight), "to_out_norm.bias": f(self.ln_out.bias),
            "to_out.weight": 2.0 * f(self.to_out.weight),  # ... undone here
        }  # fmt: skip
        ent = _CACHE[self] = (ver, w)
    return ent[1]


def forward(self, pair, mask, dropout_p):
    from miniworld_engine.modules.functional import dropout_scale, sigmoid_gate
    from miniworld_engine.modules.primitives import master_autocast

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return pair
    dropout_p = self.p_drop if dropout_p is None else dropout_p
    scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and self.training else None
    b, n = pair.shape[0], pair.shape[1]
    pm = torch.ones(b, n, n, dtype=torch.float32, device=pair.device) if mask is None else (mask[:, :, None] & mask[:, None, :]).float()
    data = (pair, pm, _weights(self), {"dim": pair.shape[-1], "hidden_dim": self.d_hidden})
    out = _entry().custom_kernel(data)  # [B, L, L, d] fp16: W_o(LN_out(t)) as mapped above
    with master_autocast(pair, self):
        out = sigmoid_gate(self.to_gate(self.ln_pair(pair)), out.to(pair.dtype))
        if scale is not None:
            out = out * scale
        return out + pair
