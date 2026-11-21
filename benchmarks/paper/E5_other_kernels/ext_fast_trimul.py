"""E5 adapter: ``fast_trimul`` 3.0.6 (tiagomonteiro0715/fast_trimul, Apache-2.0): CuTe DSL fused TriMul, fp16 kernels.

What the library computes (``TriangleMultiplicativeUpdateKernelFused``, ``residual=False`` here): LN_in, gated a / b projections (biased), the
batched fp16 tensor-core contraction (outgoing / incoming), LN_out, ``sigmoid(proj_g(LN_in)) * proj_out(...)``; the residual is added by this
adapter in the activation dtype (the library's own residual would add in fp16). Our bias-free module maps onto it with zero biases; the
weights are copied (fp16) into one library module per engine module, in place, so the library's prebound weight tensors stay valid.
Differences recorded, not hidden: fp16-only kernels (bf16 / fp16 activations are cast to fp16: a range reduction for large activations),
N % 8 == 0 (the library pads in its own dispatch; the adapter calls the kernel path directly), masks: an all-True mask is served (checked outside graph capture), a mask with padding positions is
unsupported (the library masks z before LN_in, ours masks a / b: different functions), the backward is the library's PyTorch recompute inside ``_TrainableTriMul`` (correct, not fast).
"""

from __future__ import annotations

import weakref

import torch
from ext_trimul_registry import add_paths

add_paths("fast_trimul")
PROBE = "fast_trimul._kernels"
_IMPL: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # engine module -> [library module, versions of the weights it holds]


def supports(self, pair, mask, dropout_p):
    if pair.dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {pair.dtype}: fast_trimul's kernels are fp16-only (bf16 / fp16 activations are cast; fp32 would silently lose precision)"
    if mask is not None and not torch.cuda.is_current_stream_capturing() and not bool(mask.all()):
        return "mask with padding positions: fast_trimul masks z before LN_in (not a / b after it), a different function from the module's"
    if pair.shape[1] % 8:
        return f"L={pair.shape[1]}: the CuTe contraction needs N % 8 == 0"
    return None


def _params(self):
    """The 16 tensors of the library's ``_params()`` order, as fp16 (autograd-linked to the engine's masters), biases zero."""
    h = lambda t: t.to(torch.float16)
    d, dh = self.to_gate.weight.shape[0], self.d_hidden
    z = lambda n: torch.zeros(n, dtype=torch.float16, device=self.to_gate.weight.device)
    return [
        h(self.ln_pair.weight), h(self.ln_pair.bias), h(self.ln_out.weight), h(self.ln_out.bias),
        h(self.to_left.weight), z(dh), h(self.to_left_gate.weight), z(dh),
        h(self.to_right.weight), z(dh), h(self.to_right_gate.weight), z(dh),
        h(self.to_gate.weight), z(d), h(self.to_out.weight), z(d),
    ]  # fmt: skip


def _library_module(self):
    from fast_trimul._kernels import TriangleMultiplicativeUpdateKernelFused

    src = (self.ln_pair.weight, self.ln_pair.bias, self.ln_out.weight, self.ln_out.bias, self.to_left.weight, self.to_left_gate.weight,
                       self.to_right.weight, self.to_right_gate.weight, self.to_gate.weight, self.to_out.weight)  # fmt: skip
    ver = tuple((p.data_ptr(), p._version) for p in src)
    ent = _IMPL.get(self)
    if ent is None:
        d = self.to_gate.weight.shape[0]
        mod = TriangleMultiplicativeUpdateKernelFused(d, self.d_hidden, "outgoing" if self.outgoing else "incoming", residual=False)
        mod = mod.half().to(self.to_gate.weight.device).requires_grad_(False)
        ent = _IMPL[self] = [mod, None]
    if ent[1] != ver:
        with torch.no_grad():
            for dst, val in zip(ent[0]._params(), _params(self), strict=True):
                dst.copy_(val)  # in place: the library's prebound weight views follow
            ent[0].norm_in.eps = ent[0].norm_out.eps = self.ln_pair.eps
        ent[1] = ver
    return ent[0]


def forward(self, pair, mask, dropout_p):
    from fast_trimul._kernels import _TrainableTriMul

    from miniworld_engine.modules.functional import dropout_scale

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return pair
    dropout_p = self.p_drop if dropout_p is None else dropout_p
    scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and self.training else None
    mod = _library_module(self)
    z = pair.to(torch.float16)
    if torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters())):
        delta = _TrainableTriMul.apply(mod, z, *_params(self))  # kernel forward, the library's torch-recompute backward
    else:
        delta = mod._forward_kernels(z)
    delta = delta.to(pair.dtype)
    if scale is not None:
        delta = delta * scale
    return delta + pair
