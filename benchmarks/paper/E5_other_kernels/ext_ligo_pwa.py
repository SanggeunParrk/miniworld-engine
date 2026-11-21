"""E5 external backend ``ligo_pwa``: the Triton MSA pair-weighted-averaging kernel of the Ligo Biosciences AlphaFold3 implementation (Apache-2.0; credited to Alex Zhang).

Upstream: github.com/Ligo-Biosciences/AlphaFold3, ``src/models/components/msa_kernel.py`` (``MSAWeightedAveragingFused``: ``sigmoid(g) * softmax(b) @ v`` with v / g
``[B, S, R, H, C]``, b the pair-bias logits ``[B, R, R, H]``; forward kernel + a torch-einsum backward). It has no key mask: the adapter masks the bias logits
(``-1e9``, the kernel's own INF) like the reference. The module keeps its LayerNorms, v / gate / bias projections, out-projection, row dropout and residual.
Serves ``MSAPairWeightedAveraging`` (forward + backward). The kernel allocates its output with ``torch.empty`` and accumulates into it (reads before writing):
``third_party/ligo-af3/patches/ligo-msa-kernel-fixes.patch`` (applied by fetch.sh) makes those two buffers ``torch.zeros`` (and spells the INF constant the way Triton 3.7 accepts). The kernel takes fp32 (its ``tl.dot`` mixes the fp32 softmax weights with v: bf16 v fails), so v / b / g are cast to fp32 (TF32 dots) and the result back.
"""

from __future__ import annotations

import os

import ext_attn_common as common
import torch

NAME = "ligo_pwa"


def _mod():
    return common.load_file("ligo_msa_kernel", os.path.join(common.upstream("ligo-af3"), "src", "models", "components", "msa_kernel.py"))


def _supports(self, msa, pair, mask=None):
    try:
        _mod()
    except Exception as exc:
        return f"ligo_pwa: {type(exc).__name__}: {str(exc)[:200]} (see third_party/ligo-af3)"
    if not msa.is_cuda:
        return "CUDA tensors only"
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else msa.dtype
    if dt not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {dt}"
    if msa.dim() != 4 or pair.dim() != 4:
        return "msa must be [B, M, L, d_msa] and pair [B, L, L, d_pair]"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (msa.shape[0], msa.shape[2])):
        return "mask must be a bool [B, L] key mask"
    return None


def _forward(self, msa, pair, mask=None):
    from einops import rearrange
    from miniworld_engine.modules.primitives import master_autocast

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return msa
    with master_autocast(msa, self):
        b, s, n, _ = msa.shape
        h = self.n_head
        m = self.ln_msa(msa)
        value = self.to_value(m)
        gate = self.to_gate(m)
        bias = self.to_bias(self.ln_pair(pair))  # [B, L, L, H]
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, :, None], -1e9)
        d = value.shape[-1] // h
        out = _mod().MSAWeightedAveragingFused(
            value.reshape(b, s, n, h, d).contiguous().float(), bias.contiguous().float(), gate.reshape(b, s, n, h, d).contiguous().float()
        )
        out = rearrange(out, "B M L (H D) -> B M L (H D)", H=h)
        return msa + self.drop_msa(self.to_out(out.to(value.dtype)))


def install() -> None:
    from miniworld_engine.modules import msa

    cls = msa.MSAPairWeightedAveraging
    if NAME not in cls.backends:
        cls.backends = (*cls.backends, NAME)
    setattr(cls, f"_supports_{NAME}", _supports)
    setattr(cls, f"_forward_{NAME}", _forward)
