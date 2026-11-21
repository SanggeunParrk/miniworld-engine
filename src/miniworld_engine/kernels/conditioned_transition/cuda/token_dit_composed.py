"""The fused token DiT runner with the hand-CUDA stages plugged in: CUDA row passes, the SwiGLU GEMM, the attention cores.

``triton/token_dit_runner.FusedTokenDiT`` is the pure-Triton runner (Triton row kernels, cuBLAS GEMMs, the Triton gated core). This subclass
only OVERRIDES its hooks with the hand-CUDA choices that used to be made inside that file by compute capability:

* the row passes: CUDA on sm_90 and sm_100, Triton elsewhere or when the extension does not build;
* the expand GEMM with the SwiGLU epilogue (``gemm_swiglu``): sm_100a bf16, sm_90a bf16 / fp32;
* the attention core: the upstream B200 bf16 / TF32 cores or the H100 bf16 16 x 48 core, and the H100 fp32 TF32 core.

Whoever builds a runner picks the class (``kernels/integrations/token_dit.py``): this one off H100 (the switch that picked the
Triton runner, ``force_portable_kernels``, is gone). On a card with no CUDA choice (every hook returns Triton / None) the two behave the same.
"""

import functools

import torch

from miniworld_engine.kernels._nvcc import build_failed
from miniworld_engine.kernels.conditioned_transition.triton import token_dit_kernels as K
from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import FusedTokenDiT


def _row_kernels(device):
    """The row passes between the GEMMs: CUDA on H100 and B200 (``kernels/conditioned_transition/cuda``), Triton elsewhere
    or when the extension does not build."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device) in ((9, 0), (10, 0)):
        try:
            from miniworld_engine.kernels.conditioned_transition import (
                cuda as cuda_rows,
            )

            cuda_rows.available()
            return cuda_rows
        except Exception as exc:  # a build / load failure is an error, never a fallback
            build_failed("token DiT CUDA row", exc)
    return K


class FusedTokenDiTComposed(FusedTokenDiT):
    """:class:`FusedTokenDiT` with the CUDA rows, SwiGLU GEMM and attention cores of the card where they apply."""

    @staticmethod
    def _rows(device):
        return _row_kernels(device)

    @staticmethod
    def _blackwell_bf16(device, dtype):
        return dtype is torch.bfloat16 and device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0)

    def _gemm_swiglu(self, device, xa):
        """The expand GEMM with the SwiGLU epilogue (``kernels/conditioned_transition/cuda/gemm_swiglu.py``: sm_100a, or
        sm_90a) where it fits and builds; cuBLAS + the SwiGLU row pass otherwise."""
        cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        if (
            torch.compiler.is_compiling()
            or cap not in ((9, 0), (10, 0))
            or not (self.dtype is torch.bfloat16 or (self.dtype is torch.float32 and cap == (9, 0)))
        ):  # fp32: sm_90 (TF32)
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cache = self.__dict__.setdefault("_gemm_swiglu_ops", {})
        key = (idx, xa.shape[0])  # the choice depends on M (gemm_swiglu.MIN_ROWS)
        if key not in cache:
            from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

            op = None
            wab = self.per[0]["wab"]
            if gemm_swiglu.supported(xa, wab) or gemm_swiglu.supported_sm90(xa, wab):
                try:
                    op = (
                        gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2)
                        if torch.cuda.get_device_capability(device) == (10, 0)
                        else gemm_swiglu.GemmSwigluSm90(idx)
                    )
                except Exception as exc:  # a build / load failure is an error, never a fallback
                    build_failed("SwiGLU GEMM", exc)
            cache[key] = op
        return cache[key]

    def _cuda_core(self, device, L, D, H):
        """The upstream B200 bf16 / TF32 cores and layouts, or the H100 bf16 16 x 48 core where each is supported.
        Same contract: pre-scaled logits in, sigmoid(g) * o written over q. Unsupported calls keep Triton."""
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cores = self.__dict__.setdefault("_cuda_cores", {})
        if idx not in cores:
            from miniworld_engine.kernels.augmented_attention import cuda as sm90
            from miniworld_engine.kernels.augmented_attention.cuda import sm100

            core = None
            if not torch.compiler.is_compiling():
                try:
                    if sm100.inference_core_supported(self.dtype, L, D, H, idx):
                        core = sm100.GatedInferenceCore(idx, self.dtype, H, D // H)
                    elif sm90.inference_core_supported(self.dtype, L, D, H, idx):
                        sm90._ext("attn_fwd")
                        core = sm90.gated_inference
                except Exception as exc:  # a build / load failure is an error, never a fallback
                    build_failed("CUDA token DiT core", exc)
            cores[idx] = core
        core = cores[idx]
        multiple = 128 if torch.cuda.get_device_capability(device) == (9, 0) else 8
        return core if core is not None and L % multiple == 0 else None

    def _tf32_core(self, device, L):
        """The fp32 step's TF32 CUDA core (``augmented_attention/cuda``, attn_tf32.cu) where it fits and builds: fp32,
        pre-scaled logits, H100, L a multiple of 128, the CUDA rows (they lay out its key-permuted bias). None otherwise
        (the Triton gated2 core, TF32 too)."""
        if (
            self.dtype is not torch.float32
            or self.core != "gated2"
            or not self.prescale
            or not hasattr(self.K, "key_perm")
            or L % 128
            or torch.compiler.is_compiling()
        ):
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cores = self.__dict__.setdefault("_tf32_cores", {})
        if idx not in cores:
            from miniworld_engine.kernels.augmented_attention import cuda as sm90

            core = None
            if sm90.tf32_inference_core_supported(self.dtype, L, self.d, self.h, idx):
                try:
                    nh, dh = self.h, self.d // self.h
                    sm90._ext("attn_tf32", sm90.tf32_fwd_defs(nh, dh))  # build now: a failure keeps Triton
                    core = functools.partial(sm90.tf32_gated_inference, nh=nh, dh=dh)
                except Exception as exc:  # a build / load failure is an error, never a fallback
                    build_failed("TF32 token DiT core", exc)
            cores[idx] = core
        return cores[idx]

    def _hoist(self, pair, mask):
        """The pair bias of every block. A pair wider than 128 (256: Protenix-v2 / ESMFold2, 512: IntelliFold2; B200 -- H100 takes the native
        block) is not an input of ``pair_bias_all``'s 128-wide LayerNorm rows: the CUDA LayerNorm rows over its width (``layernorm_rows``,
        128-512 columns), one cuBLAS GEMM into the head-major [nb H, L, L] planes, the masked keys -inf (as ``pair_bias_all``). Made once per
        pair (the runner's bias cache), so the passes cost nothing a solver step."""
        if self.dp == 128 or not hasattr(self.K, "layernorm_rows") or self._tf32_core(pair.device, pair.shape[1]) is not None:
            return super()._hoist(pair, mask)
        L = pair.shape[1]
        self._bias_perm = False
        z = pair.reshape(L * L, self.dp)
        zh = torch.empty(z.shape, device=z.device, dtype=self.dtype)
        self.K.layernorm_rows(z, zh, self.eps)
        out = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        torch.mm(self.pw_t.t(), zh.t(), out=out.view(self.nb * self.h, L * L))
        if mask is not None:
            out[:, :, ~mask.reshape(L)] = float("-inf")
        return out

    def _cond(self, cond, L, D):
        """The conditioning tables. A 768-wide conditioning (ESMFold2 / Boltz-2 widths) is not an input of ``layernorm_rows`` (128-512):
        the CUDA ``cond_rows`` (LN(c) and the cast, 384 or 768 columns) as at 384, then the same two GEMMs."""
        if self.dc != 768 or not hasattr(self.K, "cond_rows"):
            return super()._cond(cond, L, D)
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        c = cond[0, 0] if shared else cond.reshape(-1, self.dc)
        rows = c.shape[0]
        cn = torch.empty(rows, self.dc, device=c.device, dtype=self.dtype)
        cc = torch.empty_like(cn)
        self.K.cond_rows(c, cn, cc, self.eps)
        g1 = torch.addmm(self.b1, cn, self.w1.t()).view(rows, self.nb, 4, D)
        g2 = torch.addmm(self.b2, cc, self.w2.t()).view(rows, self.nb, 2, D)
        return g1, g2
