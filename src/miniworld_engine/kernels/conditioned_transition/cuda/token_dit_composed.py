"""The fused token DiT runner with the hand-CUDA stages plugged in: CUDA row passes, the SwiGLU GEMM, the attention cores.

``triton/token_dit_runner.FusedTokenDiT`` is the pure-Triton runner (Triton row kernels, cuBLAS GEMMs, the Triton gated core). This subclass
only OVERRIDES its hooks with the hand-CUDA choices that used to be made inside that file by compute capability:

* the row passes: CUDA on sm_90 and sm_100 (``settings.token_dit_rows_cuda``), Triton elsewhere or when the extension does not build;
* the expand GEMM with the SwiGLU epilogue (``gemm_swiglu``): sm_100a bf16, sm_90a bf16 / fp32;
* the attention core: the upstream B200 bf16 / TF32 cores or the H100 bf16 16 x 48 core, and the H100 fp32 TF32 core.

Whoever builds a runner picks the class (``integrations/token_dit.py``): this one by default, the Triton runner under
``settings.force_portable_kernels``. On a card with no CUDA choice (every hook returns Triton / None) the two behave the same.
"""

import functools

import torch

from miniworld_engine import settings
from miniworld_engine.kernels.conditioned_transition.triton import token_dit_kernels as K
from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import FusedTokenDiT


def _row_kernels(device):
    """The row passes between the GEMMs: CUDA on H100 and B200 (``kernels/conditioned_transition/cuda``), Triton elsewhere
    or when the extension does not build. settings.token_dit_rows_cuda=0 keeps Triton there too."""
    if (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) in ((9, 0), (10, 0))
        and settings.current().token_dit_rows_cuda
    ):
        try:
            from miniworld_engine.kernels.conditioned_transition import (
                cuda as cuda_rows,
            )

            cuda_rows.available()
            return cuda_rows
        except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton rows
            import warnings

            warnings.warn(
                f"token DiT CUDA row kernels unavailable, keeping Triton: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
    return K


class FusedTokenDiTComposed(FusedTokenDiT):
    """:class:`FusedTokenDiT` with the CUDA rows, SwiGLU GEMM and attention cores of the card where they apply."""

    @staticmethod
    def _rows(device):
        return _row_kernels(device)

    @staticmethod
    def _blackwell_bf16(device, dtype):
        return (
            dtype is torch.bfloat16
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == (10, 0)
        )

    def _gemm_swiglu(self, device, xa):
        """The expand GEMM with the SwiGLU epilogue (``kernels/conditioned_transition/cuda/gemm_swiglu.py``: sm_100a, or
        sm_90a) where it fits and builds; cuBLAS + the SwiGLU row pass otherwise."""
        cap = (
            torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        )
        if (
            torch.compiler.is_compiling()
            or cap not in ((9, 0), (10, 0))
            or not (
                self.dtype is torch.bfloat16
                or (self.dtype is torch.float32 and cap == (9, 0))
            )
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
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps cuBLAS + the row pass
                    import warnings

                    warnings.warn(
                        f"SwiGLU GEMM unavailable, keeping cuBLAS: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
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
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton core
                    import warnings

                    warnings.warn(
                        f"CUDA token DiT core unavailable, keeping the Triton core: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
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
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton core
                    import warnings

                    warnings.warn(
                        f"TF32 token DiT core unavailable, keeping the Triton core: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            cores[idx] = core
        return cores[idx]
