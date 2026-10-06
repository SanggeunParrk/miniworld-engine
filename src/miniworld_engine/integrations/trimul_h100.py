"""Automatic dispatch to the packaged, selected bidirectional H100 training path."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType

#: Widths with a packaged bidirectional CUDA training path (D64, D128 and the wide port).
TRAINING_WIDTHS = (64, 128, 256, 384, 512)
#: Widths with a single-direction CUDA training path (D64: port of the bidirectional D64 kernels; D128: B1/B7;
#: D256/384: the wide sources at hidden D).$
SINGLE_TRAINING_WIDTHS = (64, 128, 256, 384)


def serves(module, pair: torch.Tensor) -> bool:
    """Check the mathematical and resource contract before touching CUDA builders."""
    policy = settings.current()
    if (
        module.implementation != ImplementationType.MINIWORLD
        or policy.engine_backend == "triton"
    ):
        return False
    if (
        pair.shape[-1] not in policy.trimul_h100_training_widths
        or pair.shape[-1] not in TRAINING_WIDTHS
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or pair.shape[0] != 1
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] not in ((128, 256, 384, 512, 640, 768)
                                if pair.shape[-1] == 64 and _native_training_enabled()
                                else (384, 768))
    ):
        return False
    if (
        module.d_hidden != pair.shape[-1]
        or module.ln_pair.eps != 1e-5
        or module.ln_out.eps != 1e-5
    ):
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    # Cooperative B1/B7 grids were qualified on full 132-SM H100s, not MIG slices.
    if (
        pair.shape[-1] == 128
        and torch.cuda.get_device_properties(pair.device).multi_processor_count != 132
    ):
        return False
    return all(
        p.dtype == torch.float32
        for p in (
            module.ln_pair.weight,
            module.ln_pair.bias,
            module.ln_out.weight,
            module.ln_out.bias,
        )
    )


def update(
    module,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    dropscale: torch.Tensor | None,
    *, bidirectional: bool = True,
) -> torch.Tensor:
    """Use native FP32 master gradients where qualified; preserve cast-based fallbacks."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_training import (
        bidirectional_trimul,
    )

    n, d = pair.shape[1], pair.shape[-1]
    # Bidirectional D128 takes the token mask [n]: its front K1 and B7 form m[i] & m[j] themselves.
    # The other paths read a bf16 pair mask in forward and backward, so it is built once here.
    token_path = bidirectional and d == 128 and (mask is None or mask.ndim == 2)
    if d == 64 and _native_training_enabled():
        # Keep token masks compact until the saved forward boundary constructs
        # its owned FP32 pair mask. No intermediate BF16 square allocation.
        pair_mask = (torch.empty(0, device=pair.device, dtype=torch.bool)
                     if mask is None else mask)
    elif mask is None:
        pair_mask = (torch.ones(n, device=pair.device, dtype=torch.bool) if token_path
                     else torch.ones((n, n), device=pair.device, dtype=torch.bfloat16))
    elif token_path:
        pair_mask = mask.reshape(n)
        if pair_mask.dtype != torch.bool:
            pair_mask = pair_mask != 0
    else:
        pair_mask = (
            (mask.unsqueeze(-1) & mask.unsqueeze(-2)) if mask.ndim == 2 else mask
        )
        pair_mask = pair_mask.to(torch.bfloat16).contiguous()
    if dropscale is None and d == 64 and _native_training_enabled():
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import training_unit_scale
        scale = training_unit_scale(pair)
    else:
        scale = (pair.new_ones((n, d)) if dropscale is None
                 else dropscale.reshape(n, d).contiguous())
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    if bidirectional and d == 64:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_d64_training import (
            bidirectional_trimul as apply,
        )
    elif bidirectional:
        apply = bidirectional_trimul
    elif d in (64, 256, 384):
        if d == 64:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_d64_training import (
                single_trimul,
            )
        else:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_training import (
                single_trimul,
            )
        def apply(*args):
            return single_trimul(module.outgoing, *args)
    else:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_single import (
            single_trimul,
        )
        def apply(*args):
            return single_trimul(module.outgoing, *args)
    return apply(
        pair,
        *(w if w.dtype in (torch.bfloat16, torch.float32) else w.to(pair.dtype).contiguous()
          for w in weights),
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
        pair_mask,
        scale,
    )


def _native_training_enabled():
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import cuda_training_aux_enabled
    return cuda_training_aux_enabled()


def native_training_dropout(module, pair, probability):
    """Share the hand-CUDA Philox primitive; preserve PyTorch RNG progression."""
    if pair.shape[-1] != 64 or not _native_training_enabled() or not serves(module, pair):
        return None
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import extension
    return extension("pack").drop_scale(pair, probability).reshape(
        pair.shape[0], 1, pair.shape[1], pair.shape[-1])


def requires_cuda_training(module, pair):
    """The explicit D64 candidate never silently substitutes a Triton shape."""
    return (_native_training_enabled()
            and module.implementation == ImplementationType.MINIWORLD
            and settings.current().engine_backend != "triton"
            and pair.is_cuda and pair.shape[-1] == 64
            and torch.cuda.get_device_capability(pair.device) == (9, 0))


def _native_residual_bucket(module, pair):
    """Measured outgoing D384/L512 inference bucket (contract checked by serves)."""
    return (pair.ndim == 4 and pair.shape[1] == 512 and pair.shape[-1] == 384
            and module.d_hidden == 384 and getattr(module, "outgoing", False))


def serves_inference(
    module, pair: torch.Tensor, *, bidirectional: bool, dropscale=None
) -> bool:
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().engine_backend == "triton"
    ):
        return False
    if torch.is_grad_enabled() or dropscale is not None:
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or pair.shape[0] != 1
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] <= 0
    ):
        return False
    if module.ln_pair.eps != module.ln_out.eps:
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    if pair.shape[-1] in (64, 128) or (not bidirectional and _native_residual_bucket(module, pair)):
        if (pair.shape[1] not in (128, 256, 384, 512, 640, 768)
                or module.d_hidden != pair.shape[-1] or module.ln_pair.eps != 1e-5
                or any(p.dtype != torch.float32 for p in (
                    module.ln_pair.weight, module.ln_pair.bias,
                    module.ln_out.weight, module.ln_out.bias))):
            return False
    if bidirectional and _wide_inference_ok(module, pair):
        return True
    if not bidirectional and _uni_wide_inference_ok(module, pair):
        return True
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as table

    hidden = module.d_hidden * (2 if bidirectional else 1)
    entry = table.TILE_TABLE.get(("sm_90a", pair.shape[-1], hidden, "b"))
    return bool(entry and entry.get("k1") and entry.get("k3"))


def requires_cuda_inference(module, pair):
    """Promoted inference buckets refuse instead of silently using Triton."""
    return (module.implementation == ImplementationType.MINIWORLD
            and (settings.current().engine_backend != "triton" or _native_residual_bucket(module, pair))
            and pair.is_cuda and (pair.shape[-1] in (64, 128) or _native_residual_bucket(module, pair))
            and torch.cuda.get_device_capability(pair.device) == (9, 0)
            and not torch.is_grad_enabled())


def update_inference(module, pair, mask, *, bidirectional):
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import inference

    n = pair.shape[1]
    native_residual = not bidirectional and _native_residual_bucket(module, pair)
    if native_residual and mask is not None and (
        mask.device != pair.device or mask.dtype != torch.bool
        or tuple(mask.shape) != (1, n) or not mask.is_contiguous()
    ):
        raise ValueError("H100 D384/L512 TriMul requires a contiguous boolean [1,L] mask")
    # K1 forms the pair mask m[i] & m[j] from the token mask itself: no [n, n] tensor per call.
    token = None if mask is None else mask.reshape(n)
    if token is not None and token.dtype != torch.bool:
        token = token != 0
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    # K1 packs the first four matrices with live stride-aware CUDA operations.
    # Preserve native column-major parameters instead of copying them first.
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as runtime
    if pair.shape[-1] in (64, 128) or native_residual or runtime.cuda_aux_enabled():
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import cast_weights
        matrices = cast_weights(weights)
    else:
        matrices = [w.to(pair.dtype) if i < 4 else w.to(pair.dtype).contiguous()
                    for i, w in enumerate(weights)]
    weights = matrices + [
        module.ln_pair.weight.float().contiguous(),
        module.ln_pair.bias.float().contiguous(),
        module.ln_out.weight.float().contiguous(),
        module.ln_out.bias.float().contiguous(),
    ]
    if bidirectional and _wide_inference_ok(module, pair):
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_wide_inference import (
            wide_inference,
        )
        return wide_inference(pair, weights, token)
    if not bidirectional and _uni_wide_inference_ok(module, pair):
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_inference import (
            uni_wide_inference,
        )

        if runtime.cuda_aux_enabled():
            from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import pair_mask
            pm = pair_mask(token, n, pair.device)
        else:
            pm = (pair.new_ones((n, n)) if mask is None
                  else (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float().contiguous())
        return uni_wide_inference(pair, weights, pm, module.outgoing)
    direction = 0 if bidirectional else 1 if module.outgoing else 2
    return inference(pair, weights, token, direction, module.ln_pair.eps)


def _wide_inference_ok(module, pair: torch.Tensor) -> bool:
    """Bidirectional D256/384/512 at L384/768: fused wide K1/K3 (hidden = D per direction)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_wide_inference import supports

    return (
        supports(pair.shape[-1], pair.shape[1])
        and module.d_hidden == pair.shape[-1]
        and module.ln_pair.eps == 1e-5
        and module.ln_out.eps == 1e-5
    )


def serves_single(module, pair: torch.Tensor) -> bool:
    """Qualified single-direction native training contract; inference is separate."""
    return pair.shape[-1] in SINGLE_TRAINING_WIDTHS and serves(module, pair)


def _uni_wide_inference_ok(module, pair: torch.Tensor) -> bool:
    """Single-direction D512 at L384/768: LN + K1 + contraction + gate GEMM + folded K3."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_inference import (
        serves,
    )

    return serves(module, pair)
