"""Automatic dispatch to the packaged, selected bidirectional H100 training path."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType

#: Widths with a packaged bidirectional CUDA training path (D64, D128 and the wide port).
TRAINING_WIDTHS = (64, 128, 256, 384, 512)
#: Widths with a single-direction CUDA training path (D64: port of the bidirectional D64 kernels; D128: B1/B7;
#: D256/384: the wide sources at hidden D).$
SINGLE_TRAINING_WIDTHS = (64, 128, 256, 384, 512)


#: Cells where the preserved native path (served with the switch off) measured faster than the B200-style port on H100
#: (2026-10-13, bf16, FP32 masters, B1): training L384 / L768 at D256 / D384 / D512, and bidirectional inference at
#: (D384, L768) and (D512, L384). Every other length keeps the B200-style port, which is 1.2-2.1x faster than the
#: general path wherever the preserved path does not exist.
_LEGACY_TRAINING_WIDTHS = (256, 384, 512)
_LEGACY_BIDIR_INFERENCE_CELLS = ((384, 768), (512, 384))


def _legacy_native_faster(module, pair, inference):
    n, d = pair.shape[1], pair.shape[-1]
    if inference:
        if hasattr(module, "outgoing"):
            # single direction D256: the tmn K1 / K3 tiles beat the B200-style port by 0.7-2.5 % at every length but 384
            return d == 256 and n != 384
        return (d, n) in _LEGACY_BIDIR_INFERENCE_CELLS
    return n in (384, 768) and d in _LEGACY_TRAINING_WIDTHS


def _b200style_serves(module, pair, inference=False):
    """Explicit Hopper port; keep measured default selections until qualified."""
    if not settings.current().trimul_h100_b200style:
        return False
    if pair.ndim == 4 and _legacy_native_faster(module, pair, inference):
        return False
    if (not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4
            or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]
            or pair.shape[-1] not in (256, 384, 512)
            or pair.shape[1] not in (128, 256, 384, 512, 640, 768, 896, 1024)
            or not pair.is_contiguous() or module.d_hidden != pair.shape[-1]
            or module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5
            or torch.cuda.get_device_capability(pair.device) != (9, 0)):
        return False
    return all(p.dtype == torch.float32 and p.is_contiguous() and p.device == pair.device
               for p in (module.ln_pair.weight, module.ln_pair.bias,
                         module.ln_out.weight, module.ln_out.bias))


def _b200style_update(module, pair, mask, dropscale, bidirectional):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
        training_unit_scale,
    )
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_b200style import trimul
    n, d = pair.shape[1], pair.shape[-1]
    scale = ((training_unit_scale(pair) if torch.is_grad_enabled() else None)
             if dropscale is None else dropscale.reshape(n, d).contiguous())
    token = torch.empty(0, device=pair.device, dtype=torch.bool) if mask is None else mask.contiguous()
    leaves = [pair, module.to_left.weight, module.to_left_gate.weight,
              module.to_right.weight, module.to_right_gate.weight,
              module.to_gate.weight, module.to_out.weight,
              module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias]
    direction = 0 if bidirectional else 1 if module.outgoing else 2
    return trimul(leaves, token, scale, direction)


def _native_buckets_enabled(pair=None):
    """True = explicit native buckets. Unset (None): D64 / D128 take them at every length 128..1024 (2026-10-13: 1.2-2.4x
    over the Triton fallback they otherwise get outside L384 / L768); D256+ keep the qualified cells only (the explicit
    buckets measured 2.6x slower there at L384 / L768)."""
    policy = settings.current().trimul_h100_native_buckets
    if policy is None and pair is not None:
        return pair.shape[-1] in (64, 128)
    return policy is True


def _selected_uni_d256(module, pair):
    """Qualified outgoing D256/D384 L384 and D512 L384,768 native defaults."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import uni_training
    return (hasattr(module, "outgoing") and uni_training(pair, bool(module.outgoing)))


def _selected_d64_training(module, pair):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import d64_training
    return d64_training(pair, getattr(module, "outgoing", False),
                        not hasattr(module, "outgoing"))


def _selected_wide_training(module, pair):
    """The qualified D256+ native cells (the D128 default stays on the framework RNG so the module compiles fullgraph)."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import bidi_training
    return _selected_uni_d256(module, pair) or (not hasattr(module, "outgoing") and bidi_training(pair))


def _selected_native_training(module, pair):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        bidi_training,
        d128_training,
    )
    return (_selected_uni_d256(module, pair)
            or (not hasattr(module, "outgoing") and bidi_training(pair))
            or d128_training(pair, getattr(module, "outgoing", False),
                             not hasattr(module, "outgoing")))

def serves(module, pair: torch.Tensor) -> bool:
    """Check the mathematical and resource contract before touching CUDA builders."""
    policy = settings.current()
    if (
        module.implementation != ImplementationType.MINIWORLD
        or policy.engine_backend == "triton"
    ):
        return False
    if _b200style_serves(module, pair):
        return pair.shape[-1] in policy.trimul_h100_training_widths
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
        or pair.shape[1] not in ((128, 256, 384, 512, 640, 768, 896, 1024)
                                if _native_buckets_enabled(pair) or (pair.shape[-1] == 64 and _native_training_enabled())
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
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        preserved_d64_context,
    )
    with preserved_d64_context(_selected_d64_training(module, pair)):
        out = _update(module, pair, mask, dropscale, bidirectional=bidirectional)
    if _b200style_serves(module, pair) or _native_buckets_enabled(pair) or _selected_native_training(module, pair) or _selected_d64_training(module, pair) or (pair.shape[-1] == 64 and _native_training_enabled()):
        from miniworld_engine.integrations.token_dit_train import _native_caller_output
        out = _native_caller_output(out, pair, *module.parameters(), force=True)
    return out


def _update(module, pair, mask, dropscale, *, bidirectional=True):
    """Use native FP32 master gradients where qualified; preserve cast-based fallbacks."""
    if _b200style_serves(module, pair):
        return _b200style_update(module, pair, mask, dropscale, bidirectional)
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_training import (
        bidirectional_trimul,
    )

    n, d = pair.shape[1], pair.shape[-1]
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        preserve_training,
    )
    preserve = preserve_training(d) or _selected_native_training(module, pair)
    if _native_buckets_enabled(pair) and d != 64 and not preserve:
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
            training_unit_scale,
        )
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_training import (
            native_buckets,
            single_trimul,
        )
        empty_mask = torch.empty(0,device=pair.device,dtype=torch.bool) if mask is None else mask
        scale = training_unit_scale(pair) if dropscale is None else dropscale.reshape(n,d)
        args = (pair, module.to_left.weight, module.to_left_gate.weight,
                module.to_right.weight, module.to_right_gate.weight,
                module.to_gate.weight, module.to_out.weight,
                module.ln_pair.weight, module.ln_pair.bias,
                module.ln_out.weight, module.ln_out.bias, empty_mask, scale)
        if not bidirectional:
            return single_trimul(module.outgoing, *args)
        return native_buckets(True, True, *args)
    # Bidirectional D128 takes the token mask [n]: its front K1 and B7 form m[i] & m[j] themselves.
    # The other paths read a bf16 pair mask in forward and backward, so it is built once here.
    token_path = bidirectional and d == 128 and (mask is None or mask.ndim == 2)
    if preserve and not token_path:
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
            owned_training_mask,
        )
        with torch.no_grad():
            pair_mask = owned_training_mask(pair,mask)
    elif d == 64 and _native_training_enabled():
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
    if dropscale is None and (preserve or (d == 64 and _native_training_enabled())):
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
            training_unit_scale,
        )
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
        if settings.current().trimul_h100_b200style:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_d64_b200gate import (
                bidirectional_trimul as apply,
            )
        else:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_d64_training import (
                bidirectional_trimul as apply,
            )
    elif bidirectional:
        apply = bidirectional_trimul
    elif d in (64, 256, 384, 512):
        if d == 64:
            if settings.current().trimul_h100_b200style:
                from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_d64_b200gate import (
                    single_trimul,
                )
            else:
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
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        cuda_training_aux_enabled,
    )
    return cuda_training_aux_enabled()


def native_training_dropout(module, pair, probability):
    """Share the hand-CUDA Philox primitive; preserve PyTorch RNG progression."""
    if not (_b200style_serves(module, pair) or settings.current().trimul_h100_native_buckets is True or _selected_wide_training(module, pair) or _selected_d64_training(module, pair) or (pair.shape[-1] == 64 and _native_training_enabled())) or not serves(module, pair):
        return None
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import extension
    return extension("pack").drop_scale(pair, probability).reshape(
        pair.shape[0], 1, pair.shape[1], pair.shape[-1])


def requires_cuda_training(module, pair):
    """Selected native training never silently substitutes a Triton shape. Training only: an inference call (no grad) and an
    explicit `engine_backend="triton"` comparison are not refused here (the inference contract is `requires_cuda_inference`)."""
    if not torch.is_grad_enabled() or settings.current().engine_backend == "triton":
        return False
    # The default D64 / D128 length buckets fall back to the general path where they do not apply; only an explicit
    # `trimul_h100_native_buckets=True` or a qualified cell turns an unsupported call into an error.
    return ((settings.current().trimul_h100_native_buckets is True or _selected_wide_training(module, pair) or _selected_d64_training(module, pair) or (_native_training_enabled() and pair.shape[-1] == 64))
            and module.implementation == ImplementationType.MINIWORLD
            and pair.is_cuda
            and torch.cuda.get_device_capability(pair.device) == (9, 0))


def _native_residual_bucket(module, pair):
    """Measured outgoing D384/L512 inference bucket (contract checked by serves)."""
    return (pair.ndim == 4 and pair.shape[1] == 512 and pair.shape[-1] == 384
            and module.d_hidden == 384 and getattr(module, "outgoing", False))



def _native_wide_inference_bucket(module, pair):
    """One BIDI bucket; engine_backend=triton remains an explicit comparison."""
    if (settings.current().engine_backend == "triton" or pair.ndim != 4
            or module.d_hidden != pair.shape[-1]
            or module.to_left.weight.shape[0] != 2 * pair.shape[-1]):
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_wide_inference import (
        native_inference_bucket,
    )
    return native_inference_bucket(pair.shape[-1], pair.shape[1])


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
    if _b200style_serves(module, pair, inference=True):
        return True
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
    if ((pair.shape[-1] in (64, 128) or (not bidirectional and _native_residual_bucket(module, pair))
            or _native_wide_inference_bucket(module, pair))
            and (pair.shape[1] not in ((128, 256, 384, 512, 640, 768, 896, 1024)
                 if settings.current().trimul_h100_b200style
                 else (128, 256, 384, 512, 640, 768))
                 or module.d_hidden != pair.shape[-1] or module.ln_pair.eps != 1e-5
                 or any(p.dtype != torch.float32 for p in (
                     module.ln_pair.weight, module.ln_pair.bias,
                     module.ln_out.weight, module.ln_out.bias)))):
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
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        cuda_aux_enabled,
    )
    native = cuda_aux_enabled() or _native_wide_inference_bucket(module, pair)
    # A non-default LayerNorm eps is outside every promoted bucket: that call uses the general path, it is not refused.
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    return (module.implementation == ImplementationType.MINIWORLD
            and (native or settings.current().engine_backend != "triton" or _native_residual_bucket(module, pair))
            and pair.is_cuda and (native or pair.shape[-1] in (64, 128) or _native_residual_bucket(module, pair))
            and torch.cuda.get_device_capability(pair.device) == (9, 0)
            and not torch.is_grad_enabled())


def update_inference(module, pair, mask, *, bidirectional):
    if _b200style_serves(module, pair, inference=True):
        return _b200style_update(module, pair, mask, None, bidirectional)
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import inference

    n = pair.shape[1]
    native_residual = not bidirectional and _native_residual_bucket(module, pair)
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as runtime
    native_aux = runtime.cuda_aux_enabled() or _native_wide_inference_bucket(module, pair)
    if (native_residual or native_aux) and mask is not None and (
        mask.device != pair.device or mask.dtype != torch.bool
        or tuple(mask.shape) != (1, n) or not mask.is_contiguous()
    ):
        raise ValueError("H100 native TriMul inference requires a contiguous boolean [1,L] mask")
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
    if pair.shape[-1] in (64, 128) or native_residual or native_aux:
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import cast_weights
        matrices = cast_weights(weights)
    else:
        matrices = [w.to(pair.dtype) if i < 4 else w.to(pair.dtype).contiguous()
                    for i, w in enumerate(weights)]
    weights = [
        *matrices,
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

        if native_aux:
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
    return (pair.shape[-1] in SINGLE_TRAINING_WIDTHS
            and (pair.shape[-1] != 512 or _native_buckets_enabled(pair)
                 or _selected_native_training(module, pair) or _b200style_serves(module, pair)) and serves(module, pair))


def _uni_wide_inference_ok(module, pair: torch.Tensor) -> bool:
    """Single-direction D512 at L384/768: LN + K1 + contraction + gate GEMM + folded K3."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_inference import (
        serves,
    )

    return serves(module, pair)
