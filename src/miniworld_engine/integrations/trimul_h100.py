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

#: Largest batch the D64 path takes natively (tokens b-major, one launch per stage; MiniWorld's template embedder runs its templates
#: as the B samples of one pair stack). Same meaning as ``trimul_b200.MAX_BATCH``; see ``_batch_ok`` for why this bound.
MAX_BATCH = 48

#: Training lengths: multiples of TRAINING_GRANULE (the B1 / B2 / B7 tiles of 64 tokens must not straddle a pair row and the
#: B200-style output tiles 128 tokens) from 128 up to MAX_TRAINING_LENGTH, B200's bound (80 x 128). The kernels index planes
#: through TMA coordinates or 64-bit offsets; their only index bound is the int32 token index (``_index_ok``). Memory binds
#: first on an 80 GB card: the training step needs ~4.3 kB per pair at D128 bidirectional (L2560: 28 GB, so up to ~L4300) and
#: ~2.1 kB at D64 (L2560: 13.5 GB per sample).
TRAINING_GRANULE = 128
MAX_TRAINING_LENGTH = 10240
#: Inference takes any L >= MIN_INFERENCE_LENGTH: one sample pads to 16 inside K1 / K3 (planes [c, Np, Np], zero pad rows),
#: B samples (whose K1 / K3 stack the samples without a pad) are padded to a multiple of 16 on the host when L is not one.
MIN_INFERENCE_LENGTH = 16


def _index_ok(pair) -> bool:
    """The int32 token index of K1 / K3 / B1 / B2 (B L^2 < 2^31, with L rounded up to the 16-token pad)."""
    n = -(-pair.shape[1] // 16) * 16
    return pair.shape[0] * n * n < 2 ** 31


def _training_length(n) -> bool:
    return n % TRAINING_GRANULE == 0 and TRAINING_GRANULE <= n <= MAX_TRAINING_LENGTH


def _batch_ok(pair, inference=False) -> bool:
    """One sample at every width; B samples (B <= MAX_BATCH) at D64, whose K1 / K3 / B1 / B2 take b-major tokens (rows = B L of the
    slow token axis, a per-sample token mask and row-dropout scale) and whose contraction batches over channels x B. Training also
    needs the native D64 preparation (the per-sample pair mask is built there). Inference at L % 16 != 0 pads on the host
    (``update_inference``)."""
    if pair.ndim != 4 or not _index_ok(pair):
        return False
    bsz = pair.shape[0]
    if bsz == 1:
        return True
    return (pair.shape[-1] == 64 and 1 <= bsz <= MAX_BATCH and (inference or pair.shape[1] % 16 == 0)
            and (inference or _native_training_enabled()))


#: Cells where the preserved native path (the one served before the port) measured faster than the B200-style port on H100
#: (2026-10-13, bf16, FP32 masters, B1): training L384 / L768 at D256 / D384 / D512, and bidirectional inference at
#: (D384, L768) and (D512, L384). Every other length keeps the B200-style port, which is 1.2-2.1x faster than the
#: general path wherever the preserved path does not exist.
_LEGACY_TRAINING_WIDTHS = (256, 384, 512)
_LEGACY_BIDIR_INFERENCE_CELLS = ((384, 768), (512, 384))


def _legacy_native_faster(module, pair, inference):
    n, d = pair.shape[1], pair.shape[-1]
    if inference:
        if hasattr(module, "outgoing") and n > 1024:
            # single direction D256 / D384 beyond L1024: the packaged tmn K1 / K3 (any L) served these before the port did
            return d in (256, 384)
        if hasattr(module, "outgoing"):
            # single direction D256: the tmn K1 / K3 tiles beat the B200-style port by 0.7-2.5 % at every length but 384
            return d == 256 and n != 384
        return (d, n) in _LEGACY_BIDIR_INFERENCE_CELLS
    return n in (384, 768) and d in _LEGACY_TRAINING_WIDTHS


def _b200style_serves(module, pair, inference=False):
    """The Hopper port of the B200 algorithm, except the cells where the preserved native path measured faster."""
    if pair.ndim == 4 and _legacy_native_faster(module, pair, inference):
        return False
    if (not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4
            or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]
            or pair.shape[-1] not in (256, 384, 512)
            or not _training_length(pair.shape[1])
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
        or not _batch_ok(pair)
        or pair.shape[1] != pair.shape[2]
        or not (_training_length(pair.shape[1])
                if _native_buckets_enabled(pair) or (pair.shape[-1] == 64 and _native_training_enabled())
                else pair.shape[1] in (384, 768))
        or (pair.shape[-1] > 128 and pair.shape[1] > 1024)   # D256+ beyond L1024: the B200-style port only (above)
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
        scale = (pair.new_ones((pair.shape[0] * n, d)) if dropscale is None
                 else dropscale.reshape(pair.shape[0] * n, d).contiguous())
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    if bidirectional and d == 64:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_d64_b200gate import (
            bidirectional_trimul as apply,
        )
    elif bidirectional:
        apply = bidirectional_trimul
    elif d in (64, 256, 384, 512):
        if d == 64:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_d64_b200gate import (
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
    if _b200style_serves(module, pair, inference=True):
        return True
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or not _batch_ok(pair, inference=True)
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] <= 0
    ):
        return False
    if module.ln_pair.eps != module.ln_out.eps:
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    if ((pair.shape[-1] in (64, 128) or (not bidirectional and _native_residual_bucket(module, pair)))
            and (pair.shape[1] < MIN_INFERENCE_LENGTH
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


def update_inference(module, pair, mask, *, bidirectional):
    if _b200style_serves(module, pair, inference=True):
        return _b200style_update(module, pair, mask, None, bidirectional)
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import inference

    bsz, n = pair.shape[0], pair.shape[1]
    if bsz > 1 and n % 16:
        return _padded_inference(module, pair, mask, bidirectional)
    native_residual = not bidirectional and _native_residual_bucket(module, pair)
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as runtime
    native_aux = runtime.cuda_aux_enabled()
    if (native_residual or native_aux) and mask is not None and (
        mask.device != pair.device or mask.dtype != torch.bool
        or tuple(mask.shape) != (bsz, n) or not mask.is_contiguous()
    ):
        raise ValueError("H100 native TriMul inference requires a contiguous boolean [B,L] mask")
    # K1 forms the pair mask m[i] & m[j] from the token mask itself: no [n, n] tensor per call (B samples: [B n], b-major).
    token = None if mask is None else mask.reshape(bsz * n)
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


def _padded_inference(module, pair, mask, bidirectional):
    """B samples at L % 16 != 0. K1 / K3 stack the B samples along the slow token axis without a pad (one sample pads inside
    them), so each sample is padded to the next multiple of 16 here: zero pair entries on masked pad tokens. A pad token's
    left / right rows and columns are zero (K1's pair mask m[i] & m[j]), so every real pair (i, j) contracts the same terms as
    without the pad; the pad rows of the output are dropped."""
    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    npad = -(-n // 16) * 16
    x = pair.new_zeros((bsz, npad, npad, d))
    x[:, :n, :n] = pair
    token = torch.zeros((bsz, npad), device=pair.device, dtype=torch.bool)
    if mask is None:
        token[:, :n] = True
    else:
        token[:, :n] = mask.reshape(bsz, n) != 0
    return update_inference(module, x, token, bidirectional=bidirectional)[:, :n, :n].contiguous()


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
