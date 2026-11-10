"""Fused token DiT inference (H100, B200) wired to DiTBlock's existing parameter contract.

On B200 (sm_100) the step runs the sm_100a gated attention core (``kernels/augmented_attention/cuda/sm100``: bf16, or
TF32 tensor cores for fp32) and the CUDA row kernels; the GEMMs are cuBLAS as on H100.

Inference only: weight packs and pair bias are cached by tensor pointer / version within the current capture scope.
A capture records their construction once, so each replay recomputes them from the current weights and pair.
The conditioning may be shared by the samples or per sample. QK-norm runs on H100 and B200 with the CUDA row kernels
(one in-place pass after the projection; the logit scale folds into the q norm's weight). Hopper BF16 and FP32 token inference
use the native CUDA block by default; unsupported engine calls raise. Per-shape performance qualification is recorded separately.

Precision: the runner's, i.e. single's dtype (bf16, or fp32 on TF32 tensor cores at every head layout of LAYOUTS on B200); the
step runs with autocast off.

Lengths: on H100 L must be a multiple of 128 (the sm_90a core's tiles). On B200 any L >= 8: the sm_100a core takes a
multiple of 8 (its TMA maps are 3-D per sample, so tile tails load as zeros and stores clip), and ``update`` pads the other
lengths to the next multiple of 8 -- single / cond with zero rows, the pair with zero rows and columns, the key mask with
False -- and returns the first L rows.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque

WEIGHTS = (
    "attention.to_query.weight",
    "attention.to_query.bias",
    "attention.to_key.weight",
    "attention.to_value.weight",
    "attention.to_gate.weight",
    "attention.to_out.weight",
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_scale.bias",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_scale.weight",
    "attention.to_scale.bias",
    "attention.ln_pair.weight",
    "attention.to_bias.weight",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.to_bias.weight",
    "transition.to_scale.weight",
    "transition.to_scale.bias",
    "transition.expand_a.weight",
    "transition.expand_b.weight",
    "transition.squeeze.weight",
)
#: (heads, d_single) layouts the fused step serves: 16 x 48 everywhere; 24 x 32, 12 x 64 and 16 x 64 (d 1024) with the CUDA rows, on
#: H100 and B200 (bf16, and fp32 through the TF32 core built per layout)
LAYOUTS = ((16, 768), (24, 768), (12, 768), (16, 1024))
#: pair widths of the Hopper step (bf16 and fp32): 128, 256 (Protenix-v2's diffusion pair) and 512 (IntelliFold2's); only the hoisted
#: bias reads the pair
PAIR_WIDTHS_H100 = (128, 256, 512)
#: conditioning widths of the Hopper step: 384, and 768 (ESMFold2's token DiT, conditioned at its own width)
COND_WIDTHS_H100 = (384, 768)
#: the QK-norm weights, appended to WEIGHTS when the block has use_qk_norm
QK_WEIGHTS = ("attention.norm_query.weight", "attention.norm_key.weight")
_TUNING: dict = {}
#: Packed runners keyed by the weights' (pointer, version): see ``_infer``.
_RUNNERS: dict = {}


# ------------------------------------------------------------------------------------------------ the models' options
# ONE token DiT algorithm for every model (TokenDiTProtenix / MiniWorld / Boltz2 / RoseTTAFold3 / Chai1): a model's differences are
# options of it, read off the block's own modules (``options``), and its parameters are mapped onto the canonical leaves WEIGHTS
# (``leaves``): a part the model does not have is a constant leaf that makes the canonical math the model's --
#   bias-free query (RoseTTAFold3)        to_query.bias = 0
#   identity AdaLN (Chai-1)               ln_cond.weight = 1 (the tables read the raw conditioning), to_scale.bias = 1 (the "+ 1")
#   no gating query (Chai-1)              to_gate.weight = 0 (the kernels then skip its columns: option ``gated``)
#   flat q / k LayerNorm (RoseTTAFold3)   QK_WEIGHTS = 1 (unused), its affine appended to the leaves (option ``kq_ln``)
#   up-gate transition (Boltz-2)          transition.up.weight appended to the leaves (option ``up``)
_CONST: dict = {}


def _const(value: float, shape, device) -> torch.Tensor:
    """A cached fp32 constant leaf (one tensor per value / shape / device: its pointer and version keep the weight packs' keys stable)."""
    key = (float(value), tuple(shape), str(device))
    t = _CONST.get(key)
    if t is None:
        t = _CONST[key] = torch.full(tuple(shape), float(value), dtype=torch.float32, device=device)
    return t


def _identity_adaln(ada) -> bool:
    return not hasattr(ada, "ln_cond")   # IdentityAdaptiveLayerNorm: no cond LayerNorm, no biases


def options(module) -> tuple[float, ...] | None:
    """The block's options in ``conditioned_transition.cuda.VARIANT_DEFAULT`` order (parallel, sigmoid AdaLN, AdaLN eps, gating query,
    up-gate transition, flat q / k LayerNorm, its eps, fp32 q / k rounding), or None when the block is outside the algorithm (mixed AdaLN forms / epsilons,
    q and k norms of different epsilons)."""
    a, t = module.attention, module.transition
    ident = _identity_adaln(a.ada_ln_in)
    if ident != _identity_adaln(t.ada_ln_in) or a.ada_ln_in.ln_in.eps != t.ada_ln_in.ln_in.eps:
        return None
    if not ident and (a.ada_ln_in.ln_cond.eps != 1e-5 or t.ada_ln_in.ln_cond.eps != 1e-5):
        return None
    kq_ln = hasattr(a, "norm_query") and not a.use_qk_norm   # FlatQKNormAugmentedAttention: a LayerNorm of the flat q / k
    if kq_ln and (a.norm_query.eps != a.norm_key.eps or a.norm_query.bias is None or a.norm_key.bias is None):
        return None
    parallel, gated, up = bool(getattr(module, "parallel", False)), hasattr(a, "to_gate"), hasattr(t, "up")
    # qk_rna (fp32 q / k rounded to nearest before the TF32 core) for the blocks with any of the options: the models whose fp32
    # accuracy depends on it (Chai-1's identity AdaLN scale is unbounded); the default block keeps its tuned path
    rna = parallel or ident or not gated or up or kq_ln
    return (float(parallel), float(not ident), float(a.ada_ln_in.ln_in.eps), float(gated), float(up), float(kq_ln),
            float(a.norm_query.eps) if kq_ln else 1e-5, float(rna))


def _leaf(module, name):
    """The parameter of WEIGHTS / QK_WEIGHTS ``name``, or the constant leaf that stands for a part the block does not have."""
    a = module.attention
    part, _, rest = name.partition(".")
    owner = getattr(module, part)
    d, dev = a.to_query.weight.shape[0], a.to_query.weight.device
    if rest == "to_query.bias" and a.to_query.bias is None:
        return _const(0.0, (d,), dev)
    if rest == "to_gate.weight" and not hasattr(a, "to_gate"):
        return _const(0.0, (d, d), dev)
    if rest.startswith("ada_ln_in.") and _identity_adaln(owner.ada_ln_in):
        if rest == "ada_ln_in.ln_cond.weight":
            return _const(1.0, (owner.ada_ln_in.to_scale.weight.shape[1],), dev)
        if rest == "ada_ln_in.to_scale.bias":
            return _const(1.0, (owner.ada_ln_in.to_scale.weight.shape[0],), dev)
    if rest in ("norm_query.weight", "norm_key.weight") and not a.use_qk_norm:
        return _const(1.0, (d // a.n_head,), dev)
    return module.get_parameter(name)


def leaves(module, opts) -> list[torch.Tensor]:
    """The block's leaves for ``conditioned_transition.cuda.token_inference``: WEIGHTS (+ QK_WEIGHTS for a q / k norm), then the up-gate
    weight, then the flat q / k LayerNorm affine."""
    a = module.attention
    names = WEIGHTS + (QK_WEIGHTS if (a.use_qk_norm or opts[5]) else ())
    out = [_leaf(module, n) for n in names]
    if opts[4]:
        out.append(module.transition.up.weight)
    if opts[5]:
        out += [a.norm_query.weight, a.norm_query.bias, a.norm_key.weight, a.norm_key.bias]
    return out


def _cond_width(module) -> int:
    return module.attention.ada_ln_in.to_scale.weight.shape[1]


def _norms_ok(module, *, h100: bool) -> bool:
    """The block is inside the algorithm: its options are defined, the pair LayerNorm eps is 1e-5; off H100 (B200) only the default
    options at AdaLN eps 1e-5 (the other options are H100 kernels)."""
    from miniworld_engine.kernels.conditioned_transition.cuda import VARIANT_DEFAULT

    opts = options(module)
    if opts is None or module.attention.ln_pair.eps != 1e-5:
        return False
    return h100 or opts == VARIANT_DEFAULT


def _native_fp32_inference(module, single) -> bool:
    """The qualified Hopper FP32 inference family, independent of BF16 experiments."""

    return (
        not torch.is_grad_enabled()
        and module.attention._backend == "portable"
        and getattr(getattr(module.attention, "to_bias", None), "weight", None)
        is not None
        and module.attention.to_bias.weight.shape[1] in PAIR_WIDTHS_H100   # 128; 256: Protenix-v2's diffusion pair; 512: IntelliFold2's
        and (module.attention.n_head, single.shape[-1]) in LAYOUTS  # 16 x 48, 24 x 32, 12 x 64 (d 768), 16 x 64 (d 1024)
        and single.is_cuda
        and single.dtype is torch.float32
        and torch.cuda.get_device_capability(single.device) == (9, 0)
    )


def _check_native_fp32(module, single, cond, pair, compute_dtype):
    """Reject unsupported Hopper FP32 calls instead of entering a portable fallback."""
    if compute_dtype not in (None, torch.float32):
        raise ValueError("Hopper FP32 token inference requires FP32 compute operands")
    if (
        single.ndim != 4
        or single.shape[0] < 1
        or single.shape[1] != 1
        or single.shape[2] < 128
        or single.shape[2] % 128
        or single.shape[3] not in (768, 1024)
    ):
        raise ValueError(
            "Hopper FP32 token inference requires [A,1,L,768 or 1024] with L a multiple of 128"
        )
    A, _, L, _ = single.shape
    if (
        cond.ndim != 4
        or cond.shape[0] not in (1, A)
        or cond.shape[1:3] != (1, L)
        or cond.shape[3] not in COND_WIDTHS_H100
        or cond.shape[3] != _cond_width(module)
        or tuple(pair.shape) != (1, L, L, module.attention.to_bias.weight.shape[1])
        or cond.dtype is not torch.float32
        or pair.dtype is not torch.float32
        or cond.device != single.device
        or pair.device != single.device
    ):
        raise ValueError(
            "Hopper FP32 token inference requires matching FP32 conditioning/pair operands"
        )
    a = module.attention
    d = single.shape[3]
    if (a.n_head, d) not in LAYOUTS or tuple(module.transition.expand_a.weight.shape) != (2 * d, d):
        raise ValueError(
            "Hopper FP32 token inference requires a 16x48 / 24x32 / 12x64 (d 768) or 16x64 (d 1024) layout and a 2d-wide transition"
        )
    if not _norms_ok(module, h100=True):
        raise ValueError("Hopper FP32 token inference requires the token DiT's options (LayerNorm eps=1e-5, the AdaLN's any)")
    if any(
        p.dtype is not torch.float32 or p.device != single.device
        for p in module.parameters()
    ):
        raise ValueError(
            "Hopper FP32 token inference requires FP32 parameters on the input device"
        )


def serves(module, single, cond, pair, compute_dtype=None, extra=None):
    if extra is not None and not serves_extra_bias(single, extra):
        return False
    if _native_fp32_inference(module, single):
        try:
            _check_native_fp32(module, single, cond, pair, compute_dtype)
        except ValueError:
            # no CUDA path for this call: the block's ``cuda`` backend refuses it (BackendUnsupported; no fallback)
            return False
        return True
    if torch.is_grad_enabled():
        return False
    a = module.attention
    # the engine's kernels (the kernel view sets ``_backend``), as the attention cores and the training
    # path gate on

    if a._backend != "portable":
        return False
    if not single.is_cuda or single.dtype not in (torch.bfloat16, torch.float32):
        return False
    if compute_dtype is not None and compute_dtype != single.dtype:
        return False
    if single.ndim != 4 or single.shape[2] < 8:
        return False
    if single.shape[1] != 1 and not (
        # B trunk_seeds, each its own pair (the grouped-bias attention): bf16 on H100 only, a pair per trunk_seed
        single.dtype is torch.bfloat16
        and pair.ndim == 4
        and pair.shape[0] == single.shape[1]
        and torch.cuda.get_device_capability(single.device) == (9, 0)
    ):
        return False
    d = single.shape[-1]
    if (a.n_head, d) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * d:
        return False
    # cond 384; 768 (ESMFold2's token DiT) on H100, where the weight pack and the conditioning rows are built per cond width
    if cond.shape[-1] != _cond_width(module) or not (
        cond.shape[-1] == 384
        or (cond.shape[-1] in COND_WIDTHS_H100 and torch.cuda.get_device_capability(single.device) == (9, 0))
    ):
        return False
    # the pair enters once per sample, through the bias hoist: 128 everywhere; 256 (Protenix-v2's diffusion pair) and 512
    # (IntelliFold2's) on H100, where the hoist normalises it with the CUDA LayerNorm rows (``conditioned_transition.cuda._pair_bias_wide``;
    # fp32 takes ``_native_fp32_inference`` above)
    if pair.shape[-1] != 128 and not (
        pair.shape[-1] in PAIR_WIDTHS_H100 and single.dtype is torch.bfloat16 and torch.cuda.get_device_capability(single.device) == (9, 0)
    ):
        return False
    # One conditioning shared by the samples (a sampling step: L table rows) or one per sample (S L rows); the runner
    # takes either (FusedTokenDiT._cond). Anything else does not describe these samples.
    if (
        cond.shape[0] not in (1, single.shape[0])
        or cond.shape[1:3] != single.shape[1:3]
    ):
        return False
    if not _norms_ok(module, h100=torch.cuda.get_device_capability(single.device) == (9, 0)):
        return False
    cap = torch.cuda.get_device_capability(single.device)
    if a.use_qk_norm and not (cap in ((9, 0), (10, 0)) and _cuda_rows()):
        return False  # QK-norm needs the CUDA rows on either architecture
    if (a.n_head, d) != (16, 768) and not (cap in ((9, 0), (10, 0)) and _cuda_rows()):
        return False  # the other head layouts: the CUDA rows (H100: attn_fwd.cu / attn_tf32.cu per layout; B200: sm_100a core)
    if cap == (10, 0):
        # any length, through the sm_100a core (a length the Triton gated core cannot tile needs it)
        if single.shape[2] % 128 == 0 and (a.n_head, d) == (16, 768):
            return True
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        idx = (
            single.device.index
            if single.device.index is not None
            else torch.cuda.current_device()
        )
        return sm100.inference_core_supported(
            single.dtype, _pad8(single.shape[2]), d, a.n_head, idx
        )
    return cap == (9, 0) and single.shape[2] % 128 == 0 and (a.n_head, d) in LAYOUTS


def serves_extra_bias(single, extra) -> bool:
    """The [L, L] extra pair bias is folded into the Hopper (BF16 and FP32) CUDA path's hoisted bias, and nowhere else."""
    return (
        single.is_cuda
        and single.dtype in (torch.bfloat16, torch.float32)
        and torch.cuda.get_device_capability(single.device) == (9, 0)
        and tuple(extra.shape) == (single.shape[2], single.shape[2])
    )


def _pad8(L: int) -> int:
    return -(-L // 8) * 8


def _cuda_rows() -> bool:
    """The fused step's CUDA row kernels are on (and build): the runner takes them on H100 and B200."""

    return _cuda_rows_available()


@device_constant
def _cuda_rows_available() -> bool:
    """Resolve the CUDA extension once while tracing; compiler/file probes are not graph operations."""
    try:
        from miniworld_engine.kernels.conditioned_transition import cuda as cuda_rows

        return cuda_rows.available()
    except Exception:
        if torch.cuda.get_device_capability() == (9, 0):
            raise
        return False


def _fake(single, cond, pair, mask, weights, qk, eq, ek, heads, extra, variant):
    return torch.empty_like(single)


@opaque(fake=_fake, name="token_dit_h100_infer")
def _infer(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qk: bool,
    eq: float,
    ek: float,
    heads: int,
    extra: torch.Tensor | None,
    variant: list[float],
) -> torch.Tensor:
    """``weights``: the block's ``leaves`` (WEIGHTS, then QK_WEIGHTS when the block has QK-norm -- eq / ek its RMSNorm eps; unused
    otherwise -- or a flat q / k LayerNorm, then the option leaves); ``variant``: its ``options`` (H100; off H100 the defaults only).

    ``extra``: an [L, L] logit bias for every head (Hopper CUDA path only; ``serves`` rejects it elsewhere)."""
    if torch.cuda.get_device_capability(single.device) == (9, 0):
        from miniworld_engine.kernels.conditioned_transition import cuda
        from miniworld_engine.kernels.integrations.token_dit_train import _tf32

        with _tf32(single.dtype is torch.float32):
            return cuda.token_inference(
                single, cond, pair.contiguous(), mask, weights, qk, eq, ek, heads, extra, variant
            )
    # Off H100 the runner with the hand-CUDA rows / SwiGLU GEMM / attention cores of the card (it degrades to the Triton hooks where
    # the card has none).
    else:
        from miniworld_engine.kernels.conditioned_transition.cuda.token_dit_composed import (
            FusedTokenDiTComposed as FusedTokenDiT,
        )

    def build():
        block = SimpleNamespace()
        for path, weight in zip(
            WEIGHTS + (QK_WEIGHTS if qk else ()), weights, strict=True
        ):
            node = block
            names = path.split(".")
            for name in names[:-1]:
                if not hasattr(node, name):
                    setattr(node, name, SimpleNamespace())
                node = getattr(node, name)
            setattr(node, names[-1], weight)
        block.attention.use_qk_norm = qk
        block.attention.qk_eps = (eq, ek) if qk else None
        block.attention.n_head = heads
        return FusedTokenDiT([block], dtype=single.dtype)

    # autocast off: the step's precision is the runner's dtype (single's), and an enclosing bf16 autocast would turn the fp32
    # step's conditioning GEMMs bf16
    with torch.cuda.device(single.device), torch.autocast("cuda", enabled=False):
        # This path is inference only (the token DiT is not recycled; training takes kernels/integrations/token_dit_train), so
        # the weights do not change between the calls of a sampling run. Packing is tens of small kernels (~0.2 ms a
        # call, more than the block's step at L384): reuse the pack while every weight's (pointer, version) is the same
        # -- an in-place update bumps ``_version`` and misses. Scoped to the CUDA-graph capture (``kernels._capture``): a
        # capture packs once and records it, so a replay packs the weights as they are then (a training run that samples
        # inside a graph -- a mini-rollout -- changes them between replays), and reuses that pack for the capture's other calls.
        wkey = (
            single.dtype,
            qk,
            eq,
            ek,
            heads,
            *((w.data_ptr(), w._version) for w in weights),
        )
        runner = _capture.lookup(_RUNNERS, wkey, build)
        key = (single.device, single.dtype)
        caches = _TUNING.setdefault(key, ({}, {}))
        runner._mm_cfg, runner._gated_cfg = caches
        # The pair is the trunk's output: the same at every diffusion step of a sample, so the block's pair bias (a
        # LayerNorm over L^2 pair rows + a projection, ~40 % of a call at L = 768) is computed once per sample. Keyed by
        # the caller's pair / mask tensors (pointer, version, layout): a new pair, or an in-place change of it, misses.
        # Scoped as the pack: a capture computes the bias once (recorded), so each replay recomputes it from the pair it has.
        assert (
            mask is not None
        )  # off sm_90 ``update`` fills a missing mask with all-True
        pinned = (pair, mask)  # the cache entry below keeps the caller's tensors alive
        bkey = _capture.scoped(
            (
                pair.data_ptr(),
                pair._version,
                tuple(pair.shape),
                pair.stride(),
                mask.data_ptr(),
                mask._version,
            )
        )
        hit = runner.__dict__.get("_bias_cache")
        L, L8 = single.shape[2], _pad8(single.shape[2])
        if bkey is not None and hit is not None and hit[0] == bkey:
            bias = hit[1]
        else:
            if (
                L8 != L
            ):  # padded keys: zero pair rows / columns, masked out of every softmax
                pair = torch.nn.functional.pad(pair, (0, 0, 0, L8 - L, 0, L8 - L))
                mask = torch.nn.functional.pad(
                    mask.to(torch.bool), (0, L8 - L), value=False
                )
            bias = runner.hoist(pair.contiguous(), mask)
            if bkey is not None:
                runner._bias_cache = (bkey, bias, pinned)
        if L8 == L:
            return runner.step(single, cond, bias)
        # padded query rows: zero single / cond rows, computed and dropped (a shared cond stays one table)
        single = torch.nn.functional.pad(single, (0, 0, 0, L8 - L))
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        cond = torch.nn.functional.pad(cond[:1] if shared else cond, (0, 0, 0, L8 - L))
        return runner.step(single, cond, bias)[:, :, :L].contiguous()


_ALL_TRUE: dict = {}


def update(module, single, cond, pair, mask, extra=None):
    if single.shape[1] > 1 and mask is not None and (mask.shape[0] != single.shape[1] or mask.stride(0) != 0):
        # B trunk_seeds of one input: the key mask is [B, L], one mask expanded over B (the grouped core reads one)
        raise ValueError("B > 1 token DiT takes one key mask expanded over B (stride 0)")
    if (
        _native_fp32_inference(module, single)
        and mask is not None
        and (
            mask.dtype is not torch.bool
            or mask.device != single.device
            or tuple(mask.shape) != (1, single.shape[2])
        )
    ):
        raise ValueError(
            "Hopper FP32 token inference requires a boolean [1,L] key mask"
        )
    if mask is None and torch.cuda.get_device_capability(single.device) != (9, 0):
        # The native Hopper path accepts None without an ATen mask fill.
        k = (single.shape[2], single.device)
        mask = _ALL_TRUE.get(k)
        if mask is None:
            mask = _ALL_TRUE[k] = torch.ones(
                (1, single.shape[2]), device=single.device, dtype=torch.bool
            )
    # the pair as the caller holds it: _infer keys its pair-bias cache on this tensor and makes it contiguous on a miss
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(single.dtype)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(single.dtype)) if qk else 0.0
    opts = options(module)
    return _infer(
        single.contiguous(),
        cond,
        pair,
        mask,
        leaves(module, opts),
        qk,
        eq,
        ek,
        int(a.n_head),
        extra,
        list(opts),
    )
