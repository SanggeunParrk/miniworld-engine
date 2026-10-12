from collections.abc import Mapping
from typing import TypeVar

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float
from pydantic import BaseModel
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
from miniworld_engine.architecture.blocks.pair_bias_hoist import pair_bias_all
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import AugmentedAttentionPairBias
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.functional import sigmoid_gate

T = TypeVar("T")


def _sm90_attention():
    """The engine's sm_90 bf16 attention core (miniworld_engine.kernels.augmented_attention.cuda, engine
    feat/token-dit-train-core). Older engine revs do not have it -- say so instead of failing inside the forward."""
    try:
        from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90
    except ImportError as exc:  # pragma: no cover - depends on the installed engine rev
        raise RuntimeError(
            "attention_core_dtype='bf16_sm90' needs a miniworld-engine with "
            "kernels/augmented_attention/cuda"
        ) from exc
    return cuda_sm90


def _checkpoint_segments(fns, segments, x, context_fn):
    """``checkpoint_sequential(fns, segments, x, use_reentrant=False)`` with a ``context_fn``: the same split (all but
    the last segment checkpointed, the last run plainly)."""
    from torch.utils.checkpoint import checkpoint

    def run(start, end):
        def f(y):
            for fn in fns[start : end + 1]:
                y = fn(y)
            return y

        return f

    size = len(fns) // segments
    end = -1
    for start in range(0, size * (segments - 1), size):
        end = start + size - 1
        x = checkpoint(run(start, end), x, use_reentrant=False, context_fn=context_fn)
    return run(end + 1, len(fns) - 1)(x)


def _attention_with_bias(
    attn: AugmentedAttentionPairBias, single, cond, bias, mask, compute_dtype=None
):
    """``AugmentedAttentionPairBias.forward`` with the pair bias already computed (``pair_bias_all``): the same
    steps and the same submodules, minus ``ln_pair`` / ``to_bias``.

    Returns the residual output ``single + update``, as the module does.
    """
    in_dtype = single.dtype
    single_res = single
    single = attn.ada_ln_in(single, cond)
    query, key, value, gate = (
        attn.to_query(single),
        attn.to_key(single),
        attn.to_value(single),
        attn.to_gate(single),
    )
    num_aug, batch, len_res = query.shape[:3]
    n_head, hidden_dim = attn.n_head, query.shape[-1] // attn.n_head
    query, key, value = [
        x.view(num_aug, batch, len_res, n_head, hidden_dim) for x in (query, key, value)
    ]
    if mask is not None and mask.ndim == 2:
        mask = repeat(mask, "B L -> A B L", A=single.shape[0])
    if attn.use_qk_norm:
        query, key = attn.norm_query(query), attn.norm_key(key)
    if (
        compute_dtype == "bf16_sm90"
    ):  # the hoisted bias arrives head-major: no permute copy
        sm90 = _sm90_attention()
        if not sm90.available(query, bias, bias_head_major=True):
            raise RuntimeError(
                "attention_core_dtype='bf16_sm90': this shape / GPU is not supported (sm_90, 16 heads "
                "x 48, B == 1, L a multiple of 128) -- use 'bf16'"
            )
        out = sm90.augmented_attention_bf16_sm90(
            query, key, value, bias, mask, bias_head_major=True
        )
    else:
        out = attn._kernel_attention_pair_bias(
            query, key, value, bias, mask, compute_dtype=compute_dtype
        )
    out = rearrange(out.to(in_dtype), "A B L H D -> A B L (H D)")
    out = attn.to_out(sigmoid_gate(gate, out))
    return single_res + sigmoid_gate(attn.to_scale(cond), out)


# The attention's prefix in checkpoints written before the block became the
# engine's DiTBlock (its old attribute name).
_LEGACY = "attention_pair_bias."


def _load_legacy_names(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    *_: object,
) -> None:
    """Read ``attention_pair_bias.`` keys into ``attention``."""
    old, new = prefix + _LEGACY, prefix + "attention."
    for k in [k for k in state_dict if k.startswith(old)]:
        state_dict[new + k.removeprefix(old)] = state_dict.pop(k)


def upgrade_legacy_keys(model: nn.Module, state_dict: Mapping[str, T]) -> dict[str, T]:
    """``state_dict`` with its diffusion blocks' ``attention_pair_bias.`` keys renamed.

    Each ``DiffusionTransformerBlock``'s ``attention_pair_bias.`` keys become
    ``attention.``. ``load_state_dict`` does this through the block's load pre-hook;
    readers of a raw state dict keyed by ``named_parameters`` (the EMA shadow,
    parameter policies) call this first. Other modules' own ``attention_pair_bias``
    (the pairformer's) keep their keys.
    """
    olds = tuple(
        f"{n}.{_LEGACY}" if n else _LEGACY
        for n, m in model.named_modules()
        if isinstance(m, DiffusionTransformerBlock)
    )
    out: dict[str, T] = {}
    for k, v in state_dict.items():
        old = next((o for o in olds if k.startswith(o)), None)
        if old is not None:
            k = old.removesuffix(_LEGACY) + "attention." + k.removeprefix(old)
        out[k] = v
    return out


class DiffusionTransformerBlock(DiTBlock):
    """A single block of the Diffusion Transformer: the engine's ``DiTBlock``.

    ``AugmentedAttentionPairBias`` -> ``ConditionedTransition``, each returning its
    residual output ``x + f(x)`` (the magnitude-preserving ``mp_sum`` variant and the
    EDM2/rotation/bias-norm options were removed). Being the engine's block, a call
    takes its fused paths where they serve it (B200: token DiT training / inference,
    the AF3 atom block; H100: token DiT inference) and the module path otherwise.

    The attention is ``self.attention`` (the engine's name); its keys are
    ``attention.`` in ``named_parameters`` and ``state_dict`` alike, and
    ``attention_pair_bias`` is the same module under its old name. Checkpoints that
    spell the keys ``attention_pair_bias.`` still load (a load pre-hook;
    :func:`upgrade_legacy_keys` for raw state dicts).
    """

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        d_pair: int,
        n_head: int,
        *,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(
            d_single,
            d_cond,
            d_pair,
            n_head,
            use_qk_norm=use_qk_norm,
            implementation=to_engine_impl(implementation),
        )
        self.register_load_state_dict_pre_hook(_load_legacy_names)

    @property
    def attention_pair_bias(self) -> AugmentedAttentionPairBias:
        """The attention (``self.attention``) under this block's original name."""
        return self.attention

    @typecheck
    def forward(  # ty: ignore[invalid-method-override] - extends DiTBlock's call
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
        bias: torch.Tensor | None = None,
        core_dtype: torch.dtype | None = None,
        bias_head_major: bool = True,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Chain ``attention`` -> ``transition`` (each returns its residual output).

        With ``pair`` and a [B, L] mask (or none) this is ``DiTBlock.forward``, fused
        paths included. ``bias`` (this block's ``to_bias(ln_pair(pair))``, from
        ``pair_bias_all``) replaces ``pair`` when given. ``core_dtype`` is the
        attention core's compute dtype (the engine's ``compute_dtype``; ``None`` =
        the tensors' own).
        """
        attn = self.attention
        if bias is None:
            if pair is None:
                msg = "either `pair` or `bias` must be provided"
                raise ValueError(msg)
            if mask is None or mask.ndim == 2:
                return DiTBlock.forward(
                    self, single, cond, pair, mask, compute_dtype=core_dtype
                )
            single = attn(single, cond, pair, mask, compute_dtype=core_dtype)
        else:
            if bias_head_major:
                bias = (
                    bias.permute(1, 0, 2, 3)
                    if core_dtype == "bf16_sm90"
                    else bias.permute(0, 2, 3, 1)
                )
            single = _attention_with_bias(attn, single, cond, bias, mask, core_dtype)
        return self.transition(single, cond)


class DiffusionTransformer(nn.Module):
    """A Diffusion Transformer module."""

    class Config(BaseModel):
        """Configuration for DiffusionTransformer module."""

        d_single: int = 768
        d_cond: int = 384
        d_pair: int = 128
        n_head: int = 16
        use_qk_norm: bool = False
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 3
        n_checkpoint_segments: int | None = None
        # Main's per-block API takes precedence; legacy segmentation remains supported.
        use_checkpoint: bool = False
        # All blocks' pair biases from one LayerNorm + one GEMM (pair_bias_hoist.py), outside the checkpointed
        # blocks, instead of a LayerNorm over the whole pair tensor in every block. Same parameters, same result
        # to rounding; saves the per-block pair LayerNorm (twice under checkpointing), its backward and the 24
        # pair-gradient accumulations.
        hoist_pair_bias: bool = False
        # Precision the attention CORE (softmax(q k^T + bias) v and its backward) runs in; the projections, norms and
        # residual stay in the module's dtype. "bf16" runs a fp32 model's core in bf16 with fp32 accumulation (the
        # engine routes it to its sm_90 kernels where they apply). "bf16_sm90" (needs hoist_pair_bias) asks for those
        # kernels explicitly and hands them the hoisted bias head-major; an unsupported shape/GPU raises instead of
        # falling back. L768 A48 block fwd+bwd: 22.22 ms "bf16" (Triton) -> 18.77 ms "bf16_sm90". ("tdt" = old name.)
        attention_core_dtype: str = "model"
        # With "bf16_sm90" and per-block checkpointing: keep each block's attention output and LSE from the forward so
        # the backward's recompute skips the attention kernel (-0.33 ms a block at L768 / A48, +115 MB held per block).
        checkpoint_keep_attention: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    config.d_single,
                    config.d_cond,
                    config.d_pair,
                    config.n_head,
                    use_qk_norm=config.use_qk_norm,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def prepare_pair_biases(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
    ) -> tuple[Float[torch.Tensor, "B H L L"], ...]:
        """Preserve the main API's differentiable, head-major per-block biases."""
        return tuple(
            block.attention_pair_bias.to_bias(
                block.attention_pair_bias.ln_pair(pair)
            ).permute(0, 3, 1, 2)
            for block in self.blocks
        )

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
        *,
        pair_biases: tuple[Float[torch.Tensor, "B H L L"], ...] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Forward pass."""
        cd = {
            "model": None,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "bf16_sm90": "bf16_sm90",
            "tdt": "bf16_sm90",
        }[self.config.attention_core_dtype]
        if (
            cd == "bf16_sm90"
            and not self.config.hoist_pair_bias
            and pair_biases is None
        ):
            raise ValueError(
                "attention_core_dtype 'bf16_sm90' needs hoist_pair_bias (it takes the head-major hoisted bias)"
            )
        if pair_biases is not None:
            block_fns = [
                (
                    lambda s, block=b, bias=bb: block(
                        s, cond, pair, mask, bias=bias, core_dtype=cd
                    )
                )
                for b, bb in zip(self.blocks, pair_biases, strict=True)
            ]
        elif pair is None:
            raise ValueError("either `pair` or `pair_biases` must be provided")
        elif self.config.hoist_pair_bias:
            biases = pair_bias_all(
                [b.attention_pair_bias for b in self.blocks],
                pair,
                head_major=cd == "bf16_sm90",
            )
            block_fns = [
                (
                    lambda s, block=b, bias=bb: block(
                        s,
                        cond,
                        pair,
                        mask,
                        bias=bias,
                        core_dtype=cd,
                        bias_head_major=False,
                    )
                )
                for b, bb in zip(self.blocks, biases, strict=False)
            ]
        else:
            block_fns = [
                (lambda s, block=b: block(s, cond, pair, mask, core_dtype=cd))
                for b in self.blocks
            ]

        if self.config.use_checkpoint and torch.is_grad_enabled():
            kwargs = {}
            if self.config.checkpoint_keep_attention and cd == "bf16_sm90":
                kwargs["context_fn"] = (
                    _sm90_attention().checkpoint_context_keeping_attention
                )
            for fn in block_fns:
                single = checkpoint(fn, single, use_reentrant=False, **kwargs)
        elif self.config.n_checkpoint_segments is None or not torch.is_grad_enabled():
            for fn in block_fns:
                single = fn(single)
        elif self.config.checkpoint_keep_attention and cd == "bf16_sm90":
            # checkpoint_sequential's segmentation, with a selective-checkpoint context (it takes no context_fn)
            single = _checkpoint_segments(
                block_fns,
                self.config.n_checkpoint_segments,
                single,
                _sm90_attention().checkpoint_context_keeping_attention,
            )
        else:
            single = checkpoint_sequential(
                block_fns,
                self.config.n_checkpoint_segments,
                input=single,
                use_reentrant=False,
            )

        return single
