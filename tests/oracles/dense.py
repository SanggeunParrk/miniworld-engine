"""Frozen numerical oracle of the released checkpoints: the dense (unfused, plain fp32 PyTorch) blocks the engine's ``from_released`` modules are
checked against, self-contained so the engine's own tests need no FoldForge.

These are the same equations as FoldForge's ``foldforge.modules.dense`` modules (the released AlphaFold 3 family: AF3 / Protenix / Boltz / Chai /
RoseTTAFold3 layouts), written out once here with NO kernel routing and no import of ``foldforge``: parameter names, shapes and registration
order are the released ones (that is what ``miniworld_engine.modules.released`` reads), so a module seeded from ``_rand``-style draws holds the
SAME weights as the FoldForge class of the same name (``tests/oracles/compare_foldforge.py`` states this as a check). Forward returns the raw
delta (the residual is added by the caller), as the released blocks do.

Do not "fix" an equation here to follow a change in the engine: this file is the oracle; a change in the numbers has to be explained.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DenseSpec:
    """The family flags the released-parity cases vary (names and defaults as in FoldForge's ``DenseSpec``)."""

    adaptive_norm_eps: float = 1e-5
    adaptive_identity_scale: bool = False
    affine_norms: frozenset[str] = field(default_factory=frozenset)
    attention_kq_norm: bool = False
    parallel_attention_transition: bool = False
    parallel_pairformer_block: bool = False
    per_block_pair_layer_norm: bool = False
    per_block_atom_pair_layer_norm: bool = False
    single_attention_gate_bias: float = 0.0
    token_attention_gating_query: bool = True
    atom_attention_gating_query: bool = True
    atom_attention_project_output: bool = True
    atom_adaptive_identity_scale: bool | None = None
    atom_adaptive_zero_bias: bool = True
    transition_up_gate: bool = False
    transposed_column_pair_bias: bool = False
    triangle_attention_bias: bool = False
    triangle_mul_divide_by_length: bool = False
    untransposed_column_pair_output: bool = False


ALPHAFOLD3 = DenseSpec()


# ------------------------------------------------------------------------------------------------------------------- shared equations
def attention(q, k, v, *, bias=None, mask=None, mask_value=-1e9):
    """Softmax attention on [..., H, L, D] (scale D^-0.5, logits in the input dtype); ``bias`` is added to the logits BEFORE the key mask
    (a replacement, True = keep): a fully masked row is uniform. ``mask`` [L_k] or [B, L_k]."""
    if mask is not None:
        mask = mask[None, None, None, :] if mask.ndim == 1 else mask[:, None, None, :]
    logits = (q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), mask_value)
    return logits.softmax(dim=-1) @ v


def _heads(t, h):
    """[..., n, h * c] -> [..., h, n, c]"""
    return t.unflatten(-1, (h, -1)).transpose(-2, -3)


# ------------------------------------------------------------------------------------------------------------------- pair-track blocks
class Transition(nn.Module):
    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super().__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = nn.LayerNorm(c_x)
        self.transition1 = nn.Linear(c_x, num_intermediate_factor * c_x * 2, bias=False)
        self.transition2 = nn.Linear(num_intermediate_factor * c_x, c_x, bias=False)

    def forward(self, x):
        a, b = self.transition1(self.input_layer_norm(x)).chunk(2, dim=-1)
        return self.transition2(F.silu(a) * b)


class TriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True, hidden_dim: int | None = None, divide_by_length: bool = False) -> None:
        super().__init__()
        self.c_pair = c_pair
        self.divide_by_length = divide_by_length
        hidden = hidden_dim or c_pair
        self.left_norm_input = nn.LayerNorm(c_pair)
        self.projection = nn.Linear(c_pair, 2 * hidden, bias=False)  # interleaved: row 2k = branch a of channel k, row 2k+1 = branch b
        self.gate = nn.Linear(c_pair, 2 * hidden, bias=False)
        self.center_norm = nn.LayerNorm(hidden)
        self.output_projection = nn.Linear(hidden, c_pair, bias=False)
        self.gating_linear = nn.Linear(c_pair, c_pair, bias=False)
        self.outgoing = _outgoing

    def forward(self, pair, mask, contraction_scale=None):
        """[L, L, C] -> the raw delta. ``contraction_scale``: a 0-d tensor the contraction is multiplied by before the centre norm."""
        value = self.left_norm_input(pair)
        projection = self.projection(value).movedim(-1, -3)
        if mask is not None:
            projection = projection * mask.unsqueeze(-3)
        projection = projection * torch.sigmoid(self.gate(value).movedim(-1, -3))
        a, b = projection.unflatten(-3, (-1, 2)).unbind(-3)
        equation = "...cik,...cjk->...cij" if self.outgoing else "...ckj,...cki->...cij"
        contracted = torch.einsum(equation, a, b).movedim(-3, -1)
        if contraction_scale is not None:
            contracted = contracted * contraction_scale.to(contracted.dtype)
        return self.output_projection(self.center_norm(contracted)) * torch.sigmoid(self.gating_linear(value))


def length_divided_triangle(module: TriangleMultiplication, pair, mask):
    """RoseTTAFold3's form: the contraction divided by the real sequence length (the residues any row of the pair mask keeps)."""
    count = mask.reshape(-1, mask.shape[-1]).any(0).sum().clamp_min(1)
    return module(pair, mask, contraction_scale=count.float().reciprocal())


def triangle_residual(module: TriangleMultiplication, pair, mask):
    return pair + (length_divided_triangle(module, pair, mask) if module.divide_by_length else module(pair, mask))


class OuterProductMean(nn.Module):
    def __init__(
        self, c_msa: int = 64, num_output_channel: int = 128, num_outer_channel: int = 32, projection_bias: bool = False, groups: int = 1,
        bias_after_norm: bool = False, clamped_norm: bool = False,
    ) -> None:
        super().__init__()
        self.c_msa = c_msa
        self.bias_after_norm = bias_after_norm
        self.clamped_norm = clamped_norm
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.groups = groups
        self.epsilon = 1e-3
        self.layer_norm_input = nn.LayerNorm(c_msa)
        width = groups * num_outer_channel
        self.left_projection = nn.Linear(c_msa, width, bias=projection_bias)
        self.right_projection = nn.Linear(c_msa, width, bias=projection_bias)
        if groups > 1:
            # grouped: the depth axis is summed, never divided; the product norm (eps 0.1) absorbs the scale
            self.product_norm = nn.LayerNorm(groups * num_outer_channel**2, eps=0.1)
        self.output_w = nn.Parameter(torch.randn(*((groups,) if groups > 1 else ()), num_outer_channel, num_outer_channel, num_output_channel))
        self.output_b = nn.Parameter(torch.randn(num_output_channel))

    def forward(self, msa, mask):
        """[S, L, c_msa], [S, L] -> [L, L, c_pair]"""
        mask = mask.to(msa.dtype).unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left, right = mask * self.left_projection(msa), mask * self.right_projection(msa)
        if self.groups > 1:
            g, c = self.groups, self.num_outer_channel
            product = torch.einsum("sigk,sjgl->ijgkl", left.unflatten(-1, (g, c)), right.unflatten(-1, (g, c)))
            product = self.product_norm(product.flatten(-3))
            return product @ self.output_w.reshape(g * c * c, self.num_output_channel) + self.output_b
        outer = torch.einsum("acb,ade->dceb", left.permute(0, 2, 1), right)
        output = torch.einsum("dceb,cef->dbf", outer, self.output_w)
        norm = torch.einsum("abc,adc->bdc", mask, mask)  # the number of MSA rows that keep both residues
        if not self.bias_after_norm and not self.clamped_norm:
            # AF3: mean over MSA rows with a 1e-3 offset, the output bias BEFORE the divide
            return (output + self.output_b).permute(1, 0, 2) / (self.epsilon + norm)
        divisor = norm.clamp_min(1.0) if self.clamped_norm else self.epsilon + norm
        if self.bias_after_norm:
            return output.permute(1, 0, 2) / divisor + self.output_b
        return (output + self.output_b).permute(1, 0, 2) / divisor


class GridSelfAttention(nn.Module):
    """Triangle attention: rows (``transpose=False``, starting node) or columns (ending node) of the pair, bias from the pair."""

    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False, spec: DenseSpec = ALPHAFOLD3,
                 qkv_dim: int | None = None, dual_output: bool = False) -> None:
        super().__init__()
        self.dual_output = dual_output
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = qkv_dim or c_pair // num_head
        hidden = num_head * self.qkv_dim
        self.transpose = transpose
        self.transposed_bias = transpose and spec.transposed_column_pair_bias
        # the ending direction is not turned back: the sum of the two directions is taken in the mixed orientation
        self.untransposed_output = transpose and spec.untransposed_column_pair_output and not dual_output
        self.act_norm = nn.LayerNorm(c_pair)
        self.pair_bias_projection = nn.Linear(c_pair, num_head, bias=False)
        self.q_projection = nn.Linear(c_pair, hidden, bias=False)
        self.k_projection = nn.Linear(c_pair, hidden, bias=False)
        self.v_projection = nn.Linear(c_pair, hidden, bias=False)
        self.gating_query = nn.Linear(c_pair, hidden, bias=spec.triangle_attention_bias)
        self.output_projection = nn.Linear(hidden, c_pair, bias=spec.triangle_attention_bias)
        if dual_output:
            self.output_projection_transposed = nn.Linear(hidden, c_pair, bias=spec.triangle_attention_bias)

    def forward(self, pair, mask):
        """[L, L, C], [L, L] -> the raw delta [L, L, C]"""
        pair = self.act_norm(pair)
        bias = self.pair_bias_projection(pair).permute(2, 0, 1)
        if self.transposed_bias:
            bias = bias.transpose(-1, -2)
        if self.transpose:
            pair = pair.permute(1, 0, 2)
        h = self.num_head
        q, k, v = (_heads(p(pair), h) for p in (self.q_projection, self.k_projection, self.v_projection))
        avg = attention(q, k, v, mask=mask, bias=bias).transpose(-2, -3).flatten(-2)  # [rows, n, h * d]
        gate = torch.sigmoid(self.gating_query(pair)) * avg
        kept = self.output_projection(gate)
        if not self.dual_output:
            return kept.permute(1, 0, 2) if self.transpose and not self.untransposed_output else kept
        other = self.output_projection_transposed(gate)
        if self.transpose:
            kept, other = kept.permute(1, 0, 2), other.permute(1, 0, 2)
        return kept + other.transpose(-2, -3)


class MSAAttention(nn.Module):
    """MSA row attention with pair-only logits (MSA pair weighted averaging)."""

    def __init__(self, c_msa: int = 64, c_pair: int = 128, num_head: int = 8, value_dim: int | None = None) -> None:
        super().__init__()
        self.c_msa, self.c_pair, self.num_head = c_msa, c_pair, num_head
        self.value_dim = value_dim or c_msa // num_head
        hidden = num_head * self.value_dim
        self.act_norm = nn.LayerNorm(c_msa)
        self.pair_norm = nn.LayerNorm(c_pair)
        self.pair_logits = nn.Linear(c_pair, num_head, bias=False)
        self.v_projection = nn.Linear(c_msa, hidden, bias=False)
        self.gating_query = nn.Linear(c_msa, hidden, bias=False)
        self.output_projection = nn.Linear(hidden, c_msa, bias=False)

    def forward(self, msa, msa_mask, pair):
        """[S, L, c_msa], [S, L], [L, L, c_pair] -> the raw delta [S, L, c_msa]"""
        msa = self.act_norm(msa)
        logits = self.pair_logits(self.pair_norm(pair)).permute(2, 0, 1)
        logits = logits + 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)  # a key column no MSA row keeps is masked
        weights = torch.softmax(logits, dim=-1)
        v = self.v_projection(msa).unflatten(-1, (self.num_head, -1))
        avg = torch.einsum("hqk,bkhc->bqhc", weights, v).flatten(-2)
        return self.output_projection(torch.sigmoid(self.gating_query(msa)) * avg)


# ------------------------------------------------------------------------------------------------------------------- conditioned blocks
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, c_x: int, c_single_cond: int | None, use_single_cond: bool = False, identity_scale: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.c_x, self.c_single_cond, self.use_single_cond, self.identity_scale = c_x, c_single_cond, use_single_cond, identity_scale
        if use_single_cond:
            self.layer_norm = nn.LayerNorm(c_x, eps=eps, elementwise_affine=False, bias=False)
            if not identity_scale:
                self.single_cond_layer_norm = nn.LayerNorm(c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(c_single_cond, c_x, bias=not identity_scale)
            self.single_cond_bias = nn.Linear(c_single_cond, c_x, bias=False)
        else:
            self.layer_norm = nn.LayerNorm(c_x, eps=eps)

    def forward(self, x, single_cond=None):
        value = self.layer_norm(x)
        if not self.use_single_cond:
            return value
        # the (s + 1) form leaves the conditioning unnormalised; the sigmoid form normalises it
        context = single_cond if self.identity_scale else self.single_cond_layer_norm(single_cond)
        scale = self.single_cond_scale(context)
        return (scale + 1.0 if self.identity_scale else torch.sigmoid(scale)) * value + self.single_cond_bias(context)


class AdaLNZero(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_single_cond: int | None, use_single_cond: bool = False, project: bool = True,
                 zero_bias: bool = True) -> None:
        super().__init__()
        self.use_single_cond, self.project = use_single_cond, project
        if project:
            self.transition2 = nn.Linear(c_in, c_out, bias=False)
        if use_single_cond:
            self.adaptive_zero_cond = nn.Linear(c_single_cond, c_out, bias=zero_bias)

    def forward(self, x, single_cond=None, gate_logits=None):
        output = self.transition2(x if gate_logits is None else torch.sigmoid(gate_logits) * x)
        if self.use_single_cond:
            output = torch.sigmoid(self.adaptive_zero_cond(single_cond)) * output
        return output


class DiffusionTransition(nn.Module):
    def __init__(self, c_x: int, c_single_cond: int | None, num_intermediate_factor: int = 2, use_single_cond: bool = False,
                 identity_scale: bool = False, norm_eps: float = 1e-5, zero_bias: bool = True) -> None:
        super().__init__()
        self.c_x, self.use_single_cond, self.num_intermediate_factor = c_x, use_single_cond, num_intermediate_factor
        self.adaptive_layernorm = AdaptiveLayerNorm(c_x, c_single_cond, use_single_cond, identity_scale=identity_scale, eps=norm_eps)
        self.transition1 = nn.Linear(c_x, 2 * c_x * num_intermediate_factor, bias=False)
        self.adaptive_zero_init = AdaLNZero(num_intermediate_factor * c_x, c_x, c_single_cond, use_single_cond, zero_bias=zero_bias)

    def forward(self, x, single_cond=None):
        a, b = self.transition1(self.adaptive_layernorm(x, single_cond)).chunk(2, dim=-1)
        return self.adaptive_zero_init(F.silu(a) * b, single_cond)


class GatedDiffusionTransition(DiffusionTransition):
    """The SwiGLU output passes a linear up-gate before the zero-initialised output (``a_to_b``)."""

    def __init__(self, c_x: int, c_single_cond: int | None, **options) -> None:
        super().__init__(c_x, c_single_cond, **options)
        self.a_to_b = nn.Linear(c_x, self.num_intermediate_factor * c_x, bias=False)

    def forward(self, x, single_cond=None):
        normed = self.adaptive_layernorm(x, single_cond)
        gate, value = self.transition1(normed).chunk(2, dim=-1)
        return self.adaptive_zero_init(F.silu(gate) * value * self.a_to_b(normed), single_cond)


def conditioned_transition(spec: DenseSpec) -> type[DiffusionTransition]:
    return GatedDiffusionTransition if spec.transition_up_gate else DiffusionTransition


class SelfAttention(nn.Module):
    def __init__(self, c_x: int = 768, c_single_cond: int = 384, num_head: int = 16, use_single_cond: bool = False, kq_norm: bool = False,
                 identity_scale: bool = False, norm_eps: float = 1e-5, gating_query: bool = True, project_output: bool = True,
                 gate_bias: float = 0.0) -> None:
        super().__init__()
        self.kq_norm, self.use_gating_query, self.gate_bias = kq_norm, gating_query, gate_bias  # the gate bias is added to the LOGITS
        if kq_norm:
            self.query_layer_norm = nn.LayerNorm(c_x)
            self.key_layer_norm = nn.LayerNorm(c_x)
        self.c_x, self.c_single_cond, self.num_head = c_x, c_single_cond, num_head
        self.qkv_dim = c_x // num_head
        self.use_single_cond = use_single_cond
        self.adaptive_layernorm = AdaptiveLayerNorm(c_x, c_single_cond, use_single_cond, identity_scale=identity_scale, eps=norm_eps)
        self.q_projection = nn.Linear(c_x, c_x, bias=True)
        self.k_projection = nn.Linear(c_x, c_x, bias=False)
        self.v_projection = nn.Linear(c_x, c_x, bias=False)
        if gating_query:
            self.gating_query = nn.Linear(c_x, c_x, bias=False)
        self.adaptive_zero_init = AdaLNZero(c_x, c_x, c_single_cond, use_single_cond, project=project_output)

    def forward(self, x, mask, pair_logits=None, single_cond=None):
        """[n, c], [n], logits [H, n, n] -> the raw delta [n, c]"""
        x = self.adaptive_layernorm(x, single_cond)
        q, k, v = self.q_projection(x), self.k_projection(x), self.v_projection(x)
        if self.kq_norm:
            q, k = self.query_layer_norm(q), self.key_layer_norm(k)  # over the flattened head axis
        q, k, v = (_heads(t, self.num_head)[None] for t in (q, k, v))  # [1, H, n, c]
        avg = attention(q, k, v, mask=mask, bias=pair_logits).transpose(-2, -3).flatten(-2)[0]
        gate_logits = None
        if self.use_gating_query:
            gate_logits = self.gating_query(x)
            if self.gate_bias:
                gate_logits = gate_logits + self.gate_bias
        return self.adaptive_zero_init(avg, single_cond, gate_logits=gate_logits)


class DiffusionTransformer(nn.Module):
    """The token diffusion transformer (Boltz-2 / RoseTTAFold3 / Chai-1 layouts): ``num_blocks`` x (attention, transition) on a [n, c] activation."""

    def __init__(self, c_act: int = 768, c_single_cond: int = 384, c_pair_cond: int = 128, num_head: int = 16, num_blocks: int = 24,
                 super_block_size: int = 4, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()
        self.per_block_pair = spec.per_block_pair_layer_norm
        self.parallel = spec.parallel_attention_transition
        self.num_head, self.num_blocks, self.super_block_size = num_head, num_blocks, super_block_size
        self.num_super_blocks = num_blocks // super_block_size
        pair_norm_bias = "pair_input_layer_norm" in spec.affine_norms
        if self.per_block_pair:  # one pair norm and one head projection per block
            self.pair_input_layer_norm = nn.ModuleList([nn.LayerNorm(c_pair_cond, bias=pair_norm_bias) for _ in range(num_blocks)])
            self.pair_logits_projection = nn.ModuleList([nn.Linear(c_pair_cond, num_head, bias=False) for _ in range(num_blocks)])
        else:  # one norm; one projection per super block of ``super_block_size`` blocks
            self.pair_input_layer_norm = nn.LayerNorm(c_pair_cond, bias=pair_norm_bias)
            self.pair_logits_projection = nn.ModuleList(
                [nn.Linear(c_pair_cond, super_block_size * num_head, bias=False) for _ in range(self.num_super_blocks)])
        self.self_attention = nn.ModuleList([
            SelfAttention(c_act, c_single_cond, use_single_cond=True, kq_norm=spec.attention_kq_norm, identity_scale=spec.adaptive_identity_scale,
                          gating_query=spec.token_attention_gating_query, norm_eps=spec.adaptive_norm_eps) for _ in range(num_blocks)])
        self.transition_block = nn.ModuleList([
            conditioned_transition(spec)(c_act, c_single_cond, use_single_cond=True, identity_scale=spec.adaptive_identity_scale,
                                         norm_eps=spec.adaptive_norm_eps) for _ in range(num_blocks)])

    def _block(self, act, idx, mask, single_cond, pair_logits):
        if self.parallel:  # both deltas read the activation entering the block
            return act + self.self_attention[idx](act, mask, pair_logits, single_cond) + self.transition_block[idx](act, single_cond)
        act = act + self.self_attention[idx](act, mask, pair_logits, single_cond)
        return act + self.transition_block[idx](act, single_cond)

    def forward(self, act, mask, single_cond, pair_cond, extra_pair_bias=None):
        """[n, c], [n], [n, c_cond], [n, n, c_pair]; ``extra_pair_bias`` [n, n] is added to every head's logits of every block."""
        extra = None if extra_pair_bias is None else extra_pair_bias[None]
        if self.per_block_pair:
            for idx in range(self.num_blocks):
                logits = self.pair_logits_projection[idx](self.pair_input_layer_norm[idx](pair_cond)).permute(2, 0, 1)
                act = self._block(act, idx, mask, single_cond, logits if extra is None else logits + extra.to(logits.dtype))
            return act
        pair_act = self.pair_input_layer_norm(pair_cond)
        for sb in range(self.num_super_blocks):
            logits = self.pair_logits_projection[sb](pair_act)  # [n, n, (b h)]
            logits = logits.unflatten(-1, (self.super_block_size, self.num_head)).permute(2, 3, 0, 1)  # [b, h, n, n]
            if extra is not None:
                logits = logits + extra[None].to(logits.dtype)
            for j in range(self.super_block_size):
                act = self._block(act, sb * self.super_block_size + j, mask, single_cond, logits[j])
        return act


class CrossAttention(nn.Module):
    """The atom cross attention's PARAMETERS (the released layout): the atom-stack parity case strict-loads it; no forward here."""

    def __init__(self, key_dim: int = 128, value_dim: int = 128, c_single_cond: int = 128, num_head: int = 4, kq_norm: bool = False,
                 identity_scale: bool = False, norm_eps: float = 1e-5, gating_query: bool = True, project_output: bool = True,
                 zero_bias: bool = True) -> None:
        super().__init__()
        if kq_norm:
            self.query_layer_norm = nn.LayerNorm(key_dim)
            self.key_layer_norm = nn.LayerNorm(key_dim)
        adaln = dict(c_x=key_dim, c_single_cond=c_single_cond, use_single_cond=True, identity_scale=identity_scale, eps=norm_eps)
        self.q_adaptive_layernorm = AdaptiveLayerNorm(**adaln)
        self.k_adaptive_layernorm = AdaptiveLayerNorm(**adaln)
        self.q_projection = nn.Linear(key_dim, key_dim, bias=True)
        self.k_projection = nn.Linear(key_dim, key_dim, bias=False)
        self.v_projection = nn.Linear(value_dim, value_dim, bias=False)
        if gating_query:
            self.gating_query = nn.Linear(key_dim, value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(value_dim, value_dim, key_dim, use_single_cond=True, project=project_output, zero_bias=zero_bias)


class DiffusionCrossAttTransformer(nn.Module):
    """The atom diffusion transformer's PARAMETER layout (``cross_attention.i`` / ``transition_block.i`` / pair norm and projection); the
    parity case only strict-loads it into ``AtomDiTVariantStack``, so there is no forward."""

    def __init__(self, c_query: int = 128, c_single_cond: int = 128, c_pair_cond: int = 16, num_blocks: int = 3, num_head: int = 4,
                 spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()
        self.num_blocks, self.num_head = num_blocks, num_head
        self.per_block_pair = spec.per_block_atom_pair_layer_norm
        self.parallel = spec.parallel_attention_transition
        pair_norm_bias = "pair_input_layer_norm" in spec.affine_norms
        if self.per_block_pair:
            self.pair_input_layer_norm = nn.ModuleList([nn.LayerNorm(c_pair_cond, bias=pair_norm_bias) for _ in range(num_blocks)])
            self.pair_logits_projection = nn.ModuleList([nn.Linear(c_pair_cond, num_head, bias=False) for _ in range(num_blocks)])
        else:
            self.pair_input_layer_norm = nn.LayerNorm(c_pair_cond, bias=pair_norm_bias)
            self.pair_logits_projection = nn.Linear(c_pair_cond, num_blocks * num_head, bias=False)
        identity = spec.adaptive_identity_scale if spec.atom_adaptive_identity_scale is None else spec.atom_adaptive_identity_scale
        self.cross_attention = nn.ModuleList([
            CrossAttention(num_head=num_head, kq_norm=spec.attention_kq_norm, identity_scale=identity,
                           gating_query=spec.atom_attention_gating_query, project_output=spec.atom_attention_project_output,
                           zero_bias=spec.atom_adaptive_zero_bias, norm_eps=spec.adaptive_norm_eps) for _ in range(num_blocks)])
        self.transition_block = nn.ModuleList([
            conditioned_transition(spec)(c_query, c_single_cond, use_single_cond=True, identity_scale=identity,
                                         zero_bias=spec.atom_adaptive_zero_bias, norm_eps=spec.adaptive_norm_eps) for _ in range(num_blocks)])


# ------------------------------------------------------------------------------------------------------------------- pairformer
class PairformerBlock(nn.Module):
    """One Pairformer block (AF3 Algorithm 17): pair track (two triangle multiplications, two triangle attentions, transition) and, with
    ``with_single``, the single track (attention over the pair logits, transition)."""

    def __init__(self, n_heads: int = 16, c_pair: int = 128, c_single: int = 384, c_hidden_mul: int = 128, n_heads_pair: int = 4,
                 num_intermediate_factor: int = 4, with_single: bool = True, with_pair_attention: bool = True,
                 spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()
        self.n_heads, self.with_single, self.num_intermediate_factor = n_heads, with_single, num_intermediate_factor
        self.triangle_multiplication_outgoing = TriangleMultiplication(c_pair, True, divide_by_length=spec.triangle_mul_divide_by_length)
        self.triangle_multiplication_incoming = TriangleMultiplication(c_pair, False, divide_by_length=spec.triangle_mul_divide_by_length)
        self.with_pair_attention = with_pair_attention
        if with_pair_attention:
            self.pair_attention1 = GridSelfAttention(c_pair, n_heads_pair, transpose=False, spec=spec)
            self.pair_attention2 = GridSelfAttention(c_pair, n_heads_pair, transpose=True, spec=spec)
        self.pair_transition = Transition(c_pair, num_intermediate_factor)
        self.c_single = c_single
        self.parallel = spec.parallel_pairformer_block  # every update reads the representation ENTERING the block and the deltas are summed
        if with_single:
            self.single_pair_logits_norm = nn.LayerNorm(c_pair)
            self.single_pair_logits_projection = nn.Linear(c_pair, n_heads, bias=False)
            self.single_attention_ = SelfAttention(c_x=c_single, num_head=n_heads, use_single_cond=False, gate_bias=spec.single_attention_gate_bias)
            self.single_transition = Transition(c_single, num_intermediate_factor)

    def forward(self, pair, pair_mask, single=None, seq_mask=None, extra_pair_bias=None):
        """[L, L, C], [L, L], [L, c_single], [L] -> ``(pair, single)`` (``pair`` alone without the single track). ``extra_pair_bias`` [L, L] is
        added to the single attention's pair logits on every head."""
        pair_in = pair
        if self.parallel:
            updates = self.triangle_multiplication_outgoing(pair_in, pair_mask) + self.triangle_multiplication_incoming(pair_in, pair_mask)
            if self.with_pair_attention:
                updates = updates + self.pair_attention1(pair_in, pair_mask) + self.pair_attention2(pair_in, pair_mask)
            pair = pair_in + updates + self.pair_transition(pair_in)
        else:
            pair = triangle_residual(self.triangle_multiplication_outgoing, pair, pair_mask)
            pair = triangle_residual(self.triangle_multiplication_incoming, pair, pair_mask)
            if self.with_pair_attention:
                pair = pair + self.pair_attention1(pair, pair_mask)
                pair = pair + self.pair_attention2(pair, pair_mask)
            pair = pair + self.pair_transition(pair)
        if not self.with_single:
            return pair
        if single is None or seq_mask is None:
            raise ValueError("Single-track pairformer requires single features and a mask")
        logits_source = pair_in if self.parallel else pair  # a parallel block's single reads the pair ENTERING it
        pair_logits = self.single_pair_logits_projection(self.single_pair_logits_norm(logits_source)).permute(2, 0, 1)
        if extra_pair_bias is not None:
            pair_logits = pair_logits + extra_pair_bias[None].to(pair_logits.dtype)
        update = self.single_attention_(single, seq_mask, pair_logits=pair_logits)
        if self.parallel:
            return pair, single + update + self.single_transition(single)
        single = single + update
        return pair, single + self.single_transition(single)
