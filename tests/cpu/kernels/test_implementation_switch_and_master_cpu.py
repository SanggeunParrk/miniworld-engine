# ruff: noqa: PT018, C408
"""``set_implementation`` (a live module switches its implementation) and fp32-master weights (``set_master_weights`` / ``master_weights=True``), CPU.

* every pair-track module, ``PairformerBlock`` / ``Pairformer`` and the DiT variant stacks: after ``set_implementation(B)`` the module and every child
  hold the fields a module built with ``B`` holds (``implementation``, ``_backend``, ``_train_gate``, the norms' options), weights untouched;
  a refused option restores everything;
* fp32 master: the pin survives ``.bfloat16()``, propagates to every child, the constructor flag, an fp32 checkpoint loads exactly, a Transition
  with bf16 activations equals its bf16-cast copy, its gradients land fp32 on the master; ``h100_master.cached_pack`` is cached while the master is
  unchanged and made again after the optimizer bumps its version.

The H100 kernel side (fwd + bwd against the pure-bf16 module, pack invalidation after an optimizer step): ``tests/gpu/sm90/test_h100_master_mode_gpu.py``.
"""
from __future__ import annotations

import pytest
import torch

from miniworld_engine.modules.attention_pair_bias import AttentionPairBias
from miniworld_engine.modules.dit_variant import AtomDiTVariantStack, DiTVariantStack
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.modules.pairformer import (
    Pairformer,
    PairformerBlock,
    PairformerConfig,
)
from miniworld_engine.modules.primitives import LayerNorm, RMSNorm
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_attention.bidirectional import (
    BidirectionalTriangleAttention,
)
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_engine.utils import h100_master

_CFG = dict(d_pair=16, d_hidden_tri_multi=16, d_hidden_tri_attention=4, n_head_tri_attention=2, n_block=2, p_drop=0.0)

#: name -> factory(implementation, **kw): small modules, every pair-track module and the composites
FACTORIES = {
    "trimul": lambda impl, **kw: TriangleMultiplication(16, implementation=impl, **kw),
    "trimul_ln_miniworld": lambda impl, **kw: TriangleMultiplication(16, implementation=impl, ln_implementation=IT.MINIWORLD, **kw),
    "trimul_bidir": lambda impl, **kw: BidirectionalTriangleMultiplication(16, implementation=impl, **kw),
    "triattn": lambda impl, **kw: TriangleAttention(16, 2, d_hidden=16, implementation=impl, **kw),
    "triattn_bidir": lambda impl, **kw: BidirectionalTriangleAttention(16, 2, implementation=impl, **kw),
    "transition": lambda impl, **kw: Transition(16, 2, implementation=impl, **kw),
    "opm": lambda impl, **kw: OuterProductMean(8, 16, 4, implementation=impl, **kw),
    "pwa": lambda impl, **kw: MSAPairWeightedAveraging(8, 16, 2, implementation=impl, **kw),
    "apb": lambda impl, **kw: AttentionPairBias(16, 16, 2, implementation=impl, **kw),
    "apb_qknorm": lambda impl, **kw: AttentionPairBias(16, 16, 2, use_qk_norm=True, implementation=impl, **kw),
    "layernorm": lambda impl, **kw: LayerNorm(16, implementation=impl),
    "rmsnorm": lambda impl, **kw: RMSNorm(16, implementation=impl),
    "block": lambda impl, **kw: PairformerBlock(PairformerConfig(**_CFG), implementation=impl, **kw),
    "block_single": lambda impl, **kw: PairformerBlock(PairformerConfig(**_CFG, d_single=16, n_head_single=2), implementation=impl, **kw),
    "block_bidir": lambda impl, **kw: PairformerBlock(PairformerConfig(**_CFG, bidirectional_trimul=True), implementation=impl, **kw),
    "pairformer": lambda impl, **kw: Pairformer(PairformerConfig(**_CFG), implementation=impl, **kw),
}
SWITCH_ONLY = {"layernorm", "rmsnorm"}  # no master weights of their own (the LayerNorm affine is always fp32)
PAIRS = [(IT.PYTORCH, IT.MINIWORLD), (IT.MINIWORLD, IT.PYTORCH), (IT.PYTORCH, IT.CUEQUIVARIANCE), (IT.MINIWORLD, IT.CUEQUIVARIANCE)]
_FIELDS = ("implementation", "_backend", "_native_cuda_route", "_returns_input_dtype", "ln_implementation")


def _gate(module):
    gate = getattr(module, "_train_gate", None)
    return None if gate is None else (gate.op, gate.name, gate.capability, gate.holes)


def _state(module: torch.nn.Module) -> dict:
    return {
        name: ({f: getattr(m, f) for f in _FIELDS if hasattr(m, f)}, _gate(m) if hasattr(m, "_train_gate") else "n/a")
        for name, m in module.named_modules()
    }


# ------------------------------------------------------------------------------------------------------------ set_implementation
@pytest.mark.parametrize(("start", "target"), PAIRS)
@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_set_implementation_equals_a_module_built_with_it(name, start, target):
    """Switching a live module leaves every field of the module and of its children as the constructor would have; the weights are the same tensors."""
    live = FACTORIES[name](start)
    params = dict(live.named_parameters())
    assert live.set_implementation(target) is live
    assert _state(live) == _state(FACTORIES[name](target)), name
    assert live.implementation == target
    assert all(live.get_parameter(n) is p for n, p in params.items())


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_set_implementation_round_trip_restores_the_state(name):
    live = FACTORIES[name](IT.PYTORCH)
    before = _state(live)
    live.set_implementation(IT.MINIWORLD).set_implementation("pytorch")
    assert _state(live) == before


def test_children_follow_the_constructor_rules():
    block = PairformerBlock(PairformerConfig(**_CFG, d_single=16, n_head_single=2), implementation=IT.PYTORCH)
    block.set_implementation(IT.MINIWORLD)
    for sub in (block.tri_multi_outgoing, block.tri_multi_incoming, block.tri_atten_starting, block.tri_atten_ending,
                block.transition_pair, block.single_attention, block.single_transition):
        assert sub.implementation == IT.MINIWORLD
    # the Transition / AttentionPairBias norms follow the module; the TriMul norms follow ``ln_implementation``; the TriangleAttention ones nothing
    assert block.transition_pair.ln_in.implementation == IT.MINIWORLD
    assert block.single_attention.ln_pair.implementation == IT.MINIWORLD
    assert block.tri_multi_outgoing.ln_pair.implementation == IT.PYTORCH
    assert block.tri_atten_starting.ln_pair.implementation == IT.PYTORCH
    # an ANTHROPIC OPM builds its norms with MINIWORLD
    opm = OuterProductMean(8, 16, 4, implementation=IT.PYTORCH).set_implementation(IT.ANTHROPIC)
    assert opm.implementation == IT.ANTHROPIC and opm.ln_msa.implementation == IT.MINIWORLD
    # ln_implementation is the TriMul's own: the norms switch with it, not with the module
    tm = TriangleMultiplication(16, implementation=IT.PYTORCH, ln_implementation=IT.MINIWORLD).set_implementation(IT.PYTORCH)
    assert tm.ln_pair.implementation == IT.MINIWORLD


def test_a_refused_option_restores_everything():
    """ANTHROPIC has no Transition payload: the block's switch fails at the Transition and leaves no module half switched."""
    block = PairformerBlock(PairformerConfig(**_CFG), implementation=IT.PYTORCH)
    before = _state(block)
    with pytest.raises(ValueError, match="anthropic"):
        block.set_implementation(IT.ANTHROPIC)
    assert _state(block) == before
    assert block.implementation == IT.PYTORCH
    with pytest.raises(ValueError, match="anthropic"):
        Transition(16, 2).set_implementation(IT.ANTHROPIC)


def test_triangle_attention_keeps_its_anthropic_contract():
    ta = TriangleAttention(16, 2, d_hidden=16, use_self_attention=False, implementation=IT.PYTORCH)
    with pytest.raises(ValueError, match="use_self_attention"):
        ta.set_implementation(IT.ANTHROPIC)
    assert ta.implementation == IT.PYTORCH


def test_a_switched_pytorch_module_computes_the_same():
    """Switching back to PYTORCH gives the PyTorch module's output bit for bit (CPU: the kernel families have nothing to run)."""
    torch.manual_seed(0)
    live = TriangleMultiplication(16, implementation=IT.MINIWORLD, p_drop=0.0)
    ref = TriangleMultiplication(16, implementation=IT.PYTORCH, p_drop=0.0)
    ref.load_state_dict(live.state_dict())
    live.set_implementation(IT.PYTORCH)
    pair = torch.randn(1, 6, 6, 16)
    assert torch.equal(live(pair), ref(pair))


@pytest.mark.parametrize("cls", [DiTVariantStack, AtomDiTVariantStack])
def test_variant_stacks_switch(cls):
    stack = cls(2, implementation=IT.MINIWORLD) if cls is AtomDiTVariantStack else cls(2, 32, 16, 16, 2, implementation=IT.MINIWORLD)
    assert stack.set_implementation("pytorch") is stack
    assert stack.implementation is IT.PYTORCH
    assert stack.serving() == "pytorch"


# ------------------------------------------------------------------------------------------------------------ master weights
@pytest.mark.parametrize("name", sorted(set(FACTORIES) - SWITCH_ONLY))
def test_master_pin_survives_a_bf16_cast_and_reaches_every_child(name):
    module = FACTORIES[name](IT.PYTORCH, master_weights=True)
    assert module.master_weights is True
    assert all(p.dtype == torch.float32 for p in module.parameters())
    assert all(m.master_weights for m in module.modules() if hasattr(m, "master_weights"))
    module.bfloat16()
    assert all(p.dtype == torch.float32 for p in module.parameters()), "a bulk cast rounded a master"
    module.half()
    assert all(p.dtype == torch.float32 for p in module.parameters())


def test_default_is_unchanged_and_the_flag_can_be_set_later():
    tm = TriangleMultiplication(16)
    assert tm.master_weights is False
    tm.bfloat16()
    assert tm.to_left.weight.dtype == torch.bfloat16 and tm.ln_pair.weight.dtype == torch.float32  # the registry layout
    fresh = TriangleMultiplication(16)
    same = next(fresh.parameters())
    assert fresh.set_master_weights() is fresh
    assert fresh.master_weights
    assert next(fresh.parameters()) is same  # the same Parameter object: an optimizer built before stays valid
    fresh.bfloat16()
    assert fresh.to_left.weight.dtype == torch.float32
    fresh.set_master_weights(False)
    fresh.bfloat16()
    assert fresh.to_left.weight.dtype == torch.bfloat16


def test_master_is_set_by_the_parent_on_the_whole_stack():
    stack = Pairformer(PairformerConfig(**_CFG)).bfloat16()
    stack.set_master_weights()
    assert all(p.dtype == torch.float32 for p in stack.parameters())
    stack.bfloat16()
    assert all(p.dtype == torch.float32 for p in stack.parameters())


def test_fp32_checkpoint_loads_exactly_into_a_master_module():
    src = Transition(16, 2, implementation=IT.PYTORCH).float()
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    master = Transition(16, 2, implementation=IT.PYTORCH, master_weights=True)
    master.load_state_dict(src.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(src.parameters(), master.parameters(), strict=True))
    bf16 = Transition(16, 2, implementation=IT.PYTORCH)
    bf16.load_state_dict(src.state_dict())
    assert bf16.expand_a.weight.dtype == torch.bfloat16  # rounded: why a master exists


def test_transition_master_equals_the_bf16_cast_copy_and_grads_are_fp32():
    torch.manual_seed(1)
    master = Transition(16, 2, implementation=IT.PYTORCH, master_weights=True)
    with torch.no_grad():
        for p in master.parameters():
            p.normal_(std=0.3)
    cast = Transition(16, 2, implementation=IT.PYTORCH)
    cast.load_state_dict(master.state_dict())  # the master rounded to bf16 matrices, fp32 norm affine
    x = torch.randn(2, 5, 16).bfloat16()
    got, want = master(x.clone().requires_grad_()), cast(x.clone().requires_grad_())
    torch.testing.assert_close(got.float(), want.float(), atol=1e-6, rtol=0)
    got.float().square().sum().backward()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in master.parameters())


# ------------------------------------------------------------------------------------------------------------ cached packs
def test_cached_pack_is_cached_and_invalidated_by_the_optimizer():
    w = [torch.nn.Parameter(torch.randn(8, 8)), torch.nn.Parameter(torch.randn(4, 8))]
    first = h100_master.cached_pack("test_cpu", w)
    assert all(t.dtype == torch.bfloat16 for t in first)
    assert h100_master.cached_pack("test_cpu", w)[0] is first[0], "a second call re-cast an unchanged master"
    opt = torch.optim.SGD(w, lr=0.1)
    (w[0].sum() + w[1].sum()).backward()
    opt.step()
    second = h100_master.cached_pack("test_cpu", w)
    assert second[0] is not first[0]
    torch.testing.assert_close(second[0].float(), w[0].detach().bfloat16().float(), atol=0, rtol=0)
    with torch.no_grad():
        w[1].add_(1.0)  # an in-place update bumps the version too
    assert h100_master.cached_pack("test_cpu", w)[1] is not second[1]
    assert second[0].grad_fn is None
    assert not second[0].requires_grad
