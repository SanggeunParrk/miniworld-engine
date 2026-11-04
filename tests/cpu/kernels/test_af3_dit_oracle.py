"""AlphaFold 3's own diffusion stacks as ``DiTVariantStack`` / ``AtomDiTVariantStack`` configurations, against a FROZEN oracle.

``tests/oracles/af3_dit_oracle.pt`` holds seeded weights (FoldForge's released parameter names), inputs, and the forward output and input
gradients that FoldForge's dense ``DiffusionTransformer`` / ``DiffusionCrossAttTransformer`` (spec ``ALPHAFOLD3``) produced, once, in fp32 CPU
(``tests/oracles/make_af3_dit_oracle.py``). The test imports no FoldForge: it builds the engine stacks from the released names
(``from_released``, the PyTorch implementation) and compares the real rows' output and the gradients of the inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from miniworld_engine.modules.dit_variant import (
    AF3,
    AF3_ATOM,
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.exceptions import ImplementationType

ORACLE = Path(__file__).resolve().parents[2] / "oracles" / "af3_dit_oracle.pt"
pytestmark = pytest.mark.skipif(not ORACLE.exists(), reason="tests/oracles/af3_dit_oracle.pt missing")


@pytest.fixture(scope="module")
def oracle():
    return torch.load(ORACLE, map_location="cpu", weights_only=True)


def _state(case):
    return {k: v.float() for k, v in case["state"].items()}


def _rel(got, want):
    return ((got.double() - want.double()).norm() / want.double().norm()).item()


def _run(stack, case, inputs):
    leaves = [t.float().clone().requires_grad_() for t in inputs]
    n = case["n_real"]
    out = stack(*leaves, case["mask"])
    (out[:, :n] * case["dy"].float()[:, :n]).sum().backward()
    return out, leaves


def test_token_stack_is_the_af3_configuration(oracle):
    case = oracle["token"]
    stack = DiTVariantStack.from_released(_state(case), n_head=case["n_head"], parallel=False, super_block_size=case["super_block_size"],
                                          dtype=torch.float32)
    block = stack.blocks[0]
    assert (block.attn_adaln.identity_scale, block.gating, block.kq_norm, block.up_gate) == (
        AF3["identity_scale"], AF3["gating"], AF3["kq_norm"], AF3["up_gate"])
    assert stack.parallel == AF3["parallel"]
    assert block.pair_norm.bias is None
    # the preset builds the same parameter set the weights show
    d, d_cond, d_pair = block.to_q.weight.shape[0], block.attn_zero.weight.shape[1], block.pair_norm.weight.shape[0]
    preset = DiTVariantStack(len(stack.blocks), d, d_cond, d_pair, case["n_head"], 2,
                             **{k: v for k, v in AF3.items() if k not in ("d_cond", "d_pair")})
    assert {k: v.shape for k, v in preset.state_dict().items()} == {k: v.shape for k, v in stack.state_dict().items()}
    assert stack.serving(case["act"].float(), case["cond"].float(), case["pair"].float(), case["mask"]) == "pytorch"


@pytest.mark.parametrize("grad", [False, True])
def test_token_stack_matches_the_dense_oracle(oracle, grad):
    case = oracle["token"]
    stack = DiTVariantStack.from_released(_state(case), n_head=case["n_head"], parallel=False, super_block_size=case["super_block_size"],
                                          implementation=ImplementationType.PYTORCH, dtype=torch.float32)
    n = case["n_real"]
    if not grad:
        with torch.no_grad():
            out = stack(case["act"].float(), case["cond"].float(), case["pair"].float(), case["mask"])
        assert _rel(out[:, :n], case["out"][:, :n]) < 1e-4
        return
    out, (act, cond, pair) = _run(stack, case, (case["act"], case["cond"], case["pair"]))
    assert _rel(out[:, :n], case["out"][:, :n]) < 1e-4
    assert _rel(act.grad[:, :n], case["grad_act"][:, :n]) < 1e-3
    assert _rel(cond.grad, case["grad_cond"]) < 1e-3
    assert _rel(pair.grad, case["grad_pair"]) < 1e-3


def test_atom_stack_is_the_af3_configuration(oracle):
    case = oracle["atom"]
    stack = AtomDiTVariantStack.from_released(_state(case), parallel=False, key_window="slide", dtype=torch.float32)
    block = stack.blocks[0]
    assert (block.q_adaln.identity_scale, block.gating, block.project_output, block.kq_norm, block.up_gate) == (
        AF3_ATOM["identity_scale"], AF3_ATOM["gating"], AF3_ATOM["project_output"], AF3_ATOM["kq_norm"], AF3_ATOM["up_gate"])
    assert (stack.parallel, stack.per_block_pair_norm) == (AF3_ATOM["parallel"], AF3_ATOM["per_block_pair_norm"])
    d, d_cond, d_pair = block.to_q.weight.shape[0], block.attn_zero.weight.shape[1], stack.pair_norm.weight.shape[0]
    preset = AtomDiTVariantStack(len(stack.blocks), d, d_cond, d_pair, 2, key_window="slide", **AF3_ATOM)
    assert {k: v.shape for k, v in preset.state_dict().items()} == {k: v.shape for k, v in stack.state_dict().items()}


@pytest.mark.parametrize("grad", [False, True])
def test_atom_stack_matches_the_dense_oracle(oracle, grad):
    """The slid key window (160 atoms, 150 real: the last windows are shifted back inside the real atoms), batch of two samples."""
    case = oracle["atom"]
    stack = AtomDiTVariantStack.from_released(_state(case), parallel=False, key_window="slide", implementation=ImplementationType.PYTORCH,
                                              dtype=torch.float32)
    n = case["n_real"]
    if not grad:
        with torch.no_grad():
            out = stack(case["act"].float(), case["cond"].float(), case["pair"].float(), case["mask"])
        assert _rel(out[:, :n], case["out"][:, :n]) < 1e-4
        return
    out, (act, cond, pair) = _run(stack, case, (case["act"], case["cond"], case["pair"]))
    assert _rel(out[:, :n], case["out"][:, :n]) < 1e-4
    assert _rel(act.grad[:, :n], case["grad_act"][:, :n]) < 1e-3
    assert _rel(cond.grad, case["grad_cond"]) < 1e-3
    assert _rel(pair.grad, case["grad_pair"].float()) < 2e-3      # the stored oracle gradient is fp16
