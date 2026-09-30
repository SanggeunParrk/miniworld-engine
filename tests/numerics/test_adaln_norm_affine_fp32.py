"""AdaptiveLayerNorm keeps its norm affine (ln_cond.weight) in fp32 under any dtype cast.

A bf16 gamma at 1.0 cannot move by one Adam step, so a DiT cast to bf16 must leave every norm
affine in fp32 (primitives._Fp32ParamsMixin). ConditionedTransition reaches the same parameter
through its own AdaptiveLayerNorm.
"""
import pytest
import torch

from miniworld_engine.modules import AdaptiveLayerNorm, ConditionedTransition


@pytest.mark.parametrize("build", ["cast", "constructed"])
def test_ln_cond_weight_stays_fp32(build):
    if build == "cast":
        ln = AdaptiveLayerNorm(16, 8, implementation="pytorch").to(torch.bfloat16)
    else:
        ln = AdaptiveLayerNorm(16, 8, implementation="pytorch", dtype=torch.bfloat16)
    assert ln.ln_cond.weight.dtype == torch.float32
    assert ln.to_scale.weight.dtype == torch.bfloat16
    assert ln.to_bias.weight.dtype == torch.bfloat16


def test_conditioned_transition_norm_affine_stays_fp32():
    tr = ConditionedTransition(16, 8, 2, implementation="pytorch").to(torch.bfloat16)
    assert tr.ada_ln_in.ln_cond.weight.dtype == torch.float32


def test_pytorch_path_runs_bf16_and_matches_fp32():
    torch.manual_seed(0)
    ln = AdaptiveLayerNorm(16, 8, implementation="pytorch")
    with torch.no_grad():
        ln.ln_cond.weight.uniform_(0.5, 1.5)
        ln.to_bias.weight.normal_()
    x, cond = torch.randn(4, 16), torch.randn(4, 8)
    expected = ln(x, cond)
    ln_bf16 = ln.to(torch.bfloat16)
    actual = ln_bf16(x.bfloat16(), cond.bfloat16())
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)
    actual.float().sum().backward()
    assert ln_bf16.ln_cond.weight.grad.dtype == torch.float32
