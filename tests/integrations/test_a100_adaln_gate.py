"""The A100 AdaLN / ConditionedTransition gates (``integrations/adaln_sm80.py``, ``integrations/conditioned_transition_sm80.py``): CPU-runnable, they read shapes, dtypes, devices and
the switches only.  The kernels' numerics are ``test_a100_adaln_gpu.py`` / ``test_a100_conditioned_transition_gpu.py``."""

import pytest
import torch

from miniworld_engine.integrations import adaln_sm80, conditioned_transition_sm80
from miniworld_engine.modules import AdaptiveLayerNorm, ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType

MW = ImplementationType.MINIWORLD
BF = torch.bfloat16


def test_period_is_the_rows_per_conditioning_row():
    x = torch.empty(5, 1, 7, 8)
    per_sample = torch.empty(5, 1, 7, 4)
    shared = torch.empty(1, 1, 7, 4).expand(5, 1, 7, 4)
    one_row = torch.empty(1, 1, 7, 4)
    period = adaln_sm80._period
    assert period(x, per_sample, False) == 35
    assert period(x, per_sample, True) == 35
    assert period(x, shared, False) == 7                       # no gradient: the L rows of one sample are the table
    assert period(x, shared, True) == 35                       # gradient: one conditioning per row (autograd sums the expanded gradient)
    assert period(x, one_row, False) == 7                      # a size-1 leading dim broadcasts like the PyTorch module
    assert period(x, one_row, True) is None                    # ... which the training path does not take
    assert period(x, torch.empty(5, 1, 6, 4), False) is None   # a conditioning that does not describe the rows
    assert period(x, torch.empty(4, 1, 7, 4), False) is None
    assert period(torch.empty(3, 8), torch.empty(3, 4), False) == 3
    assert period(torch.empty(1, 7, 8), torch.empty(1, 7, 4), False) == 7


@pytest.mark.parametrize(("d", "dc"), [(128, 128), (768, 384), (768, 768)])
def test_cpu_tensors_are_never_served(d, dc):
    x, cond = torch.randn(2, 1, 4, d), torch.randn(2, 1, 4, dc)
    ada = AdaptiveLayerNorm(d, dc, implementation=MW)
    ct = ConditionedTransition(d, dc, 2, implementation=MW)
    for grad in (False, True):
        assert not adaln_sm80.serves(ada, x, cond, torch.float32, grad)
        assert not conditioned_transition_sm80.serves(ct, x, cond, torch.float32, grad)


def test_the_widths_are_the_registry_widths():
    assert adaln_sm80.WIDTHS == (128, 384, 768)
    assert conditioned_transition_sm80.WIDTHS == adaln_sm80.WIDTHS


def test_the_atom_gate_reads_dtype_and_width_only():
    pytest.importorskip("miniworld_engine.kernels.adaln.cuda.sm80")
    from miniworld_engine.kernels.adaln.cuda import sm80 as ka
    from miniworld_engine.kernels.conditioned_transition.cuda import sm80 as kt

    x, c = torch.empty(4, 128, dtype=BF), torch.empty(4, 128, dtype=BF)
    assert ka.atom_supported(x, c)
    assert kt.atom_supported(x, c, torch.empty(256, 128))
    assert not ka.atom_supported(x.float(), c.float())                              # fp32 has its own (TF32) kernels
    assert not ka.atom_supported(torch.empty(4, 768, dtype=BF), torch.empty(4, 384, dtype=BF))
    assert not kt.atom_supported(x, c, torch.empty(512, 128))                       # expansion 4 is not the fused tail's contract
    xf, cf = x.float(), c.float()
    assert ka.atom_tf32_supported(xf, cf)                                           # fp32 at the atom width: the fused TF32 kernels (inference)
    assert kt.atom_tf32_supported(xf, cf, torch.empty(256, 128))
    assert not ka.atom_tf32_supported(x, c)                                         # bf16 has its own kernels
    assert not ka.atom_tf32_supported(torch.empty(4, 768), torch.empty(4, 384))
    assert not kt.atom_tf32_supported(xf, cf, torch.empty(512, 128))
