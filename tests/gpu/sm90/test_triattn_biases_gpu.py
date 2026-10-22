"""RoseTTAFold3's TriangleAttention (a bias on the gate and on the out-projection) on the H100 sm90 kernels, inference: against the
module's fp64 torch reference, and that reference (biases zero) against the engine's own PyTorch path."""

import copy

import pytest
import torch

from miniworld_engine.integrations import triattn_h100_sm90
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]


def _module(starting, d_pair=128, n_head=4, d_hidden=None, biases=True):
    torch.manual_seed(0)
    m = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, transposed_bias=False, gate_bias=biases,
                          out_bias=biases, implementation=ImplementationType.MINIWORLD, p_drop=0.0).cuda().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name == "ln_pair.weight" else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize(("b", "n", "real", "d_pair"), [(1, 256, 256, 128), (2, 384, 300, 128), (1, 256, 200, 64)])
def test_biased_triattn_h100(starting, b, n, real, d_pair):
    m = _module(starting, d_pair=d_pair)
    g = torch.Generator(device="cuda").manual_seed(1)
    z = torch.randn(b, n, n, d_pair, device="cuda", generator=g)
    mask = (torch.arange(n, device="cuda") < real).expand(b, n).contiguous()
    gold64 = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    zb = z.to(torch.bfloat16)
    with torch.no_grad():
        assert triattn_h100_sm90.serves(eng, zb, mask)
        gold = z.double() + gold64._biased_reference(z.double(), mask)
        base = zb + ref16._biased_reference(zb, mask)
        got = eng(zb, mask)
    rows = (slice(None), slice(0, real), slice(0, real))
    err, err16 = _rel(got[rows], gold[rows]), _rel(base[rows], gold[rows])
    assert torch.isfinite(got[rows]).all()
    assert err <= max(1.5 * err16, 2e-3), f"engine {err:.2e} vs bf16 reference {err16:.2e}"


@pytest.mark.parametrize("starting", [True, False])
def test_biased_reference_matches_the_engine_reference(starting):
    """With zero biases the biased reference is the module's own math (the ending node on the pair-as-is bias included)."""
    m = _module(starting)
    with torch.no_grad():
        m.to_gate.bias.zero_()
        m.to_out.bias.zero_()
    plain = TriangleAttention(128, 4, starting=starting, transposed_bias=False, implementation=ImplementationType.PYTORCH,
                              p_drop=0.0).cuda().eval()
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.endswith(".bias") or k == "ln_pair.bias"})
    z = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.float64)
    mask = (torch.arange(128, device="cuda") < 100)[None]
    with torch.no_grad():
        ours = z + m.double()._biased_reference(z, mask)
        theirs = plain.double()(z, mask)
    rows = (slice(None), slice(0, 100), slice(0, 100))
    torch.testing.assert_close(ours[rows], theirs[rows], rtol=1e-6, atol=1e-7)


def test_attention_pair_bias_gate_bias():
    """Chai-1's single attention: ``gate_bias`` adds a constant to the gate logits (the same as a to_gate bias of that value)."""
    from miniworld_engine.modules.attention_pair_bias import AttentionPairBias

    torch.manual_seed(0)
    m = AttentionPairBias(384, 256, 16, 24, gate_bias=1.0, implementation=ImplementationType.MINIWORLD).cuda().eval()
    ref = AttentionPairBias(384, 256, 16, 24, implementation=ImplementationType.PYTORCH).cuda().eval()
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 1 + 0.1 * torch.randn_like(p))
        ref.load_state_dict(m.state_dict())
        ref.to_gate = torch.nn.Linear(384, 384).cuda()
        ref.to_gate.weight.copy_(m.to_gate.weight)
        ref.to_gate.bias.fill_(1.0)
        s = torch.randn(1, 256, 384, device="cuda")
        z = torch.randn(1, 256, 256, 256, device="cuda")
        mask = (torch.arange(256, device="cuda") < 200)[None]
        torch.testing.assert_close(m(s, z, mask), ref(s, z, mask), rtol=1e-5, atol=1e-5)
