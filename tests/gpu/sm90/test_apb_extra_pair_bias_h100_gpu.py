"""AttentionPairBias(extra_pair_bias=) on H100 (the structural-token refiner's additive term on the pair-bias logits).

The native CUDA paths (kernels/integrations/attention_pair_bias_b200.py) compute the bias from ``pair`` inside the kernel and cannot take the
term, so a call with it must skip them (asserted: the native entry points are not called, and are called without the term) and run the attention
on the summed bias. The bar is the guide's: output and gradients no worse than 1.3x the bf16 PyTorch module's error against the fp32 module.
Only declared shapes (kernels/registry/registry_module.csv, attention_pair_bias rows: 8 heads d_pair 128, train + eval; the Chai-1 rows 16 heads
d_pair 256 / 128, gate_bias 1, eval); ``extra_pair_bias`` is an input option, not a shape.
"""

import copy

import pytest
import torch

from miniworld_engine.kernels.integrations import attention_pair_bias_b200 as native
from miniworld_engine.modules import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

# (heads, d_single, d_pair, gate_bias)
CASES = [(8, 384, 128, 0.0), (16, 384, 256, 1.0), (16, 384, 128, 1.0)]


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _reference(case, seed=0):
    heads, d, dp, gb = case
    torch.manual_seed(seed)
    ref = AttentionPairBias(d, dp, heads, gate_bias=gb, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():
        for p in ref.parameters():  # non-default parameters: the zero-initialised to_out and unit LayerNorms move
            p.add_(torch.randn_like(p) * 0.1)
    return ref.cuda()


def _native(ref, master):
    ours = copy.deepcopy(ref)
    ours.implementation = ImplementationType.MINIWORLD
    return ours if master else ours.bfloat16()


def _inputs(L, case, masked, seed=1):
    heads, d, dp, _ = case
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(1, L, d, device="cuda", generator=g)
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g)
    extra = 2 * torch.randn(1, heads, L, L, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
        mask[0, 0] = True
    return single, pair, mask, extra


@pytest.fixture
def calls(monkeypatch):
    """Counts the native CUDA entry points the module calls."""
    n = {"infer": 0, "train": 0}
    infer, train = native.update_inference, native.update_train

    def update_inference(*a, **k):
        n["infer"] += 1
        return infer(*a, **k)

    def update_train(*a, **k):
        n["train"] += 1
        return train(*a, **k)

    monkeypatch.setattr(native, "update_inference", update_inference)
    monkeypatch.setattr(native, "update_train", update_train)
    return n


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c[0]}p{c[2]}g{c[3]}")
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("L", [128, 384])
@pytest.mark.parametrize("master", [False, True], ids=["bf16params", "fp32master"])
def test_extra_pair_bias_inference(case, masked, L, master, calls):
    ref = _reference(case)
    single, pair, mask, extra = _inputs(L, case, masked)
    sb, pb, eb = single.bfloat16(), pair.bfloat16(), extra.bfloat16()
    ours = _native(ref, master)
    torch_bf = copy.deepcopy(ref)
    if not master:
        torch_bf = torch_bf.bfloat16()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
        assert native.serves_inference(ours, sb, pb, mask), "the call is one the native CUDA path serves without the term"
        ours(sb, pb, mask)
        assert calls["infer"] == 1
        y_ours = ours(sb, pb, mask, eb)
        assert calls["infer"] == 1, "a call with extra_pair_bias must skip the native path (it cannot take the term)"
        y_ref = ref(single, pair, mask, extra)
        y_bf = torch_bf(sb, pb, mask, eb)
    assert torch.isfinite(y_ours).all()
    assert _rel(y_ours, y_ref) < 1.3 * _rel(y_bf, y_ref) + 1e-3, (_rel(y_ours, y_ref), _rel(y_bf, y_ref))
    # the term moves the output (it is not silently ignored)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
        y_plain = ours(sb, pb, mask)
    assert _rel(y_ours, y_plain) > 5 * _rel(y_ours, y_ref)


@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("L", [128, 384])
def test_extra_pair_bias_training(masked, L, calls):
    """The training path (d_pair 128, 8 heads: the native train kernels serve it without the term): output, input and parameter gradients, and
    the gradient of the term itself."""
    case = CASES[0]
    ref = _reference(case)
    single, pair, mask, extra = _inputs(L, case, masked)
    ours = _native(ref, False)
    torch_bf = copy.deepcopy(ref).bfloat16()

    def run(module, dtype):
        s, p, e = (t.to(dtype).clone().requires_grad_(True) for t in (single, pair, extra))
        module.zero_grad()
        y = module(s, p, mask, e)
        (y.float()[:, : L - 8].square().sum()).backward()
        return y, [s.grad, p.grad, e.grad, module.to_bias.weight.grad, module.to_query.weight.grad]

    with torch.enable_grad():
        assert native.serves_train(ours, single.bfloat16(), pair.bfloat16(), mask) == (L % 128 == 0)
        y_ref, g_ref = run(ref, torch.float32)
        y_bf, g_bf = run(torch_bf, torch.bfloat16)
        y_ours, g_ours = run(ours, torch.bfloat16)
    assert calls["train"] == 0, "a call with extra_pair_bias must skip the native train path"
    assert _rel(y_ours, y_ref) < 1.3 * _rel(y_bf, y_ref) + 1e-3
    for name, a, b, c in zip(("single", "pair", "extra", "to_bias", "to_query"), g_ours, g_bf, g_ref):
        assert a is not None and torch.isfinite(a).all(), name
        assert _rel(a, c) < 1.3 * _rel(b, c) + 1e-3, (name, _rel(a, c), _rel(b, c))
