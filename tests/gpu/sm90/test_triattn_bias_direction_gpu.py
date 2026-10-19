"""``TriangleAttention(transposed_bias=False)`` (AlphaFold 3's ending-node bias) on the H100 hand-CUDA kernels and the whole op.

The kernels serve the orientation for inference, by transposing the bias between the front and the core; a call with grad falls
to the composed legs. Accuracy is judged against the PyTorch module of the same orientation in fp32, with the bf16 PyTorch
module's own error as the yardstick.
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from miniworld_engine.integrations import triattn_h100_sm90 as triattn_h100
from miniworld_engine.kernels.triangle_attention.whole_op import triangle_attention
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def relative(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def make(impl, d_pair, heads, transposed_bias, dtype=torch.bfloat16):
    torch.manual_seed(0)
    module = TriangleAttention(
        d_pair, heads, starting=False, transposed_bias=transposed_bias, p_drop=0.0, implementation=impl
    )
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    # the bias projection decides the orientation: make it count
    module.to_bias.weight.data.mul_(4.0)
    return module.cuda().to(dtype)


def trio(d_pair, heads, transposed_bias):
    ours = make(ImplementationType.MINIWORLD, d_pair, heads, transposed_bias)
    ref16 = make(ImplementationType.PYTORCH, d_pair, heads, transposed_bias)
    ref16.load_state_dict(ours.state_dict())
    ref32 = make(ImplementationType.PYTORCH, d_pair, heads, transposed_bias, dtype=torch.float32)
    ref32.load_state_dict({k: v.float() for k, v in ours.state_dict().items()})
    return ours, ref16, ref32


def key_mask(length):
    mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
    mask[0, length // 2 : length // 2 + 9] = False
    mask[0, -5:] = False
    return mask


def kernels(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    return {k.key for k in prof.key_averages() if k.device_type.name == "CUDA"}


# (d_pair, heads): the fused kernels (128, 4 x 32) and the width-generic path (heads of 32 channels)
SHAPES = [(128, 4), (256, 8)]


@pytest.mark.parametrize(("d_pair", "heads"), SHAPES)
@pytest.mark.parametrize("length", [128, 208, 256])
@pytest.mark.parametrize("masked", [False, True])
def test_inference_matches_the_reference_orientation(d_pair, heads, length, masked):
    ours, ref16, ref32 = trio(d_pair, heads, transposed_bias=False)
    torch.manual_seed(1)
    pair = torch.randn(1, length, length, d_pair, device="cuda", dtype=torch.bfloat16)
    mask = key_mask(length) if masked else None
    with torch.no_grad():
        assert triattn_h100.serves(ours, pair, mask)
        got = ours(pair, mask)
        names = kernels(lambda: ours(pair, mask))
        want = ref32(pair.float(), mask)
        base = ref16(pair, mask)
    assert any("tri" in n for n in names), names
    live = slice(None) if mask is None else mask[0]
    err, base_err = relative(got[:, live][:, :, live], want[:, live][:, :, live]), relative(
        base[:, live][:, :, live], want[:, live][:, :, live]
    )
    assert err <= 2 * base_err + 2e-3, (err, base_err)


@pytest.mark.parametrize(("d_pair", "heads"), SHAPES)
def test_orientations_differ_and_the_default_is_unchanged(d_pair, heads):
    torch.manual_seed(2)
    pair = torch.randn(1, 128, 128, d_pair, device="cuda", dtype=torch.bfloat16)
    transposed, _, ref32_t = trio(d_pair, heads, transposed_bias=True)
    untransposed, _, ref32_u = trio(d_pair, heads, transposed_bias=False)
    with torch.no_grad():
        a, b = transposed(pair), untransposed(pair)
        want_t, want_u = ref32_t(pair.float()), ref32_u(pair.float())
    assert relative(a, want_t) < 2e-2
    assert relative(b, want_u) < 2e-2
    # the update (output minus the residual) is what the orientation changes
    assert relative(b - pair, a - pair) > 5e-2


def test_a_call_with_grad_takes_the_composed_legs_and_keeps_the_gradients():
    ours, ref16, _ = trio(128, 4, transposed_bias=False)
    torch.manual_seed(3)
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = key_mask(128)
    x_ours, x_ref = pair.clone().requires_grad_(True), pair.clone().requires_grad_(True)
    assert not triattn_h100.serves(ours, x_ours, mask)
    out_ours, out_ref = ours(x_ours, mask), ref16(x_ref, mask)
    assert relative(out_ours, out_ref) < 1e-2
    grad = torch.randn_like(out_ours)
    g_ours = torch.autograd.grad(out_ours, (x_ours, ours.to_bias.weight), grad)
    g_ref = torch.autograd.grad(out_ref, (x_ref, ref16.to_bias.weight), grad)
    for got, want in zip(g_ours, g_ref, strict=True):
        assert relative(got, want) < 3e-2


@pytest.mark.parametrize("transposed_bias", [True, False])
def test_whole_op_matches_the_module(transposed_bias):
    ours = make(ImplementationType.PYTORCH, 128, 4, transposed_bias, dtype=torch.float32)
    torch.manual_seed(4)
    pair = torch.randn(1, 64, 64, 128, device="cuda")
    mask = key_mask(64)
    with torch.no_grad():
        update = triangle_attention(
            pair,
            mask,
            n_head=4,
            ln_pair_weight=ours.ln_pair.weight,
            ln_pair_bias=ours.ln_pair.bias,
            to_value_weight=ours.to_value.weight,
            to_bias_weight=ours.to_bias.weight,
            to_gate_weight=ours.to_gate.weight,
            to_out_weight=ours.to_out.weight,
            to_query_weight=ours.to_query.weight,
            to_key_weight=ours.to_key.weight,
            starting=False,
            transposed_bias=transposed_bias,
        )
        torch.testing.assert_close(pair + update, ours(pair, mask), rtol=2e-3, atol=2e-3)
