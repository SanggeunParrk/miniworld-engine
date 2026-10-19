"""``TriangleAttention(transposed_bias=...)``: which pair the ending node projects its bias from.

The ending node attends in the transposed frame ``y = x^T``. ``transposed_bias=True`` (the default, and what the module always
did) projects the bias from ``y``; ``False`` projects it from ``x`` as it is and reads it in the transposed frame, the
orientation of AlphaFold 3's released code. The reference below is written from that definition alone, with no engine code.
"""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

D_PAIR, HEADS, LENGTH = 16, 4, 7


def build(starting, transposed_bias=True):
    torch.manual_seed(0)
    module = TriangleAttention(
        D_PAIR,
        HEADS,
        starting=starting,
        transposed_bias=transposed_bias,
        p_drop=0.0,
        implementation=ImplementationType.PYTORCH,
    )
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.1)
    return module.eval()


def reference(module, x, mask, bias_from_transposed):
    """Triangle attention around the starting node of ``y`` with a given pair-bias orientation, written out."""
    h, d = module.n_head, x.shape[-1] // module.n_head

    def project(w, t):
        return F.linear(t, w.weight)

    def head_major(t):  # [B, I, J, H*D] -> [B, H, I, J, D]
        return t.unflatten(-1, (h, d)).permute(0, 3, 1, 2, 4)

    ln = lambda t: F.layer_norm(t, (t.shape[-1],), module.ln_pair.weight, module.ln_pair.bias, module.ln_pair.eps)
    ending = not module.starting
    y = x.transpose(1, 2) if ending else x
    z = ln(y)
    q, k, v = (head_major(project(w, z)) for w in (module.to_query, module.to_key, module.to_value))
    # bias[b, h, j, k]: the bias a query j puts on key k, shared by every row i
    bias_source = z if (bias_from_transposed or not ending) else ln(x)
    bias = project(module.to_bias, bias_source).permute(0, 3, 1, 2)
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
    logits = torch.einsum("bhijd,bhikd->bhijk", q * d**-0.5, k) + bias[:, :, None]
    out = torch.einsum("bhijk,bhikd->bhijd", logits.softmax(-1), v)
    out = out.permute(0, 2, 3, 1, 4).flatten(-2)
    out = F.linear(torch.sigmoid(project(module.to_gate, z)) * out, module.to_out.weight)
    return x + (out.transpose(1, 2) if ending else out)


def inputs():
    torch.manual_seed(1)
    x = torch.randn(2, LENGTH, LENGTH, D_PAIR)
    mask = torch.ones(2, LENGTH, dtype=torch.bool)
    mask[0, -2:] = False
    mask[1, 1] = False
    return x, mask


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_default_keeps_the_transposed_bias(starting, masked):
    x, mask = inputs()
    mask = mask if masked else None
    module = build(starting)
    assert module.transposed_bias is True
    torch.testing.assert_close(module(x, mask), reference(module, x, mask, True), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("masked", [False, True])
def test_ending_node_takes_the_bias_of_the_pair_as_it_is(masked):
    x, mask = inputs()
    mask = mask if masked else None
    module = build(starting=False, transposed_bias=False)
    actual = module(x, mask)
    torch.testing.assert_close(actual, reference(module, x, mask, False), rtol=1e-4, atol=1e-4)
    # the two orientations are different functions for an asymmetric pair
    assert (actual - reference(module, x, mask, True)).abs().max() > 1e-3


def test_starting_node_ignores_the_option():
    x, mask = inputs()
    a, b = build(True, True), build(True, False)
    torch.testing.assert_close(a(x, mask), b(x, mask), rtol=0, atol=0)


def test_symmetric_pair_makes_both_orientations_equal():
    x, mask = inputs()
    x = x + x.transpose(1, 2)
    a, b = build(False, True), build(False, False)
    torch.testing.assert_close(a(x, mask), b(x, mask), rtol=1e-5, atol=1e-5)


def test_gradients_follow_the_orientation():
    x, mask = inputs()
    x.requires_grad_(True)
    module = build(starting=False, transposed_bias=False)
    out = module(x, mask)
    expected = reference(module, x, mask, False)
    grads = torch.autograd.grad(out.square().sum(), (x, module.to_bias.weight), retain_graph=True)
    ref_grads = torch.autograd.grad(expected.square().sum(), (x, module.to_bias.weight))
    for got, want in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)

