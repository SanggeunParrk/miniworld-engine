"""CPU-runnable parts of the A100 AugmentedAttentionPairBias path (integrations/augattn_sm80.py): the gate predicates, the staging layouts (b-major rows, zero padding to a multiple of 128)
and the per-sample key penalties, which are plain tensor code around the kernels."""

import pytest
import torch

from miniworld_engine.integrations import augattn_sm80 as m
from miniworld_engine.kernels.augmented_attention.cuda import sm80
from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType


def test_the_padded_length_is_the_next_multiple_of_128():
    assert [m._pad(n) for n in (1, 127, 128, 129, 1024, 1025)] == [128, 128, 128, 256, 1024, 1152]


def test_mask_kinds():
    a, b, length = 3, 2, 16
    shared = torch.ones(b, length, dtype=torch.bool)
    assert m._mask_kind(None, a, b, length) == 0
    assert m._mask_kind(shared, a, b, length) == 1
    assert m._mask_kind(shared[None].expand(a, b, length), a, b, length) == 1          # stride 0 over the samples: still one mask
    assert m._mask_kind(torch.ones(a, b, length, dtype=torch.bool), a, b, length) == 2
    assert m._mask_kind(shared.to(torch.uint8), a, b, length) is None
    assert m._mask_kind(torch.ones(a, length, dtype=torch.bool), a, b, length) is None
    assert m._mask_kind(torch.ones(b, length + 1, dtype=torch.bool), a, b, length) is None


@pytest.mark.parametrize(("a", "b", "length"), [(3, 1, 128), (2, 2, 200), (1, 3, 130), (4, 1, 1)])
def test_staging_roundtrips_and_pads_with_zero_rows(a, b, length):
    padded = m._pad(length)
    x = torch.randn(a, b, length, 8)
    s = m._stage(x, b, padded)
    assert s.shape == (b * a * padded, 8)
    s4 = s.view(b, a, padded, 8)
    assert torch.equal(s4[:, :, :length].permute(1, 0, 2, 3), x)                          # b-major, (sample, token) inside
    assert not s4[:, :, length:].any()
    assert torch.equal(m._unstage(s.clone(), a, b, length, padded, 8), x)


@pytest.mark.parametrize("dim", [24, 32, 48])
def test_head_staging_pads_the_head_dim_and_the_length(dim):
    a, b, h, length = 2, 2, 3, 70
    padded, dpad = m._pad(length), m._dpad(dim)
    x = torch.randn(a, b, h, length, dim)
    s = m._stage_heads(x, a, b, length, padded, h, dim)
    assert s.shape == (b * a * padded, h * dpad)
    padding = s.view(b, a, padded, h, dpad)
    assert not padding[:, :, length:].any()
    assert not padding[..., dim:].any()
    assert torch.equal(m._unstage_heads(s.clone(), a, b, length, padded, h, dim), x)


def test_penalties_are_zero_on_valid_keys_and_minus_infinity_elsewhere():
    a, b, length = 2, 2, 10
    kmask = torch.rand(a, b, length) > 0.4
    pen = m._penalties(kmask, a, b, length, 128)
    assert pen.shape == (b, a, 128)
    assert pen.dtype is torch.float32
    assert torch.equal(pen[:, :, :length] == 0, kmask.permute(1, 0, 2))
    assert torch.isinf(pen[:, :, :length][~kmask.permute(1, 0, 2)]).all()
    assert torch.isinf(pen[:, :, length:]).all()
    flat = sm80.key_penalty(kmask.reshape(-1, length))
    assert torch.equal(flat == 0, kmask.reshape(-1, length))


def test_the_packed_bias_masks_and_pads_when_the_length_is_not_a_multiple_of_8():
    b, h, length, dim = 2, 3, 10, 48
    bias = torch.randn(b, h, length, length)
    kmask = torch.rand(b, length) > 0.3
    kmask[:, 0] = True
    out = m._pack_bias(bias, kmask, length, 128, h, dim).float()
    scale, fill = dim ** 0.5, m.MASKED * dim ** 0.5
    want = (bias * scale).to(torch.bfloat16).float()
    for bi in range(b):
        valid = kmask[bi]
        assert torch.equal(out[bi, :, :length, :length][:, :, valid], want[bi][:, :, valid])
        assert (out[bi, :, :length, :length][:, :, ~valid] == torch.tensor(fill).bfloat16().float()).all()
        assert (out[bi, :, :length, length:] == torch.tensor(fill).bfloat16().float()).all()
        assert not out[bi, :, length:].any()


def test_the_gate_refuses_what_it_cannot_serve_without_a_gpu():
    module = AugmentedAttentionPairBias(128, 128, 16, 4, implementation=ImplementationType.MINIWORLD)
    single, cond, pair = torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128), torch.randn(1, 128, 128, 16)
    assert not m.serves(module, single, pair, None, None, cond=cond), "CPU tensors"
    assert not m.serves(module, single.bfloat16(), pair.bfloat16(), None, None, cond=cond.bfloat16()), "CPU tensors"
    assert not m.serves(module, single.double(), pair.double(), None, None, cond=cond.double()), "fp64"
    assert not m.serves_ops(*(torch.randn(2, 1, 4, 128, 32).bfloat16() for _ in range(3)), torch.randn(1, 4, 128, 128).bfloat16(), None), "CPU tensors"


def test_the_kind_of_pair_bias_producer():
    assert m._kind(16, 4) == "atom"
    assert m._kind(16, 4, torch.float32) == "atom"
    assert [m._kind(128, h) for h in (16, 8, 12)] == ["apb"] * 3
    assert m._kind(128, 16, torch.float32) == "generic"            # the tensor-core kernel is bf16
    assert m._kind(256, 16) == "generic"
    assert m._kind(128, 4) == "generic"
