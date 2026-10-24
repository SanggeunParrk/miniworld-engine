"""The Transition weight re-layout cache serves a tensor, not an address.

``transposed_contiguous`` keys its cache on the weight's address, version, shape and dtype. An address is reused as soon as a tensor is
freed, so another weight (the next model, the next test) can land on a dead one's address with the same version and be served the dead
one's transposed copy -- silently wrong output. A tensor that shares the first one's storage but has its own version counter gives the
same key with different contents, deterministically.
"""

import torch

from miniworld_engine.kernels.transition.cuda import _weights


def aliased_pair():
    first = torch.randn(8, 4)
    second = torch.empty(0).set_(first.untyped_storage(), 0, first.shape, first.stride())
    first.add_(0)                           # set_ bumped the second's version: bring the first level with it
    assert second.data_ptr() == first.data_ptr()
    assert second._version == first._version
    return first, second


def test_a_different_tensor_at_the_same_address_and_version_is_not_served_the_old_copy():
    _weights._CACHE.clear()
    first, second = aliased_pair()
    with torch.no_grad():
        old = _weights.transposed_contiguous(first)
        first.zero_()                      # its version moves; the other tensor's does not
        assert (second == 0).all()
        got = _weights.transposed_contiguous(second)
    torch.testing.assert_close(got, second.t().contiguous())
    assert not torch.equal(got, old)


def test_the_same_tensor_still_hits_and_a_change_misses():
    _weights._CACHE.clear()
    w = torch.randn(8, 4)
    with torch.no_grad():
        a = _weights.transposed_contiguous(w)
        assert _weights.transposed_contiguous(w) is a
        w.mul_(2)
        b = _weights.transposed_contiguous(w)
    assert b is not a
    torch.testing.assert_close(b, w.t().contiguous())


def test_dead_entries_are_dropped_and_a_newer_version_replaces_the_older():
    _weights._CACHE.clear()
    with torch.no_grad():
        for _ in range(200):
            _weights.transposed_contiguous(torch.randn(4, 4))
        assert len(_weights._CACHE) <= 1
        w = torch.randn(4, 4)
        _weights.transposed_contiguous(w)
        w.add_(1)
        _weights.transposed_contiguous(w)
    assert sum(ref() is w for ref, _ in _weights._CACHE.values()) == 1


def test_live_weights_keep_their_entries_past_a_model_s_worth_of_blocks():
    # a captured CUDA graph reads the entry it hit: evicting a live weight's transpose freed memory a replay still read
    _weights._CACHE.clear()
    weights = [torch.randn(4, 4) for _ in range(200)]
    with torch.no_grad():
        first = [_weights.transposed_contiguous(w) for w in weights]
        assert all(_weights.transposed_contiguous(w) is t for w, t in zip(weights, first))
