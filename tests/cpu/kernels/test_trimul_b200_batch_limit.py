"""``integrations.trimul_b200``: the samples-per-launch limit callers (MiniWorld's template embedder) ask for, per width, and the
deterministic chunking of a B-sample call (``_plan``), a pure function of shapes, mode and the device's total memory."""

import types

import pytest
import torch

from miniworld_engine.integrations import trimul_b200


def test_the_batch_limit_is_public_and_positive():
    assert isinstance(trimul_b200.MAX_BATCH, int)
    assert trimul_b200.MAX_BATCH >= 1


def test_max_batch_per_width():
    """``max_batch(width)`` is the per-width limit; MAX_BATCH keeps the D64 value; an unserved width takes one."""
    assert trimul_b200.max_batch(64) == trimul_b200.MAX_BATCH == 48
    for width in (64, 128, 256, 384, 512):
        assert trimul_b200.max_batch(width) == 48
    assert trimul_b200.max_batch(96) == 1


def _module(d, hidden=None, bidirectional=True):
    ln = types.SimpleNamespace(weight=torch.empty(d))
    m = types.SimpleNamespace(d_hidden=hidden or d, ln_pair=ln)
    if not bidirectional:
        m.outgoing = True
    return m


class _Pair:
    """Shape-only stand-in for a CUDA pair tensor (the plan reads shapes and the device index only)."""

    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)
        self.is_cuda = True
        self.device = types.SimpleNamespace(index=0)


@pytest.mark.parametrize("inference", [True, False])
@pytest.mark.parametrize("d", [64, 128, 256, 384, 512])
def test_plan_is_deterministic_and_bounded(monkeypatch, d, inference):
    monkeypatch.setattr(trimul_b200, "_total_memory", lambda index: 180 * 10 ** 9)
    m = _module(d)
    for length in (128, 384, 768, 1536):
        for bsz in (1, 2, 16, 48, 49, 200):
            pair = _Pair(bsz, length, length, d)
            chunk, checkpoint = trimul_b200._plan(m, pair, inference, True)
            assert (chunk, checkpoint) == trimul_b200._plan(m, pair, inference, True)
            assert 1 <= chunk <= max(1, min(bsz, trimul_b200.max_batch(d)))
            assert chunk * length * length < 2 ** 31
            assert not (inference and checkpoint)
            sizes = trimul_b200._split(bsz, chunk)
            assert sum(sizes) == bsz and max(sizes) <= chunk and max(sizes) - min(sizes) <= 1


def test_plan_small_batches_take_one_launch(monkeypatch):
    monkeypatch.setattr(trimul_b200, "_total_memory", lambda index: 180 * 10 ** 9)
    for d in (64, 128, 256, 384, 512):
        for inference in (True, False):
            assert trimul_b200._plan(_module(d), _Pair(4, 128, 128, d), inference, True) == (4, False)
        assert trimul_b200._plan(_module(d), _Pair(48, 128, 128, d), False, True) == (48, False)     # 786432 pair tokens
        assert trimul_b200._plan(_module(d), _Pair(36, 128, 128, d), True, True) == (36, False)      # 589824


def test_plan_caps_the_tokens_per_launch(monkeypatch):
    """Inference launches take at most 589824 pair tokens, training at D128+ 786432 (speed; D64 training: one launch)."""
    monkeypatch.setattr(trimul_b200, "_total_memory", lambda index: 180 * 10 ** 9)
    assert trimul_b200._plan(_module(128), _Pair(8, 768, 768, 128), True, True) == (1, False)
    assert trimul_b200._plan(_module(128), _Pair(48, 384, 384, 128), True, True) == (4, False)
    assert trimul_b200._plan(_module(128), _Pair(48, 384, 384, 128), False, True) == (5, False)
    assert trimul_b200._plan(_module(64), _Pair(48, 384, 384, 64), False, True) == (48, False)
    assert trimul_b200._plan(_module(512, bidirectional=False), _Pair(4, 768, 768, 512), False, False) == (1, False)
