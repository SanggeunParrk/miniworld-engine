"""``integrations.trimul_h100.MAX_BATCH`` is the samples-per-call limit that callers (MiniWorld's template embedder) ask for."""

from miniworld_engine.integrations import trimul_h100


def test_the_batch_limit_is_public_and_positive():
    assert isinstance(trimul_h100.MAX_BATCH, int)
    assert trimul_h100.MAX_BATCH >= 1


def test_max_batch_per_width():
    """``max_batch(width)`` is the per-width limit MiniWorld asks for; MAX_BATCH keeps the D64 value; an unserved width takes one."""
    assert trimul_h100.max_batch(64) == trimul_h100.MAX_BATCH
    for width in (64, 128, 256, 384, 512):
        assert isinstance(trimul_h100.max_batch(width), int)
        assert trimul_h100.max_batch(width) >= 1
    assert trimul_h100.max_batch(96) == 1
