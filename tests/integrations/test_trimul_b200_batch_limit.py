"""``integrations.trimul_b200.MAX_BATCH`` is the samples-per-call limit that callers (MiniWorld's template embedder) ask for."""

from miniworld_engine.integrations import trimul_b200


def test_the_batch_limit_is_public_and_positive():
    assert isinstance(trimul_b200.MAX_BATCH, int)
    assert trimul_b200.MAX_BATCH >= 1
