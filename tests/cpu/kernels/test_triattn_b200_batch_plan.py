"""``integrations.triattn_b200``: samples per launch (``max_batch``) and the memory chunking of a B-sample call (``plan``), a pure
function of the path, the shapes, the node, the mode and the device's total memory."""

import pytest

from miniworld_engine.integrations import triattn_b200 as tri

B200 = 191_503_138_816  # torch.cuda.get_device_properties(0).total_memory of a B200 (2026-10-20)
FUSED = (tri.FUSED, 128, 4)
WIDE = (tri.GENERIC, 512, 16)
TRUNK = (tri.GENERIC, 256, 8)
TEMPLATE = (tri.GENERIC, 64, 4)
PATHS = [FUSED, WIDE, TRUNK, TEMPLATE]


def test_max_batch_per_path():
    for kind in (tri.FUSED, tri.GENERIC):
        assert tri.max_batch(kind) == tri.MAX_BATCH[kind] >= 48
        assert tri.max_batch(kind, 768) == tri.MAX_BATCH[kind]
        assert tri.max_batch(kind, 2560) == tri.MAX_BATCH[kind]
        # the int32 row coordinate of the TMA maps: B Lp^2 < 2^31
        assert tri.max_batch(kind, 46_000) * 46_080**2 < 2**31
        assert tri.max_batch(kind, 10**6) == 1


@pytest.mark.parametrize("ending", [False, True], ids=["starting", "ending"])
@pytest.mark.parametrize("path", PATHS, ids=["fused", "wide", "trunk", "template"])
@pytest.mark.parametrize("training", [False, True])
def test_plan_is_deterministic_and_bounded(path, training, ending):
    kind, c, h = path
    for length in (16, 100, 128, 208, 384, 768, 1536, 2560):
        for batch in (1, 2, 4, 16, 48, 96):
            chunk, checkpoint = tri.plan(kind, batch, length, c, h, training, B200, ending)
            assert (chunk, checkpoint) == tri.plan(kind, batch, length, c, h, training, B200, ending)
            assert 1 <= chunk <= tri.max_batch(kind, length)
            assert not checkpoint or training
            sizes = tri._split(batch, min(chunk, batch))
            assert sum(sizes) == batch
            assert max(sizes) - min(sizes) <= 1
            assert max(sizes) <= chunk


def test_small_calls_run_in_one_launch():
    for kind, c, h in PATHS:
        for training in (False, True):
            assert tri.plan(kind, 1, 768, c, h, training, B200)[0] == 1
            assert tri.plan(kind, 48, 128, c, h, training, B200) == (48, False)
            assert tri.plan(kind, 4, 384, c, h, training, B200) == (4, False)
            assert tri.plan(kind, 1, 2560, c, h, training, B200) == (1, False)
    assert tri.plan(tri.FUSED, 48, 768, 128, 4, False, B200) == (48, False)
    assert tri.plan(tri.FUSED, 48, 768, 128, 4, True, B200) == (48, False)   # fits on a B200 (it did not on an H100 80GB)


def test_more_samples_than_one_launch_takes():
    chunk, checkpoint = tri.plan(tri.FUSED, 96, 128, 128, 4, False, B200)
    assert chunk == tri.max_batch(tri.FUSED)
    assert not checkpoint
    assert tri._split(96, chunk) == [48, 48]
    assert tri._split(100, chunk) == [34, 33, 33]


def test_calls_that_do_not_fit_are_split():
    """The cases the GPU test runs split by memory (``SPLIT_CASES`` of tests/gpu/sm100/test_triattn_b200_batched_gpu.py)."""
    assert tri.plan(tri.FUSED, 96, 768, 128, 4, True, B200) == (27, False)
    assert tri.plan(tri.GENERIC, 32, 768, 256, 8, True, B200)[0] < 32
    assert tri.plan(tri.GENERIC, 40, 768, 512, 16, False, B200)[0] < 40
    assert tri.plan(tri.GENERIC, 16, 768, 512, 16, True, B200) == (11, False)
    assert tri.plan(tri.GENERIC, 96, 768, 64, 4, True, B200)[0] < 48


def test_recompute_when_the_saved_tensors_do_not_fit():
    chunk, checkpoint = tri.plan(tri.FUSED, 192, 768, 128, 4, True, B200)
    assert checkpoint
    assert chunk < 48
    chunk, checkpoint = tri.plan(tri.GENERIC, 32, 768, 512, 16, True, B200)
    assert checkpoint
    assert 1 <= chunk < 32


def test_smaller_device_smaller_chunks():
    for kind, c, h in PATHS:
        for training in (False, True):
            big = tri.plan(kind, 48, 768, c, h, training, B200)[0]
            small = tri.plan(kind, 48, 768, c, h, training, B200 // 4)[0]
            assert small <= big


def test_working_set_counts_the_padded_length():
    """L = 720 runs at 768: its per-token bytes include the padded copies. The ending node reads and writes the caller's layout
    directly (no transposed copies): the same working set as the starting node."""
    for kind, c, h in PATHS:
        base = tri._bytes_per_token(kind, c, h, False, False)
        padded = tri._bytes_per_token(kind, c, h, True, False)
        assert all(p > e for p, e in zip(padded, base, strict=True))
        assert tri._bytes_per_token(kind, c, h, False, True) == base
        assert tri._bytes_per_token(kind, c, h, True, True) == padded
        assert tri.plan(kind, 4, 720, c, h, True, B200) == tri.plan(kind, 4, 768, c, h, True, B200)
