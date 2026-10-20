"""``integrations.triattn_h100_sm90``: samples per launch (``max_batch``) and the memory chunking of a B-sample call (``plan``), a pure
function of the path, the shapes, the mode and the device's total memory."""

import pytest

from miniworld_engine.integrations import triattn_h100_sm90 as tri

H100 = 85_018_279_936  # torch.cuda.get_device_properties(0).total_memory of an H100 80GB HBM3 (2026-10-20)
FUSED = (tri.FUSED, 128, 4)
WIDE = (tri.GENERIC, 512, 16)
TEMPLATE = (tri.GENERIC, 64, 4)


def test_max_batch_per_path():
    for kind in (tri.FUSED, tri.GENERIC):
        assert tri.max_batch(kind) == tri.MAX_BATCH[kind] >= 48
        assert tri.max_batch(kind, 768) == tri.MAX_BATCH[kind]
        # the int32 row coordinate of the TMA maps: B Lp^2 < 2^31
        assert tri.max_batch(kind, 46_000) * 46_080**2 < 2**31
        assert tri.max_batch(kind, 10**6) == 1


@pytest.mark.parametrize(
    "path", [FUSED, WIDE, TEMPLATE], ids=["fused", "wide", "template"]
)
@pytest.mark.parametrize("training", [False, True])
def test_plan_is_deterministic_and_bounded(path, training):
    kind, c, h = path
    for length in (128, 208, 384, 768):
        for batch in (1, 2, 4, 16, 48, 96):
            chunk, checkpoint = tri.plan(kind, batch, length, c, h, training, H100)
            assert (chunk, checkpoint) == tri.plan(
                kind, batch, length, c, h, training, H100
            )
            assert 1 <= chunk <= tri.max_batch(kind, length)
            assert not checkpoint or training
            sizes = tri._split(batch, min(chunk, batch))
            assert sum(sizes) == batch
            assert max(sizes) - min(sizes) <= 1
            assert max(sizes) <= chunk


def test_small_calls_run_in_one_launch():
    for kind, c, h in (FUSED, WIDE, TEMPLATE):
        for training in (False, True):
            assert tri.plan(kind, 1, 768, c, h, training, H100)[0] == 1
            assert tri.plan(kind, 48, 128, c, h, training, H100) == (48, False)
            assert tri.plan(kind, 4, 384, c, h, training, H100) == (4, False)
    assert tri.plan(*FUSED[:1], 48, 768, 128, 4, False, H100) == (
        48,
        False,
    )  # fits: 1.8 kB per token


def test_more_samples_than_one_launch_takes():
    chunk, checkpoint = tri.plan(tri.FUSED, 96, 128, 128, 4, False, H100)
    assert chunk == tri.max_batch(tri.FUSED)
    assert not checkpoint
    assert tri._split(96, chunk) == [48, 48]
    assert tri._split(100, chunk) == [34, 33, 33]


def test_calls_that_ran_out_of_memory_are_split():
    """The one-launch OOMs measured on an H100 80GB before chunking (2026-10-19)."""
    chunk, checkpoint = tri.plan(tri.FUSED, 48, 768, 128, 4, True, H100)
    assert chunk < 48
    assert (
        not checkpoint
    )  # the 48 samples' saved tensors fit next to one chunk's backward
    assert tri.plan(tri.GENERIC, 48, 768, 256, 8, False, H100)[0] < 48
    assert tri.plan(tri.GENERIC, 16, 768, 512, 16, False, H100)[0] < 16
    assert tri.plan(tri.GENERIC, 48, 768, 64, 4, True, H100)[0] < 48
    assert tri.plan(tri.GENERIC, 16, 768, 256, 8, True, H100)[0] < 16


def test_recompute_when_the_saved_tensors_do_not_fit():
    chunk, checkpoint = tri.plan(tri.FUSED, 96, 768, 128, 4, True, H100)
    assert checkpoint
    assert chunk < 48
    chunk, checkpoint = tri.plan(tri.GENERIC, 16, 768, 512, 16, True, H100)
    assert checkpoint
    assert chunk >= 1


def test_smaller_device_smaller_chunks():
    for kind, c, h in (FUSED, WIDE, TEMPLATE):
        for training in (False, True):
            big = tri.plan(kind, 48, 768, c, h, training, H100)[0]
            small = tri.plan(kind, 48, 768, c, h, training, H100 // 4)[0]
            assert small <= big


def test_working_set_counts_the_padded_length():
    """L = 720 runs at 768: its per-token bytes include the padded copies, on top of L = 768's."""
    for kind, c, h in (FUSED, WIDE, TEMPLATE):
        padded, exact = (
            tri._bytes_per_token(kind, c, h, True),
            tri._bytes_per_token(kind, c, h, False),
        )
        assert all(p > e for p, e in zip(padded, exact, strict=True))
        assert tri.plan(kind, 4, 720, c, h, True, H100) == tri.plan(
            kind, 4, 768, c, h, True, H100
        )
