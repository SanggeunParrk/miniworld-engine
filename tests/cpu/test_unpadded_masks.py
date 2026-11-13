"""Speed inputs have no padding position (``unpadded_masks``); accuracy inputs and tests keep the default L // 8 padding."""

import torch

from miniworld_engine.measure import speed
from miniworld_engine.modules.primitives import key_mask, pair_inputs, unpadded_masks


def test_default_mask_pads_the_tail():
    m = key_mask(2, 64, device="cpu")
    assert m.shape == (2, 64) and m.dtype == torch.bool
    assert m[:, : 64 - 8].all() and not m[:, 64 - 8 :].any()


def test_unpadded_masks_is_all_true_and_nests():
    with unpadded_masks():
        assert key_mask(3, 40, device="cpu").all()
        assert key_mask(1, 40, device="cpu", pad=10).all()  # an explicit pad is dropped too
        assert pair_inputs(8, length=16, samples=2, dtype=torch.float32, device="cpu", generator=torch.Generator())["mask"].all()
        with unpadded_masks():
            assert key_mask(1, 16, device="cpu").all()
        assert key_mask(1, 16, device="cpu").all()
    assert not key_mask(1, 16, device="cpu").all()  # restored on exit


def test_speed_measurer_version_records_the_unpadded_inputs():
    assert speed.SPEED_VERSION == "4"
