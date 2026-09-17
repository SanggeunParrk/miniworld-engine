"""The largest declared bidirectional width must remain buildable."""

import pytest

from miniworld_engine.autotune.shape_key import TOKEN_SHAPES, token_key
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import dual_shape_key


@pytest.mark.parametrize("length", TOKEN_SHAPES)
def test_wide_projection_has_a_distinct_int64_key(length):
    wide = dual_shape_key(length, 512, 4096, 512)
    assert 0 < wide < 2**63
    assert wide != dual_shape_key(length, 512, 512, 512)
    assert wide != dual_shape_key(length, 512, 4088, 512)
    assert wide != dual_shape_key(length, 256, 4096, 512)
    assert wide != dual_shape_key(length, 512, 4096, 256)


@pytest.mark.parametrize("kp", [67, 259, 512, 1024, 2048, 4093])
def test_existing_projection_keys_do_not_change(kp):
    assert dual_shape_key(384, 128, kp, 128) == token_key(384, KG=128, KP=kp, N=128)
