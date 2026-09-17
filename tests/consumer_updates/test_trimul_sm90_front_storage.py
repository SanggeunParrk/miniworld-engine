"""Native coverage must use the same saved-output storage budget as runtime."""
import pytest


@pytest.mark.parametrize("save_preact", [False, True])
def test_front_native_bucket_preserves_training_storage_contract(save_preact):
    pytest.importorskip("cutlass")
    from miniworld_engine.autotune.trimul_sm90_config import partition_for_bucket
    from miniworld_engine.kernels.trimul_inproj.cute.parity_front import (
        front_config_rejection,
    )

    # Six stages fit for inference but saved preactivations exceed the Hopper
    # shared-memory budget. Treating both buckets as training loses legal
    # inference candidates; treating both as inference launches an invalid CTA.
    candidate = {
        "BLOCK_M1": 128, "BLOCK_K_D": 64, "BLOCK_K_H2": 64,
        "num_warps": 4, "num_stages": 6,
    }
    tensors = [
        ((512, 128), (128, 1), "torch.bfloat16"),
        ((128, 1024), (1024, 1), "torch.bfloat16"),
        None,
    ]
    bucket = repr((tensors, (save_preact,)))
    kept, rejected = partition_for_bucket(
        "trimul_inproj_gemm_gate_mmajor_sm90_cute", bucket
    )
    reason = front_config_rejection(
        candidate, m=512, k=128, h2=256, save_preact=save_preact
    )
    if save_preact:
        assert reason is not None and "shared" in reason
        assert candidate not in kept
        assert any(row["config"] == candidate and row["reason"] == reason
                   for row in rejected)
    else:
        assert reason is None
        assert candidate in kept
