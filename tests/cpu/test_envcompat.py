"""The pre-1.0.0 environment names (MINIWORLD_*, MINIWORLD_ENGINE_*) map to FOLDJET_ENGINE_*; kernel-read switches map back."""

from foldjet_engine import _envcompat


def test_old_names_fill_unset_new_names():
    env = {"MINIWORLD_FP8": "safe", "MINIWORLD_ENGINE_JIT_ROOT": "/x", "FOLDJET_ENGINE_PWA_FP8": "1", "MINIWORLD_PWA_FP8": "0"}
    copied = _envcompat.sync(env)
    assert env["FOLDJET_ENGINE_FP8"] == "safe"
    assert env["FOLDJET_ENGINE_JIT_ROOT"] == "/x"
    assert env["FOLDJET_ENGINE_PWA_FP8"] == "1"  # a set new name wins
    assert sorted(copied) == ["MINIWORLD_ENGINE_JIT_ROOT", "MINIWORLD_FP8"]


def test_kernel_getenv_names_map_back():
    env = {"FOLDJET_ENGINE_TRIATTN_B200_PDL": "0", "FOLDJET_ENGINE_A100_CLEANUP": "1", "MINIWORLD_TRIATTN_B200_L128_GB": "1"}
    _envcompat.sync(env)
    assert env["MINIWORLD_TRIATTN_B200_PDL"] == "0"
    assert env["MINIWORLD_A100_CLEANUP"] == "1"
    assert env["FOLDJET_ENGINE_TRIATTN_B200_L128_GB"] == "1"


def test_new_name():
    assert _envcompat.new_name("MINIWORLD_TRIMUL_FP8") == "FOLDJET_ENGINE_TRIMUL_FP8"
    assert _envcompat.new_name("MINIWORLD_ENGINE_PREBUILT_DIR") == "FOLDJET_ENGINE_PREBUILT_DIR"
