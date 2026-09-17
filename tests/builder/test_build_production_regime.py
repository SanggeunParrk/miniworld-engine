"""The real build and its fake derivation must reach the measured diffusion shapes."""
import dataclasses

import pytest
import torch

from miniworld_engine.autotune import builder, derive, plan
from miniworld_engine.autotune.module_registry import module_rows


@pytest.mark.parametrize(("train", "augmentation"), [(False, 5), (True, 48)])
def test_diffusion_inputs_match_production(monkeypatch, train, augmentation):
    # Retain the real input factories; replace only allocation device for this CPU test.
    monkeypatch.setattr(builder, "_single", lambda b, l, d, dt: torch.empty(b, l, d, dtype=dt, device="meta"))
    monkeypatch.setattr(builder, "_pair", lambda b, l, d, dt: torch.empty(b, l, l, d, dtype=dt, device="meta"))
    monkeypatch.setattr(builder, "_mask", lambda b, l: torch.empty(b, l, device="meta"))
    monkeypatch.setattr(builder, "_swa_params", lambda l, dims, dt, b=1: (b, l))
    cases = {c.name: c for c in builder.cases()}
    for name in ("augmented_attention", "conditioned_transition", "adaptive_layernorm"):
        case = cases[name]
        di = next(i for i, d in enumerate(case.dims) if d.get("d_single", d.get("d_hidden")) == 768)
        args = case.input_args(di, 384, torch.bfloat16, train=train)
        assert args[0].shape == (augmentation, 1, 384, 768)
        assert args[1].shape == (augmentation, 1, 384, 384)
        if name == "augmented_attention":
            assert args[2].shape == (1, 384, 384, 128)
            assert args[3].shape == (1, 384)
    swa = cases["swa_atom_attention"]
    args = swa.input_args(0, 3072, torch.bfloat16, train=train)
    assert args[0].shape == (augmentation, 3072, 128)
    assert args[1] == (augmentation, 3072)
    pair = cases["triangle_multiplication"]
    assert pair.input_args(0, 384, torch.bfloat16, train=train)[0].shape == (1, 384, 384, 128)


def test_real_and_fake_units_have_identical_identities(monkeypatch):
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    cases = builder.cases()
    by_name = {c.name: c for c in cases}
    real = {plan.label(u, by_name[u.case]) for u in builder.units(cases)}
    fake = {u.label for u in derive.units(module_rows(), arch="sm86")}
    assert real == fake
    assert any("L=3072 A=48 train" in label for label in real)


def test_augmentation_changes_the_plan_identity():
    row = next(r for r in module_rows() if r.module == "swa_atom_attention")
    before = {u.label for u in derive.units([row])}
    after = {u.label for u in derive.units([dataclasses.replace(row, train_augmentation=2)])}
    assert before != after


def test_native_recorder_does_not_compile_and_retains_dtype_restrictions(monkeypatch):
    import cutlass.cute as cute
    import quack.cache as quack_cache
    import quack.cute_dsl_utils as cute_utils
    from torch._subclasses.fake_tensor import FakeTensorMode

    from miniworld_engine.kernels.layernorm import cuda
    from miniworld_engine.modules.swa_atom_attention import module as swa
    monkeypatch.setattr(quack_cache, "CACHE_ENABLED", quack_cache.CACHE_ENABLED)
    monkeypatch.setattr(swa, "_flash_window_core", swa._flash_window_core)
    monkeypatch.setattr(swa, "_FA2_SPEC", False)
    monkeypatch.setattr(swa, "_FA4_SPEC", False)
    from miniworld_engine.kernels.transition import cuda as transition_cuda
    monkeypatch.setattr(cute, "compile", cute.compile)
    monkeypatch.setattr(cute_utils, "get_max_active_clusters", cute_utils.get_max_active_clusters)
    monkeypatch.setattr(transition_cuda, "_ext", transition_cuda._ext)
    monkeypatch.setattr(torch._dynamo.config, "disable", torch._dynamo.config.disable)
    original = cuda.layer_norm_bwd_cuda
    monkeypatch.setattr(cuda, "layer_norm_bwd_cuda", original)
    def forbidden():
        pytest.fail("fake derivation compiled a CUDA extension")
    monkeypatch.setattr(cuda, "_ext", forbidden)
    derive.install_native_recorders()
    assert swa._FA2_SPEC
    assert swa._FA4_SPEC
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 6))
    assert swa._flash_backend(torch.device("cuda")) == "fa2"
    with FakeTensorMode():
        x = torch.empty(48 * 384, 384, dtype=torch.bfloat16)
        w = torch.empty(384, dtype=torch.bfloat16)
        stats = torch.empty(48 * 384)
        dx, dw, db = cuda.layer_norm_bwd_cuda(x, x, w, stats, stats)
        assert dx.shape == x.shape
        assert dw.shape == db.shape == w.shape
        assert dx.dtype == dw.dtype == db.dtype == w.dtype
        with pytest.raises(RuntimeError, match="matching"):
            cuda.layer_norm_bwd_cuda(x, x, w.float(), stats, stats)


@pytest.mark.parametrize("augmentation", [5, 48])
def test_swa_rotary_tables_match_native_benchmark(monkeypatch, augmentation):
    for name in ("ones", "zeros", "full", "arange"):
        original = getattr(torch, name)
        def allocate(*args, _original=original, **kwargs):
            return _original(*args, **{**kwargs, "device": "meta"})
        monkeypatch.setattr(torch, name, allocate)
    cos, sin, counts, offsets, length, valid = builder._swa_params(
        3072, {"d_model": 128, "n_heads": 4}, torch.bfloat16, augmentation)
    assert cos.shape == sin.shape == (augmentation, 3072, 16)
    assert cos.dtype == sin.dtype == torch.float32
    assert counts.shape == (augmentation,)
    assert offsets.shape == (augmentation + 1,)
    assert valid.shape == (augmentation, 3072)
    assert length == 3072
