"""Where the portable DiT paths run, card by card -- CPU only, with a fake GPU and a fake manifest directory (``tests/fake_card.py``).

Three ops hold the attention cores of the portable paths: ``local_dit`` (the AF3 / Protenix / OpenFold3 / ESMFold2 atom block's window
attention), ``dit_variant_atom`` (the Boltz-2 / RoseTTAFold3 / Chai-1 atom stacks, same kernels) and ``dit_variant`` (their token stacks:
the pair-bias attention of ``augmented_attention/triton/main.py``). The rule is ``docs/dispatch.md``'s: a card with ``ok`` manifest rows
for every kernel the call launches runs the Triton kernels; a card without them warns once and runs the PyTorch reference; H100 / B200
never reach Triton through ``miniworld`` unless ``settings.force_portable_kernels`` is on. The backward kernels only matter to a call that
needs a gradient.
"""
from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch
from fake_card import fake_card, write_manifest

from miniworld_engine import settings
from miniworld_engine.autotune import records
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import UnrecordedCardWarning

CARD = "NVIDIA RTX A6000"
ATOM_OPS = ("local_dit", "dit_variant_atom")


@pytest.fixture(autouse=True)
def _clean():
    previous = settings.current()
    settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False, b200_portable_kernels=False, capture=False)
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()
    yield
    settings.configure(**dataclasses.asdict(previous))
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()


@pytest.fixture
def manifests(tmp_path):
    with records.use_manifest_dir(tmp_path):
        yield tmp_path


def test_the_atom_ops_hold_the_window_kernels_and_the_token_ops_the_token_ones():
    window = {"augmented_attention_fwd_window_triton", "augmented_attention_bwd_window_triton",
              "augmented_attention_bwd_reduce_window_triton"}
    bias = {"layernorm_linear_fwd_triton"}        # the pair-bias producer of an inference call
    for op in ATOM_OPS:
        assert set(records.triton_kernels(op, "bf16", "training")) == window | bias
        assert set(records.triton_kernels(op, "bf16", "inference")) == {"augmented_attention_fwd_window_triton"} | bias
    for op in ("dit_variant", "augmented_attention", "augmented_attention_pair_bias"):
        token = set(records.triton_kernels(op, "bf16", "training"))
        assert token, op
        assert not (token & window), f"{op} would be held against the window kernels"
    assert set(records.triton_kernels("dit_variant", "bf16", "inference")) == {"augmented_attention_fwd_triton"} | bias


def test_a_recorded_card_runs_the_window_kernels_and_an_unrecorded_one_warns_once(monkeypatch, manifests):
    device = fake_card(monkeypatch, CARD, (8, 6))
    for op in ATOM_OPS:
        with pytest.warns(UnrecordedCardWarning, match="augmented_attention_fwd_window_triton"):
            assert not dispatch.portable_leg_allowed(device, op)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                      # once per (op, card, mode)
            assert not dispatch.portable_leg_allowed(device, op)
    write_manifest(manifests, CARD, (8, 6), [*ATOM_OPS, "dit_variant"])
    records._load.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in (*ATOM_OPS, "dit_variant"):
            assert dispatch.portable_leg_allowed(device, op, training=False)
            assert dispatch.portable_leg_allowed(device, op, training=True)


def test_an_inference_record_serves_inference_and_blocks_only_a_call_that_needs_a_backward(monkeypatch, manifests):
    device = fake_card(monkeypatch, CARD, (8, 6))
    write_manifest(manifests, CARD, (8, 6), [*ATOM_OPS, "dit_variant"], training=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in (*ATOM_OPS, "dit_variant"):
            assert dispatch.portable_leg_allowed(device, op, training=False)
    with pytest.warns(UnrecordedCardWarning, match="backward"):
        assert not dispatch.portable_leg_allowed(device, "local_dit", training=True)


def test_the_token_variant_op_is_not_unrecorded_by_the_missing_window_kernels(monkeypatch, manifests):
    """A card recorded for the token attention only runs the token stacks, not the atom ones."""
    device = fake_card(monkeypatch, CARD, (8, 6))
    write_manifest(manifests, CARD, (8, 6), ["dit_variant"])
    assert dispatch.portable_leg_allowed(device, "dit_variant", training=True)
    with pytest.warns(UnrecordedCardWarning):
        assert not dispatch.portable_leg_allowed(device, "dit_variant_atom")


@pytest.mark.parametrize(("name", "capability"), [("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))])
def test_h100_and_b200_never_reach_the_portable_path_without_the_ab_switch(monkeypatch, manifests, name, capability):
    device = fake_card(monkeypatch, name, capability)
    write_manifest(manifests, name, capability, [*ATOM_OPS, "dit_variant"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in (*ATOM_OPS, "dit_variant"):
            assert not dispatch.portable_leg_allowed(device, op)
    settings.configure(force_portable_kernels=True)
    for op in (*ATOM_OPS, "dit_variant"):
        assert dispatch.portable_leg_allowed(device, op)


def test_the_ab_switch_forces_the_portable_path_on_an_unrecorded_card(monkeypatch, manifests):
    device = fake_card(monkeypatch, "Brand New GPU", (8, 9))
    settings.configure(force_portable_kernels=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in (*ATOM_OPS, "dit_variant"):
            assert dispatch.portable_leg_allowed(device, op, training=True)


@pytest.mark.filterwarnings("ignore::miniworld_engine.modules.dispatch.UnrecordedCardWarning")
def test_a_call_the_window_kernel_refuses_warns_once_and_a_cpu_tensor_does_not(monkeypatch, manifests):
    """A head wider than 64 (or more than 256 keys) is a refusal, not an error: ``KernelRefusedWarning`` once, then the reference."""
    from miniworld_engine.modules.dispatch import KernelRefusedWarning
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.local_dit import LocalDiTBlock

    class Cuda:
        """What ``_window_triton`` reads of an activation: ``is_cuda``, ``device``, ``dtype`` -- no device behind it."""
        is_cuda, dtype, device = True, torch.bfloat16, torch.device("cuda", 0)
        requires_grad = False

    fake_card(monkeypatch, CARD, (8, 6))
    write_manifest(manifests, CARD, (8, 6), ["local_dit"])
    block = LocalDiTBlock(512, 128, 16, 4, n=2, implementation=ImplementationType.MINIWORLD)   # 4 heads x 128: wider than the kernel
    with pytest.warns(KernelRefusedWarning, match="head width 128"):
        assert not block._window_triton(Cuda(), 128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not block._window_triton(Cuda(), 128)
        assert not block._window_triton(torch.zeros(1, 1, 4, 512), 128)     # a CPU tensor: nothing to refuse
    narrow = LocalDiTBlock(128, 128, 16, 4, n=2, implementation=ImplementationType.MINIWORLD)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert narrow._window_triton(Cuda(), 32)
    assert not LocalDiTBlock(128, 128, 16, 4, n=2, implementation=ImplementationType.PYTORCH)._window_triton(Cuda(), 32)
