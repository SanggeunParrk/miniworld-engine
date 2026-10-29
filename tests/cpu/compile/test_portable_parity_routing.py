"""CPU-only routing of the module-level parity paths (fake GPU, fake manifests: ``tests/fake_card.py``):

  * the bias-only DiT block builds its parts with the block's implementation, so they route per card like any module (portable kernels
    on a recorded card, PyTorch + one warning on an unrecorded one, the PyTorch leg on H100 / B200);
  * Transition fp32 goes to the portable residual path only where torch allows TF32 and the card has fp32 rows for what the call launches;
  * ``DiTBlock(extra_pair_bias=)`` is an additive logit term on every path (no ValueError), differentiable;
  * the TriAttn variant ``refusal`` takes fp32 master parameters under bf16 activations.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from fake_card import fake_card, write_manifest
from routing_helpers import clean_routing_state as _clean  # noqa: F401

from miniworld_engine.autotune import records
from miniworld_engine.modules import Transition, dispatch
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock, pair_bias_all
from miniworld_engine.modules.dispatch import KernelBackend, UnrecordedCardWarning
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType as I

A6000 = "NVIDIA RTX A6000"
BO_OPS = ["adaptive_layernorm", "conditioned_transition", "layernorm"]


def _parts(block):
    return (block.attention.ada_ln_in, block.transition, block.transition.ada_ln_in)


# ---- (1) bias-only DiT ------------------------------------------------------------------------------------------------------
def test_bias_only_parts_follow_the_block_implementation_on_a_recorded_card(monkeypatch, manifests):
    fake_card(monkeypatch, A6000, (8, 6))
    write_manifest(manifests, A6000, (8, 6), BO_OPS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        block = BiasOnlyDiTBlock(64, 32, 16, 4, implementation=I.MINIWORLD)
    assert all(part._backend is KernelBackend.PORTABLE for part in _parts(block))
    assert block.attention.ln_pair._backend is not KernelBackend.PYTORCH     # the pair LayerNorm: its own family (CUDA / portable forward)


def test_bias_only_parts_warn_and_run_pytorch_on_an_unrecorded_card(monkeypatch, manifests):
    fake_card(monkeypatch, A6000, (8, 6))
    with pytest.warns(UnrecordedCardWarning):
        block = BiasOnlyDiTBlock(64, 32, 16, 4, implementation=I.MINIWORLD)
    assert all(part._backend is KernelBackend.PYTORCH for part in _parts(block))
    assert block.attention.ln_pair._backend is KernelBackend.PYTORCH


@pytest.mark.parametrize(("name", "capability"), [("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))])
def test_bias_only_parts_take_their_pytorch_leg_on_h100_and_b200(monkeypatch, manifests, name, capability):
    fake_card(monkeypatch, name, capability)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        block = BiasOnlyDiTBlock(64, 32, 16, 4, implementation=I.MINIWORLD)
    cuda_parts = {"adaptive_layernorm", "conditioned_transition"}   # hand-written AdaLN / ConditionedTransition (H100 sm_90a; B200 rows + tcgen05 tail)
    for part in _parts(block):
        expected = KernelBackend.CUDA if type(part).__name__.lower() in {n.replace("_", "") for n in cuda_parts} else KernelBackend.PYTORCH
        assert part._backend is expected


@pytest.mark.parametrize("impl", [I.PYTORCH, I.CUEQUIVARIANCE])
def test_other_implementations_keep_the_pytorch_statements(monkeypatch, manifests, impl):
    fake_card(monkeypatch, A6000, (8, 6))
    write_manifest(manifests, A6000, (8, 6), BO_OPS)
    block = BiasOnlyDiTBlock(64, 32, 16, 4, implementation=impl)
    assert all(part._backend is KernelBackend.PYTORCH for part in _parts(block))


def test_bias_only_block_on_a_cpu_matches_the_reference_in_every_layout():
    """No card: the Triton label is kept and the parts run their statements; samples, psb and the hoisted bias all agree with PYTORCH."""
    torch.manual_seed(0)
    ref = BiasOnlyDiTBlock(64, 32, 16, 4, implementation=I.PYTORCH)
    fast = BiasOnlyDiTBlock(64, 32, 16, 4, implementation=I.MINIWORLD)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(std=0.2)
    fast.load_state_dict(ref.state_dict())
    for A, B in ((3, 1), (1, 3)):
        single, cond, pair = torch.randn(A, B, 24, 64), torch.randn(A, B, 24, 32), torch.randn(B, 24, 24, 16)
        mask = torch.rand(B, 24) > 0.1
        torch.testing.assert_close(fast(single, cond, pair, mask), ref(single, cond, pair, mask))
        bias = pair_bias_all([fast], pair)[0]
        torch.testing.assert_close(fast(single, cond, None, mask, bias=bias), ref(single, cond, pair, mask), rtol=1e-4, atol=1e-5)


# ---- (2) DiTBlock extra_pair_bias ---------------------------------------------------------------------------------------------
def test_extra_pair_bias_is_an_additive_logit_term_on_every_path():
    """On a CPU the composed path is the PyTorch statements; the GPU twin (tests/gpu/any/test_dit_extra_pair_bias_portable_gpu.py) runs it on the kernels."""
    torch.manual_seed(1)
    block = DiTBlock(32, 16, 8, 4, implementation=I.PYTORCH)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(std=0.3)
    single, cond, pair = torch.randn(2, 1, 12, 32), torch.randn(2, 1, 12, 16), torch.randn(1, 12, 12, 8)
    mask = torch.rand(1, 12) > 0.2
    extra = torch.randn(12, 12, requires_grad=True)
    base = block(single, cond, pair, mask)
    assert torch.allclose(block(single, cond, pair, mask, extra_pair_bias=torch.full((12, 12), 2.5)), base, atol=1e-5)
    out = block(single, cond, pair, mask, extra_pair_bias=extra)                    # no ValueError
    assert not torch.allclose(out, base, atol=1e-4)
    out.square().sum().backward()
    assert extra.grad is not None and extra.grad.abs().sum() > 0
    # the term is the attention bias's: moving it into the pair projection gives the same answer
    attn = block.attention
    with torch.no_grad():
        direct = attn.to_bias(attn.ln_pair(pair))
        shifted = direct + extra[None, :, :, None]
        assert shifted.shape == direct.shape
    with pytest.raises(ValueError, match="extra_pair_bias"):
        block(single, cond, pair, mask, extra_pair_bias=torch.zeros(12, 11))


# ---- (3) Transition fp32 ------------------------------------------------------------------------------------------------------
def test_the_fp32_transition_kernels_declare_fp32_and_the_inference_set_is_a_subset():
    inference = set(records.triton_kernels("transition", "fp32", "inference"))
    training = set(records.triton_kernels("transition", "fp32", "training"))
    assert {"transition_expand_swiglu_triton", "transition_squeeze_residual_triton"} <= inference
    assert {"transition_bwd_swiglu_recompute_triton"} <= training - inference
    assert inference < training
    bf16 = set(records.triton_kernels("transition", "bf16", "training"))
    assert training <= bf16, "an fp32 kernel is a bf16 kernel the module also launches"


@pytest.fixture
def cuda_tensors(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


@pytest.fixture
def tf32(monkeypatch):
    previous = torch.get_float32_matmul_precision()
    yield lambda on: torch.set_float32_matmul_precision("high" if on else "highest")
    torch.set_float32_matmul_precision(previous)


def _transition_route(monkeypatch, x, training=False):
    """Which leg Transition.forward takes for ``x`` on the faked card: 'portable' or 'pytorch'."""
    legs = []
    monkeypatch.setattr(Transition, "_residual_forward", lambda self, x: (legs.append("portable"), x)[1])
    monkeypatch.setattr(Transition, "_torch_forward", lambda self, x: (legs.append("pytorch"), torch.zeros_like(x))[1])
    m = Transition(64, n=2, implementation=I.MINIWORLD)
    m(x)
    return legs[-1]


def _write_both(directory, name, capability, ops, dtypes):
    """A manifest with ``ok`` rows for ``ops`` at every dtype in ``dtypes`` (``write_manifest`` writes one dtype per file)."""
    import csv

    rows = None
    for dtype in dtypes:
        path = write_manifest(directory, name, capability, ops, dtype=dtype)
        with path.open(newline="") as handle:
            part = list(csv.reader(handle))
        rows = part if rows is None else rows + part[2:]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    records._load.cache_clear()


def test_fp32_transition_goes_portable_only_with_tf32_and_fp32_rows(monkeypatch, manifests, cuda_tensors, tf32):
    fake_card(monkeypatch, A6000, (8, 6))
    x = torch.randn(1, 8, 8, 64)
    monkeypatch.setattr(dispatch, "_card_of", lambda device: (A6000, (8, 6)))        # the CPU tensors stand in for tensors on the faked card
    tf32(True)
    _write_both(manifests, A6000, (8, 6), ["transition", "layernorm"], ["bf16"])        # recorded for bf16 only
    with pytest.warns(UnrecordedCardWarning, match="transition"):
        assert _transition_route(monkeypatch, x) == "pytorch"                       # the fp32 call has no rows: PyTorch, one warning
    dispatch._unrecorded_warned.clear()
    _write_both(manifests, A6000, (8, 6), ["transition", "layernorm"], ["bf16", "fp32"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCardWarning)
        assert _transition_route(monkeypatch, x) == "portable"
    tf32(False)
    assert _transition_route(monkeypatch, x) == "pytorch"                           # full-precision fp32 keeps cuBLAS


def test_fp32_transition_h100_is_unchanged(monkeypatch, manifests, cuda_tensors, tf32):
    """H100 resolves Transition before any portable rule: the portable fp32 gate is never consulted."""
    fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    monkeypatch.setattr(dispatch, "dtype_gate", lambda *a, **k: pytest.fail("the portable fp32 gate ran on H100"))
    tf32(True)
    monkeypatch.setattr("miniworld_engine.kernels.transition.cuda.fused_generic_sm90a.available", lambda *a, **k: False)
    monkeypatch.setattr(Transition, "_torch_forward", lambda self, x: torch.zeros_like(x))
    m = Transition(64, n=2, implementation=I.MINIWORLD)
    m(torch.randn(1, 8, 8, 64))


def test_the_dot_precision_follows_torchs_tf32_switch(tf32):
    from miniworld_engine.kernels._dot_precision import dot_precision

    tf32(False)
    assert dot_precision(torch.float32) == "ieee" and dot_precision(torch.bfloat16) == "ieee"
    tf32(True)
    assert dot_precision(torch.float32) == "tf32" and dot_precision(torch.bfloat16) == "ieee"


# ---- (4) TriAttn variants with fp32 master parameters ------------------------------------------------------------------------------
def test_variant_refusal_takes_fp32_master_parameters_under_bf16(cuda_tensors):
    from miniworld_engine.kernels.triangle_attention.triton.variants import refusal

    z = torch.zeros(1, 16, 16, 64, dtype=torch.bfloat16)
    w32, w16 = torch.zeros(8, 64), torch.zeros(8, 64, dtype=torch.bfloat16)
    assert refusal(z, (w32, w16, None), 4, 128) is None
    assert refusal(z.float(), (w32,), 4, 128) is not None            # activations stay bf16
    assert refusal(z, (w32.half(),), 4, 128) is not None
