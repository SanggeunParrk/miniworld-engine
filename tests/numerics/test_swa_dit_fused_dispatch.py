"""The fused SWA atom DiT block (kernels/swa_dit): who takes it, what a call it cannot serve is told, and what it means.

CPU only. Every refusal below is decided from tensor metadata before a kernel could launch; the reference is checked
against the module's own PyTorch statements, which is what makes it the right thing for the GPU checkers to compare
the kernels with.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.swa_dit.interface import (
    refusal,
    swa_dit_hoist_modulation,
)
from miniworld_engine.kernels.swa_dit.reference import (
    swa_dit_block_reference,
    swa_dit_hoist_modulation_reference,
)
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_dit import SWADiTBlock


@pytest.fixture(autouse=True)
def restore_policy():
    previous = settings.current()
    yield
    settings.configure(**asdict(previous))


def _modulation_weight(block: SWADiTBlock) -> torch.Tensor:
    """The adaLN projection's weight (``adaln_modulation`` is SiLU -> Linear)."""
    linear = block.adaln_modulation[1]
    assert isinstance(linear, torch.nn.Linear)
    return linear.weight


def _operands(dtype=torch.bfloat16, c=128, hidden=256, n=3, s=11, half=16):
    q = torch.randn(n, s, c, dtype=dtype)
    cos = torch.randn(n, s, half)
    weights = [torch.randn(*shape, dtype=dtype) for shape in ((3 * c, c), (c, c), (c, c), (2 * hidden, c), (c, hidden))]
    seqused = torch.full((n,), s, dtype=torch.int32)
    return q, cos, cos.clone(), seqused, weights


def test_every_launch_keys_on_the_augment_count():
    """The input feature embedder calls the block with A = 1 and diffusion with its num_augment; the backward row tiles
    (SP augments x AT atoms) want different shapes for the two, so every Triton launch folds A into its shape key,
    A = 1 included."""
    import ast
    from pathlib import Path

    import miniworld_engine.kernels.swa_dit as family

    tree = ast.parse(Path(family.__file__).with_name("dispatch.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "atom_key"]
    launches = [n for n in ast.walk(tree)
                if isinstance(n, ast.Subscript) and getattr(n.value, "id", "").endswith("_kernel")]
    assert launches, "no Triton launches found in dispatch.py"
    assert len(calls) == len(launches), f"{len(launches)} Triton launches but {len(calls)} atom_key calls"
    missing = [n.lineno for n in calls if {"A", "C"} - {k.arg for k in n.keywords}]
    assert not missing, f"atom_key calls without A= / C= at dispatch.py lines {missing}"
    from miniworld_engine.kernels.swa_dit.dispatch import _augments

    assert (_augments(48, 1), _augments(6, 6), _augments(5, 1), _augments(10**6, 1)) == (48, 1, 5, 4095)


def test_the_defaults_are_team_gms():
    """The switches replace team-gm's environment variables one for one, with the same defaults."""
    s = settings.Settings()
    assert s.swa_dit_fused is True
    assert (s.swa_dit_qkvg_fwd_cuda, s.swa_dit_ffn_fwd_cuda, s.swa_dit_ffn_bwd_cuda) == (True, True, True)
    assert s.swa_dit_ffn_dw == "mat"
    assert s.swa_dit_dq1 == "bf16"


@pytest.mark.parametrize(("change", "reason"), [
    ({}, "CUDA"),
    ({"dtype": torch.float32}, "CUDA"),          # all-fp32 is served (off-GPU is the only reason left)
    ({"dtype": torch.float16}, "bf16 or fp32"),
    ({"c": 64}, "d_atom=128"),
    ({"hidden": 512}, "SwiGLU hidden"),
    ({"half": 8}, "fp32 [..., 16]"),
])
def test_refusals_name_the_reason(change, reason):
    q, cos, sin, seqused, w = _operands(**change)
    assert reason in str(refusal(q, cos, sin, seqused, *w, n_head=4, half_window=64))


def test_mixed_dtypes_are_refused():
    """bf16 or fp32 activations and conditioning, with the weights in that dtype or fp32 (an fp32 master over bf16
    activations, cast for the kernels outside autograd): an fp32 stream against bf16 weights, or bf16 conditioning against an
    fp32 stream, is not a call either kernel set was written for."""
    q, cos, sin, seqused, w = _operands()
    assert "mixed dtypes" in str(refusal(q.float(), cos, sin, seqused, *w, n_head=4, half_window=64))
    assert "mixed dtypes" not in str(refusal(q, cos, sin, seqused, w[0], w[1], w[2], w[3], w[4].float(), n_head=4, half_window=64))
    assert "mixed dtypes" in str(refusal(q, cos, sin, seqused, w[0], w[1], w[2], w[3], w[4].half(), n_head=4, half_window=64))
    cond = torch.randn(3, 11, 128)
    assert "mixed dtypes" in str(refusal(q, cos, sin, seqused, *w, n_head=4, half_window=64, cond=cond,
                                         wmod=torch.randn(768, 128, dtype=torch.bfloat16)))


def test_the_contract_widths_are_refused_by_name():
    q, cos, sin, seqused, w = _operands()
    assert "half_window" in str(refusal(q, cos, sin, seqused, *w, n_head=4, half_window=32))
    assert "4 heads" in str(refusal(q, cos, sin, seqused, *w, n_head=8, half_window=64))
    assert "int32" in str(refusal(q, cos, sin, seqused.long(), *w, n_head=4, half_window=64))
    cond = torch.randn(3, 11, 96, dtype=torch.bfloat16)
    assert "adaLN weight" in str(refusal(q, cos, sin, seqused, *w, n_head=4, half_window=64, cond=cond,
                                         wmod=torch.randn(768, 128, dtype=torch.bfloat16)))


def test_the_module_refuses_off_gpu_and_names_why():
    block = SWADiTBlock(128, 128, 4, implementation=I.MINIWORLD).bfloat16()
    x = torch.randn(2, 9, 128, dtype=torch.bfloat16)
    angle = torch.randn(1, 9, 16)
    ap = build_attention_params(angle.cos(), angle.sin(), torch.ones(2, 9, dtype=torch.bool), 2)
    assert "CUDA" in str(block.fused_refusal(x, x, ap))
    settings.configure(swa_dit_fused=False)
    assert "swa_dit_fused" in str(block.fused_refusal(x, x, ap))
    assert "pytorch" in str(SWADiTBlock(128, 128, 4).fused_refusal(x, x, ap))


def test_forward_hoisted_is_forward_on_the_repeated_conditioning():
    """Where the fused block cannot serve (CPU here), forward_hoisted is literally forward(x, c_base.repeat(A))."""
    torch.manual_seed(5)
    block = SWADiTBlock(32, 24, 4, half_window=2)
    with torch.no_grad():
        _modulation_weight(block).normal_(std=0.1)
    a, b, s = 3, 2, 7
    x = torch.randn(a * b, s, 32)
    c_base = torch.randn(b, s, 24)
    angle = torch.randn(b, s, 4)
    valid = torch.ones(a * b, s, dtype=torch.bool)
    valid[1, 5:] = False
    ap = build_attention_params(angle.cos(), angle.sin(), valid, a)
    torch.testing.assert_close(block.forward_hoisted(x, c_base, ap), block(x, c_base.repeat(a, 1, 1), ap),
                               rtol=0, atol=0)


def test_the_hoisted_modulation_is_the_reference():
    c = torch.randn(2, 5, 24)
    w = torch.randn(6 * 32, 24)
    torch.testing.assert_close(swa_dit_hoist_modulation(c, w), swa_dit_hoist_modulation_reference(c, w))


@pytest.mark.parametrize("hoisted", [False, True])
def test_the_reference_is_the_modules_pytorch_statements(hoisted):
    """reference.py against SWADiTBlock(implementation=pytorch), fp32, outputs and every gradient.

    ``hoisted`` runs the reference with one modulation row per batch element (B < N) against the module on the
    repeated conditioning, which is the augment structure the fused block exploits.
    """
    torch.manual_seed(7)
    d, dc, heads, hw = 32, 24, 4, 2
    a, b, s = (3, 2, 9) if hoisted else (1, 3, 9)
    n = a * b
    block = SWADiTBlock(d, dc, heads, half_window=hw)
    with torch.no_grad():
        _modulation_weight(block).normal_(std=0.2)
    angle = torch.randn(b, s, d // heads // 2)
    valid = torch.ones(n, s, dtype=torch.bool)
    valid[1, 6:] = False
    valid[n - 1, 3:] = False
    ap = build_attention_params(angle.cos(), angle.sin(), valid, a)
    x = torch.randn(n, s, d, requires_grad=True)
    c_base = torch.randn(b, s, dc, requires_grad=True)
    expected = block(x, c_base.repeat(a, 1, 1), ap)
    dy = torch.randn_like(expected)
    expected.backward(dy)
    params = [_modulation_weight(block), block.attn.Wqkv.weight, block.attn.gate_proj.weight,
              block.attn.out_proj.weight, block.ffn.w_up.weight, block.ffn.w_down.weight]
    want = [x.grad, c_base.grad, *(p.grad for p in params)]

    leaves = [t.detach().clone().requires_grad_() for t in (x, c_base, *params)]
    rx, rc, rmod_w, rwqkv, rwg, rwo, rwu, rwd = leaves
    mod = swa_dit_hoist_modulation_reference(rc, rmod_w)
    cos, sin, seqused = ap[0][:b], ap[1][:b], ap[2]
    got = swa_dit_block_reference(rx, mod, cos, sin, seqused, rwqkv, rwg, rwo, rwu, rwd, b, half_window=hw, n_head=heads)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    got.backward(dy)
    for i, (g, w) in enumerate(zip([t.grad for t in leaves], want, strict=True)):
        assert g is not None, i
        assert w is not None, i
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4, msg=f"gradient {i}")
