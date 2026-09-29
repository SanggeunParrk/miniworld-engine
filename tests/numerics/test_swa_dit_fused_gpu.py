"""The fused SWA atom DiT block (kernels/swa_dit) on a GPU: against its fp32 reference, and against the module.

Moved from team-gm (commit 14f2c73), so what is pinned here is the contract team-gm's ``SWAAtomTransformer`` called it
with: A augments x B batch elements on N = A*B rows, the adaLN modulation hoisted once per (b, atom) from the
augment-invariant conditioning, ragged front-packed ``seqused`` (including a row shorter than the half window), bf16,
d_atom 128 / 4 heads / half window 64 / SwiGLU hidden 256. Every differentiable input is checked -- q, the five block
weights, and the conditioning and adaLN weight THROUGH the hoisted modulation (so dmod is covered).

Paths: the Triton kernels everywhere (the hand-CUDA stages pinned off), both FFN weight-gradient modes, the fp32 dq1
option, and the hand-CUDA stages on sm_90 (skipped elsewhere, or when the extension does not build here). A = 1 (the
input feature embedder's call) is among the shapes.

fp32 (the second half): the fp32 kernels against the reference, against the module's own per-op fp32 path (both matmul
precisions), against team-gm's unfused fp32 SWAAtomTransformer when team-gm is importable, compiled fullgraph, and the
ops' schema / fakes.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.swa_dit.interface import (
    swa_dit_block,
    swa_dit_hoist_modulation,
)
from miniworld_engine.kernels.swa_dit.reference import (
    swa_dit_block_reference,
    swa_dit_hoist_modulation_reference,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_atom_attention.module import _flash_backend
from miniworld_engine.modules.swa_dit import SWADiTBlock

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF = "cuda", torch.bfloat16
C, H, HIDDEN, HW = 128, 4, 256, 64

TRITON = {"swa_dit_qkvg_fwd_cuda": False, "swa_dit_ffn_fwd_cuda": False, "swa_dit_ffn_bwd_cuda": False}
CUDA = {"engine_backend": "auto", "swa_dit_qkvg_fwd_cuda": True, "swa_dit_ffn_fwd_cuda": True,
        "swa_dit_ffn_bwd_cuda": True, "swa_dit_ffn_dw": "mat"}
PATHS = {
    "triton": TRITON,
    "triton_fused_dw": {**TRITON, "swa_dit_ffn_dw": "fused"},
    "triton_dq1_fp32": {**TRITON, "swa_dit_dq1": "fp32"},
    "cuda": CUDA,
}


@pytest.fixture(autouse=True)
def restore_settings():
    previous = settings.current()
    yield
    settings.configure(**asdict(previous))


def _needs_cuda_stages():
    from miniworld_engine.kernels.swa_dit.cuda.loader import ERRORS, extension, is_sm90

    if not is_sm90():
        pytest.skip("the hand-CUDA stages are sm_90a kernels")
    missing = {w: ERRORS.get(w) for w in ("qkvg", "fwd", "bwd") if extension(w) is None}
    if missing:
        pytest.skip(f"the swa_dit CUDA extensions did not build here: {missing}")


def _use(path: str) -> None:
    if path == "cuda":
        _needs_cuda_stages()
    settings.configure(**PATHS[path])


def _case(a: int, b: int, s: int, seed: int = 0, dtype: torch.dtype = BF):
    """Leaves (``dtype``) for one block call: q, c_base, wmod, and the five weights; plus cos/sin/seqused (fp32, int32)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, dtype)

    n = a * b
    leaves = [r(n, s, C), r(b, s, C), r(6 * C, C, scale=0.05),
              r(3 * C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5),
              r(2 * HIDDEN, C, scale=C ** -0.5), r(C, HIDDEN, scale=HIDDEN ** -0.5)]
    angles = torch.randn(b, s, C // H // 2, generator=g) * 3.0
    lengths = [s, max(1, s - 17), 5, s - 1] * n
    seqused = torch.tensor(lengths[:n], dtype=torch.int32, device=DEV)
    return leaves, angles.cos().to(DEV), angles.sin().to(DEV), seqused


def _fused(leaves, cos, sin, seqused, b):
    q, c_base, wmod, *weights = leaves
    mod = swa_dit_hoist_modulation(c_base, wmod)
    return swa_dit_block(q, mod, cos, sin, seqused, *weights, b, half_window=HW)


def _reference(leaves, cos, sin, seqused, b):
    q, c_base, wmod, *weights = leaves
    mod = swa_dit_hoist_modulation_reference(c_base, wmod)
    return swa_dit_block_reference(q, mod, cos, sin, seqused, *weights, b, half_window=HW)


def _run(fn, leaves, dy, *args):
    live = [t.detach().clone().requires_grad_() for t in leaves]
    out = fn(live, *args)
    out.backward(dy.to(out.dtype))
    return out.detach(), [t.grad for t in live]


def _rel(a, e):
    a, e = a.float(), e.float()
    return ((a - e).norm() / e.norm().clamp_min(1e-30)).item(), ((a - e).abs().max() / e.abs().max().clamp_min(1e-30)).item()


NAMES = ("q", "c_base", "w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize(("a", "b", "s"), [(48, 1, 257), (3, 2, 333), (8, 2, 130), (1, 3, 300)])
def test_the_block_matches_its_fp32_reference(path, a, b, s):
    """Output and every gradient within bf16's band of the fp32 reference, on the same bf16 values."""
    _use(path)
    torch.manual_seed(0)
    leaves, cos, sin, seqused = _case(a, b, s)
    dy = torch.randn(a * b, s, C, device=DEV, dtype=BF)      # bf16, so both sides see the same upstream values
    out, grads = _run(_fused, leaves, dy, cos, sin, seqused, b)
    ref, ref_grads = _run(_reference, [t.float() for t in leaves], dy, cos, sin, seqused, b)
    failures = []
    for name, got, want in [("out", out, ref), *zip(NAMES, grads, ref_grads, strict=True)]:
        assert got is not None, name
        assert torch.isfinite(got).all(), name
        frob, peak = _rel(got, want)
        if frob > 3e-2 or peak > 5e-2:
            failures.append(f"{name}: rel_frob={frob:.2e} rel_max={peak:.2e}")
    assert not failures, f"{path} A={a} B={b} S={s}:\n  " + "\n  ".join(failures)


def test_the_cuda_stages_agree_with_the_triton_kernels():
    """Same rounding points by construction (team-gm wrote the CUDA stages against the Triton ones)."""
    _needs_cuda_stages()
    leaves, cos, sin, seqused = _case(48, 1, 257, seed=3)
    dy = torch.randn(48, 257, C, device=DEV, dtype=BF)
    settings.configure(**TRITON)
    out_t, grads_t = _run(_fused, leaves, dy, cos, sin, seqused, 1)
    settings.configure(**CUDA)
    out_c, grads_c = _run(_fused, leaves, dy, cos, sin, seqused, 1)
    for name, got, want in [("out", out_c, out_t), *zip(NAMES, grads_c, grads_t, strict=True)]:
        frob, _peak = _rel(got, want)
        assert frob < 1e-2, (name, frob)


@pytest.mark.parametrize("path", ["triton", "cuda"])
def test_inference_and_training_forward_agree(path):
    """No gradient recorded -> the inference forward (nothing saved); same numbers as the training forward."""
    _use(path)
    leaves, cos, sin, seqused = _case(3, 2, 333, seed=4)
    with torch.no_grad():
        inference = _fused(leaves, cos, sin, seqused, 2)
    training, _ = _run(_fused, leaves, torch.randn(6, 333, C, device=DEV, dtype=BF), cos, sin, seqused, 2)
    torch.testing.assert_close(inference, training, rtol=1e-2, atol=1e-2)


def _block(active=True, dtype=BF):
    torch.manual_seed(11)
    block = SWADiTBlock(C, C, H, implementation=ImplementationType.MINIWORLD).to(DEV, dtype)
    if active:
        with torch.no_grad():
            block.adaln_modulation[1].weight.normal_(std=0.05)
    return block


def _module_case(a, b, s, seed=5, dtype=BF):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(a * b, s, C, generator=g).to(DEV, dtype)
    c_base = torch.randn(b, s, C, generator=g).to(DEV, dtype)
    angles = torch.randn(b, s, C // H // 2, generator=g).to(DEV) * 3.0
    lengths = torch.tensor([s, s - 9, 7][: a * b] + [s] * max(0, a * b - 3), device=DEV)
    valid = torch.arange(s, device=DEV)[None] < lengths[:, None]
    return x, c_base, build_attention_params(angles.cos(), angles.sin(), valid, a)


def _module_grads(block, fn, x, c, ap, dy):
    block = copy.deepcopy(block)
    x = x.detach().clone().requires_grad_()
    c = c.detach().clone().requires_grad_()
    out = fn(block, x, c, ap)
    out.backward(dy)
    return out.detach(), [x.grad, c.grad, *(p.grad for p in block.parameters())]


@pytest.mark.parametrize("path", ["triton", "cuda"])
@pytest.mark.parametrize("hoisted", [False, True])
def test_the_module_takes_the_fused_block_and_matches_its_per_op_path(path, hoisted):
    """SWADiTBlock(miniworld): the fused block against the same module with `swa_dit_fused=False` (the per-op path,
    FlashAttention windowed attention). ``hoisted`` goes through `forward_hoisted` with the [B, S] conditioning."""
    if _flash_backend(torch.device(DEV)) is None:
        pytest.skip("the per-op path needs a flash backend")
    _use(path)
    a, b, s = 3, 2, 300
    block = _block()
    x, c_base, ap = _module_case(a, b, s)
    c = c_base if hoisted else c_base.repeat(a, 1, 1)
    assert block.fused_refusal(x, c, ap) is None
    fn = (lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_)) if hoisted else (lambda m, x_, c_, ap_: m(x_, c_, ap_))
    dy = torch.randn(a * b, s, C, device=DEV, dtype=BF)
    out, grads = _module_grads(block, fn, x, c, ap, dy)
    settings.configure(swa_dit_fused=False)
    ref, ref_grads = _module_grads(block, fn, x, c, ap, dy)
    names = ["x", "cond", *(n for n, _ in block.named_parameters())]
    failures = []
    for name, got, want in [("out", out, ref), *zip(names, grads, ref_grads, strict=True)]:
        assert got is not None, name
        assert want is not None, name
        frob, _peak = _rel(got, want)
        if frob > 3e-2:
            failures.append(f"{name}: rel_frob={frob:.2e}")
    assert not failures, "\n  ".join(failures)


def test_zero_initialised_gates_keep_the_block_the_identity():
    """adaLN-Zero: with the modulation weight at its zero init the fused block returns its input exactly."""
    _use("triton")
    block = _block(active=False)
    x, c_base, ap = _module_case(2, 1, 200)
    c = c_base.repeat(2, 1, 1)
    assert block.fused_refusal(x, c, ap) is None
    with torch.no_grad():
        torch.testing.assert_close(block(x, c, ap), x, rtol=0, atol=0)


@pytest.mark.skipif(settings.current().compile_wrap != "custom_op", reason="fullgraph needs the custom_op launches")
def test_the_fused_module_compiles_fullgraph():
    """Two opaque ops and an autograd.Function: torch.compile(fullgraph=True) traces the module with no break."""
    _use("triton")
    block = _block()
    x, c_base, ap = _module_case(3, 2, 200)
    c = c_base.repeat(3, 1, 1)
    dy = torch.randn(6, 200, C, device=DEV, dtype=BF)
    eager, eager_grads = _module_grads(block, lambda m, x_, c_, ap_: m(x_, c_, ap_), x, c, ap, dy)
    compiled, compiled_grads = _module_grads(
        block, lambda m, x_, c_, ap_: torch.compile(m, fullgraph=True)(x_, c_, ap_), x, c, ap, dy)
    # Inductor reorders the per-row modulation (silu + matmul, plain torch ops), so a few bf16 outputs round the other
    # way (1 ULP, 7 of 153600 elements measured on H100); compare the output the way the gradients are compared.
    frob, _peak = _rel(compiled, eager)
    assert frob < 1e-2, frob
    for got, want in zip(compiled_grads, eager_grads, strict=True):
        frob, _peak = _rel(got, want)
        assert frob < 1e-2, frob


# ---------------------------------------------------------------------------------------------------------------------
# fp32: the kernels in triton/forward_fp32.py / backward_fp32.py around the shared bf16-operand window attention.
# ---------------------------------------------------------------------------------------------------------------------
F32 = torch.float32


@pytest.fixture
def matmul_precision():
    """Restore the process's fp32 matmul precision after a test that sets it."""
    previous = torch.get_float32_matmul_precision()
    yield torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize(("a", "b", "s"), [(48, 1, 257), (3, 2, 333), (1, 3, 300)])
def test_the_fp32_block_matches_its_fp32_reference(a, b, s):
    """fp32 in, fp32 out, every gradient in its input's dtype, within the band the bf16 attention operands leave.

    The attention core runs on bf16 q / k / v / P / dO exactly as the per-op path's FlashAttention-4 does, and that is
    the whole error budget: measured on an H100, out ~1.3e-4 and the q / conditioning / FFN gradients <= 7e-4 relative
    Frobenius, the attention-side weight gradients (Wqkv, gate, out) ~3-4e-3 -- the same as the per-op fp32 path. The
    bounds are ~5x those measurements.
    """
    torch.manual_seed(0)
    leaves, cos, sin, seqused = _case(a, b, s, dtype=F32)
    dy = torch.randn(a * b, s, C, device=DEV)
    out, grads = _run(_fused, leaves, dy, cos, sin, seqused, b)
    assert out.dtype == F32
    assert all(g.dtype == F32 for g in grads)
    ref, ref_grads = _run(_reference, leaves, dy, cos, sin, seqused, b)
    attention_side = {"w_qkv", "w_gate", "w_out"}
    failures = []
    for name, got, want in [("out", out, ref), *zip(NAMES, grads, ref_grads, strict=True)]:
        assert torch.isfinite(got).all(), name
        frob, peak = _rel(got, want)
        bound = 2e-2 if name in attention_side else 5e-3
        if frob > bound or peak > 2 * bound:
            failures.append(f"{name}: rel_frob={frob:.2e} rel_max={peak:.2e} (bound {bound:.0e})")
    assert not failures, f"fp32 A={a} B={b} S={s}:\n  " + "\n  ".join(failures)


def _reference_module(block, x, c, ap):
    """reference.py on a module's weights, with one modulation row per row of ``c`` (B = c.shape[0])."""
    b = c.shape[0]
    mod = swa_dit_hoist_modulation_reference(c, block.adaln_modulation[1].weight)
    return swa_dit_block_reference(x, mod, ap[0][:b], ap[1][:b], ap[2], block.attn.Wqkv.weight, block.attn.gate_proj.weight,
                                   block.attn.out_proj.weight, block.ffn.w_up.weight, block.ffn.w_down.weight, b,
                                   half_window=HW)


@pytest.mark.parametrize("precision", ["highest", "medium"])
def test_the_fp32_fused_block_is_as_accurate_as_the_per_op_path(precision, matmul_precision):
    """The fp32 module on the fused block against the same module on its per-op path (``swa_dit_fused=False``: engine
    rmsnorm_adamod, cuBLAS projections, FlashAttention-4 in bf16, triton_swiglu_ffn), each against the fp32 reference.
    ``medium`` is MiniWorld's trainer setting (cuBLAS TF32). The fused error may not exceed the per-op error by more than
    1.5x (plus 1e-4 for tensors whose error is at rounding level)."""
    if _flash_backend(torch.device(DEV)) is None:
        pytest.skip("the per-op path needs a flash backend")
    a, b, s = 3, 2, 300
    block = _block(dtype=F32)
    x, c_base, ap = _module_case(a, b, s, dtype=F32)
    assert block.fused_refusal(x, c_base, ap) is None
    dy = torch.randn(a * b, s, C, device=DEV)
    matmul_precision(precision)
    fused, fused_grads = _module_grads(block, lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_), x, c_base, ap, dy)
    settings.configure(swa_dit_fused=False)
    perop, perop_grads = _module_grads(block, lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_), x, c_base, ap, dy)
    matmul_precision("highest")
    ref, ref_grads = _module_grads(block, _reference_module, x, c_base, ap, dy)
    names = ["out", "x", "cond", *(n for n, _ in block.named_parameters())]
    failures = []
    for name, f, p, r in zip(names, [fused, *fused_grads], [perop, *perop_grads], [ref, *ref_grads], strict=True):
        assert f.dtype == F32, name
        err_f, err_p = _rel(f, r)[0], _rel(p, r)[0]
        if err_f > 1.5 * err_p + 1e-4:
            failures.append(f"{name}: fused {err_f:.2e} vs per-op {err_p:.2e}")
    assert not failures, f"precision={precision}:\n  " + "\n  ".join(failures)


def test_the_fp32_module_takes_the_fused_block_and_mixed_dtypes_do_not():
    block = _block(dtype=F32)
    x, c_base, ap = _module_case(2, 1, 150, dtype=F32)
    assert block.fused_refusal(x, c_base, ap) is None
    assert "mixed dtypes" in str(block.fused_refusal(x.to(BF), c_base, ap))


@pytest.mark.skipif(settings.current().compile_wrap != "custom_op", reason="fullgraph needs the custom_op launches")
def test_the_fp32_fused_module_compiles_fullgraph():
    block = _block(dtype=F32)
    x, c_base, ap = _module_case(3, 2, 200, dtype=F32)
    c = c_base.repeat(3, 1, 1)
    dy = torch.randn(6, 200, C, device=DEV)
    eager, eager_grads = _module_grads(block, lambda m, x_, c_, ap_: m(x_, c_, ap_), x, c, ap, dy)
    compiled, compiled_grads = _module_grads(
        block, lambda m, x_, c_, ap_: torch.compile(m, fullgraph=True)(x_, c_, ap_), x, c, ap, dy)
    frob, _peak = _rel(compiled, eager)
    assert frob < 1e-4, frob
    for got, want in zip(compiled_grads, eager_grads, strict=True):
        assert got.dtype == F32
        frob, _peak = _rel(got, want)
        assert frob < 1e-3, frob


@pytest.mark.skipif(settings.current().compile_wrap != "custom_op", reason="opcheck needs the custom_op launches")
def test_the_fp32_ops_satisfy_their_contract():
    """Schema and fake against the fp32 forward (both save modes) and backward: the fakes carry the per-output dtypes
    (bf16 attention operands, q's dtype elsewhere), which only an fp32 call can tell apart."""
    from miniworld_engine.kernels.swa_dit.dispatch import swa_dit_block_fwd

    leaves, cos, sin, seqused = _case(3, 2, 200, dtype=F32)
    q, c_base, wmod, *weights = (t.detach() for t in leaves)
    mod = swa_dit_hoist_modulation(c_base, wmod)
    cos, sin = cos.reshape(-1, cos.shape[-1]), sin.reshape(-1, sin.shape[-1])
    fwd_op = torch.ops.miniworld_engine.swa_dit_block_fwd.default
    bwd_op = torch.ops.miniworld_engine.swa_dit_block_bwd.default
    for save in (False, True):
        torch.library.opcheck(fwd_op, (q, mod, cos, sin, seqused, *weights, 2, HW, 1e-7, save),
                              test_utils=("test_schema", "test_faketensor"))
    saved = swa_dit_block_fwd(q, mod, cos, sin, seqused, *weights, 2, HW, 1e-7, True)
    dy = torch.randn_like(q)
    torch.library.opcheck(bwd_op, (dy, q, mod, cos, sin, seqused, *weights, *saved[1:], 2, HW, 1e-7),
                          test_utils=("test_schema", "test_faketensor"))


def test_the_fp32_block_matches_team_gms_unfused_atom_transformer():
    """team-gm's SWAAtomTransformer (block_style esmfold2, fp32, fused path off) is the fp32 path MiniWorld v1.3 runs.
    The fused block, driven the way team-gm's `_forward_fused` drives it, against it: 3 blocks, A=3 x B=2, ragged; every
    gradient within the per-op-vs-reference band measured above. Skipped without team-gm on the path."""
    tg = pytest.importorskip("team_gm.modules.blocks.swa_atom_transformer")
    if _flash_backend(torch.device(DEV)) is None:
        pytest.skip("team-gm's unfused path needs a flash backend")
    torch.manual_seed(13)
    a, b, s = 3, 2, 300
    model = tg.SWAAtomTransformer(tg.SWAAtomTransformer.Config(n_block=3, fused_triton=False)).to(DEV)
    with torch.no_grad():
        for blk in model.blocks:
            blk.adaln_modulation[1].weight.normal_(std=0.05)
    x, c_base, ap = _module_case(a, b, s, dtype=F32)
    dy = torch.randn(a * b, s, C, device=DEV)

    def fused(m, x_, c_, ap_):
        cos, sin, seqused = ap_[0][:b], ap_[1][:b], ap_[2]
        for blk in m.blocks:
            mod = swa_dit_hoist_modulation(c_, blk.adaln_modulation[1].weight)
            x_ = swa_dit_block(x_, mod, cos, sin, seqused, blk.attn.Wqkv.weight, blk.attn.gate_proj.weight,
                               blk.attn.out_proj.weight, blk.ffn.w_up.weight, blk.ffn.w_down.weight, b, half_window=HW)
        return x_

    def unfused(m, x_, c_, ap_):
        return m(x_, c_.repeat(a, 1, 1), ap_, c_base=c_)

    got, got_grads = _module_grads(model, fused, x, c_base, ap, dy)
    want, want_grads = _module_grads(model, unfused, x, c_base, ap, dy)
    names = ["out", "x", "cond", *(n for n, _ in model.named_parameters())]
    failures = []
    for name, g, w in zip(names, [got, *got_grads], [want, *want_grads], strict=True):
        frob, _peak = _rel(g, w)
        if frob > (3e-2 if "attn." in name else 1e-2):
            failures.append(f"{name}: rel_frob={frob:.2e}")
    assert not failures, "\n  ".join(failures)
