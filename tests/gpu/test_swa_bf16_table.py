"""H100: the SWA atom block's training modulation table in bf16 (3.0.1).

* kernel level: the slab kernels read a bf16 modulation (and write its gradient in bf16) with the same arithmetic as an fp32 table holding the
  same bf16 values: forward outputs / saved tensors, dq and the weight gradients are bit-identical, dmod equals the fp32 dmod rounded to bf16;
* module level: a master training step with the bf16 table (``dispatch._BF16_TABLE``) is deterministic (two calls, the same bytes) and within the
  engine tolerance of the float64 PyTorch statements for the output and every gradient, as the fp32 table of 3.0.0 is;
* the table is bf16 only where the stages are the composed ones (``dispatch.bf16_table``): fp32 blocks and the A/B switch keep the fp32 table.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from foldjet_engine.kernels.swa_dit import dispatch
from foldjet_engine.modules.atom_dit import AtomDiTMiniWorld

pytestmark = [pytest.mark.gpu]

_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _module(half_window: int = 64) -> AtomDiTMiniWorld:
    torch.manual_seed(0)
    m = AtomDiTMiniWorld(half_window=half_window).cuda()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            r = torch.randn(p.shape, generator=g)
            p.copy_(1.0 + 0.1 * r if p.ndim == 1 else r * p.shape[-1] ** -0.5)
    return m


def _inputs(m, length: int, samples: int):
    return m.make_inputs(
        length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )


def _rel(x: torch.Tensor, gold: torch.Tensor) -> float:
    return float((x.double() - gold.double()).abs().max() / gold.double().abs().max().clamp_min(1e-30))


@pytest.fixture
def table_switch():
    saved = dispatch._BF16_TABLE[0]
    yield
    dispatch._BF16_TABLE[0] = saved


@pytest.mark.skipif(not _H100, reason="the H100 (sm_90) path")
@pytest.mark.parametrize(("length", "samples"), [(256, 1), (200, 4)])  # A = 1 per module call (B = N); the augment count is the kernels' own
def test_slab_kernels_bf16_table_matches_widened(length, samples):
    m = _module()
    inp = _inputs(m, length, samples)
    x, cond, params = inp["x"].to(torch.bfloat16), inp["cond"].to(torch.bfloat16), inp["attention_params"]
    cos, sin, seqused = (t.reshape(-1, t.shape[-1]).contiguous() if t.ndim > 1 else t for t in params[:3])
    n, _s, _ = x.shape
    w = [p.detach().to(torch.bfloat16).contiguous() for p in (m.attn.Wqkv.weight, m.attn.gate_proj.weight, m.attn.out_proj.weight, m.ffn.w_up.weight, m.ffn.w_down.weight)]
    c = cond.reshape(-1, cond.shape[-1])
    mod_bf = F.linear(F.silu(c), m.adaln_modulation[1].weight.detach().to(torch.bfloat16))  # [N*S, 768] bf16  # pyright: ignore[reportCallIssue]  -- adaln_modulation[1] is the Linear (nn.Sequential indexing is untyped)
    mod_f = mod_bf.float()
    q = x.contiguous()
    fwd = {}
    for tag, mod in (("bf", mod_bf), ("f32", mod_f)):
        fwd[tag] = dispatch.swa_dit_block_fwd(q, mod, cos, sin, seqused, *w, n, m.attn.half_window, 1e-5, True)  # pyright: ignore[reportCallIssue]  -- the weight list unpacked into the launcher's fixed parameters
    for a, b in zip(fwd["bf"], fwd["f32"], strict=True):
        assert torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a, b.view(torch.int16) if b.dtype == torch.bfloat16 else b)
    dy = torch.randn(q.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(3)).to(torch.bfloat16)
    bwd = {tag: dispatch.swa_dit_block_bwd(dy, q, mod, cos, sin, seqused, *w, *fwd[tag][1:], n, m.attn.half_window, 1e-5) for tag, mod in (("bf", mod_bf), ("f32", mod_f))}  # pyright: ignore[reportCallIssue]
    assert bwd["bf"][1].dtype == torch.bfloat16 and bwd["f32"][1].dtype == torch.float32
    assert torch.equal(bwd["bf"][1].view(torch.int16), bwd["f32"][1].to(torch.bfloat16).view(torch.int16))
    for a, b in zip(bwd["bf"][:1] + bwd["bf"][2:], bwd["f32"][:1] + bwd["f32"][2:], strict=True):
        assert torch.equal(a, b)


@pytest.mark.skipif(not _H100, reason="the H100 (sm_90) path")
@pytest.mark.parametrize("half_window", [64, 500000])
@pytest.mark.parametrize("samples", [1, 3])
def test_block_step_bf16_table(table_switch, half_window, samples):
    m = _module(half_window)
    inp = _inputs(m, 320, samples)
    x, cond, params = inp["x"], inp["cond"], inp["attention_params"]
    ref = copy.deepcopy(m).double().set_backend("pytorch")
    wide = tuple(t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in params)
    xr, cr = (t.detach().double().requires_grad_() for t in (x, cond))
    out_ref = ref(xr, cr, wide)
    up = torch.randn(out_ref.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64).cuda()
    out_ref.backward(up)
    m.set_backend("cuda")

    def step(table: bool):
        dispatch._BF16_TABLE[0] = table
        xb, cb = (t.detach().to(torch.bfloat16).requires_grad_() for t in (x, cond))
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(xb, cb, params)
        out.backward(up.to(out.dtype))
        return [out.detach(), xb.grad, cb.grad, *[p.grad for p in m.parameters()]]

    assert dispatch.bf16_table(x.to(torch.bfloat16), cond.to(torch.bfloat16), m.attn.half_window) is True
    new_a, new_b, old = step(True), step(True), step(False)
    names = ["out", "dx", "dcond", *[n for n, _ in m.named_parameters()]]
    golds = [out_ref.detach(), xr.grad, cr.grad, *[p.grad for p in ref.parameters()]]
    tol = 6e-2  # bf16 operands, as the other block tests
    for name, a, b, o, gd in zip(names, new_a, new_b, old, golds, strict=True):
        if half_window < 65536:
            assert torch.equal(a, b), f"{name}: the bf16 table must be deterministic"
        else:  # global attention: FlashAttention-4's backward accumulates dq with atomics (as with the fp32 table)
            assert _rel(a, b) < 2e-2, f"{name}: repeat differs by {_rel(a, b):.3g}"
        assert _rel(a, gd) < tol, f"{name} vs fp64: {_rel(a, gd):.3g}"
        # the table's rounding is the reference's own (a bf16 Linear): no worse than the fp32 table beyond a small factor
        assert _rel(a, gd) <= max(2.0 * _rel(o, gd), 8e-3), f"{name}: bf16 table {_rel(a, gd):.3g} vs fp32 table {_rel(o, gd):.3g}"


@pytest.mark.skipif(not _H100, reason="the H100 (sm_90) path")
def test_bf16_table_predicate(table_switch):
    xb = torch.zeros(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    assert dispatch.bf16_table(xb, xb, 64)
    assert not dispatch.bf16_table(xb.float(), xb.float(), 64)  # fp32 blocks keep the fp32 table
    assert not dispatch.bf16_table(xb, xb.float(), 64)
    dispatch._BF16_TABLE[0] = False
    assert not dispatch.bf16_table(xb, xb, 64)
