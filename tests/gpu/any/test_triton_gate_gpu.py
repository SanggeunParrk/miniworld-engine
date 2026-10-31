"""The record gate on the call sites wired by the 2026-10-31 audit (docs/records/triton_gate_audit_20261031.md), on a real card.

The card the gate sees is emulated (``dispatch._card_of`` names a card that is not H100 / B200, ``records.use_manifest_dir`` holds its
manifest; ``portable_on_hopper_blackwell`` lifts the H100 / B200 restriction so the kernels can run here). For each site:

* unrecorded card: one ``UnrecordedCardWarning`` and NO Triton launch of the gated kernels -- the PyTorch statements;
* recorded card: no warning, the Triton kernels launch, and the result agrees with the PyTorch statements (which validates them).
"""
from __future__ import annotations

import collections
import warnings

import pytest
import torch
from fake_card import write_manifest

from miniworld_engine import settings
from miniworld_engine.autotune import records
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import UnrecordedCardWarning

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")]

CARD, CAP = "Brand New GPU", (8, 9)


def _clear():
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()
    records._load.cache_clear()


@pytest.fixture
def card(monkeypatch, tmp_path):
    """``card(ops, dtype=, training=)`` records ``ops`` for the emulated card (``card(())``: nothing recorded)."""
    previous = settings.configure(portable_on_hopper_blackwell=True, force_portable_kernels=False)
    monkeypatch.setattr(dispatch, "_card_of", lambda device=None: (CARD, CAP))

    def record(ops, *, dtype="bf16", training=True, extra=()):
        path = tmp_path / f"{records.card_key(CARD, CAP)}.csv"
        path.unlink(missing_ok=True)
        if ops or extra:
            write_manifest(tmp_path, CARD, CAP, list(ops), dtype=dtype, training=training)
            for more_dtype, more_ops in extra:   # a second precision appended to the same manifest
                import csv

                rows = list(csv.reader(path.open()))
                for op in more_ops:
                    rows += [[k, "triton", "x", "x.py", "ok", "", more_dtype] for k in records.triton_kernels(op, more_dtype)]
                with path.open("w", newline="") as handle:
                    csv.writer(handle).writerows(rows)
        _clear()

    with records.use_manifest_dir(tmp_path):
        _clear()
        yield record
    settings.configure(portable_on_hopper_blackwell=previous.portable_on_hopper_blackwell,
                       force_portable_kernels=previous.force_portable_kernels)
    _clear()


@pytest.fixture
def launches(monkeypatch):
    import triton.runtime.jit as jit

    seen = collections.Counter()
    original = jit.JITFunction.run

    def run(self, *args, **kwargs):
        seen[f"{getattr(self.fn, '__module__', '?')}.{getattr(self.fn, '__name__', '?')}"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(jit.JITFunction, "run", run)
    return seen


def _of(seen, *parts):
    return sum(n for k, n in seen.items() if any(p in k for p in parts))


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _blocked_then_recorded(card, launches, ops, fn, parts, *, match, tol=2e-2, dtype="bf16"):
    """Run ``fn`` on the unrecorded card (warns, no launch of ``parts``), then recorded (silent, launches); the two outputs agree."""
    card(())
    with pytest.warns(UnrecordedCardWarning, match=match):
        ref = fn()
    assert _of(launches, *parts) == 0, dict(launches)
    card(ops, dtype=dtype)
    launches.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCardWarning)
        out = fn()
    torch.cuda.synchronize()
    assert _of(launches, *parts) > 0, dict(launches)
    for o, r in zip(out if isinstance(out, tuple) else (out,), ref if isinstance(ref, tuple) else (ref,), strict=True):
        assert _rel(o, r) < tol, (_rel(o, r), match)
    return ref, out


@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
def test_rms_norm_modulation(card, launches, grad):
    from miniworld_engine import ops

    torch.manual_seed(0)
    q = torch.randn(4, 512, 128, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    c = torch.randn(4, 512, 128, device="cuda", dtype=torch.bfloat16)
    w = [(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_(grad) for _ in range(3)]

    def fn():
        with torch.set_grad_enabled(grad):
            y, g = ops.rms_norm_modulation(q, c, *w, eps=1e-6)
            if grad:
                (y.float().square().sum() + g.float().sum()).backward()
                out = (y.detach(), g.detach(), q.grad.clone(), w[0].grad.clone())
                q.grad = None
                for t in w:
                    t.grad = None
                return out
            return y, g

    _blocked_then_recorded(card, launches, ["rms_norm_modulation"], fn, ["kernels.rmsnorm"], match="rms_norm_modulation")


@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
def test_layer_norm_linear(card, launches, grad):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x = torch.randn(1, 96, 96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    lnw = (1 + 0.1 * torch.randn(128, device="cuda")).requires_grad_(grad)
    pw = (torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(grad)

    def fn():
        with torch.set_grad_enabled(grad):
            out = ops.layer_norm_linear(x, lnw, pw, 1e-5)
            if grad:
                out.float().square().sum().backward()
                res = (out.detach(), x.grad.clone(), pw.grad.clone())
                x.grad = pw.grad = lnw.grad = None
                return res
            return out

    _blocked_then_recorded(card, launches, ["layer_norm_linear"], fn, ["layernorm_linear"], match="layer_norm_linear")


def test_gated_residual_is_never_recorded(card, launches):
    from miniworld_engine import ops

    x, g, b = (torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    card(["swa_dit", "rms_norm_modulation"])
    with pytest.warns(UnrecordedCardWarning, match="unregistered"):
        ref = ops.gated_residual(x, g, b)
    assert _of(launches, "gated_projection.triton.residual") == 0
    settings.configure(force_portable_kernels=True)
    out = ops.gated_residual(x, g, b)
    torch.cuda.synchronize()
    assert _of(launches, "gated_projection.triton.residual") > 0
    torch.testing.assert_close(out, ref)


def _trimul_weights(d, bidirectional=False):
    w = 2 * d if bidirectional else d
    gen = lambda *s: (torch.randn(*s, device="cuda") * s[-1] ** -0.5).to(torch.bfloat16)
    return {
        "norm_in": (1 + 0.1 * torch.randn(d, device="cuda"), 0.1 * torch.randn(d, device="cuda")),
        "left": gen(2 * d if not bidirectional else w, d), "left_gate": gen(2 * d if not bidirectional else w, d),
        "right": gen(w, d), "right_gate": gen(w, d),
        "norm_out": (1 + 0.1 * torch.randn(w, device="cuda"), 0.1 * torch.randn(w, device="cuda")),
        "out": gen(d, w), "gate": gen(d, d),
    }


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_triangle_multiplicative_update(card, launches, direction):
    from miniworld_engine import ops

    torch.manual_seed(0)
    d, L = 128, 96
    x = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, L, device="cuda") > 0.1
    w = _trimul_weights(d)

    def fn():
        with torch.no_grad():
            return ops.triangle_multiplicative_update(
                x, direction, mask, norm_in_weight=w["norm_in"][0], norm_in_bias=w["norm_in"][1], p_in_weight=w["left"],
                g_in_weight=w["left_gate"], norm_out_weight=w["norm_out"][0], norm_out_bias=w["norm_out"][1], p_out_weight=w["out"],
                g_out_weight=w["gate"])

    _blocked_then_recorded(card, launches, ["triangle_multiplication"], fn, ["trimul_inproj"], match="triangle_multiplication")


def test_bidirectional_triangle_multiplicative_update(card, launches):
    from miniworld_engine import ops

    torch.manual_seed(0)
    d, L = 128, 96
    x = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, L, device="cuda") > 0.1
    w = _trimul_weights(d, bidirectional=True)

    def fn():
        with torch.no_grad():
            return ops.bidirectional_triangle_multiplicative_update(
                x, mask, norm_in_weight=w["norm_in"][0], norm_in_bias=w["norm_in"][1], to_left_weight=w["left"],
                to_left_gate_weight=w["left_gate"], to_right_weight=w["right"], to_right_gate_weight=w["right_gate"],
                norm_out_weight=w["norm_out"][0], norm_out_bias=w["norm_out"][1], to_out_weight=w["out"], to_gate_weight=w["gate"])

    _blocked_then_recorded(card, launches, ["triangle_multiplication"], fn, ["trimul_inproj"], match="triangle_multiplication")


def test_conditioned_transition_tail(card, launches):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x = torch.randn(2, 384, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(2, 384, 64, device="cuda", dtype=torch.bfloat16)
    gen = lambda *s: (torch.randn(*s, device="cuda") * s[-1] ** -0.5).to(torch.bfloat16)
    kw = {"expand_a_weight": gen(256, 128), "expand_b_weight": gen(256, 128), "squeeze_weight": gen(128, 256), "to_scale_weight": gen(128, 64),
              "to_scale_bias": torch.zeros(128, device="cuda", dtype=torch.bfloat16), "n": 2}

    def fn():
        with torch.no_grad():
            return ops.conditioned_transition(x, cond, **kw)

    _blocked_then_recorded(card, launches, ["swiglu_ffn"], fn, ["kernels.transition"], match="swiglu_ffn")


def test_triangle_attention_qk_norm_asks_the_rmsnorm_record(card, launches):
    from miniworld_engine import ops

    torch.manual_seed(0)
    d, L, h = 64, 64, 4
    pair = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16)
    gen = lambda *s: (torch.randn(*s, device="cuda") * s[-1] ** -0.5).to(torch.bfloat16)
    kw = {"n_head": h, "ln_pair_weight": torch.ones(d, device="cuda", dtype=torch.bfloat16), "ln_pair_bias": torch.zeros(d, device="cuda", dtype=torch.bfloat16),
              "to_value_weight": gen(d, d), "to_bias_weight": gen(h, d), "to_gate_weight": gen(d, d), "to_out_weight": gen(d, d),
              "to_query_weight": gen(d, d), "to_key_weight": gen(d, d), "norm_query_weight": torch.ones(d // h, device="cuda", dtype=torch.bfloat16),
              "norm_key_weight": torch.ones(d // h, device="cuda", dtype=torch.bfloat16)}
    card(["triangle_attention"])                                   # the attention kernels, not the RMSNorm ones
    with torch.no_grad(), pytest.warns(UnrecordedCardWarning, match="rmsnorm"):
        ops.triangle_attention(pair, None, **kw)
    torch.cuda.synchronize()
    assert _of(launches, "kernels.rmsnorm") == 0, dict(launches)
    assert _of(launches, "kernels.triangle_attention") > 0, dict(launches)


@pytest.mark.parametrize("fused", [False, True], ids=["per_block", "fused"])
def test_the_esmfold2_atom_transformer(card, launches, fused):
    previous = settings.configure(swa_dit_h100_native=False)       # H100 runs the fused block's Triton kernels, as another card would
    try:
        _esmfold2(card, launches, fused)
    finally:
        settings.configure(swa_dit_h100_native=previous.swa_dit_h100_native)


def _esmfold2(card, launches, fused):
    from miniworld_engine.architecture.blocks.swa_atom_transformer import (
        SWAAtomTransformer,
    )
    from miniworld_engine.modules.swa_atom_attention import build_attention_params

    torch.manual_seed(0)
    config = SWAAtomTransformer.Config(n_block=2, block_style="esmfold2", fused_triton=fused)
    model = SWAAtomTransformer(config).cuda()
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    s, a = 256, 2
    q = torch.randn(a, s, 128, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(a, s, 128, device="cuda", dtype=torch.bfloat16)
    cos, sin = model.build_rope(torch.randn(1, s, 3, device="cuda"), torch.arange(s, device="cuda")[None])
    params = build_attention_params(cos, sin, torch.ones(a, s, dtype=torch.bool, device="cuda"), a)

    def fn():
        # fp32 master weights under a bf16 activation: the caller's autocast, as in production
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(q, c, params)

    gated = ["kernels.rmsnorm", "kernels.transition", "kernels.swa_dit", "kernels.rope", "kernels.gated_projection"]
    card(())
    with pytest.warns(UnrecordedCardWarning):
        ref = fn()
    assert not model._fused_ok(q, c, params)                       # warned once above: asked again, silently
    torch.cuda.synchronize()
    assert _of(launches, *gated) == 0, dict(launches)
    card(["swiglu_ffn", "rms_norm_modulation", "swa_atom_attention", "swa_dit", "rmsnorm"])
    launches.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCardWarning)
        assert model._fused_ok(q, c, params) is fused        # the gate passes; the fused block takes the call where it is asked to
        if fused:
            # the fused block itself is not run here: on H100 its stages are this card's own CUDA, not the portable kernels, and
            # they serve fp32 modulation only (test_swa_dit_fused_gpu.py covers the block); what is under test is the gate above
            return
        out = fn()
    torch.cuda.synchronize()
    assert _of(launches, *gated) > 0, dict(launches)
    assert _rel(out, ref) < 3e-2


def test_bias_only_attention_training_core(card, launches):
    from miniworld_engine.modules.bias_only_dit.module import BiasOnlyAttention
    from miniworld_engine.modules.exceptions import ImplementationType

    torch.manual_seed(0)
    card(())
    with pytest.warns(UnrecordedCardWarning):
        module = BiasOnlyAttention(128, 64, 32, 4, implementation=ImplementationType.MINIWORLD).cuda()
    for p in module.parameters():
        p.data.normal_(0, 0.05)
    single = torch.randn(2, 1, 128, 128, device="cuda", requires_grad=True)
    cond = torch.randn(2, 1, 128, 64, device="cuda")
    pair = torch.randn(1, 128, 128, 32, device="cuda", requires_grad=True)

    def fn():
        out = module(single, cond, pair)
        out.square().sum().backward()
        res = (out.detach(), single.grad.clone(), pair.grad.clone())
        single.grad = pair.grad = None
        module.zero_grad()
        return res

    with pytest.warns(UnrecordedCardWarning, match="bias_only"):
        ref = fn()
    torch.cuda.synchronize()
    assert _of(launches, "bias_only_attention", "gated_projection") == 0, dict(launches)
    settings.configure(force_portable_kernels=True)                 # the A/B switch: the core and the pair-bias kernels run
    out = fn()
    torch.cuda.synchronize()
    assert _of(launches, "bias_only_attention", "gated_projection") > 0, dict(launches)
    for o, r in zip(out, ref, strict=True):
        assert _rel(o, r) < 2e-2


def test_mpnn_message_auto(card, launches):
    from miniworld_engine.kernels.mpnn_message.interface import message_hidden_reduce

    torch.manual_seed(0)
    p = torch.randn(512, 48, 128, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(128, 128, device="cuda") * 0.1).to(torch.bfloat16)
    b = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    m = (torch.rand(512, 48, device="cuda") > 0.2).float()

    def fn():
        with torch.no_grad():
            return message_hidden_reduce(p, w, b, m, 30, backend="auto")

    _blocked_then_recorded(card, launches, ["mpnn_message"], fn, ["mpnn_message"], match="mpnn_message")
    card(())                                                         # an explicit backend is the caller's choice: not asked
    launches.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCardWarning)
        with torch.no_grad():
            message_hidden_reduce(p, w, b, m, 30, backend="triton")
    assert _of(launches, "mpnn_message") > 0


def test_layernorm_module_asks_the_training_record_per_call(card, launches):
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.primitives import LayerNorm

    card(["layernorm"], training=False)                              # the forward kernels only
    # rows of 4 KiB: wider than the H100 pipelined CUDA LayerNorm serves (144 - 2048 bytes), so the portable forward runs here
    norm = LayerNorm(2048, implementation=ImplementationType.MINIWORLD).cuda()
    x = torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCardWarning)
        norm(x)
    torch.cuda.synchronize()
    assert _of(launches, "kernels.layernorm") > 0
    launches.clear()
    with pytest.warns(UnrecordedCardWarning, match="layernorm"):
        norm(x).float().sum().backward()
    torch.cuda.synchronize()
    assert _of(launches, "kernels.layernorm") == 0, dict(launches)


def test_adaptive_layernorm_asks_fp32_rows_for_an_fp32_call(card, launches):
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType

    card(["adaptive_layernorm", "layernorm"])                         # bf16 rows only
    module = AdaptiveLayerNorm(128, 128, implementation=ImplementationType.MINIWORLD).cuda()
    x, cond = torch.randn(2, 512, 128, device="cuda"), torch.randn(2, 512, 128, device="cuda")
    with torch.no_grad(), pytest.warns(UnrecordedCardWarning, match="adaptive_layernorm"):
        ref = module(x, cond)
    torch.cuda.synchronize()
    assert _of(launches, "kernels.adaln") == 0, dict(launches)
    card(["adaptive_layernorm", "layernorm"], extra=(("fp32", ["adaptive_layernorm", "layernorm"]),))
    launches.clear()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCardWarning)
        out = module(x, cond)
    torch.cuda.synchronize()
    assert _of(launches, "kernels.adaln") > 0, dict(launches)
    assert _rel(out, ref) < 1e-2
