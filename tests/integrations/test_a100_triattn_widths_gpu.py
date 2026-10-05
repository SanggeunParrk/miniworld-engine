"""TriangleAttention on A100 at every registered width (integrations/triattn_sm80.py, kernels/triangle_attention/cuda/sm80_wide.py): d_pair 64 (hidden 64 / 128, 4 / 2 heads: head
dim 16 / 32 / 32), 256 (8 x 32), 384 (12 x 32) and the d_pair-128 geometry through the generic path, inference and training, starting and ending node, against the fp32 PyTorch
module.  The errors are held to the Triton path's own error in the same regime (the kernels compute in bf16 with fp32 accumulation, as the Triton path does)."""

import os

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

#: (d_pair, d_hidden, n_head): the registered geometries (head dim 16 / 32 / 32 / 32 / 32)
GEOMETRIES = [(64, 64, 4), (64, 128, 4), (64, 64, 2), (256, 256, 8), (384, 384, 12)]
IDS = [f"d{d}_h{hid}_{h}heads" for d, hid, h in GEOMETRIES]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-20))


def _module(geom, starting, *, ln_fp32=True, p_drop=0.0, seed=5):
    d, hidden, heads = geom
    torch.manual_seed(seed)
    m = TriangleAttention(d, heads, d_hidden=hidden, starting=starting, implementation=ImplementationType.MINIWORLD, p_drop=p_drop).cuda()
    for lin in (m.to_query, m.to_key, m.to_value, m.to_gate, m.to_bias, m.to_out):
        torch.nn.init.normal_(lin.weight, std=d ** -0.5)          # to_out is zero-initialised: a zero update would make every comparison vacuous
    m.ln_pair.weight.data.uniform_(0.7, 1.3)
    m.ln_pair.bias.data.normal_(0, 0.3)
    m = m.to(torch.bfloat16)
    if ln_fp32:
        m.ln_pair.float()
    return m


def _reference(m, geom, starting, p_drop=0.0):
    d, hidden, heads = geom
    ref = TriangleAttention(d, heads, d_hidden=hidden, starting=starting, implementation=ImplementationType.PYTORCH, p_drop=p_drop).cuda().float()
    ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    return ref


def _inputs(geom, length, batch=1, masked=True, seed=3):
    torch.manual_seed(seed)
    pair = torch.randn(batch, length, length, geom[0], device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(batch, length, device="cuda") > 0.1) if masked else None
    return pair, mask


def _loss_and_grads(m, pair, mask, cot, seed):
    """Output and the gradients (pair tensor, then every parameter) of ``sum(out * cot)``; the RNG is re-seeded so the dropout draw is the same."""
    pair = pair.detach().clone().requires_grad_()
    torch.manual_seed(seed)
    out = m(pair, mask)
    params = list(m.parameters())
    grads = torch.autograd.grad((out.float() * cot).sum(), [pair, *params])
    return out.detach(), grads


def _serves(m, pair, mask, *, train):
    from miniworld_engine.integrations import triattn_sm80

    if train:
        return triattn_sm80.serves_train(m, pair.detach().clone().requires_grad_(), mask)
    with torch.no_grad():
        return triattn_sm80.serves_module(m, pair, mask)


def _spy(monkeypatch, module, name, calls):
    real = getattr(module, name)

    def wrapper(*a, **k):
        calls.append(name)
        return real(*a, **k)

    monkeypatch.setattr(module, name, wrapper)


# ------------------------------------------------------------------------------------------------------------------------------------------------- the gate
def test_gate_declines_what_it_is_not_built_for(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide as w

    geom = (256, 256, 8)
    m = _module(geom, True)
    pair, mask = _inputs(geom, 128)
    ln = m.ln_pair
    weights = (m.to_query.weight, m.to_key.weight, m.to_value.weight, m.to_gate.weight, m.to_bias.weight)
    cfg = w.cfg_for(*geom)
    assert w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask)
    assert not w.supports(pair.float(), cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask)               # fp32 activations
    assert not w.supports(pair[:, :, :96].contiguous(), cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask)  # not square
    assert not w.supports(pair.transpose(1, 2), cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask)         # not contiguous
    p2, mk2 = _inputs(geom, 192)
    assert not w.supports(p2, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mk2)                          # L is not a multiple of 128
    assert not w.supports(pair, cfg, weights, m.to_out.weight.float(), ln.weight, ln.bias, mask)               # fp32 weight
    assert not w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias.bfloat16(), mask)            # mixed LayerNorm dtypes
    assert not w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask.float())              # mask dtype
    assert not w.supports(pair, None, weights, m.to_out.weight, ln.weight, ln.bias, mask)
    monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
    assert not w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask)                       # the switch


def test_the_training_gate_declines_when_the_bias_gradient_partials_do_not_fit(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_core2
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide as w

    geom = (256, 256, 8)
    m = _module(geom, True)
    pair, mask = _inputs(geom, 128)
    ln = m.ln_pair
    weights = (m.to_query.weight, m.to_key.weight, m.to_value.weight, m.to_gate.weight, m.to_bias.weight)
    cfg = w.cfg_for(*geom)
    assert w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask, train=True)
    monkeypatch.setattr(sm80_core2, "MAX_DBP_BYTES", 1 << 20)                                        # the partials of this step: 128 / 4 groups x 8 heads x 128^2 x 2 B = 8 MiB
    assert not w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask, train=True)
    assert w.supports(pair, cfg, weights, m.to_out.weight, ln.weight, ln.bias, mask)                 # inference does not need them


def test_the_module_gate_keeps_the_triton_path_for_what_the_kernels_do_not_serve():
    m = _module((64, 128, 4), True)
    pair, mask = _inputs((64, 128, 4), 128)
    assert _serves(m, pair, mask, train=False)
    assert _serves(m, pair, mask, train=True)
    assert not _serves(m, pair.float(), mask, train=False)                            # fp32 activations
    p2, mk2 = _inputs((64, 128, 4), 192)
    assert not _serves(m, p2, mk2, train=False)                                       # L % 128
    m._sm80_cuda = False
    assert not _serves(m, pair, mask, train=False)                                    # the module's own switch
    m._sm80_cuda = True
    q = _module((64, 128, 4), True)
    q.use_qk_norm = True
    assert not _serves(q, pair, mask, train=False)
    m.train()
    m.p_drop = 0.25
    with torch.no_grad():
        assert not triattn_serves(m, pair, mask)                                      # dropout active without autograd keeps the Triton path


def triattn_serves(m, pair, mask):
    from miniworld_engine.integrations import triattn_sm80

    return triattn_sm80.serves_module(m, pair, mask)


def test_env_switch_turns_every_geometry_off(monkeypatch):
    for geom in GEOMETRIES:
        m = _module(geom, True)
        pair, mask = _inputs(geom, 128)
        assert _serves(m, pair, mask, train=False)
        monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
        assert not _serves(m, pair, mask, train=False)
        assert not _serves(m, pair, mask, train=True)
        monkeypatch.delenv("MINIWORLD_TRIATTN_SM80")


# --------------------------------------------------------------------------------------------------------------------------------------------- inference
@pytest.mark.parametrize("geom", GEOMETRIES, ids=IDS)
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [128, 256])
def test_inference_is_no_less_accurate_than_the_triton_path(geom, starting, masked, length):
    m = _module(geom, starting).eval()
    ref = _reference(m, geom, starting).eval()
    pair, mask = _inputs(geom, length, masked=masked)
    assert _serves(m, pair, mask, train=False)
    with torch.no_grad():
        got = m(pair, mask)
        m._sm80_cuda = False
        tri = m(pair, mask)
        m._sm80_cuda = True
        want = ref(pair.float(), mask)
    assert got.dtype is torch.bfloat16
    assert got.shape == pair.shape
    mine, base = _rel(got, want), _rel(tri, want)
    assert mine <= max(1.25 * base, 3e-3), f"cuda {mine:.3e} vs triton path {base:.3e}"


@pytest.mark.parametrize("geom", [(64, 64, 4), (64, 128, 4), (256, 256, 8)], ids=["d64_hd16", "d64_hd32", "d256"])
@pytest.mark.parametrize("starting", [True, False])
def test_a_batch_of_two_and_a_length_that_is_not_a_power_of_two(geom, starting):
    m = _module(geom, starting).eval()
    ref = _reference(m, geom, starting).eval()
    pair, mask = _inputs(geom, 384, batch=2)
    assert _serves(m, pair, mask, train=False)
    with torch.no_grad():
        got = m(pair, mask)
        m._sm80_cuda = False
        tri = m(pair, mask)
        m._sm80_cuda = True
        want = ref(pair.float(), mask)
    assert _rel(got, want) <= max(1.25 * _rel(tri, want), 3e-3)
    for b in range(2):                                                            # a batch element does not see the other
        with torch.no_grad():
            alone = m(pair[b:b + 1].contiguous(), mask[b:b + 1].contiguous())
        assert _rel(got[b:b + 1], alone) < 3e-3


@pytest.mark.parametrize("geom", GEOMETRIES, ids=IDS)
def test_the_layernorm_affine_may_be_bf16_or_fp32(geom):
    ref = _reference(_module(geom, True), geom, True).eval()
    pair, mask = _inputs(geom, 128)
    errs = {}
    for fp32 in (True, False):
        m = _module(geom, True, ln_fp32=fp32).eval()
        assert _serves(m, pair, mask, train=False)
        with torch.no_grad():
            errs[fp32] = _rel(m(pair, mask), ref(pair.float(), mask))
    assert errs[True] < 6e-3, errs
    assert errs[False] < 6e-3, errs


@pytest.mark.parametrize("geom", GEOMETRIES, ids=IDS)
def test_inference_replays_bit_identically_and_does_not_touch_its_inputs(geom):
    m = _module(geom, False).eval()
    pair, mask = _inputs(geom, 256)
    keep = pair.clone()
    with torch.no_grad():
        a = m(pair, mask)
        b = m(pair, mask)
    assert torch.equal(a, b)
    assert torch.equal(pair, keep)


@pytest.mark.parametrize("geom", [(64, 64, 4), (256, 256, 8)], ids=["d64_hd16", "d256"])
def test_a_fully_masked_row_of_keys_gives_finite_output(geom):
    m = _module(geom, True).eval()
    pair, _ = _inputs(geom, 128)
    mask = torch.zeros(1, 128, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        out = m(pair, mask)
    assert torch.isfinite(out.float()).all()


# ----------------------------------------------------------------------------------------------------------------------------------------------- training
@pytest.mark.parametrize("geom", GEOMETRIES, ids=IDS)
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (2, 128), (1, 256)])
def test_training_matches_the_fp32_module_as_well_as_the_triton_path(geom, starting, batch, length, monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    m = _module(geom, starting)
    ref = _reference(m, geom, starting)
    pair, mask = _inputs(geom, length, batch=batch)
    cot = torch.randn(batch, length, length, geom[0], device="cuda")
    assert _serves(m, pair, mask, train=True)
    calls = []
    _spy(monkeypatch, sm80_wide, "forward_ops", calls)
    _spy(monkeypatch, sm80_wide, "backward_ops", calls)
    out, grads = _loss_and_grads(m, pair, mask, cot, 1)
    assert calls == ["forward_ops", "backward_ops"], calls
    m._sm80_cuda = False
    out_t, grads_t = _loss_and_grads(m, pair, mask, cot, 1)
    m._sm80_cuda = True
    out_r, grads_r = _loss_and_grads(ref, pair.float(), mask, cot, 1)
    assert _rel(out, out_r) <= max(1.25 * _rel(out_t, out_r), 3e-3)
    names = ["pair", *(n for n, _ in m.named_parameters())]
    for name, mine_g, tri_g, ref_g in zip(names, grads, grads_t, grads_r, strict=True):
        mine, base = _rel(mine_g, ref_g), _rel(tri_g, ref_g)
        assert mine <= max(1.25 * base, 6e-3), f"d{name}: cuda {mine:.3e} vs triton path {base:.3e}"
        assert mine_g.dtype == tri_g.dtype, name                                  # gradients come back in the parameters' dtypes (fp32 LayerNorm affine)


@pytest.mark.parametrize("geom", [(64, 64, 4), (64, 128, 4), (256, 256, 8)], ids=["d64_hd16", "d64_hd32", "d256"])
@pytest.mark.parametrize("starting", [True, False])
def test_training_with_the_broadcast_dropout_matches_the_triton_path(geom, starting):
    m = _module(geom, starting, p_drop=0.25).train()
    pair, mask = _inputs(geom, 128)
    cot = torch.randn(1, 128, 128, geom[0], device="cuda")
    out, grads = _loss_and_grads(m, pair, mask, cot, 5)
    m._sm80_cuda = False
    out_t, grads_t = _loss_and_grads(m, pair, mask, cot, 5)             # the same draw: the same dropped channels
    m._sm80_cuda = True
    assert _rel(out, out_t) < 6e-3
    names = ["pair", *(n for n, _ in m.named_parameters())]
    for name, g, gt in zip(names, grads, grads_t, strict=True):
        assert _rel(g, gt) < 2e-2, f"d{name}: {_rel(g, gt):.3e}"


@pytest.mark.parametrize("geom", [(64, 128, 4), (256, 256, 8)], ids=["d64", "d256"])
def test_training_replays_bit_identically(geom):
    m = _module(geom, True)
    pair, mask = _inputs(geom, 128)
    cot = torch.randn(1, 128, 128, geom[0], device="cuda")
    out_a, grads_a = _loss_and_grads(m, pair, mask, cot, 1)
    out_b, grads_b = _loss_and_grads(m, pair, mask, cot, 1)
    assert torch.equal(out_a, out_b)
    for a, b in zip(grads_a, grads_b, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("geom", [(64, 128, 4), (256, 256, 8)], ids=["d64", "d256"])
def test_only_the_parameters_or_only_the_pair_want_gradients(geom, monkeypatch):
    from miniworld_engine.integrations import triattn_sm80

    m = _module(geom, True)
    pair, mask = _inputs(geom, 128)
    cot = torch.randn(1, 128, 128, geom[0], device="cuda")
    for p in m.parameters():
        p.requires_grad_(False)
    assert not triattn_sm80.serves_train(m, pair.requires_grad_(False), mask)             # nothing wants a gradient: the inference path serves it
    p1 = pair.detach().clone().requires_grad_()
    out = m(p1, mask)                                                                    # only the pair tensor
    g, = torch.autograd.grad((out.float() * cot).sum(), [p1])
    assert torch.isfinite(g.float()).all()
    for p in m.parameters():
        p.requires_grad_(True)
    pair2 = pair.detach().clone()
    out = m(pair2, mask)                                                                 # only the parameters
    gs = torch.autograd.grad((out.float() * cot).sum(), list(m.parameters()))
    assert all(torch.isfinite(t.float()).all() for t in gs)


# ------------------------------------------------------------------------------------------------------------------------------------- engines agree
@pytest.mark.parametrize("starting", [True, False])
def test_the_two_engines_agree_at_d64(starting, monkeypatch):
    """The fused kernels and the row kernels + cuBLAS are two implementations of the same module: the same bf16 accuracy against the fp32 module."""
    geom = (64, 128, 4)
    m = _module(geom, starting).eval()
    ref = _reference(m, geom, starting).eval()
    pair, mask = _inputs(geom, 256)
    with torch.no_grad():
        fused = m(pair, mask)
        monkeypatch.setenv("MINIWORLD_TRIATTN_SM80_ROWS", "1")
        rows = m(pair, mask)
        want = ref(pair.float(), mask)
    assert _rel(fused, want) < 3e-3
    assert _rel(rows, want) < 3e-3
    assert _rel(fused, rows) < 4e-3


@pytest.mark.parametrize("starting", [True, False])
def test_the_generic_path_at_d128_matches_the_dedicated_kernels(starting, monkeypatch):
    geom = (128, 128, 4)
    m = _module(geom, starting).eval()
    pair, mask = _inputs(geom, 256)
    with torch.no_grad():
        dedicated = m(pair, mask)
        monkeypatch.setenv("MINIWORLD_TRIATTN_SM80_WIDE", "1")
        generic = m(pair, mask)
        monkeypatch.setenv("MINIWORLD_TRIATTN_SM80_ROWS", "1")
        rows = m(pair, mask)
    assert _rel(generic, dedicated) < 2e-3
    assert _rel(rows, dedicated) < 4e-3


# --------------------------------------------------------------------------------------------------------------------------- compile and CUDA graphs
@pytest.mark.parametrize("geom", [(64, 64, 4), (256, 256, 8)], ids=["d64_hd16", "d256"])
@pytest.mark.parametrize("starting", [True, False])
def test_the_compiled_module_matches_eager_in_inference_and_training(geom, starting):
    m = _module(geom, starting)
    pair, mask = _inputs(geom, 128)
    cot = torch.randn(1, 128, 128, geom[0], device="cuda")
    with torch.no_grad():
        eager = m.eval()(pair, mask)
    compiled = torch.compile(m, fullgraph=True)
    with torch.no_grad():
        got = compiled.eval()(pair, mask)
    assert torch.equal(got, eager)
    m.train()
    out_e, grads_e = _loss_and_grads(m, pair, mask, cot, 1)
    out_c, grads_c = _loss_and_grads(compiled, pair, mask, cot, 1)
    assert torch.equal(out_c, out_e)
    for a, b in zip(grads_c, grads_e, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("geom", [(64, 128, 4), (256, 256, 8)], ids=["d64", "d256"])
def test_a_cuda_graph_of_the_inference_call_replays_the_eager_result(geom):
    m = _module(geom, False).eval()
    pair, mask = _inputs(geom, 128)
    with torch.no_grad():
        for _ in range(2):
            want = m(pair, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = m(pair, mask)
        pair.copy_(torch.randn_like(pair))
        want = m(pair, mask)
        graph.replay()
    assert torch.equal(out, want)


@pytest.mark.parametrize("geom", [(64, 128, 4), (256, 256, 8)], ids=["d64", "d256"])
def test_a_cuda_graph_of_a_training_step_replays_the_eager_gradients(geom):
    m = _module(geom, True)
    pair, mask = _inputs(geom, 128)
    cot = torch.randn(1, 128, 128, geom[0], device="cuda")
    x = pair.detach().clone().requires_grad_()
    params = list(m.parameters())

    def step():
        out = m(x, mask)
        return torch.autograd.grad((out.float() * cot).sum(), [x, *params])

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = step()
    x.data.copy_(torch.randn_like(pair))
    want = step()
    graph.replay()
    for got, ref in zip(static, want, strict=True):
        assert torch.equal(got, ref)


# --------------------------------------------------------------------------------------------------------------------- parameters are repacked after a step
@pytest.mark.parametrize("geom", [(64, 64, 4), (256, 256, 8)], ids=["d64_hd16", "d256"])
def test_an_optimizer_step_is_seen_by_the_next_call(geom):
    m = _module(geom, True).eval()
    pair, mask = _inputs(geom, 128)
    with torch.no_grad():
        before = m(pair, mask)
        m.to_query.weight.mul_(1.5)
        m.to_out.weight.mul_(0.5)
        m.ln_pair.weight.mul_(1.1)
        after = m(pair, mask)
        m._sm80_cuda = False
        triton = m(pair, mask)
    assert not torch.equal(before, after)
    assert _rel(after, triton) < 6e-3


def test_the_env_switch_leaves_the_triton_numbers_untouched(monkeypatch):
    geom = (64, 128, 4)
    m = _module(geom, True).eval()
    pair, mask = _inputs(geom, 128)
    with torch.no_grad():
        m._sm80_cuda = False
        a = m(pair, mask)
        m._sm80_cuda = True
        monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
        b = m(pair, mask)
    assert torch.equal(a, b)
    assert os.environ["MINIWORLD_TRIATTN_SM80"] == "0"


# ------------------------------------------------------------------------------------------------------------------ the widest width the gate accepts
def test_d512_width_inference_and_training_match_the_fp32_module():
    """d_pair 512 (16 heads x 32) is the upper end of the row engine's range (not a registry row): inference and every gradient no further from the fp32 module than 1.5x the Triton path."""
    geom = (512, 512, 16)
    m = _module(geom, True)
    ref = _reference(m, geom, True)
    pair, mask = _inputs(geom, 128)
    cot = torch.randn(1, 128, 128, geom[0], device="cuda")
    assert _serves(m, pair, mask, train=False)
    assert _serves(m, pair, mask, train=True)
    out, grads = _loss_and_grads(m, pair, mask, cot, 1)
    m._sm80_cuda = False
    out_t, grads_t = _loss_and_grads(m, pair, mask, cot, 1)
    m._sm80_cuda = True
    out_r, grads_r = _loss_and_grads(ref, pair.float(), mask, cot, 1)
    assert _rel(out, out_r) <= max(1.5 * _rel(out_t, out_r), 3e-3)
    names = ["pair", *(n for n, _ in m.named_parameters())]
    for name, g, gt, gr in zip(names, grads, grads_t, grads_r, strict=True):
        assert _rel(g, gr) <= max(1.5 * _rel(gt, gr), 6e-3), f"d{name}: cuda {_rel(g, gr):.3e} vs triton path {_rel(gt, gr):.3e}"
