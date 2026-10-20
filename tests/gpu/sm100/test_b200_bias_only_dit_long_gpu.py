"""The bf16 bias-only token DiT on B200 past L 768 (MiniWorld's token rungs, up to L 3072): the streaming attention core
(``pv_stream_sm100.cu``) against its reference, rerun bit-identically over a NaN-poisoned allocator; the long ``softmax_t``; the
inference block (three-kernel and 12-launch steps) and the training block against the PyTorch block; CUDA graphs; and the
dispatch: L <= 768 keeps the tensor-memory core (``pv_gate_inf.cu``), bit for bit."""

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit
from miniworld_engine.integrations import bias_only_dit_train as TR
from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.kernels.bias_only_dit import reference as R
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        settings.configure(**vars(old))


def relative(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


def blocks(seed=811, n_head=16, d_head=None):
    torch.manual_seed(seed)
    mk = lambda impl: BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=impl).cuda()  # noqa: E731
    ref = randomize(mk(ImplementationType.PYTORCH))
    fast, ref_bf = mk(ImplementationType.MINIWORLD), mk(ImplementationType.PYTORCH)
    fast.load_state_dict(ref.state_dict())
    ref_bf.load_state_dict(ref.state_dict())
    return ref, fast.to(BF), ref_bf.to(BF)


def _poison(mb=512):
    """Fill the caching allocator's next blocks with NaN: a buffer read before it is written would show it."""
    junk = torch.full((mb << 18,), float("nan"), device="cuda")
    torch.cuda.synchronize()
    del junk


# ------------------------------------------------------------------------------------------------------------- the streaming core
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "S"), [(896, 2), (1024, 1), (1024, 5), (1536, 3), (2048, 16), (2560, 1), (3072, 5)])
@pytest.mark.parametrize(("gate", "pdl"), [(True, False), (False, False), (True, True), (False, True)])
def test_stream_core_matches_its_reference_and_reruns_bitwise(n_head, d_head, L, S, gate, pdl):
    """sigmoid(g) (P v) (or P v) within the bf16 rounding of the output of the fp32 reference, finite, and bit-identical over a
    rerun into a NaN-poisoned buffer."""
    torch.manual_seed(L + S + n_head)
    M, DA = S * L, n_head * d_head
    vg = torch.randn(M, 2 * DA, device="cuda", dtype=BF)
    bias = 2 * torch.randn(n_head * L, L, device="cuda", dtype=BF)
    P = torch.empty_like(bias)
    C.softmax_rows(bias, P, torch.rand(L, device="cuda") > 0.2)
    core = C.PvStreamCore(torch.cuda.current_device(), nh=n_head, dh=d_head, pdl=pdl)
    outs = []
    for _ in range(2):
        _poison()
        a = torch.full((M, DA), float("nan"), device="cuda", dtype=BF)
        core(vg[:, :DA], P, a, S, g=vg[:, DA:] if gate else None)
        torch.cuda.synchronize()
        outs.append(a)
    v = vg[:, :DA].float().view(S, L, n_head, d_head)
    o = torch.einsum("hij,sjhd->sihd", P.float().view(n_head, L, L), v).reshape(M, DA)
    want = torch.sigmoid(vg[:, DA:].float()) * o if gate else o
    assert torch.isfinite(outs[0].float()).all()
    assert relative(outs[0], want) < 2e-3  # the output's bf16 rounding: measured 1.5e-4 (gated, against the rounded reference) - 1.7e-3
    assert torch.equal(outs[0], outs[1])
    if gate:
        assert relative(outs[0], R.pv_gate(vg, P, S)) < 2e-3


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [384, 768])
def test_short_lengths_keep_the_tensor_memory_core(monkeypatch, n_head, d_head, L):
    """PvGateCore at L <= 768 runs pv_gate_inf.cu as before (the streaming core is never made), at L > 768 it hands over."""
    core = C.PvGateCore(torch.cuda.current_device(), nh=n_head, dh=d_head)
    called = []
    monkeypatch.setattr(C.PvStreamCore, "__call__", lambda self, *a, **k: called.append(1))
    S, DA = 3, n_head * d_head
    vg = torch.randn(S * L, 2 * DA, device="cuda", dtype=BF)
    P = torch.softmax(torch.randn(n_head * L, L, device="cuda"), -1).to(BF)
    a = torch.empty(S * L, DA, device="cuda", dtype=BF)
    core(vg[:, :DA], P, a, S, g=vg[:, DA:])
    assert not called and core._stream is None
    L2 = 1024
    vg2 = torch.randn(S * L2, 2 * DA, device="cuda", dtype=BF)
    P2 = torch.softmax(torch.randn(n_head * L2, L2, device="cuda"), -1).to(BF)
    core(vg2[:, :DA], P2, torch.empty(S * L2, DA, device="cuda", dtype=BF), S, g=vg2[:, DA:])
    assert called == [1]


def test_stream_core_sample_groups():
    """The cost model's groups: one sample group for few samples (P read once), N = 256 products for many."""
    for nh, dh in LAYOUTS:
        assert C.pick_group_stream(1, 3072, 148, nh, dh) == 1
        assert C.pick_group_stream(2, 3072, 148, nh, dh) == 2
        assert C.pick_group_stream(48, 1536, 148, nh, dh) == 4


@pytest.mark.parametrize("L", [1024, 1536, 2048, 2560, 3072])
def test_softmax_t_long_rows(L):
    """softmax_t past 1024 keys (8 / 12 chunks per lane, a 32-row tile of up to 192 KB): P equal to softmax_rows, P^T its transpose."""
    H = 16
    bias = 2 * torch.randn(H * L, L, device="cuda", dtype=BF)
    mask = torch.rand(L, device="cuda") > 0.2
    P, Pt, Pr = (torch.empty(H, L, L, device="cuda", dtype=BF) for _ in range(3))
    C.softmax_t(bias, P.view(H * L, L), Pt.view(H * L, L), mask)
    C.softmax_rows(bias, Pr.view(H * L, L), mask)
    assert torch.equal(P, Pr)
    assert torch.equal(Pt.transpose(1, 2), P)
    assert relative(P.view(H * L, L), R.attention_weights(bias, mask)) < 1e-3


# ------------------------------------------------------------------------------------------------------------- inference block
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(
    ("L", "S", "shared", "masked"),
    [(1024, 5, False, True), (1152, 3, False, True), (1536, 1, True, False), (2560, 5, True, True), (3072, 1, False, False)],
)
@pytest.mark.parametrize("inf3", ["1", "0"])
def test_long_inference_block_matches_the_pytorch_block(monkeypatch, n_head, d_head, L, S, shared, masked, inf3):
    """The long rungs against the fp32 PyTorch block within the PyTorch bf16 block's own error, with the three-kernel switch on or off
    (past L 768 the runner takes the 12-launch step either way)."""
    override_setting(monkeypatch, "bias_only_dit_inf3_bf16", inf3)
    ref, fast, ref_bf = blocks(n_head=n_head, d_head=d_head)
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    with torch.no_grad():
        want = ref(x, c, p, mask)
        args = (x.to(BF), c.to(BF), p.to(BF), mask)
        assert bias_only_dit.serves(fast, *args)
        got = fast(*args)
        base = ref_bf(*args)
    assert relative(got, want) <= 1.05 * relative(base, want), (relative(got, want), relative(base, want))
    assert relative(got, base) < 0.01


def test_long_lengths_switch_and_bound():
    """settings.bias_only_dit_stream=0 restores the old bound (L > 768 runs the module); L 3200 is past the served rungs; fp32
    stays within 768."""
    _, fast, _ = blocks(seed=4)
    L = 1024
    x, c = torch.randn(2, 1, L, 768, device="cuda", dtype=BF), torch.randn(2, 1, L, 384, device="cuda", dtype=BF)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=BF)
    with torch.no_grad():
        assert bias_only_dit.serves(fast, x, c, p)
        settings.configure(bias_only_dit_stream=False)
        try:
            assert not bias_only_dit.serves(fast, x, c, p)
        finally:
            settings.configure(bias_only_dit_stream=True)
        L2 = 3200
        x2, c2 = torch.empty(1, 1, L2, 768, device="cuda", dtype=BF), torch.empty(1, 1, L2, 384, device="cuda", dtype=BF)
        assert not bias_only_dit.serves(fast, x2, c2, torch.empty(1, L2, L2, 128, device="cuda", dtype=BF))
        f32 = blocks(seed=4)[0]
        f32.implementation = ImplementationType.MINIWORLD
        assert not bias_only_dit.serves(f32, x.float(), c.float(), p.float())
    xs = x.clone().requires_grad_(True)
    assert TR.serves(fast, xs, c, p)
    assert not TR.serves(f32, xs.float(), c.float(), p.float())


@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (24, 32)])
@pytest.mark.parametrize("L", [1536, 3072])
def test_long_inference_bit_identical_reruns_with_a_poisoned_allocator(n_head, d_head, L):
    """Five steps from scratch (runner, buffers, tables, bound launches and P dropped; free memory NaN-filled before each): finite and
    bit-identical."""
    _, fast, _ = blocks(seed=9, n_head=n_head, d_head=d_head)
    S = 5
    x = torch.randn(S, 1, L, 768, device="cuda", dtype=BF)
    c = torch.randn(S, 1, L, 384, device="cuda", dtype=BF)
    p, mask = torch.randn(1, L, L, 128, device="cuda", dtype=BF), torch.rand(1, L, device="cuda") > 0.2
    outs = []
    for _ in range(5):
        bias_only_dit._RUNNERS.clear()
        _poison(2048)
        with torch.no_grad():
            outs.append(fast(x, c, p, mask).clone())
        torch.cuda.synchronize()
    assert torch.isfinite(outs[0].float()).all()
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
@pytest.mark.parametrize("L", [1024, 2560])
def test_long_inference_graph_is_the_12_launch_step_and_replays(n_head, d_head, L):
    """Past L 768 the runner takes the 12-launch step (cuBLAS GEMMs + rows around the streaming core; the three-kernel step's
    clusters run several rounds there and lose), also with the three-kernel switch on: the captured step (static weights and
    conditioning) holds the streaming core once and no front / tail kernel, and replays bit-identically to the eager call."""
    from tests.cuda_graph_nodes import graph_kernels

    from miniworld_engine.kernels import _capture

    settings.configure(bias_only_dit_inf3_bf16=True)
    _, fast, _ = blocks(seed=5, n_head=n_head, d_head=d_head)
    S = 5
    x, c = torch.randn(S, 1, L, 768, device="cuda", dtype=BF), torch.randn(S, 1, L, 384, device="cuda", dtype=BF)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=BF)
    xs = x.clone()
    with _capture.static_weights(), _capture.static_inputs(), torch.no_grad():
        eager = fast(x, c, p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fast(xs, c, p)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out = fast(xs, c, p)
        names = graph_kernels(graph)
        graph.replay()
        torch.cuda.synchronize()
    assert sum("bo_pv_stream_sm100" in n for n in names) == 1, names
    assert not any("bo_front_bf16" in n or "bo_tail" in n or "pv_gate_inf" in n for n in names), names
    assert torch.equal(out, eager)


# ------------------------------------------------------------------------------------------------------------- training block
def _inputs(L, A, masked, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(A, 1, L, 768, device="cuda")
    c = torch.randn(A, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    dy = torch.randn(A, 1, L, 768, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    return x, c, p, dy, mask


def _step(m, x, c, p, mask, dy, dtype):
    leaves = [t.detach().to(dtype).requires_grad_(True) for t in (x, c, p)]
    m.zero_grad(set_to_none=True)
    y = m(*leaves, mask)
    y.backward(dy.to(dtype))
    return [y.detach()] + [t.grad for t in leaves] + [q.grad for q in m.parameters()]


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "A", "masked"), [(1024, 3, True), (1152, 2, True), (1536, 2, False), (3072, 1, True)])
def test_long_training_every_gradient_within_the_pytorch_bf16_error(n_head, d_head, L, A, masked):
    """The output and every gradient against the fp32 PyTorch block within 1.1x the PyTorch bf16 block's error + 1e-4 (the short
    lengths' bound). L1152: 256-key tiles do not divide L, so dbias runs on single CTAs (dpb_sm100.cu), elsewhere on CTA pairs."""
    ref, fast, ref_bf = blocks(seed=3, n_head=n_head, d_head=d_head)
    x, c, p, dy, mask = _inputs(L, A, masked)
    xb, cb, pb = (t.to(BF).requires_grad_(True) for t in (x, c, p))
    assert TR.serves(fast, xb, cb, pb, mask)
    want = _step(ref, x, c, p, mask, dy, torch.float32)
    got = _step(fast, x, c, p, mask, dy, BF)
    base = _step(ref_bf, x, c, p, mask, dy, BF)
    names = ["out", "d single", "d cond", "d pair"] + [n for n, _ in fast.named_parameters()]
    for n, g, b, w in zip(names, got, base, want, strict=True):
        assert relative(g, w) <= 1.1 * relative(b, w) + 1e-4, (n, relative(g, w), relative(b, w))


@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
def test_long_training_steps_bit_identical_with_a_poisoned_allocator(n_head, d_head):
    """Five training steps at L 1536 (forward + backward), the allocator's free memory NaN-filled before each: every gradient finite
    and bit-identical."""
    _, fast, _ = blocks(seed=11, n_head=n_head, d_head=d_head)
    x, c, p, dy, mask = _inputs(1536, 4, True, seed=2)
    runs = []
    for _ in range(5):
        _poison(2048)
        runs.append([t.clone() for t in _step(fast, x, c, p, mask, dy, BF)])
        torch.cuda.synchronize()
    for i, g in enumerate(runs[0]):
        assert torch.isfinite(g.float()).all(), i
        for r in runs[1:]:
            assert torch.equal(r[i], g), i


def test_long_training_graph_capture_and_compile():
    """A captured training step at L 2048 replays to the eager gradients; torch.compile keeps the opaque ops."""
    _, fast, _ = blocks(seed=7)
    x, c, p, dy, _ = _inputs(2048, 2, False, seed=1)
    x, c, p, dy = (t.to(BF) for t in (x, c, p, dy))
    xs, cs, ps = (t.clone().requires_grad_(True) for t in (x, c, p))

    def run(m):
        for t in (xs, cs, ps):
            t.grad = None
        for q in m.parameters():
            q.grad = None if q.grad is None else q.grad.zero_()
        m(xs, cs, ps).backward(dy)
        return [t.grad.clone() for t in (xs, cs, ps)] + [q.grad.clone() for q in m.parameters()]

    eager = run(fast)
    for q in fast.parameters():
        q.grad = torch.zeros_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run(fast)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for t in (xs, cs, ps):
            t.grad = None
        for q in fast.parameters():
            q.grad.zero_()
        fast(xs, cs, ps).backward(dy)
    graph.replay()
    torch.cuda.synchronize()
    replayed = [t.grad for t in (xs, cs, ps)] + [q.grad for q in fast.parameters()]
    for g, e in zip(replayed, eager, strict=True):
        assert relative(g, e) < 2e-3
    compiled = torch.compile(fast, options={"triton.cudagraphs": False})
    for g, e in zip(run(compiled), eager, strict=True):
        assert relative(g, e) < 2e-3
