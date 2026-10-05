"""Four-family CUDA routing, general shapes, autograd and compiled entry points."""

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels import cuda_native as C
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_attention.bidirectional import (
    BidirectionalTriangleAttention,
)
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A100 required")
    old = settings.configure(engine_backend="auto")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    settings.configure(**vars(old))


def randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif name.endswith("weight"):
                p.uniform_(0.8, 1.2)
            else:
                p.normal_(std=0.1)
    return m


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", range(4))
def test_vector_rows_handle_unaligned_inputs_and_tail(dtype, mode):
    a, b = [
        torch.randn(260, device="cuda", dtype=dtype)[1:].requires_grad_()
        for _ in range(2)
    ]
    aa, bb = a.detach().float().requires_grad_(), b.detach().float().requires_grad_()
    out = [C.add, C.mul, C.gate, C.swiglu][mode](a, b)
    ref = [
        lambda: aa + bb,
        lambda: aa * bb,
        lambda: aa.sigmoid() * bb,
        lambda: aa * aa.sigmoid() * bb,
    ][mode]()
    dy = torch.randn_like(out)
    got = torch.autograd.grad(out, (a, b), dy)
    want = torch.autograd.grad(ref, (aa, bb), dy.float())
    tol = 0.006 if dtype is torch.bfloat16 else 2e-6
    assert rel(out, ref) < tol
    for g, w in zip(got, want, strict=True):
        assert rel(g, w) < tol


def check(m, ref, args, tol=0.03):
    xs = [
        x.detach().clone().requires_grad_(True) if x.is_floating_point() else x
        for x in args
    ]
    rs = [
        x.detach().float().clone().requires_grad_(True) if x.is_floating_point() else x
        for x in args
    ]
    out, want = m(*xs), ref(*rs)
    assert rel(out, want) < tol
    cot = torch.randn_like(out)
    leaves = [x for x in xs if x.requires_grad] + list(m.parameters())
    rleaves = [x for x in rs if x.requires_grad] + list(ref.parameters())
    grads = torch.autograd.grad(out, leaves, cot, allow_unused=True)
    refs = torch.autograd.grad(want, rleaves, cot.float(), allow_unused=True)
    for i, (g, r) in enumerate(zip(grads, refs, strict=True)):
        assert (g is None) == (r is None), i
        if g is not None:
            # Near-zero softmax-invariant bias gradients need an absolute bound.
            assert rel(g, r) < tol * 2 or (g.float() - r.float()).abs().max() < 0.015, (
                i,
                rel(g, r),
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sa", [False, True])
def test_bidirectional(dtype, sa):
    torch.manual_seed(198)
    m = randomize(
        BidirectionalTriangleAttention(
            64, 2, use_self_attention=sa, implementation="miniworld"
        )
        .cuda()
        .to(dtype)
    )
    r = BidirectionalTriangleAttention(
        64, 2, use_self_attention=sa, implementation="pytorch"
    ).cuda()
    r.load_state_dict(m.state_dict())
    x = torch.randn(2, 17, 17, 64, device="cuda", dtype=dtype)
    mask = torch.rand(2, 17, device="cuda") > 0.2
    check(m, r, (x, mask), tol=0.02 if dtype is torch.bfloat16 else 2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("starting", [False, True])
@pytest.mark.parametrize("sa", [False, True])
def test_triangle_general(dtype, starting, sa):
    torch.manual_seed(199)
    kw = {
        "d_pair": 64,
        "n_head": 2,
        "starting": starting,
        "use_self_attention": sa,
        "p_drop": 0.0,
    }
    m = randomize(TriangleAttention(**kw, implementation="miniworld").cuda().to(dtype))
    r = TriangleAttention(**kw, implementation="pytorch").cuda()
    r.load_state_dict(m.state_dict())
    check(
        m,
        r,
        (
            torch.randn(2, 17, 17, 64, device="cuda", dtype=dtype),
            torch.rand(2, 17, device="cuda") > 0.2,
        ),
        tol=0.02 if dtype is torch.bfloat16 else 2e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_conditioned_general(dtype):
    m = randomize(
        ConditionedTransition(64, 32, n=2, implementation="miniworld").cuda().to(dtype)
    )
    r = ConditionedTransition(64, 32, n=2, implementation="pytorch").cuda()
    r.load_state_dict(m.state_dict())
    check(
        m,
        r,
        (
            torch.randn(2, 1, 17, 64, device="cuda", dtype=dtype),
            torch.randn(1, 1, 17, 32, device="cuda", dtype=dtype),
        ),
        tol=0.025 if dtype is torch.bfloat16 else 2e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bidir", [False, True])
def test_trimul_general(dtype, bidir):
    cls = BidirectionalTriangleMultiplication if bidir else TriangleMultiplication
    m = randomize(cls(64, p_drop=0.0, implementation="miniworld").cuda().to(dtype))
    r = cls(64, p_drop=0.0, implementation="pytorch").cuda()
    r.load_state_dict(m.state_dict())
    check(
        m,
        r,
        (
            torch.randn(2, 17, 17, 64, device="cuda", dtype=dtype),
            torch.rand(2, 17, device="cuda") > 0.2,
        ),
        tol=0.03 if dtype is torch.bfloat16 else 3e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("qknorm", [False, True])
def test_token_general(dtype, qknorm):
    kw = {
        "d_single": 64,
        "d_cond": 32,
        "d_pair": 16,
        "n_head": 2,
        "use_qk_norm": qknorm,
    }
    m = randomize(DiTBlock(**kw, implementation="miniworld").cuda().to(dtype))
    r = DiTBlock(**kw, implementation="pytorch").cuda()
    r.load_state_dict(m.state_dict())
    check(
        m,
        r,
        (
            torch.randn(2, 2, 17, 64, device="cuda", dtype=dtype),
            torch.randn(1, 2, 17, 32, device="cuda", dtype=dtype),
            torch.randn(2, 17, 17, 16, device="cuda", dtype=dtype),
            torch.rand(2, 17, device="cuda") > 0.2,
        ),
        tol=0.025 if dtype is torch.bfloat16 else 3e-5,
    )


def test_compiled_bidir_matches_eager():
    m = randomize(
        BidirectionalTriangleAttention(
            64, 2, use_self_attention=True, implementation="miniworld"
        ).cuda()
    )
    r = copy.deepcopy(m)
    x = torch.randn(1, 17, 17, 64, device="cuda")
    mask = torch.rand(1, 17, device="cuda") > 0.2
    try:
        check(torch.compile(m, fullgraph=True), r, (x, mask), tol=2e-5)
    finally:
        torch._dynamo.reset()


def test_attention_masks_replay():
    q, k, v = [torch.randn(2, 2, 2, 17, 16, device="cuda") for _ in range(3)]
    bias = torch.randn(2, 2, 17, 17, device="cuda")
    mask = torch.ones(2, 2, 17, device="cuda", dtype=torch.bool)
    for _ in range(3):
        C.attention(q, k, v, bias, mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = C.attention(q, k, v, bias, mask)
    for keys in (0, 1, 17):
        mask.zero_()
        mask[:, :, :keys] = True
        graph.replay()
        torch.testing.assert_close(
            out, C.attention(q, k, v, bias, mask), atol=0, rtol=0
        )
        if keys == 0:
            assert not out.count_nonzero()
        if keys == 1:
            torch.testing.assert_close(
                out, v[:, :, :, :1].expand_as(out), atol=0, rtol=0
            )


class WholeOp(torch.nn.Module):
    def __init__(self, module, family, reference=False):
        super().__init__()
        self.module, self.family, self.reference = module, family, reference

    def forward(self, x, aux):
        from miniworld_engine import ops

        m, family = self.module, self.family
        if family == "conditioned":
            if self.reference:
                y = m.squeeze(torch.nn.functional.silu(m.expand_a(x)) * m.expand_b(x))
                return torch.sigmoid(m.to_scale(aux)) * y
            return ops.conditioned_transition(
                x,
                aux,
                expand_a_weight=m.expand_a.weight,
                expand_b_weight=m.expand_b.weight,
                squeeze_weight=m.squeeze.weight,
                to_scale_weight=m.to_scale.weight,
                to_scale_bias=m.to_scale.bias,
                n=2,
            )
        if self.reference:
            out = m(x, aux)
            return out - x if family == "triangle" else out
        if family == "triangle":
            return ops.triangle_attention(
                x,
                aux,
                n_head=m.n_head,
                ln_pair_weight=m.ln_pair.weight,
                ln_pair_bias=m.ln_pair.bias,
                to_value_weight=m.to_value.weight,
                to_bias_weight=m.to_bias.weight,
                to_gate_weight=m.to_gate.weight,
                to_out_weight=m.to_out.weight,
                to_query_weight=m.to_query.weight,
                to_key_weight=m.to_key.weight,
                starting=m.starting,
            )
        norms = {
            "norm_in_weight": m.ln_pair.weight,
            "norm_in_bias": m.ln_pair.bias,
            "norm_out_weight": m.ln_out.weight,
            "norm_out_bias": m.ln_out.bias,
        }
        if family == "trimul":
            return ops.triangle_multiplicative_update(
                x,
                "outgoing",
                aux,
                **norms,
                p_in_weight=torch.cat([m.to_left.weight, m.to_right.weight]),
                g_in_weight=torch.cat([m.to_left_gate.weight, m.to_right_gate.weight]),
                p_out_weight=m.to_out.weight,
                g_out_weight=m.to_gate.weight,
            )
        return ops.bidirectional_triangle_multiplicative_update(
            x,
            aux,
            **norms,
            to_left_weight=m.to_left.weight,
            to_left_gate_weight=m.to_left_gate.weight,
            to_right_weight=m.to_right.weight,
            to_right_gate_weight=m.to_right_gate.weight,
            to_out_weight=m.to_out.weight,
            to_gate_weight=m.to_gate.weight,
        )


@pytest.mark.parametrize(
    "family", ["conditioned", "triangle", "trimul", "trimul_bidir"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("length", [17, 128])
def test_public_ops(family, dtype, length):
    torch.manual_seed(202)

    def make(impl):
        if family == "conditioned":
            return ConditionedTransition(64, 32, n=2, implementation=impl)
        if family == "triangle":
            return TriangleAttention(64, 2, p_drop=0.0, implementation=impl)
        cls = (
            TriangleMultiplication
            if family == "trimul"
            else BidirectionalTriangleMultiplication
        )
        return cls(64, p_drop=0.0, implementation=impl)

    m = WholeOp(randomize(make("miniworld").cuda().to(dtype)), family)
    r = WholeOp(make("pytorch").cuda(), family, True)
    r.load_state_dict(m.state_dict())
    x = torch.randn(
        (2, length, 64) if family == "conditioned" else (1, length, length, 64),
        device="cuda",
        dtype=dtype,
    )
    aux = (
        torch.randn(1, length, 32, device="cuda", dtype=dtype)
        if family == "conditioned"
        else torch.rand(1, length, device="cuda") > 0.2
    )
    check(m, r, (x, aux), tol=0.035 if dtype is torch.bfloat16 else 0.004)


def test_general_forward_backward_has_no_triton_launches():
    m = randomize(
        BidirectionalTriangleAttention(
            64, 2, use_self_attention=True, implementation="miniworld"
        ).cuda()
    )
    x = torch.randn(1, 17, 17, 64, device="cuda", requires_grad=True)
    m(x).sum().backward()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        m(x).sum().backward()
    names = [
        e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert names
    assert not [n for n in names if "triton" in n.lower()], names


@pytest.mark.parametrize(
    "family",
    [
        "token",
        "conditioned",
        "trimul",
        "trimul_bidir",
        "triangle",
        "triangle_bidir",
        "triangle_bias_bidir",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_registered_paths_execute_cuda_without_triton(family, dtype):
    torch.manual_seed(209)
    length = 128
    mask = torch.rand(1, length, device="cuda") > 0.2
    if family == "token":
        m = DiTBlock(implementation="miniworld")
        args = (
            torch.randn(1, 1, length, 768, device="cuda", dtype=dtype),
            torch.randn(1, 1, length, 384, device="cuda", dtype=dtype),
            torch.randn(1, length, length, 128, device="cuda", dtype=dtype),
            mask,
        )
    elif family == "conditioned":
        m = ConditionedTransition(128, 128, n=2, implementation="miniworld")
        args = (
            torch.randn(2, 1, length, 128, device="cuda", dtype=dtype),
            torch.randn(1, 1, length, 128, device="cuda", dtype=dtype),
        )
    else:
        args = (torch.randn(1, length, length, 128, device="cuda", dtype=dtype), mask)
        if family == "trimul":
            m = TriangleMultiplication(128, p_drop=0.0, implementation="miniworld")
        elif family == "trimul_bidir":
            m = BidirectionalTriangleMultiplication(
                128, p_drop=0.0, implementation="miniworld"
            )
        elif family == "triangle":
            m = TriangleAttention(128, 4, p_drop=0.0, implementation="miniworld")
        else:
            m = BidirectionalTriangleAttention(
                128,
                4,
                use_self_attention=family == "triangle_bidir",
                implementation="miniworld",
            )
    m = randomize(m.cuda().to(dtype))
    args = tuple(x.requires_grad_(True) if x.is_floating_point() else x for x in args)

    def train():
        out = m(*args)
        out.sum().backward()

    train()
    with torch.no_grad():
        m(*args)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        train()
        with torch.no_grad():
            m(*args)
    names = [
        e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert names
    assert not [n for n in names if "triton" in n.lower()], names


@pytest.mark.parametrize("family", ["token", "conditioned", "trimul", "triangle"])
def test_compiled_general_families(family):
    if family == "token":
        m = DiTBlock(
            d_single=64, d_cond=32, d_pair=16, n_head=2, implementation="miniworld"
        )
        args = (
            torch.randn(1, 1, 17, 64, device="cuda"),
            torch.randn(1, 1, 17, 32, device="cuda"),
            torch.randn(1, 17, 17, 16, device="cuda"),
            torch.ones(1, 17, device="cuda", dtype=torch.bool),
        )
    elif family == "conditioned":
        m = ConditionedTransition(64, 32, n=2, implementation="miniworld")
        args = (
            torch.randn(2, 17, 64, device="cuda"),
            torch.randn(1, 17, 32, device="cuda"),
        )
    else:
        cls = TriangleMultiplication if family == "trimul" else TriangleAttention
        m = cls(64, p_drop=0.0, implementation="miniworld")
        args = (
            torch.randn(1, 17, 17, 64, device="cuda"),
            torch.ones(1, 17, device="cuda", dtype=torch.bool),
        )
    m = randomize(m.cuda())
    ref = copy.deepcopy(m)
    try:
        # The token block's fp32 attention / AdaLN / transition run the dedicated TF32 tensor-core kernels first (integrations.augattn_sm80 and friends, as the
        # Triton path's tl.dot does); the exact-fp32 composition of a100_families only takes what they decline, so the token block is held to TF32 accuracy.
        check(torch.compile(m, fullgraph=True), ref, args, tol=5e-4 if family == "token" else 2e-5)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("bidir", [False, True])
def test_compiled_public_trimul(bidir):
    cls = BidirectionalTriangleMultiplication if bidir else TriangleMultiplication
    family = "trimul_bidir" if bidir else "trimul"
    m = WholeOp(
        randomize(cls(64, p_drop=0.0, implementation="miniworld").cuda().bfloat16()),
        family,
    )
    r = WholeOp(cls(64, p_drop=0.0, implementation="pytorch").cuda(), family, True)
    r.load_state_dict(m.state_dict())
    args = (
        torch.randn(1, 128, 128, 64, device="cuda", dtype=torch.bfloat16),
        torch.ones(1, 128, device="cuda", dtype=torch.bool),
    )
    try:
        check(torch.compile(m, fullgraph=True), r, args, tol=0.035)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("family", ["triangle", "trimul"])
def test_dropout_general_preserves_broadcast_axis(family):
    cls = TriangleAttention if family == "triangle" else TriangleMultiplication
    m = randomize(cls(64, p_drop=0.25, implementation="miniworld").cuda())
    r = cls(64, p_drop=0.25, implementation="pytorch").cuda()
    r.load_state_dict(m.state_dict())
    shape = (1, 1, 17, 64) if family == "trimul" else (1, 17, 1, 64)
    scale = (torch.rand(shape, device="cuda") > 0.25).float() / 0.75
    attr = "_make_drop_row_scale" if family == "trimul" else "_make_drop_scale"
    setattr(m, attr, lambda *args: scale)
    setattr(r, attr, lambda *args: scale)
    check(
        m,
        r,
        (
            torch.randn(1, 17, 17, 64, device="cuda"),
            torch.ones(1, 17, device="cuda", dtype=torch.bool),
        ),
        tol=2e-5,
    )


@pytest.mark.parametrize("autocast", [False, True])
def test_triangle_qknorm_and_amp(autocast):
    torch.manual_seed(213)
    kw = {"d_pair": 64, "n_head": 2, "use_qk_norm": True, "p_drop": 0.0}
    m = randomize(TriangleAttention(**kw, implementation="miniworld").cuda())
    r = TriangleAttention(**kw, implementation="pytorch").cuda()
    r.load_state_dict(m.state_dict())
    args = (
        torch.randn(1, 17, 17, 64, device="cuda"),
        torch.ones(1, 17, device="cuda", dtype=torch.bool),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        check(m, r, args, tol=0.03 if autocast else 3e-5)
