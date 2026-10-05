"""LayerNorm, RMSNorm, 3D RoPE and the fused Q/K norm + RoPE on A100 (``kernels/{layernorm,rmsnorm,rope}/cuda/sm80.py``) through the modules and kernel entry points that dispatch to
them: numerics (forward and every gradient) against the fp32 reference and as accurate as the bf16 PyTorch path, eager and ``torch.compile(fullgraph=True)``, a CUDA-graph step, the
``MINIWORLD_NORMS_SM80`` switch and the gates.  Errors are held to the bf16 PyTorch module's own error in the same regime."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.modules import LayerNorm, RMSNorm
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

MW = ImplementationType.MINIWORLD
BF, F32 = torch.bfloat16, torch.float32


@pytest.fixture(autouse=True)
def ampere(monkeypatch):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")
    # the small shapes of these tests include the tiny training steps the Triton path wins (routed away by default): the tests exercise the CUDA rows themselves
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "force")


@pytest.fixture(autouse=True)
def restore_settings():
    previous = settings.current()
    yield
    settings.configure(**{f.name: getattr(previous, f.name) for f in dataclasses.fields(previous)})


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _all_close(pairs, tol):
    return all(_rel(a, b) < tol for a, b in pairs)


@pytest.fixture
def spy(monkeypatch):
    """Counts the calls that reach the A100 kernels (the family entry points the dispatching code calls)."""
    from miniworld_engine.kernels.layernorm.cuda import sm80 as ln
    from miniworld_engine.kernels.rmsnorm.cuda import sm80 as rms
    from miniworld_engine.kernels.rope.cuda import sm80 as rope

    calls = {"layernorm": 0, "rmsnorm": 0, "qk_norm_rope": 0, "modulation": 0}

    def wrap(module, name, key):
        real = getattr(module, name)

        def counted(*args, **kwargs):
            calls[key] += 1
            return real(*args, **kwargs)
        monkeypatch.setattr(module, name, counted)

    wrap(ln, "layernorm", "layernorm")
    wrap(rms, "rmsnorm", "rmsnorm")
    wrap(rms, "rmsnorm_adamod", "modulation")
    wrap(rope, "qk_norm_rope_3d", "qk_norm_rope")
    return calls


# ------------------------------------------------------------------------------------------------------------------------------ LayerNorm
# the registry's layernorm_native rows: (width, has_bias) and one activation shape per stream (small lengths; the staged odd-width kernels need >= 4096 rows)
WIDTHS = [(16, 0), (64, 1), (128, 0), (128, 1), (256, 0), (256, 1), (267, 0), (384, 0), (384, 1), (451, 1), (512, 1), (768, 0), (768, 1), (831, 0), (833, 0), (2560, 1)]
STREAMS = {
    "token_pair": lambda d: (1, 72, 72, d),
    "token_single": lambda d: (2, 96, d),
    "msa_token": lambda d: (1, 8, 50, d),
    "atom_single": lambda d: (1, 640, d),
    "atom_pair": lambda d: (1, 8, 32, 128, d),
    "noise": lambda d: (48, 1, d),
}


def _norm(d, bias, seed=0):
    torch.manual_seed(seed)
    ref = LayerNorm(d, bias=bool(bias), implementation=ImplementationType.PYTORCH)
    with torch.no_grad():                                   # non-default affine parameters
        ref.weight.copy_(torch.randn(d) * 0.5 + 1.0)
        if ref.bias is not None:
            ref.bias.copy_(torch.randn(d) * 0.3)
    ours = LayerNorm(d, bias=bool(bias), implementation=MW)
    ours.load_state_dict(ref.state_dict())
    return ref.cuda(), ours.cuda().to(BF)


@pytest.mark.parametrize("stream", sorted(STREAMS))
@pytest.mark.parametrize(("d", "bias"), WIDTHS)
def test_layernorm_module_forward_and_gradients(d, bias, stream, spy):
    """Forward and the x / weight / bias gradients through the MINIWORLD module match the fp32 module, as closely as the bf16 PyTorch path does."""
    if stream == "atom_pair" and d != 16:
        pytest.skip("the atom pair stream is the 16-wide norm")
    if stream != "atom_pair" and d == 16:
        pytest.skip("the 16-wide norm lives on the atom pair stream")
    ref, ours = _norm(d, bias)
    shape = STREAMS[stream](d)
    torch.manual_seed(1)
    x = torch.randn(*shape, device="cuda") * 2 + 0.5
    dy = torch.randn(*shape, device="cuda")
    xr = x.clone().requires_grad_()
    xo = x.to(BF).requires_grad_()
    xb = x.to(BF).requires_grad_()
    yr = ref(xr)
    yr.backward(dy)
    yo = ours(xo)
    yo.backward(dy.to(BF))
    wb = ours.weight.detach().clone().requires_grad_()                 # the bf16 path on copies of the parameters: it must not accumulate into the module's gradients
    bb = None if ours.bias is None else ours.bias.detach().clone().requires_grad_()
    yb = F.layer_norm(xb.float(), (d,), wb, bb, ours.eps).to(BF)        # the bf16 module path
    yb.backward(dy.to(BF))
    assert spy["layernorm"] == 1, "the A100 kernels did not serve the call"
    assert yo.dtype == BF
    assert xo.grad.dtype == BF
    assert ours.weight.grad.dtype == F32
    e, e0 = _rel(yo, yr), _rel(yb, yr)
    assert e < 1.1 * e0 + 1e-6, (e, e0)
    e, e0 = _rel(xo.grad, xr.grad), _rel(xb.grad, xr.grad)
    assert e < 1.1 * e0 + 1e-6, (e, e0)
    assert _rel(ours.weight.grad, ref.weight.grad) < 2.1e-2      # bf16 operands of the column sums: the same bound the Triton path meets
    if bias:
        assert _rel(ours.bias.grad, ref.bias.grad) < 2.1e-2


@pytest.mark.parametrize("shape", [(1, 100, 100, 267), (3, 1500, 451), (2, 1, 256), (1, 130, 130, 64)])
def test_layernorm_dx_is_reproducible(shape):
    """The input gradient (and the forward) are bit-reproducible; the parameter gradients are fixed-order sums (equal to fp32 rounding between runs: a dynamic work counter
    decides which CTA takes which rows at large M)."""
    d = shape[-1]
    _, ours = _norm(d, 1)
    x = torch.randn(*shape, device="cuda").to(BF)
    dy = torch.randn_like(x)
    outs = []
    for _ in range(3):
        xx = x.clone().requires_grad_()
        ours.zero_grad()
        y = ours(xx)
        y.backward(dy)
        outs.append((y.detach(), xx.grad, ours.weight.grad.clone(), ours.bias.grad.clone()))
    for o in outs[1:]:
        assert torch.equal(o[0], outs[0][0])
        assert torch.equal(o[1], outs[0][1])
        assert _rel(o[2], outs[0][2]) < 1e-5
        assert _rel(o[3], outs[0][3]) < 1e-5


@pytest.mark.parametrize(("d", "bias"), [(64, 1), (128, 0), (267, 0), (384, 1), (768, 1)])
def test_layernorm_and_rmsnorm_fp32_modules(d, bias, spy):
    """fp32 activations (the rows read as 4-element chunks): forward and gradients against the fp32 PyTorch module to fp32 rounding."""
    torch.manual_seed(0)
    ref = LayerNorm(d, bias=bool(bias), implementation=ImplementationType.PYTORCH).cuda()
    ours = LayerNorm(d, bias=bool(bias), implementation=MW).cuda()
    with torch.no_grad():
        ref.weight.copy_(torch.randn(d) * 0.5 + 1.0)
        if bias:
            ref.bias.copy_(torch.randn(d) * 0.3)
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(3, 70, d, device="cuda") * 2 + 0.5
    dy = torch.randn_like(x)
    xr, xo = x.clone().requires_grad_(), x.clone().requires_grad_()
    ref(xr).backward(dy)
    yo = ours(xo)
    yo.backward(dy)
    assert spy["layernorm"] == 1
    assert yo.dtype == F32
    assert _rel(yo, ref(x)) < 1e-6
    assert _rel(xo.grad, xr.grad) < 1e-5
    assert _rel(ours.weight.grad, ref.weight.grad) < 1e-5
    if bias:
        assert _rel(ours.bias.grad, ref.bias.grad) < 1e-5
    # RMSNorm over the same width in fp32
    rref = RMSNorm(d, implementation=ImplementationType.PYTORCH).cuda()
    rours = RMSNorm(d, implementation=MW).cuda()
    with torch.no_grad():
        rref.weight.copy_(torch.randn(d) * 0.5 + 1.0)
    rours.load_state_dict(rref.state_dict())
    xr, xo = x.clone().requires_grad_(), x.clone().requires_grad_()
    rref(xr).backward(dy)
    ro = rours(xo)
    ro.backward(dy)
    assert spy["rmsnorm"] == 1
    assert _rel(ro, rref(x)) < 1e-6
    assert _rel(xo.grad, xr.grad) < 1e-5
    assert _rel(rours.weight.grad, rref.weight.grad) < 1e-5


def test_layernorm_without_affine_and_with_bf16_parameters():
    from miniworld_engine.kernels.layernorm.cuda import sm80

    x = torch.randn(4, 130, 384, device="cuda").to(BF).requires_grad_()
    dy = torch.randn_like(x)
    # elementwise_affine=False: no weight, no bias
    y = LayerNorm(384, elementwise_affine=False, implementation=MW).cuda()(x)
    want = F.layer_norm(x.detach().double(), (384,), eps=1e-5)
    assert _rel(y, want) < 5e-3
    y.backward(dy)
    xr = x.detach().double().requires_grad_()
    F.layer_norm(xr, (384,), eps=1e-5).backward(dy.double())
    assert _rel(x.grad, xr.grad) < 5e-3
    # bf16 weight and bias (the kernel-level bench's contract)
    w, b = torch.randn(384, device="cuda").to(BF), torch.randn(384, device="cuda").to(BF)
    assert sm80.supports(x, w, b)
    y2 = sm80.layernorm(x.detach(), w, b, 1e-5)
    assert _rel(y2, F.layer_norm(x.detach().double(), (384,), w.double(), b.double(), 1e-5)) < 5e-3


def test_layernorm_row_scale():
    """The pair-mask fold: y = LN(x) * rs, the backward scales dy by rs (dx, dw, db follow)."""
    from miniworld_engine.kernels.layernorm.cuda import sm80

    torch.manual_seed(0)
    x = torch.randn(1, 40, 40, 128, device="cuda").to(BF).requires_grad_()
    w, b = torch.randn(128, device="cuda").requires_grad_(), torch.randn(128, device="cuda").requires_grad_()
    rs = (torch.rand(1, 40, 40, device="cuda") > 0.3).to(BF)
    dy = torch.randn_like(x)
    y = sm80.layernorm(x, w, b, 1e-5, rs)
    dx, dw, db = torch.autograd.grad(y, [x, w, b], dy)
    xr, wr, br = x.detach().double().requires_grad_(), w.detach().double().requires_grad_(), b.detach().double().requires_grad_()
    yr = F.layer_norm(xr, (128,), wr, br, 1e-5) * rs.double()[..., None]
    gx, gw, gb = torch.autograd.grad(yr, [xr, wr, br], dy.double())
    assert _rel(y, yr) < 5e-3
    assert _rel(dx, gx) < 5e-3
    assert _rel(dw, gw) < 1e-4
    assert _rel(db, gb) < 1e-4


def test_layernorm_gate_conditions(monkeypatch):
    """The path declines what it does not implement: fp16, a CPU tensor, a width past 4096, mixed parameter dtypes, an empty batch; the engine backend forced to Triton; the switch."""
    from miniworld_engine.kernels.layernorm.cuda import sm80

    x = torch.randn(8, 128, device="cuda").to(BF)
    w = torch.ones(128, device="cuda")
    assert sm80.supports(x, w, w)
    assert sm80.supports(x, w, None)
    assert sm80.supports(x, None, None)
    assert not sm80.supports(x.half(), w, w)
    assert not sm80.supports(x.cpu(), w.cpu(), w.cpu())
    assert not sm80.supports(torch.randn(2, 4097, device="cuda").to(BF), torch.ones(4097, device="cuda"), None)
    assert not sm80.supports(x, w, w.to(BF))                                  # one dtype for weight and bias
    assert not sm80.supports(x, torch.ones(64, device="cuda"), None)          # wrong width
    assert not sm80.supports(x[:0], w, w)
    settings.configure(engine_backend="triton")
    assert not sm80.supports(x, w, w)
    settings.configure(engine_backend="auto")
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "0")
    assert not sm80.supports(x, w, w)


def test_layernorm_tiny_training_steps_keep_the_triton_path(monkeypatch, spy):
    """A training step over 20 - 100 K elements of a vector width is launch-bound and the Triton path wins it: routed away by default; inference, larger steps and ``MINIWORLD_NORMS_SM80=force`` run the CUDA rows."""
    monkeypatch.delenv("MINIWORLD_NORMS_SM80")
    _, ours = _norm(384, 1)
    small = torch.randn(2, 96, 384, device="cuda").to(BF)            # 73.7 K elements
    large = torch.randn(2, 400, 384, device="cuda").to(BF)           # 307 K elements
    ours(small.clone().requires_grad_()).sum().backward()
    assert spy["layernorm"] == 0
    with torch.no_grad():
        ours(small)
    assert spy["layernorm"] == 1                                      # inference: the CUDA rows
    ours(large.clone().requires_grad_()).sum().backward()
    assert spy["layernorm"] == 2                                      # a larger step: the CUDA rows
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "force")
    ours(small.clone().requires_grad_()).sum().backward()
    assert spy["layernorm"] == 3                                      # forced


def test_layernorm_env_switch_keeps_the_triton_path(monkeypatch, spy):
    """MINIWORLD_NORMS_SM80=0: the same module call runs the Triton kernels (the A100 kernels see no call) and agrees to bf16 accuracy."""
    _, ours = _norm(384, 1)
    x = torch.randn(2, 128, 384, device="cuda").to(BF)
    with torch.no_grad():
        y = ours(x)
    assert spy["layernorm"] == 1
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "0")
    with torch.no_grad():
        y0 = ours(x)
    assert spy["layernorm"] == 1
    assert _rel(y, y0) < 1e-2


@pytest.mark.parametrize(("d", "bias", "shape"), [(128, 1, (1, 64, 64, 128)), (267, 0, (1, 72, 72, 267)), (451, 1, (2, 96, 451))])
def test_layernorm_compiled_matches_eager(d, bias, shape):
    """torch.compile(fullgraph=True) over a training step equals eager: the forward and dx bit for bit, the parameter gradients to fp32 rounding."""
    torch._dynamo.reset()
    _, ours = _norm(d, bias)
    x = torch.randn(*shape, device="cuda").to(BF)
    dy = torch.randn_like(x)

    def step(module):
        xx = x.clone().requires_grad_()
        module.zero_grad()
        y = module(xx)
        y.backward(dy)
        return y.detach(), xx.grad, [p.grad.clone() for p in module.parameters()]

    comp = torch.compile(ours, fullgraph=True)
    ye, dxe, ge = step(ours)
    yc, dxc, gc = step(comp)
    assert torch.equal(yc, ye)
    assert torch.equal(dxc, dxe)
    assert _all_close(zip(gc, ge, strict=True), 1e-5)


def test_layernorm_cuda_graph_step():
    """A captured forward + backward step replays correctly on new data (extension built and buffers allocated before the capture)."""
    _, ours = _norm(384, 1)
    x = torch.randn(1, 96, 96, 384, device="cuda").to(BF)
    dy = torch.randn_like(x)
    xg = x.clone().requires_grad_()

    def step():
        ours.zero_grad(set_to_none=False)
        y = ours(xg)
        y.backward(dy)
        return y

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    ours.weight.grad = torch.zeros_like(ours.weight)
    ours.bias.grad = torch.zeros_like(ours.bias)
    xg.grad = torch.zeros_like(xg)
    with torch.cuda.graph(graph):
        out = step()
    x2 = torch.randn_like(x)
    xg.data.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    xr = x2.double().requires_grad_()
    w, b = ours.weight.detach().double().requires_grad_(), ours.bias.detach().double().requires_grad_()
    yr = F.layer_norm(xr, (384,), w, b, 1e-5)
    gx, gw, gb = torch.autograd.grad(yr, [xr, w, b], dy.double())
    assert _rel(out, yr) < 5e-3
    assert _rel(xg.grad, gx) < 5e-3
    assert _rel(ours.weight.grad, gw) < 1e-2
    assert _rel(ours.bias.grad, gb) < 1e-2


# ------------------------------------------------------------------------------------------------------------------------------ RMSNorm
@pytest.mark.parametrize("d", [32, 48, 64, 96, 128, 384])
@pytest.mark.parametrize("affine", [False, True])
def test_rmsnorm_module(d, affine, spy):
    """The q / k head norms (32..128) and the DiT width: forward, dx and dweight against fp32, as accurate as the bf16 module."""
    torch.manual_seed(0)
    ref = RMSNorm(d, elementwise_affine=affine, implementation=ImplementationType.PYTORCH)
    if affine:
        with torch.no_grad():
            ref.weight.copy_(torch.randn(d) * 0.5 + 1.0)
    ours = RMSNorm(d, elementwise_affine=affine, implementation=MW)
    ours.load_state_dict(ref.state_dict())
    ref, ours = ref.cuda(), ours.cuda().to(BF)
    x = torch.randn(2, 130, 4, d, device="cuda") * 2 + 0.5
    dy = torch.randn_like(x)
    xr = x.clone().requires_grad_()
    xo = x.to(BF).requires_grad_()
    xb = x.to(BF).requires_grad_()
    yr = ref(xr)
    yr.backward(dy)
    yo = ours(xo)
    yo.backward(dy.to(BF))
    eps = ours.effective_eps(BF)
    wb = ours.weight.detach().clone().requires_grad_() if affine else None      # copies: the bf16 path must not accumulate into the module's gradient
    yb = F.rms_norm(xb.float(), (d,), wb, eps).to(BF)
    yb.backward(dy.to(BF))
    assert spy["rmsnorm"] == 1, "the A100 kernels did not serve the call"
    assert yo.dtype == BF
    e, e0 = _rel(yo, yr), _rel(yb, yr)
    assert e < 1.1 * e0 + 1e-6, (e, e0)
    e, e0 = _rel(xo.grad, xr.grad), _rel(xb.grad, xr.grad)
    assert e < 1.1 * e0 + 1e-6, (e, e0)
    if affine:
        assert ours.weight.grad.dtype == ours.weight.dtype
        assert _rel(ours.weight.grad, ref.weight.grad) < 1e-2


def test_rmsnorm_explicit_triton_request_stays_triton(spy):
    d = 64
    norm = RMSNorm(d, implementation=ImplementationType.TRITON).cuda().to(BF)
    x = torch.randn(4, 64, d, device="cuda").to(BF)
    with torch.no_grad():
        norm(x)
    assert spy["rmsnorm"] == 0


def test_rmsnorm_gate_and_compile():
    from miniworld_engine.kernels.rmsnorm.cuda import sm80

    w = torch.ones(48, device="cuda")
    x = torch.randn(8, 48, device="cuda").to(BF)
    assert sm80.supports(x, w)
    assert sm80.supports(x, None)
    assert not sm80.supports(x.half(), w)
    assert not sm80.supports(x, torch.ones(32, device="cuda"))
    torch._dynamo.reset()
    norm = RMSNorm(48, implementation=MW).cuda().to(BF)
    xx = torch.randn(2, 64, 4, 48, device="cuda").to(BF)
    dy = torch.randn_like(xx)

    def step(module):
        norm.zero_grad()
        x1 = xx.clone().requires_grad_()
        y = module(x1)
        y.backward(dy)
        return y.detach(), x1.grad, norm.weight.grad.clone()

    ye, dxe, ge = step(norm)
    yc, dxc, gc = step(torch.compile(norm, fullgraph=True))
    assert torch.equal(yc, ye)
    assert torch.equal(dxc, dxe)
    assert _rel(gc, ge) < 1e-5


# ------------------------------------------------------------------------------------------------------------------------------ RoPE and the fused Q/K norm + RoPE
def _angles(n, s, half, bcast):
    ang = torch.rand(1 if bcast else n, s, half, device="cuda") * 6.0 - 3.0
    return ang.cos(), ang.sin()


def _rot_ref(x, cos, sin):
    half = cos.shape[-1]
    n = x.shape[0]
    c, s = cos.double().expand(n, -1, -1)[:, :, None, :], sin.double().expand(n, -1, -1)[:, :, None, :]
    lo, hi = x[..., :half], x[..., half:2 * half]
    return torch.cat([lo * c - hi * s, hi * c + lo * s, x[..., 2 * half:]], -1)


@pytest.mark.parametrize(("n", "s", "h", "d"), [(5, 100, 4, 32), (48, 17, 4, 32), (2, 50, 2, 64), (1, 33, 1, 128)])
@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize("bcast", [False, True])
def test_qk_norm_rope_fused(n, s, h, d, dtype, bcast):
    """Fused Q/K RMSNorm + RoPE on the strided views of an interleaved QKV projection: forward and both input gradients against fp64, as accurate as the Triton kernel."""
    from miniworld_engine.kernels.rope.cuda import sm80 as rope
    from miniworld_engine.kernels.rope.triton.qk_norm import (
        qk_norm_rope_3d as triton_qk,
    )

    torch.manual_seed(0)
    qkv = torch.randn(n, s, 3 * h * d, device="cuda").to(dtype)
    q, k, _ = qkv.view(n, s, 3, h, d).permute(2, 0, 1, 3, 4).unbind(0)
    cos, sin = _angles(n, s, d // 2, bcast)
    assert rope.supports_qk(q, k, cos, sin)
    gq, gk = torch.randn(n, s, h, d, device="cuda").to(dtype), torch.randn(n, s, h, d, device="cuda").to(dtype)
    eps = torch.finfo(F32).eps

    def norm_rot(t):
        td = t.double().requires_grad_()
        return td, _rot_ref(td * torch.rsqrt(td.pow(2).mean(-1, keepdim=True) + eps), cos, sin)

    qd, oqd = norm_rot(q)
    kd, okd = norm_rot(k)
    dqd, dkd = torch.autograd.grad([oqd, okd], [qd, kd], [gq.double(), gk.double()])
    qo, ko = q.clone().requires_grad_(), k.clone().requires_grad_()
    oq, ok = rope.qk_norm_rope_3d(qo, ko, cos, sin)
    dq, dk = torch.autograd.grad([oq, ok], [qo, ko], [gq, gk])
    qt, kt = q.clone().requires_grad_(), k.clone().requires_grad_()
    tq, tk = triton_qk(qt, kt, cos, sin)
    dtq, dtk = torch.autograd.grad([tq, tk], [qt, kt], [gq, gk])
    assert oq.dtype == dtype
    assert dq.dtype == dtype
    assert oq.is_contiguous()
    assert _rel(oq, oqd) < 1.1 * _rel(tq, oqd) + 1e-7
    assert _rel(ok, okd) < 1.1 * _rel(tk, okd) + 1e-7
    assert _rel(dq, dqd) < 1.1 * _rel(dtq, dqd) + 1e-7
    assert _rel(dk, dkd) < 1.1 * _rel(dtk, dkd) + 1e-7


@pytest.mark.parametrize("half", [8, 16])
def test_qk_norm_rope_partial_rotary_prefix(half):
    """Only the leading 2 HALF channels rotate; the tail of the head is the normalised input."""
    from miniworld_engine.kernels.rope.cuda import sm80 as rope
    from miniworld_engine.kernels.rope.triton.qk_norm import (
        qk_norm_rope_3d as triton_qk,
    )

    q = torch.randn(2, 40, 4, 64, device="cuda").to(BF)
    k = torch.randn_like(q)
    cos, sin = _angles(2, 40, half, False)
    assert rope.supports_qk(q, k, cos, sin)
    oq, ok = rope.qk_norm_rope_3d(q, k, cos, sin)
    tq, tk = triton_qk(q, k, cos, sin)
    assert _rel(oq, tq) < 4e-3
    assert _rel(ok, tk) < 4e-3


@pytest.mark.parametrize("dtype", [BF, F32])
def test_rope_standalone(dtype):
    from miniworld_engine.kernels.rope.cuda import sm80 as rope

    x = torch.randn(3, 50, 4, 64, device="cuda").to(dtype).requires_grad_()
    cos, sin = _angles(3, 50, 32, False)
    assert rope.supports_rope(x, cos, sin)
    g = torch.randn_like(x)
    y = rope.rope_3d(x, cos, sin)
    dx, = torch.autograd.grad(y, x, g)
    xd = x.detach().double().requires_grad_()
    yd = _rot_ref(xd, cos, sin)
    gd, = torch.autograd.grad(yd, xd, g.double())
    tol = 5e-3 if dtype is BF else 1e-6
    assert _rel(y, yd) < tol
    assert _rel(dx, gd) < tol


def test_rope_gate_conditions():
    from miniworld_engine.kernels.rope.cuda import sm80 as rope

    qkv = torch.randn(2, 8, 3 * 4 * 32 + 8, device="cuda").to(BF)         # a row of the projection padded by one chunk
    cos, sin = _angles(2, 8, 16, False)
    q = qkv[..., :128].reshape(2, 8, 4, 32)
    assert rope.supports_qk(q, q, cos, sin)
    odd = qkv[..., 1:129].reshape(2, 8, 4, 32)             # a storage offset of one element: the op copies it before it reads with 16-byte loads (the gate cannot tell: torch.compile traces it)
    assert rope.supports_qk(odd, odd, cos, sin)
    oq, ok = rope.qk_norm_rope_3d(odd, odd, cos, sin)
    wq, wk = rope.qk_norm_rope_3d(odd.contiguous().clone(), odd.contiguous().clone(), cos, sin)
    assert torch.equal(oq, wq)
    assert torch.equal(ok, wk)
    ragged = torch.randn(2, 8, 3 * 4 * 32 + 1, device="cuda").to(BF)[..., :128].reshape(2, 8, 4, 32)      # a row stride of 385 elements: not whole 16-byte chunks
    assert not rope.supports_qk(ragged, ragged, cos, sin)
    assert not rope.supports_qk(q, q, cos.double(), sin.double())
    assert not rope.supports_qk(q.half(), q.half(), cos, sin)
    assert not rope.supports_qk(q[..., :24], q[..., :24], cos[..., :12], sin[..., :12])      # head dim 24
    assert not rope.supports_qk(q, q, cos[:, :4], sin[:, :4])                                 # table shorter than the sequence
    cos.requires_grad_()
    assert not rope.supports_qk(q, q, cos, sin)


def _attention(implementation, seed=0):
    from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention

    torch.manual_seed(seed)
    return SWA3DRoPEAttention(128, 4, implementation=implementation).cuda().to(BF)


def _swa_params(n_atoms, a):
    from miniworld_engine.modules.swa_atom_attention import build_3d_rope
    from miniworld_engine.modules.swa_atom_attention.module import (
        build_attention_params,
    )

    cos, sin = build_3d_rope(torch.randn(1, n_atoms, 3, device="cuda") * 10, torch.zeros(1, n_atoms, dtype=torch.long, device="cuda"), 32)
    valid = torch.ones(a, n_atoms, dtype=torch.bool, device="cuda")
    return build_attention_params(cos, sin, valid, a)


def test_swa_attention_module_uses_the_fused_kernel(spy):
    """modules/swa_atom_attention: the qk-norm + RoPE of MINIWORLD runs the A100 kernel, and the module agrees with the PYTORCH implementation of the same weights."""
    n_atoms, a = 256, 5
    ours, ref = _attention(MW), _attention(ImplementationType.PYTORCH)
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(a, n_atoms, 128, device="cuda").to(BF)
    params = _swa_params(n_atoms, a)
    with torch.no_grad():
        y = ours(x, params)
        yr = ref(x, params)
    assert spy["qk_norm_rope"] == 1
    assert _rel(y, yr) < 3e-2


def test_swa_attention_gradients(spy):
    n_atoms, a = 192, 4
    ours, ref = _attention(MW), _attention(ImplementationType.PYTORCH)
    ref.load_state_dict(ours.state_dict())
    params = _swa_params(n_atoms, a)
    x = torch.randn(a, n_atoms, 128, device="cuda").to(BF)
    dy = torch.randn_like(x)
    xo, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
    ours(xo, params).backward(dy)
    ref(xr, params).backward(dy)
    assert spy["qk_norm_rope"] == 1
    assert _rel(xo.grad, xr.grad) < 5e-2
    assert _rel(ours.Wqkv.weight.grad, ref.Wqkv.weight.grad) < 5e-2


def test_qk_norm_rope_compiled_and_graph_replay():
    from miniworld_engine.kernels.rope.cuda import sm80 as rope

    torch._dynamo.reset()
    n, s, h, d = 5, 64, 4, 32
    qkv = torch.randn(n, s, 3 * h * d, device="cuda").to(BF)
    cos, sin = _angles(n, s, d // 2, False)
    gq, gk = torch.randn(n, s, h, d, device="cuda").to(BF), torch.randn(n, s, h, d, device="cuda").to(BF)

    def forward(qkv_):
        q, k, _ = qkv_.view(n, s, 3, h, d).permute(2, 0, 1, 3, 4).unbind(0)
        return rope.qk_norm_rope_3d(q, k, cos, sin)

    def step(qkv_, fwd=forward):
        oq, ok = fwd(qkv_)
        return oq, ok, *torch.autograd.grad([oq, ok], [qkv_], [gq, gk])      # the backward runs outside the compiled region

    eager = step(qkv.clone().requires_grad_())
    got = step(qkv.clone().requires_grad_(), fwd=torch.compile(forward, fullgraph=True))
    for a, b in zip(got, eager, strict=True):
        assert torch.equal(a, b)
    # CUDA graph: capture the forward + backward, replay on new data
    static = qkv.clone().requires_grad_()
    for _ in range(2):
        step(static)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step(static)
    new = torch.randn_like(qkv)
    static.data.copy_(new)
    g.replay()
    torch.cuda.synchronize()
    want = step(new.clone().requires_grad_())
    for a, b in zip(out, want, strict=True):
        assert torch.equal(a, b)


def test_norms_env_switch_is_read_at_call_time(monkeypatch):
    """One switch turns the whole family off: LayerNorm, RMSNorm and the RoPE kernels refuse together."""
    from miniworld_engine.kernels.layernorm.cuda import sm80 as ln
    from miniworld_engine.kernels.rmsnorm.cuda import sm80 as rms
    from miniworld_engine.kernels.rope.cuda import sm80 as rope

    x = torch.randn(4, 8, 4, 32, device="cuda").to(BF)
    cos, sin = _angles(4, 8, 16, False)
    w = torch.ones(32, device="cuda")
    served = (ln.supports(x, w, w), rms.supports(x, w), rope.supports_qk(x, x, cos, sin), rope.supports_rope(x, cos, sin))
    assert all(served)
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "0")
    refused = (ln.supports(x, w, w), rms.supports(x, w), rope.supports_qk(x, x, cos, sin), rope.supports_rope(x, cos, sin))
    assert not any(refused)
    monkeypatch.delenv("MINIWORLD_NORMS_SM80")
    assert ln.supports(x, w, w)


# ------------------------------------------------------------------------------------------------------------------------------ RMSNorm + adaLN modulation
def _modulation_inputs(m, seed=0):
    torch.manual_seed(seed)
    q = (torch.randn(m, 128, device="cuda") * 2 + 0.3).to(BF)
    c = F.silu(torch.randn(m, 128, device="cuda")).to(BF)
    ws = [(torch.randn(128, 128, device="cuda") * 128 ** -0.5).to(BF) for _ in range(3)]
    dy, dg = torch.randn(m, 128, device="cuda").to(BF), torch.randn(m, 128, device="cuda").to(BF)
    return q, c, ws, dy, dg


def _modulation_reference(q, c, wsc, wsh, wg, weight, eps):
    """The function in whatever dtype its inputs carry (fp64 for the ground truth)."""
    normed = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * weight
    return normed * (1 + c @ wsc.mT) + c @ wsh.mT, c @ wg.mT


def _modulation_torch_bf16(q, c, wsc, wsh, wg, weight, eps):
    """The PyTorch path at bf16: fp32 norm, bf16 linears (the accuracy the CUDA path is held to)."""
    qf = q.float()
    normed = qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * weight.float()
    sc, sh, g = F.linear(c, wsc), F.linear(c, wsh), F.linear(c, wg)
    return (normed * (1 + sc.float()) + sh.float()).to(q.dtype), g


@pytest.mark.parametrize("m", [1, 5, 64, 100, 20481])
@pytest.mark.parametrize("wdtype", [None, F32, BF])
def test_rms_norm_modulation_forward_and_gradients(m, wdtype, spy):
    """``ops.rms_norm_modulation`` on the atom stream (128 / 128, bf16): y, gate and the gradients of q, c, the three weights and the norm weight against fp64, as accurate as the
    bf16 PyTorch path."""
    from miniworld_engine import ops

    q, c, ws, dy, dg = _modulation_inputs(m)
    w = None if wdtype is None else torch.randn(128, device="cuda").to(wdtype)
    eps = torch.finfo(F32).eps
    leaves = [t.double().requires_grad_() for t in (q, c, *ws)] + ([] if w is None else [w.double().requires_grad_()])
    yd, gd = _modulation_reference(*leaves[:5], None if w is None else leaves[5], eps)
    want = torch.autograd.grad([yd, gd], leaves, [dy.double(), dg.double()])

    def run(fn):
        ls = [t.clone().requires_grad_() for t in (q, c, *ws)] + ([] if w is None else [w.clone().requires_grad_()])
        y, g = fn(ls[0], ls[1], ls[2], ls[3], ls[4], None if w is None else ls[5], eps)
        return y, g, torch.autograd.grad([y, g], ls, [dy, dg])

    def engine(*a):
        y, g = ops.rms_norm_modulation(a[0].view(1, -1, 128), a[1].view(1, -1, 128), a[2], a[3], a[4], a[5], a[6])
        return y.view(m, 128), g.view(m, 128)

    yo, go, gradso = run(engine)
    assert spy["modulation"] == 1, "the A100 kernels did not serve the call"
    yb, gb, gradsb = run(_modulation_torch_bf16)
    assert yo.dtype == BF
    assert go.dtype == BF
    assert _rel(yo, yd) < 1.1 * _rel(yb, yd) + 1e-6
    assert _rel(go, gd) < 1.1 * _rel(gb, gd) + 1e-6
    names = ["dq", "dc", "dwsc", "dwsh", "dwg"] + ([] if w is None else ["dw"])
    for i, name in enumerate(names):
        assert gradso[i].dtype == (w.dtype if name == "dw" else BF)
        # the norm-weight gradient is a column sum over all rows in fp32 atomics: held to the bf16 bound the PyTorch path meets
        slack = 2e-3 if name == "dw" else 1e-6
        assert _rel(gradso[i], want[i]) < 1.1 * _rel(gradsb[i], want[i]) + slack, name


def test_rms_norm_modulation_inference_matches_training_forward():
    """The no-grad kernel (nothing saved) and the training forward (rstd saved) produce the same bits."""
    from miniworld_engine import ops

    q, c, ws, _, _ = _modulation_inputs(777)
    w = torch.randn(128, device="cuda")
    with torch.no_grad():
        yi, gi = ops.rms_norm_modulation(q, c, *ws, w, 1e-5)
    qq = q.clone().requires_grad_()
    yt, gt = ops.rms_norm_modulation(qq, c, *ws, w, 1e-5)
    assert torch.equal(yi, yt)
    assert torch.equal(gi, gt)


def test_rms_norm_modulation_gate_conditions(monkeypatch):
    """The path serves bf16 128-wide rows and the weight views the SWA DiT slices out of its adaLN projection; it declines the rest (Triton keeps those)."""
    from miniworld_engine.kernels.rmsnorm.cuda import sm80

    q, c, ws, _, _ = _modulation_inputs(64)
    assert sm80.supports_adamod(q, c, *ws)
    assert sm80.supports_adamod(q, c, *ws, torch.ones(128, device="cuda"))
    assert sm80.supports_adamod(q, c, *ws, torch.ones(128, device="cuda").to(BF))
    projection = torch.randn(6 * 128, 128, device="cuda").to(BF)
    sh_a, sc_a, g_a, *_ = projection.chunk(6, dim=0)
    assert sm80.supports_adamod(q, c, sc_a, sh_a, g_a)                      # row slices of one projection: offsets are whole tiles
    assert not sm80.supports_adamod(q.float(), c, *ws)
    assert not sm80.supports_adamod(q[:, :64], c[:, :64], *ws)
    assert not sm80.supports_adamod(q, c[:32], *ws)
    assert not sm80.supports_adamod(q[:0], c[:0], *ws)
    assert not sm80.supports_adamod(q, c, ws[0], ws[1].T, ws[2])           # a transposed weight
    assert not sm80.supports_adamod(q, c, *ws, torch.ones(64, device="cuda"))
    settings.configure(engine_backend="triton")
    assert not sm80.supports_adamod(q, c, *ws)
    settings.configure(engine_backend="auto")
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "0")
    assert not sm80.supports_adamod(q, c, *ws)


def test_rms_norm_modulation_env_switch_keeps_the_triton_path(monkeypatch, spy):
    from miniworld_engine import ops

    q, c, ws, _, _ = _modulation_inputs(200)
    q, c = q.view(1, 200, 128), c.view(1, 200, 128)                 # (B, L, D): what the Triton launcher keys on
    with torch.no_grad():
        y, g = ops.rms_norm_modulation(q, c, *ws, None, 1e-5)
    assert spy["modulation"] == 1
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "0")
    with torch.no_grad():
        y0, g0 = ops.rms_norm_modulation(q, c, *ws, None, 1e-5)
    assert spy["modulation"] == 1
    assert _rel(y, y0) < 1e-2
    assert _rel(g, g0) < 1e-2


def test_rms_norm_modulation_compiled_matches_eager_and_graph_replay():
    from miniworld_engine import ops

    torch._dynamo.reset()
    m = 300
    q, c, ws, dy, dg = _modulation_inputs(m)
    w = torch.randn(128, device="cuda")
    leaves = [q, c, *ws, w]

    def forward(*ls):
        return ops.rms_norm_modulation(ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], 1e-5)

    def step(*xs, fwd=forward):
        ls = [t.clone().requires_grad_() for t in xs]
        y, g = fwd(*ls)
        return y, g, *torch.autograd.grad([y, g], ls, [dy, dg])      # the backward runs outside the compiled region

    eager = step(*leaves)
    got = step(*leaves, fwd=torch.compile(forward, fullgraph=True))
    for i, (a, b) in enumerate(zip(got, eager, strict=True)):
        if i == 7:                                                          # the norm-weight gradient: fp32 atomics, equal to rounding
            assert _rel(a, b) < 1e-5
        else:
            assert torch.equal(a, b)
    # CUDA graph over forward + backward (cuBLAS workspaces warmed up by the eager steps above)
    static = [t.clone() for t in leaves]
    for _ in range(2):
        step(*static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(*static)
    fresh = _modulation_inputs(m, seed=5)
    new = [fresh[0], fresh[1], *fresh[2], torch.randn(128, device="cuda")]
    for dst, src in zip(static, new, strict=True):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    want = step(*new)
    for i, (a, b) in enumerate(zip(out, want, strict=True)):
        if i == 7:
            assert _rel(a, b) < 1e-5
        else:
            assert torch.equal(a, b)
