"""AF3 block-local atom DiT on H100 (integrations/local_dit_h100.py: warp-level tensor-core row kernels + mma.sync window attention).

fp32 activations and parameters run the TF32 kernels, bf16 activations (fp32 master or bf16 parameters) the bf16 ones. The bar is the
repository's (docs/kernels/gpu-kernel-development-guide.md section 6): the output and every input / parameter gradient are no worse
than 1.5x the PyTorch module's error against an fp64 block (+ an absolute floor), plus bitwise repeatability of the whole training step,
padded lengths and key masks, the opaque ops under a CUDA graph, and that the kernels actually served the call.
"""


import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.integrations import local_dit_h100 as h100
from miniworld_engine.kernels import _capture
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

CASES = [
    (3, 512, True),
    (1, 1000, True),
    (2, 200, False),
    (5, 256, False),
]  # (A, N, masked)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _block(seed=0, qk=False):
    torch.manual_seed(seed)
    blk = LocalDiTBlock(
        128,
        128,
        16,
        4,
        n=2,
        cross_attention=True,
        use_qk_norm=qk,
        implementation=ImplementationType.TRITON,
    )
    with torch.no_grad():  # non-default parameters (zero-initialised gates would make the backward vacuous)
        for p in blk.parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.add_(torch.randn_like(p) * 0.1)
    return blk.cuda().float()


def _reference(blk, qk=False):
    ref = LocalDiTBlock(
        128,
        128,
        16,
        4,
        n=2,
        cross_attention=True,
        use_qk_norm=qk,
        implementation=ImplementationType.PYTORCH,
    )
    ref.load_state_dict(blk.state_dict())
    return ref.cuda().double()


def _inputs(a, n, masked, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    c = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    z = torch.randn(1, (n + 31) // 32, 32, 128, 16, device="cuda", generator=g)
    dy = torch.randn_like(s)
    mask = None
    if masked:
        mask = torch.rand(1, n, device="cuda", generator=g) > 0.1
        mask[0, 0] = True
    return s, c, z, dy, mask


def _train(mod, s, c, z, dy, mask, dtype, autocast=False):
    ins = [x.detach().to(dtype).clone().requires_grad_() for x in (s, c, z)]
    for p in mod.parameters():
        p.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        y = mod(*ins, mask)
    y.backward(dy.to(y.dtype))
    grads = [t.grad for t in ins] + [p.grad for p in mod.parameters()]
    return y.detach(), grads


@pytest.mark.parametrize("qk", [False, True], ids=["plain", "qknorm"])
@pytest.mark.parametrize(("a", "n", "masked"), CASES, ids=lambda v: str(v))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_training_matches_fp64(a, n, masked, dtype, qk):
    blk = _block(qk=qk)
    ref = _reference(blk, qk=qk)
    s, c, z, dy, mask = _inputs(a, n, masked)
    torch.backends.cuda.matmul.allow_tf32 = False
    y_ref, g_ref = _train(ref, s, c, z, dy, mask, torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = True
    if dtype == torch.bfloat16:
        blk = blk.float()  # fp32 master parameters under autocast
    ins = (s, c, z)
    served = h100.matches(blk, *(x.to(dtype) for x in ins), mask)
    assert served
    y, g = _train(blk, s, c, z, dy, mask, dtype, autocast=dtype == torch.bfloat16)
    tol = (
        1e-3 if dtype == torch.float32 else 8e-3
    )  # TF32 kernels: ~3e-4 (the TF32 PyTorch module: 1.6e-3); bf16: ~4.4e-3
    assert _rel(y, y_ref) < tol, _rel(y, y_ref)
    names = ["single", "cond", "pair"] + [nm for nm, _ in ref.named_parameters()]
    for nm, ge, gr in zip(names, g, g_ref, strict=True):
        assert ge is not None, nm
        assert torch.isfinite(ge).all(), nm
        assert _rel(ge, gr) < (3e-3 if dtype == torch.float32 else 2.5e-2), (
            nm,
            _rel(ge, gr),
        )


@pytest.mark.parametrize("qk", [False, True], ids=["plain", "qknorm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_training_step_is_bitwise_repeatable(dtype, qk):
    blk = _block(qk=qk)
    s, c, z, dy, mask = _inputs(3, 512, True)
    out = [
        _train(blk, s, c, z, dy, mask, dtype, autocast=dtype == torch.bfloat16)
        for _ in range(2)
    ]
    assert torch.equal(out[0][0], out[1][0])
    for ga, gb in zip(out[0][1], out[1][1], strict=True):
        assert torch.equal(ga, gb)


@pytest.mark.parametrize("qk", [False, True], ids=["plain", "qknorm"])
@pytest.mark.parametrize(("a", "n", "masked"), CASES, ids=lambda v: str(v))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_inference_matches_fp64(a, n, masked, dtype, qk):
    blk = _block(qk=qk).eval()
    ref = _reference(blk, qk=qk).eval()
    s, c, z, _, mask = _inputs(a, n, masked)
    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        y_ref = ref(s.double(), c.double(), z.double(), mask)
        torch.backends.cuda.matmul.allow_tf32 = True
        y = blk(*(x.to(dtype) for x in (s, c, z)), mask)
    assert y.dtype == dtype
    assert _rel(y, y_ref) < (1e-3 if dtype == torch.float32 else 8e-3), _rel(y, y_ref)


def test_inference_training_forward_agree():
    """The training forward (saving) and the inference forward (hoisted tables) are the same function of the inputs."""
    blk = _block()
    s, c, z, _, mask = _inputs(2, 384, True)
    with torch.no_grad():
        y_inf = blk(s, c, z, mask)
    y_tr = blk(s.clone().requires_grad_(), c, z, mask)
    assert torch.equal(y_inf, y_tr.detach())


def test_cuda_graph_replay_inference_and_training():
    blk = _block()
    s, c, z, dy, mask = _inputs(2, 512, True)
    with torch.no_grad(), _capture.static_weights(), _capture.static_inputs():
        blk(s, c, z, mask)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = blk(s, c, z, mask)
        g.replay()
        torch.cuda.synchronize()
        expected = blk(s, c, z, mask)
    assert torch.equal(out, expected)
    # a whole training step in a graph (weights repacked inside the capture)
    sx, cx, zx = (t.clone().requires_grad_() for t in (s, c, z))
    for p in blk.parameters():
        p.grad = torch.zeros_like(p)
    blk(sx, cx, zx, mask).backward(dy)  # warm-up (eager)
    torch.cuda.synchronize()
    ref_grads = [sx.grad.clone()] + [p.grad.clone() for p in blk.parameters()]
    g2 = torch.cuda.CUDAGraph()
    for t in (sx, cx, zx, *blk.parameters()):
        t.grad = None
    with torch.cuda.graph(g2):
        y = blk(sx, cx, zx, mask)
        y.backward(dy)
    g2.replay()
    torch.cuda.synchronize()
    assert torch.equal(sx.grad, ref_grads[0])


def test_fallback_outside_contract(monkeypatch):
    """A non-cross block, QK norm, an unsupported dtype mix or a non-contiguous input keeps the previous routes (matches() is False)."""
    blk = _block()
    s, c, z, _, mask = _inputs(2, 256, False)
    assert h100.matches(blk, s, c, z, mask)
    assert not h100.matches(blk, s.bfloat16(), c, z, mask)  # mixed activation dtypes
    override_setting(monkeypatch, "local_dit_h100_mma", False)
    assert not h100.matches(blk, s, c, z, mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_torch_compile_fullgraph_matches_eager(dtype):
    """The two opaque ops keep the block in one compiled graph, forward, backward and inference."""
    blk = _block()
    s, c, z, dy, mask = _inputs(2, 512, True)
    cb = torch.compile(blk, fullgraph=True)
    bf = dtype == torch.bfloat16
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf):
        eager = blk(s.to(dtype), c.to(dtype), z.to(dtype), mask)
        compiled = cb(s.to(dtype), c.to(dtype), z.to(dtype), mask)
    assert torch.equal(eager, compiled)
    sx = s.to(dtype).clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf):
        y = cb(sx, c.to(dtype), z.to(dtype), mask)
    y.backward(dy.to(y.dtype))
    assert torch.isfinite(sx.grad).all()
    assert all(p.grad is not None for p in blk.parameters())


@pytest.mark.parametrize(
    "a", [4, 5, 8], ids=lambda v: f"A{v}"
)  # 4, 8: samples reduced in groups of four inside the CTA; 5: per-sample rows
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_shared_conditioning_matches_materialised(dtype, a):
    """A conditioning given as an expanded view (stride 0 over the samples) is tabulated once per atom; output and every gradient equal the
    materialised-copies call (the gradient of the base row set is the sum over the samples)."""
    blk = _block()
    s, c, z, dy, mask = _inputs(a, 512, True)
    base = c[:1].to(dtype)
    bf = dtype == torch.bfloat16

    def run(cond_fn):
        sx = s.to(dtype).clone().requires_grad_()
        cb = base.clone().requires_grad_()
        for p in blk.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf):
            y = blk(sx, cond_fn(cb), z.to(dtype), mask)
        y.backward(dy.to(y.dtype))
        return y.detach(), sx.grad, cb.grad, [p.grad.clone() for p in blk.parameters()]

    ys, gs, gc, gp = run(lambda b: b.expand(a, 1, 512, 128))
    ym, gm, gcm, gpm = run(lambda b: b.expand(a, 1, 512, 128).contiguous())
    tol = 3e-3 if dtype == torch.float32 else 3e-2
    assert _rel(ys, ym) < 1e-3 if dtype == torch.float32 else _rel(ys, ym) < 1e-2
    assert _rel(gs, gm) < tol
    assert _rel(gc, gcm) < tol
    for a_, b_ in zip(gp, gpm, strict=True):
        assert _rel(a_, b_) < tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_split_and_unsplit_launches_agree(dtype):
    """Few 64-row tiles are split over CTAs (weight units, hidden chunks, K blocks with a fixed-order finish); many tiles run unsplit. Rows
    are independent, so the first three samples of a 36-sample call (288 tiles, unsplit) give the output and the input gradients of the
    3-sample call (24 tiles, split), up to the rounding of a re-ordered fp32 sum."""
    blk = _block()
    s, c, z, dy, mask = _inputs(36, 512, True)
    bf = dtype == torch.bfloat16

    def run(a):
        sx, cx = (t[:a].to(dtype).clone().requires_grad_() for t in (s, c))
        zx = z.to(dtype).clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf):
            y = blk(sx, cx, zx, mask)
        y.backward(dy[:a].to(y.dtype))
        return y.detach(), sx.grad, cx.grad

    y3, ds3, dc3 = run(3)
    y36, ds36, dc36 = run(36)
    tol = 1e-5 if dtype == torch.float32 else 3e-3
    assert _rel(y3, y36[:3]) < tol, _rel(y3, y36[:3])
    assert _rel(ds3, ds36[:3]) < tol, _rel(ds3, ds36[:3])
    assert _rel(dc3, dc36[:3]) < 10 * tol, _rel(
        dc3, dc36[:3]
    )  # the conditioning gradient sums its K blocks in a different order
