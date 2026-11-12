"""``AtomDiTESMFold2Pair`` (FoldForge R6) on the ``cuda`` (H100 / A100) and ``triton`` backends against the fp64 PyTorch module: the RMS /
SiLU row kernels (H100 inference) or the train_rows block, the q / k RMSNorm + rotary and the 192-key rank-window attention, inference and
training. Adapted from the 2.4.x ``tests/gpu/sm90/test_h100_local_dit_esmfold2_gpu.py``."""

import pytest
import torch

from miniworld_engine.modules import AtomDiTESMFold2Pair

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


needs_card = pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80) required")

# (A, N, real atoms): full, ragged, the 192-key window starting before atom 0 (under 192 real atoms), N not a multiple of 128
CASES = [(1, 256, 256), (2, 512, 400), (5, 1024, 1000), (1, 384, 150), (3, 640, 193), (2, 300, 300), (1, 1536, 1500)]


def _block(backend="pytorch"):
    torch.manual_seed(0)
    blk = AtomDiTESMFold2Pair(backend=backend)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 1 + torch.randn_like(p) * 0.1)
    return blk


def _inputs(a, n, real, seed=1, shared=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    c = torch.randn(1 if shared else a, 1, n, 128, device="cuda", generator=g).expand(a, 1, n, 128)
    z = torch.randn(1, (n + 31) // 32, 32, 192, 16, device="cuda", generator=g)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    angle = 3 * torch.randn(1, n, 16, device="cuda", generator=g)
    return s, c, z, mask, (angle.cos(), angle.sin())


@pytest.fixture(autouse=True)
def _tf32():
    """fp32 runs with TF32 products allowed, as the measurements judge them (``measure/config.TF32_BACKENDS``): the kernels' fp32 path
    computes its attention in TF32, and the module's own fp32 error is then the TF32 one."""
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@needs_card
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize(("a", "n", "real"), CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "master"])
def test_inference_matches_the_pytorch_module(backend, a, n, real, dtype):
    blk = _block(backend).cuda().eval()
    ref = _block().cuda().double().eval()
    own = _block().cuda().eval()
    s, c, z, mask, rope = _inputs(a, n, real)
    with torch.no_grad():
        expected = ref(s.double(), c.double(), z.double(), mask, rope=tuple(t.double() for t in rope))
        ins = (s.to(dtype), c.to(dtype), z.to(dtype))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            assert blk.unsupported(*ins, mask, rope=rope) is None
            actual = blk(*ins, mask, rope=rope)
            base = own(*ins, mask, rope=rope)
    e, b = _rel(actual[:, :, :real], expected[:, :, :real]), _rel(base[:, :, :real], expected[:, :, :real])
    assert e < 2 * b + (1e-3 if dtype == torch.bfloat16 else 2e-3), f"{backend}: {e:.3e} vs the module's own {b:.3e}"
    assert torch.isfinite(actual).all()


@needs_card
def test_bitwise_deterministic():
    blk = _block("cuda").cuda().eval()
    s, c, z, mask, rope = _inputs(5, 1024, 1000)
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            ins = (s.to(dtype), c.to(dtype), z.to(dtype))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                first = blk(*ins, mask, rope=rope)
                for _ in range(3):
                    assert torch.equal(blk(*ins, mask, rope=rope), first)


@needs_card
def test_refusals():
    blk = _block("cuda").cuda()
    s, c, z, mask, rope = _inputs(1, 256, 200)
    with torch.no_grad():
        assert blk.unsupported(s, c, z, mask, rope) is None
        assert "rope" in str(blk.unsupported(s, c, z, mask, None))
        assert blk.unsupported(s[:, :, :, :64], c, z, mask, rope) is not None
        s2 = s.expand(1, 2, 256, 128)
        assert blk.unsupported(s2, c.expand(1, 2, 256, 128), z.expand(2, -1, -1, -1, -1), mask, rope) is not None  # one fold only


@needs_card
@pytest.mark.parametrize("real", [512, 400, 150])
def test_captured_in_a_cuda_graph(real):
    """The real atom count stays on the device: the block runs inside a graph capture and replays with another mask and inputs."""
    blk = _block("cuda").cuda().requires_grad_(False)
    s, c, z, mask, rope = _inputs(2, 512, 512, seed=3)
    s2, c2, z2, _, rope2 = _inputs(2, 512, 512, seed=4)
    static = [t.clone() for t in (s, c, z, mask, *rope)]
    with torch.no_grad():
        blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))  # build and pack outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))
        m2 = torch.zeros_like(mask)
        m2[:, :real] = True
        for t, new in zip(static, (s2, c2, z2, m2, *rope2), strict=True):
            t.copy_(new)
        graph.replay()
        eager = blk(s2, c2, z2, m2, rope=rope2)
    torch.testing.assert_close(out[:, :, :real], eager[:, :, :real], rtol=0, atol=0)


# ----------------------------------------------------------------------------------------------------------------- training
TRAIN_CASES = [(5, 512, 512, True), (3, 1024, 1000, False), (2, 640, 193, True), (1, 300, 300, True)]


def _train_run(blk, a, n, real, dtype, autocast, shared, seed=1):
    """Output and every gradient (single, the conditioning, the pair, every parameter) of one forward + backward, in fp64."""
    s, c, z, mask, rope = _inputs(a, n, real, seed, shared)
    g = torch.Generator(device="cuda").manual_seed(seed + 100)
    dy = torch.randn(a, 1, n, 128, device="cuda", generator=g).to(dtype)
    s = s.to(dtype).detach().requires_grad_()
    cb = (c[:1] if shared else c).to(dtype).detach().requires_grad_()
    z = z.to(dtype).detach().requires_grad_()
    blk.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        y = blk(s, cb.expand(a, 1, n, 128), z, mask, rope=rope)
    (y * dy).sum().backward()
    out = {"out": y.detach()[:, :, :real].double(), "dsingle": s.grad[:, :, :real].double(), "dcond": cb.grad[:, :, :real].double(),
           "dpair": z.grad.double()}  # fmt: skip
    for name, p in blk.named_parameters():
        out["d." + name] = p.grad.double()
    return out


@needs_card
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize(("a", "n", "real", "shared"), TRAIN_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "master"])
def test_training_matches_fp64_as_well_as_the_module(backend, a, n, real, shared, dtype):
    """Output and every gradient within 1.5x of the module's own error (bf16 autocast, or fp32 with TF32 allowed) against the fp64 module."""
    blk = _block(backend).cuda()
    own = _block().cuda()
    ref = _block().cuda().double()
    expected = _train_run(ref, a, n, real, torch.float64, False, shared)
    master = dtype == torch.bfloat16
    actual = _train_run(blk, a, n, real, dtype, master, shared)
    base = _train_run(own, a, n, real, dtype, master, shared)
    floor = 1e-3 if master else 1e-4
    for name, want in expected.items():
        e, b = _rel(actual[name], want), _rel(base[name], want)
        # the pair LayerNorm weight gradient sums every window's bias gradient: 2x (A100 master, 193 real atoms: 1.7e-2 vs 1.0e-2)
        slack = 2.0 if name.endswith("ln_pair.weight") else 1.5
        assert e <= slack * b + floor, f"{backend} {name}: {e:.3e} vs module {b:.3e}"
    assert all(torch.isfinite(v).all() for v in actual.values())


@needs_card
def test_training_is_bitwise_repeatable():
    blk = _block("cuda").cuda()
    first = _train_run(blk, 5, 512, 500, torch.bfloat16, True, True)
    again = _train_run(blk, 5, 512, 500, torch.bfloat16, True, True)
    for name, v in first.items():
        assert torch.equal(v, again[name]), name
