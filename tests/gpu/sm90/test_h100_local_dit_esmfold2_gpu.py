"""ESMFold2's atom block (``LocalDiTBlock(**ESMFOLD2)``) on the H100 kernels against the fp64 PyTorch module, inference: the RMS / SiLU row
kernels, the q / k RMSNorm + rotary and the 192-key rank-window attention."""

import pytest
import torch

from miniworld_engine.integrations import local_dit_h100 as h100
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import ESMFOLD2, LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

# (A, N, real atoms): full, ragged, the 192-key window starting before atom 0 (under 192 real atoms), N not a multiple of 128
CASES = [(1, 256, 256), (2, 512, 400), (5, 1024, 1000), (1, 384, 150), (3, 640, 193), (2, 300, 300), (1, 1536, 1500)]


def _block(impl):
    torch.manual_seed(0)
    blk = LocalDiTBlock(128, 128, 16, 4, n=2, implementation=impl, **ESMFOLD2)
    with torch.no_grad():
        for name, p in blk.named_parameters():
            if "norm_query" in name or "norm_key" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1)
    return blk


def _inputs(a, n, real, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    c = torch.randn(1, 1, n, 128, device="cuda", generator=g).expand(a, 1, n, 128)
    z = torch.randn(1, (n + 31) // 32, 32, 192, 16, device="cuda", generator=g)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    angle = 3 * torch.randn(1, n, 16, device="cuda", generator=g)
    return s, c, z, mask, (angle.cos(), angle.sin())


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize(("a", "n", "real"), CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_matches_the_pytorch_module(a, n, real, dtype):
    blk = _block(ImplementationType.MINIWORLD).cuda().eval()
    ref = _block(ImplementationType.PYTORCH).cuda().double().eval()
    ref.load_state_dict(blk.state_dict())
    s, c, z, mask, rope = _inputs(a, n, real)
    with torch.no_grad():
        expected = ref(s.double(), c.double(), z.double(), mask, rope=tuple(t.double() for t in rope))
        ins = (s.to(dtype), c.to(dtype), z.to(dtype))
        if dtype == torch.bfloat16:
            module_bf16 = _block(ImplementationType.PYTORCH).cuda().eval()
            module_bf16.load_state_dict(blk.state_dict())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert h100.matches(blk, *ins, mask, rope)
                actual = blk(*ins, mask, rope=rope)
                own = module_bf16(*ins, mask, rope=rope)
            # the kernels' bf16 error is that of the module's own bf16 run, give or take
            assert _rel(actual[:, :, :real], expected[:, :, :real]) < 2 * _rel(own[:, :, :real], expected[:, :, :real]) + 1e-3
        else:
            assert h100.matches(blk, *ins, mask, rope)
            actual = blk(*ins, mask, rope=rope)
            assert _rel(actual[:, :, :real], expected[:, :, :real]) < 2e-3
    assert torch.isfinite(actual).all()


def test_bitwise_deterministic():
    blk = _block(ImplementationType.MINIWORLD).cuda().eval()
    s, c, z, mask, rope = _inputs(5, 1024, 1000)
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            ins = (s.to(dtype), c.to(dtype), z.to(dtype))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                first = blk(*ins, mask, rope=rope)
                for _ in range(3):
                    assert torch.equal(blk(*ins, mask, rope=rope), first)


def test_a_gradient_or_a_missing_rope_is_not_served():
    blk = _block(ImplementationType.MINIWORLD).cuda().float()
    s, c, z, mask, rope = _inputs(1, 256, 200)
    assert not h100.matches(blk, s, c, z, mask, rope)            # the parameters record a gradient
    with torch.no_grad():
        assert h100.matches(blk, s, c, z, mask, rope)
        assert not h100.matches(blk, s, c, z, mask, None)


@pytest.mark.parametrize("real", [512, 400, 150])
def test_captured_in_a_cuda_graph(real):
    """The real atom count stays on the device: the block runs inside a graph capture and replays with another mask and inputs."""
    blk = _block(ImplementationType.MINIWORLD).cuda().float().requires_grad_(False)
    s, c, z, mask, rope = _inputs(2, 512, 512, seed=3)
    s2, c2, z2, _, rope2 = _inputs(2, 512, 512, seed=4)
    static = [t.clone() for t in (s, c, z, mask, *rope)]
    with torch.no_grad():
        blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))   # build and pack outside the capture
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


# ----------------------------------------------------------------------------------------------------------------- training (bf16)
TRAIN_CASES = [(5, 512, 512), (3, 1024, 1000), (2, 640, 193), (1, 300, 300)]


def _train_run(blk, a, n, real, dtype, autocast, seed=1):
    """Output and every gradient (single, the shared conditioning, the pair, every parameter) of one forward + backward, in fp64."""
    s, c, z, mask, rope = _inputs(a, n, real, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 100)
    dy = torch.randn(a, 1, n, 128, device="cuda", generator=g).to(dtype)
    s = s.to(dtype).detach().requires_grad_()
    cb = c[:1].to(dtype).detach().requires_grad_()
    z = z.to(dtype).detach().requires_grad_()
    blk.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        y = blk(s, cb.expand(a, 1, n, 128), z, mask, rope=rope)
    (y * dy).sum().backward()
    out = {"out": y.detach()[:, :, :real].double(), "dsingle": s.grad[:, :, :real].double(), "dcond": cb.grad[:, :, :real].double(),
           "dpair": z.grad.double()}
    for name, p in blk.named_parameters():
        out["d." + name] = p.grad.double()
    return out, (s, cb, z, mask, rope)


@pytest.mark.parametrize(("a", "n", "real"), TRAIN_CASES)
def test_training_matches_fp64_as_well_as_the_module(a, n, real):
    """The hand-CUDA training step: output and every gradient within 1.5x of the module's own bf16-autocast error against the fp64 module."""
    blk = _block(ImplementationType.MINIWORLD).cuda()
    own = _block(ImplementationType.PYTORCH).cuda()
    ref = _block(ImplementationType.PYTORCH).cuda().double()
    own.load_state_dict(blk.state_dict())
    ref.load_state_dict(blk.state_dict())
    expected, _ = _train_run(ref, a, n, real, torch.float64, False)
    actual, ins = _train_run(blk, a, n, real, torch.bfloat16, True)
    base, _ = _train_run(own, a, n, real, torch.bfloat16, True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert h100.matches(blk, ins[0], ins[1].expand(a, 1, n, 128), ins[2], ins[3], ins[4])
    for name, want in expected.items():
        e, b = _rel(actual[name], want), _rel(base[name], want)
        assert e <= 1.5 * b + 1e-6, f"{name}: engine {e:.3e} vs module {b:.3e}"
    assert all(torch.isfinite(v).all() for v in actual.values())


def test_training_is_bitwise_repeatable():
    blk = _block(ImplementationType.MINIWORLD).cuda()
    first, _ = _train_run(blk, 5, 512, 500, torch.bfloat16, True)
    again, _ = _train_run(blk, 5, 512, 500, torch.bfloat16, True)
    for name, v in first.items():
        assert torch.equal(v, again[name]), name


def test_training_routing():
    from miniworld_engine import settings

    blk = _block(ImplementationType.MINIWORLD).cuda()
    s, c, z, mask, rope = _inputs(4, 256, 256)
    bf = lambda t: t.to(torch.bfloat16).detach().requires_grad_()  # noqa: E731
    sb, cb, zb = bf(s), bf(c[:1]), bf(z)
    shared = cb.expand(4, 1, 256, 128)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert h100.matches(blk, sb, shared, zb, mask, rope)
        # a conditioning of its own per sample, and fp32 activations, keep the composed path under a gradient
        assert not h100.matches(blk, sb, shared.contiguous(), zb, mask, rope)
    assert not h100.matches(blk.float(), s.requires_grad_(), c, z, mask, rope)
