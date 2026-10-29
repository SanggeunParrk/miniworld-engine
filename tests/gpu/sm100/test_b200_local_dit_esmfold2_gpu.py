"""ESMFold2's atom block (``LocalDiTBlock(**ESMFOLD2)``) on the B200 kernels (``integrations.local_dit``, bf16 inference) against the fp64
PyTorch module: the RMS / SiLU row kernels (``-DRMS=1``), the q / k RMSNorm + rotary and position-indexed pair bias of ``lesm.cu`` and the
160-position rank-window attention of ``lattn_esm.cu``."""

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.integrations import local_dit
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import ESMFOLD2, LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100)",
    ),
]

# (A, N, real atoms): full, ragged, the 192-key window starting before atom 0 (under 192 real atoms), N not a multiple of 128; FoldForge
# runs 5 samples with one conditioning (an expanded view)
CASES = [(1, 256, 256), (2, 512, 400), (5, 1024, 1000), (1, 384, 150), (3, 640, 193), (2, 300, 300), (1, 1536, 1500), (5, 4096, 4000)]


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


def _spy(monkeypatch):
    calls = []
    orig = local_dit._esm_block
    monkeypatch.setattr(local_dit, "_esm_block", lambda *x: calls.append("esm") or orig(*x))
    return calls


@pytest.mark.parametrize(("a", "n", "real"), CASES)
@pytest.mark.parametrize("params", ["fp32-autocast", "bf16"])
def test_matches_the_pytorch_module(a, n, real, params, monkeypatch):
    blk = _block(ImplementationType.MINIWORLD).cuda().eval()
    ref = _block(ImplementationType.PYTORCH).cuda().double().eval()
    ref.load_state_dict(blk.state_dict())
    autocast = params == "fp32-autocast"
    if not autocast:
        blk = blk.bfloat16()
    s, c, z, mask, rope = _inputs(a, n, real)
    ins = (s.bfloat16(), c.bfloat16(), z.bfloat16())
    with torch.no_grad():
        expected = ref(s.double(), c.double(), z.double(), mask, rope=tuple(t.double() for t in rope))
        calls = _spy(monkeypatch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            assert local_dit.serves(blk, *ins, mask, rope)
            actual = blk(*ins, mask, rope=rope)
            assert calls == ["esm"], calls
            override_setting(monkeypatch, "local_dit_sm100", "0")
            own = blk(*ins, mask, rope=rope)  # the module path, same parameters and inputs
    assert actual.dtype == torch.bfloat16 and actual.shape == s.shape
    assert torch.isfinite(actual).all()
    ef, em = _rel(actual[:, :, :real], expected[:, :, :real]), _rel(own[:, :, :real], expected[:, :, :real])
    # the kernels' bf16 error is that of the module's own bf16 run, give or take
    assert ef < 2 * em + 1e-3, f"sm_100a {ef:.2e} vs module path {em:.2e}"


def test_bitwise_deterministic():
    blk = _block(ImplementationType.MINIWORLD).cuda().bfloat16().eval()
    s, c, z, mask, rope = _inputs(5, 1024, 1000)
    ins = (s.bfloat16(), c.bfloat16(), z.bfloat16())
    with torch.no_grad():
        first = blk(*ins, mask, rope=rope)
        for _ in range(3):
            assert torch.equal(blk(*ins, mask, rope=rope), first)


def test_a_gradient_fp32_or_a_missing_rope_is_not_served():
    blk = _block(ImplementationType.MINIWORLD).cuda().bfloat16()
    s, c, z, mask, rope = _inputs(1, 256, 200)
    ins = (s.bfloat16(), c.bfloat16(), z.bfloat16())
    assert not local_dit.serves(blk, *ins, mask, rope)            # the parameters record a gradient
    with torch.no_grad():
        assert local_dit.serves(blk, *ins, mask, rope)
        assert not local_dit.serves(blk, *ins, mask, None)
        assert not local_dit.serves(blk.float(), s, c, z, mask, rope)   # fp32 activations: the module path


@pytest.mark.parametrize("real", [512, 400, 150])
def test_captured_in_a_cuda_graph(real):
    """The real atom count stays on the device: the block runs inside a graph capture and replays with another mask and inputs."""
    blk = _block(ImplementationType.MINIWORLD).cuda().bfloat16().requires_grad_(False)
    s, c, z, mask, rope = _inputs(2, 512, 512, seed=3)
    s2, c2, z2, _, rope2 = _inputs(2, 512, 512, seed=4)
    static = [t.clone() for t in (s.bfloat16(), c.bfloat16(), z.bfloat16(), mask, *rope)]
    with torch.no_grad():
        blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))   # build and pack outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))
        m2 = torch.zeros_like(mask)
        m2[:, :real] = True
        for t, new in zip(static, (s2.bfloat16(), c2.bfloat16(), z2.bfloat16(), m2, *rope2), strict=True):
            t.copy_(new)
        graph.replay()
        eager = blk(s2.bfloat16(), c2.bfloat16(), z2.bfloat16(), m2, rope=rope2)
    torch.testing.assert_close(out[:, :, :real], eager[:, :, :real], rtol=0, atol=0)
