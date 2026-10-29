"""ESMFold2's atom block (``LocalDiTBlock(**ESMFOLD2)``) on the B200 kernels (``integrations.local_dit``, bf16 inference and training)
against the fp64 PyTorch module: the RMS / SiLU row kernels (``-DRMS=1``), the q / k RMSNorm + rotary and position-indexed pair bias of
``lesm.cu``, the 160-position rank-window attention of ``lattn_esm.cu`` and its backward (``lattn_esm_dq.cu``, ``lattn_esm_dkv.cu``)."""

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


def test_what_is_served():
    blk = _block(ImplementationType.MINIWORLD).cuda().bfloat16()
    s, c, z, mask, rope = _inputs(1, 256, 200)
    ins = (s.bfloat16(), c.bfloat16(), z.bfloat16())
    assert local_dit.serves(blk, *ins, mask, rope)                # training (the parameters record a gradient): _EsmBlock
    with torch.no_grad():
        assert local_dit.serves(blk, *ins, mask, rope)
        assert not local_dit.serves(blk, *ins, mask, None)
        assert not local_dit.serves(blk, s, c, z, mask, rope)     # fp32 activations, bf16 parameters: the module path
        assert local_dit.serves(blk.float(), s, c, z, mask, rope)   # fp32 activations and parameters: the TF32 kernels
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not local_dit.serves(blk.float(), s, c, z, mask, rope)   # fp32 under autocast: the module path
    assert local_dit.serves(blk.float(), s, c, z, mask, rope)       # ... with a gradient too (_EsmBlock32)


# ------------------------------------------------------------------------------------------------ the fp32 path (TF32), inference
@pytest.mark.parametrize(("a", "n", "real"), CASES)
def test_fp32_matches_the_pytorch_module(a, n, real, monkeypatch):
    """fp32 activations and parameters on the TF32 kernels against the fp64 module, no worse than the fp32 module path with TF32 on."""
    blk = _block(ImplementationType.MINIWORLD).cuda().eval()
    ref = _block(ImplementationType.PYTORCH).cuda().double().eval()
    ref.load_state_dict(blk.state_dict())
    s, c, z, mask, rope = _inputs(a, n, real)
    calls = []
    orig = local_dit._esm_fwd32
    monkeypatch.setattr(local_dit, "_esm_fwd32", lambda *x: calls.append("fp32") or orig(*x))
    with torch.no_grad():
        expected = ref(s.double(), c.double(), z.double(), mask, rope=tuple(t.double() for t in rope))
        actual = blk(s, c, z, mask, rope=rope)
        assert calls == ["fp32"], calls
        override_setting(monkeypatch, "local_dit_sm100", "0")
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            own = blk(s, c, z, mask, rope=rope)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    ef, em = _rel(actual[:, :, :real], expected[:, :, :real]), _rel(own[:, :, :real], expected[:, :, :real])
    assert ef < 1.5 * em + 1e-3, f"sm_100a {ef:.2e} vs module path {em:.2e}"


def test_fp32_bitwise_deterministic_and_captured():
    """Repeatable bit for bit, and a CUDA-graph capture replays right with new inputs and another mask."""
    blk = _block(ImplementationType.MINIWORLD).cuda().requires_grad_(False)
    s, c, z, mask, rope = _inputs(2, 512, 512, seed=3)
    s2, c2, z2, _, rope2 = _inputs(2, 512, 512, seed=4)
    static = [t.clone() for t in (s, c, z, mask, *rope)]
    with torch.no_grad():
        first = blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))
        assert torch.equal(blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5])), first)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = blk(static[0], static[1], static[2], static[3], rope=(static[4], static[5]))
        m2 = torch.zeros_like(mask)
        m2[:, :400] = True
        for t, new in zip(static, (s2, c2, z2, m2, *rope2), strict=True):
            t.copy_(new)
        graph.replay()
        eager = blk(s2, c2, z2, m2, rope=rope2)
    torch.testing.assert_close(out[:, :, :400], eager[:, :, :400], rtol=0, atol=0)


# ------------------------------------------------------------------------------------------------ training (bf16)
def _train(m, ins, w, mask, rope, *, autocast=False, shared=False, dtype=torch.bfloat16):
    """Forward + backward of a weighted sum of the output; -> the output and every gradient. ``shared``: the conditioning is one [1, 1, N, d]
    leaf expanded over the samples (FoldForge's call), its gradient the samples' sum."""
    leaves = [t.detach().clone().to(dtype).requires_grad_() for t in ins]
    if shared:
        leaves[1] = ins[1][:1].detach().clone().to(dtype).requires_grad_()
    args = list(leaves)
    if shared:
        args[1] = leaves[1].expand(ins[0].shape[0], *leaves[1].shape[1:])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = m(*args, mask, rope=rope)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "dsingle": leaves[0].grad, "dcond": leaves[1].grad, "dpair": leaves[2].grad}
    res.update({n: p.grad.clone() for n, p in m.named_parameters()})
    m.zero_grad(set_to_none=True)
    return res


TRAIN_CASES = [(2, 512, 400), (3, 300, 300), (1, 384, 150), (2, 640, 193), (5, 1024, 1000), (1, 256, 256), (2, 1024, 1024)]


@pytest.mark.parametrize(("a", "n", "real"), TRAIN_CASES)
@pytest.mark.parametrize("params", ["fp32-autocast", "bf16"])
def test_training_matches_the_pytorch_module(a, n, real, params, monkeypatch):
    """Every gradient (and the output) against the fp64 module, no worse than the module path's own bf16 run, give or take. The loss reads
    the real atoms only."""
    blk = _block(ImplementationType.MINIWORLD).cuda()
    ref = _block(ImplementationType.PYTORCH).cuda().double()
    ref.load_state_dict(blk.state_dict())
    autocast = params == "fp32-autocast"
    if not autocast:
        blk = blk.bfloat16()
    s, c, z, mask, rope = _inputs(a, n, real, seed=5)
    ins = (s, c, z)
    w = torch.randn(a, 1, n, 128, device="cuda", dtype=torch.float64)
    w[:, :, real:] = 0
    shared = a > 1 and n % 2 == 0
    truth = _train(ref, ins, w, mask, tuple(t.double() for t in rope), dtype=torch.float64, shared=shared)
    calls = []
    orig = local_dit._EsmBlock.apply
    monkeypatch.setattr(local_dit._EsmBlock, "apply", lambda *x: calls.append("train") or orig(*x))
    fused = _train(blk, ins, w, mask, rope, autocast=autocast, shared=shared)
    assert calls == ["train"], calls
    override_setting(monkeypatch, "local_dit_sm100", "0")
    module = _train(blk, ins, w, mask, rope, autocast=autocast, shared=shared)
    assert set(fused) == set(truth)
    for k in truth:
        f, t, m = fused[k], truth[k], module[k]
        if k == "out":
            f, t, m = f[:, :, :real], t[:, :, :real], m[:, :, :real]
        assert f.dtype == m.dtype and f.shape == m.shape, k
        assert torch.isfinite(f).all(), k
        ef, em = _rel(f, t), _rel(m, t)
        assert ef < 1.5 * em + 3e-3, f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize(("a", "n", "real"), TRAIN_CASES)
def test_fp32_training_matches_the_pytorch_module(a, n, real, monkeypatch):
    """fp32 activations and parameters (the TF32 kernels): every gradient against the fp64 module, no worse than the fp32 module path with TF32 on."""
    blk = _block(ImplementationType.MINIWORLD).cuda()
    ref = _block(ImplementationType.PYTORCH).cuda().double()
    ref.load_state_dict(blk.state_dict())
    s, c, z, mask, rope = _inputs(a, n, real, seed=5)
    w = torch.randn(a, 1, n, 128, device="cuda", dtype=torch.float64)
    w[:, :, real:] = 0
    shared = a > 1 and n % 2 == 0
    truth = _train(ref, (s, c, z), w, mask, tuple(t.double() for t in rope), dtype=torch.float64, shared=shared)
    calls = []
    orig = local_dit._EsmBlock32.apply
    monkeypatch.setattr(local_dit._EsmBlock32, "apply", lambda *x: calls.append("train32") or orig(*x))
    fused = _train(blk, (s, c, z), w, mask, rope, dtype=torch.float32, shared=shared)
    assert calls == ["train32"], calls
    override_setting(monkeypatch, "local_dit_sm100", "0")
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        module = _train(blk, (s, c, z), w, mask, rope, dtype=torch.float32, shared=shared)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert set(fused) == set(truth)
    for k in truth:
        f, t, m = fused[k], truth[k], module[k]
        if k == "out":
            f, t, m = f[:, :, :real], t[:, :, :real], m[:, :, :real]
        assert f.dtype == torch.float32 and torch.isfinite(f).all(), k
        ef, em = _rel(f, t), _rel(m, t)
        assert ef < 1.5 * em + 1e-3, f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"


def test_fp32_training_is_deterministic():
    blk = _block(ImplementationType.MINIWORLD).cuda()
    s, c, z, mask, rope = _inputs(3, 640, 600, seed=14)
    w = torch.randn(3, 1, 640, 128, device="cuda", dtype=torch.float64)
    first = _train(blk, (s, c, z), w, mask, rope, dtype=torch.float32, shared=True)
    again = _train(blk, (s, c, z), w, mask, rope, dtype=torch.float32, shared=True)
    for k in first:
        assert torch.equal(first[k], again[k]), k


def test_training_is_deterministic():
    """Every gradient but the query bias's (its column sum uses atomics, as in the AF3 block) is bitwise repeatable."""
    blk = _block(ImplementationType.MINIWORLD).cuda().bfloat16()
    s, c, z, mask, rope = _inputs(4, 1024, 1000, seed=11)
    w = torch.randn(4, 1, 1024, 128, device="cuda", dtype=torch.float64)
    first = _train(blk, (s, c, z), w, mask, rope, shared=True)
    for _ in range(2):
        again = _train(blk, (s, c, z), w, mask, rope, shared=True)
        for k in first:
            if k != "attention.to_query.bias":
                assert torch.equal(first[k], again[k]), k


def test_training_step_in_a_cuda_graph():
    """A whole training step (forward and backward) captured in a CUDA graph and replayed with new inputs and another mask matches eager."""
    blk = _block(ImplementationType.MINIWORLD).cuda().bfloat16()
    s, c, z, mask, rope = _inputs(2, 512, 512, seed=12)
    s2, c2, z2, _, rope2 = _inputs(2, 512, 512, seed=13)
    w = torch.randn(2, 1, 512, 128, device="cuda", dtype=torch.bfloat16)
    st = [t.bfloat16().clone().requires_grad_(i != 3) if t.is_floating_point() else t.clone()
          for i, t in enumerate((s, c, z, mask))]
    cs = [t.clone() for t in rope]

    def step():
        out = blk(st[0], st[1], st[2], st[3], rope=(cs[0], cs[1]))
        (out * w).float().sum().backward()
        return out

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    for t in (*st[:3], *blk.parameters()):
        t.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    m2 = torch.zeros_like(mask)
    m2[:, :400] = True
    with torch.no_grad():
        for t, new in zip((*st, *cs), (s2.bfloat16(), c2.bfloat16(), z2.bfloat16(), m2, *rope2), strict=True):
            t.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    got = {"out": out.detach().clone(), "ds": st[0].grad.clone(), "dz": st[2].grad.clone(),
           **{n: p.grad.clone() for n, p in blk.named_parameters()}}
    for t in (*st[:3], *blk.parameters()):
        t.grad = None
    eager = step()
    ref = {"out": eager.detach(), "ds": st[0].grad, "dz": st[2].grad, **{n: p.grad for n, p in blk.named_parameters()}}
    for k in ref:
        if k == "attention.to_query.bias":
            torch.testing.assert_close(got[k], ref[k], rtol=1e-2, atol=1e-2)
        else:
            assert torch.equal(got[k], ref[k]), k


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


def test_fp32_kernels_do_not_spill():
    """The TF32 attention kernels (512 threads: at most 128 registers) keep everything in registers."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local

    index = torch.cuda.current_device()
    for name in ("esm32", "esm_dq32", "esm_dkv32"):
        k = sm100_atom_local._load_esm32(name, index)
        assert k.lmem == 0, f"{name}: {k.lmem} B of local memory (spills)"
        assert k.regs <= 128, f"{name}: {k.regs} registers"
