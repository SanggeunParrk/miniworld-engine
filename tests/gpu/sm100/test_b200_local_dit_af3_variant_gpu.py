"""AlphaFold 3's atom block options on the B200 kernels (``integrations.local_dit``): the K/V AdaLN of the raw atoms
(``kv_norm="parallel"``: ``pre_fwd`` / ``pre_bwd`` ``-DPARALLEL=1``, the fp32 kernels likewise) in inference and training, and the slid key
window (``key_window="slide"``: edge CTAs of the attention kernels, ``sm100_atom_local/slide.cuh``) -- against the fp64 PyTorch module,
no worse than the module path in the same dtype, on the real atoms (which come first, as the slid window requires)."""

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.integrations import local_dit
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100)",
    ),
]

BF, F32 = torch.bfloat16, torch.float32
OPTIONS = [("parallel", "pad"), ("chained", "slide"), ("parallel", "slide")]
# (A, N, real atoms): FoldForge runs the atoms of the query subsets as one flat run of windows, real atoms first, at 5 samples
# (inference) / 48 (training); the small cases reach the edge rules (under 128 real atoms, windows listed twice)
CASES = [(5, 1024, 1024), (5, 1024, 1000), (2, 512, 301), (1, 384, 140), (3, 256, 256), (1, 300, 300), (1, 256, 100), (2, 128, 77)]


def _blocks(kv_norm, key_window, seed=0):
    torch.manual_seed(seed)
    kw = dict(cross_attention=True, kv_norm=kv_norm, key_window=key_window)
    ref = LocalDiTBlock(implementation=ImplementationType.PYTORCH, **kw)
    with torch.no_grad():  # the zero inits (to_out, squeeze, to_bias) would make the block an identity
        for name, p in ref.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = LocalDiTBlock(implementation=ImplementationType.MINIWORLD, **kw)
    eng.load_state_dict(ref.state_dict())
    return ref.cuda().double(), eng.cuda()


def _inputs(a, n, real, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    c = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    z = torch.randn(1, (n + 31) // 32, 32, 128, 16, device="cuda", generator=g)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    return s, c, z, mask


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _spy(monkeypatch):
    calls = []
    for name in ("_infer", "_infer32"):
        orig = getattr(local_dit, name)
        monkeypatch.setattr(local_dit, name, lambda *a, _o=orig, _n=name: calls.append(_n) or _o(*a))
    return calls


@pytest.mark.parametrize(("kv_norm", "key_window"), OPTIONS)
@pytest.mark.parametrize(("a", "n", "real"), CASES)
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_inference_matches_the_module_path(kv_norm, key_window, a, n, real, dtype, monkeypatch):
    ref, eng = _blocks(kv_norm, key_window)
    eng = eng.to(dtype)
    s, c, z, mask = _inputs(a, n, real)
    ins = (s.to(dtype), c.to(dtype), z.to(dtype))
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        assert local_dit.serves(eng, *ins, mask)
        calls = _spy(monkeypatch)
        fused = eng(*ins, mask)
        assert calls == ["_infer32" if dtype == F32 else "_infer"], calls
        override_setting(monkeypatch, "local_dit_sm100", "0")
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True  # the fp32 bar: the module path on TF32 tensor cores
        try:
            module = eng(*ins, mask)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
    assert fused.dtype == dtype and fused.shape == s.shape
    ef, em = _rel(fused[:, :, :real], truth[:, :, :real]), _rel(module[:, :, :real], truth[:, :, :real])
    assert ef < 1.5 * em + (3e-3 if dtype == BF else 1e-3), f"sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_fp32_parameters_with_bf16_activations(dtype, monkeypatch):
    """FoldForge keeps fp32 master parameters next to bf16 activations: the bf16 kernels serve them (inference)."""
    ref, eng = _blocks("parallel", "slide", seed=4)
    s, c, z, mask = _inputs(5, 1024, 990, seed=5)
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        calls = _spy(monkeypatch)
        fused = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
        assert calls == ["_infer"], calls
    assert _rel(fused[:, :, :990], truth[:, :, :990]) < 1.5e-2


@pytest.mark.parametrize("real", [1024, 1000, 700])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_the_slid_window_is_captured_in_a_cuda_graph(real, dtype):
    """The real atom count stays on the device: the block runs inside a graph capture and replays right with another mask."""
    _, eng = _blocks("parallel", "slide", seed=3)
    eng = eng.to(dtype).requires_grad_(False)
    s, c, z, mask = _inputs(2, 1024, real, seed=6)
    s, c, z = s.to(dtype), c.to(dtype), z.to(dtype)
    with torch.no_grad():
        expected = eng(s, c, z, mask)
        for _ in range(2):  # warm the kernels and the caches outside the capture
            eng(s, c, z, mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = eng(s, c, z, mask)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        mask.zero_()
        mask[:, :613] = True  # another real atom count through the same captured graph
        graph.replay()
        torch.cuda.synchronize()
        again = eng(s, c, z, mask)
        torch.cuda.synchronize()
    assert torch.equal(out, again)


def _train(m, ins, dtype, w, mask):
    leaves = [t.detach().clone().to(dtype).requires_grad_() for t in ins]
    out = m(*leaves, mask)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "dsingle": leaves[0].grad, "dcond": leaves[1].grad, "dpair": leaves[2].grad}
    res.update({n: p.grad.clone() for n, p in m.named_parameters()})
    m.zero_grad(set_to_none=True)
    return res


TRAIN_CASES = [(3, 300, 300), (2, 384, 330), (4, 130, 120), (2, 1024, 1024), (2, 512, 400), (1, 256, 100), (3, 640, 193)]


@pytest.mark.parametrize(("kv_norm", "key_window"), OPTIONS)
@pytest.mark.parametrize(("a", "n", "real"), TRAIN_CASES)
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_training_matches_the_module_path(kv_norm, key_window, a, n, real, dtype, monkeypatch):
    ref, eng = _blocks(kv_norm, key_window, seed=7)
    eng = eng.to(dtype)
    s, c, z, mask = _inputs(a, n, real, seed=8)
    ins = (s, c, z)
    w = torch.randn(a, 1, n, 128, device="cuda", dtype=torch.float64)
    w[:, :, real:] = 0  # the loss reads the real atoms (a padding query's output differs between the window rules)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = []
    for cls in (local_dit._LocalBlock, local_dit._LocalBlock32):
        orig = cls.apply
        monkeypatch.setattr(cls, "apply", lambda *x, _o=orig, _c=cls.__name__: calls.append(_c) or _o(*x))
    fused = _train(eng, ins, dtype, w, mask)
    assert calls == ["_LocalBlock32" if dtype == F32 else "_LocalBlock"], calls
    override_setting(monkeypatch, "local_dit_sm100", "0")
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        module = _train(eng, ins, dtype, w, mask)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert set(fused) == set(truth)
    for k in truth:
        f, t, m = fused[k], truth[k], module[k]
        if k == "out":
            f, t, m = f[:, :, :real], t[:, :, :real], m[:, :, :real]
        ef, em = _rel(f, t), _rel(m, t)
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + (3e-3 if dtype == BF else 1e-3), f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_slid_window_training_is_deterministic(dtype):
    """The slid windows' gradients are summed in a fixed order: what flows from dQ / dK / dV / dbias is bitwise repeatable (the block's
    LayerNorm-weight and bias-column gradients use atomics already without the slide, so they are not compared)."""
    _, eng = _blocks("parallel", "slide", seed=9)
    eng = eng.to(dtype)
    s, c, z, mask = _inputs(4, 512, 450, seed=10)
    w = torch.randn(4, 1, 512, 128, device="cuda", dtype=torch.float64)
    first = _train(eng, (s, c, z), dtype, w, mask)
    for _ in range(2):
        again = _train(eng, (s, c, z), dtype, w, mask)
        for k in ("out", "dsingle", "dpair", "attention.to_query.weight", "attention.to_key.weight", "attention.to_value.weight",
                  "attention.to_bias.weight", "attention.ln_pair.weight"):
            assert torch.equal(first[k], again[k]), k


# ------------------------------------------------------------------------------------------------ a conditioning shared by the samples
def _train_shared(m, s, c0, z, w, mask, dtype):
    """Forward + backward with the conditioning one [1, 1, N, 128] leaf expanded over the samples (the diffusion trainers' call)."""
    ls, lc, lz = (t.detach().clone().to(dtype).requires_grad_() for t in (s, c0, z))
    out = m(ls, lc.expand(s.shape[0], -1, -1, -1), lz, mask)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "dsingle": ls.grad, "dcond": lc.grad, "dpair": lz.grad}
    res.update({n: p.grad.clone() for n, p in m.named_parameters()})
    m.zero_grad(set_to_none=True)
    return res


@pytest.mark.parametrize("cross", [True, False], ids=["cross", "self"])
@pytest.mark.parametrize(("kv_norm", "key_window"), [("chained", "pad"), ("parallel", "slide")])
@pytest.mark.parametrize(("a", "n", "real"), [(4, 512, 450), (3, 300, 300), (16, 256, 256)])
def test_shared_conditioning_training(cross, kv_norm, key_window, a, n, real, monkeypatch):
    """The conditioning tables made once ([N, ..], read at row r % N) and their gradient summed over the samples: every gradient against the
    fp64 module, no worse than the module path in bf16."""
    if not cross and kv_norm == "parallel":
        pytest.skip("the parallel K/V norm needs the cross mode")
    torch.manual_seed(7)
    kw = dict(cross_attention=cross, kv_norm=kv_norm, key_window=key_window)
    ref = LocalDiTBlock(implementation=ImplementationType.PYTORCH, **kw)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = LocalDiTBlock(implementation=ImplementationType.MINIWORLD, **kw)
    eng.load_state_dict(ref.state_dict())
    ref, eng = ref.cuda().double(), eng.cuda().to(BF)
    s, c, z, mask = _inputs(a, n, real, seed=8)
    c0 = c[:1]
    w = torch.randn(a, 1, n, 128, device="cuda", dtype=torch.float64)
    w[:, :, real:] = 0
    truth = _train_shared(ref, s, c0, z, w, mask, torch.float64)
    calls = []
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    orig = local.TableSums.finish
    monkeypatch.setattr(local.TableSums, "finish", lambda *x, **k: calls.append("sum") or orig(*x, **k))
    fused = _train_shared(eng, s, c0, z, w, mask, BF)
    assert calls == ["sum"], calls                       # the shared path ran (the tables' gradient summed over the samples in the row kernels)
    override_setting(monkeypatch, "local_dit_sm100", "0")
    module = _train_shared(eng, s, c0, z, w, mask, BF)
    assert set(fused) == set(truth)
    for k in truth:
        f, t, m = fused[k], truth[k], module[k]
        if k == "out":
            f, t, m = f[:, :, :real], t[:, :, :real], m[:, :, :real]
        assert torch.isfinite(f).all(), k
        ef, em = _rel(f, t), _rel(m, t)
        assert ef < 1.5 * em + 3e-3, f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize("cross", [True, False], ids=["cross", "self"])
@pytest.mark.parametrize(("kv_norm", "key_window"), [("chained", "pad"), ("parallel", "slide")])
@pytest.mark.parametrize(("a", "n", "real"), [(4, 512, 450), (3, 300, 300), (16, 256, 256)])
def test_shared_conditioning_training_fp32(cross, kv_norm, key_window, a, n, real, monkeypatch):
    """The fp32 path (TF32) with a shared conditioning: the tables made once ([N, ..] read at row r % N by the fp32 row kernels and the
    gated-residual GEMMs), their gradient summed over the samples in the AdaLN / tail backward kernels, the conditioning's backward on N
    rows; every gradient against the fp64 module, no worse than the module path with TF32 on."""
    if not cross and kv_norm == "parallel":
        pytest.skip("the parallel K/V norm needs the cross mode")
    torch.manual_seed(7)
    kw = dict(cross_attention=cross, kv_norm=kv_norm, key_window=key_window)
    ref = LocalDiTBlock(implementation=ImplementationType.PYTORCH, **kw)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = LocalDiTBlock(implementation=ImplementationType.MINIWORLD, **kw)
    eng.load_state_dict(ref.state_dict())
    ref, eng = ref.cuda().double(), eng.cuda().float()
    s, c, z, mask = _inputs(a, n, real, seed=8)
    c0 = c[:1]
    w = torch.randn(a, 1, n, 128, device="cuda", dtype=torch.float64)
    w[:, :, real:] = 0
    truth = _train_shared(ref, s, c0, z, w, mask, torch.float64)
    from miniworld_engine.integrations import local_dit

    seen = []
    orig = local_dit._LocalBlock32.apply
    monkeypatch.setattr(local_dit._LocalBlock32, "apply", lambda *x: seen.append(x[4].shape[0]) or orig(*x))
    fused = _train_shared(eng, s, c0, z, w, mask, F32)
    assert seen == [1], seen                             # the fp32 kernels ran on the one [1, 1, N] conditioning
    override_setting(monkeypatch, "local_dit_sm100", "0")
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        module = _train_shared(eng, s, c0, z, w, mask, F32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert set(fused) == set(truth)
    for k in truth:
        f, t, m = fused[k], truth[k], module[k]
        if k == "out":
            f, t, m = f[:, :, :real], t[:, :, :real], m[:, :, :real]
        assert torch.isfinite(f).all(), k
        ef, em = _rel(f, t), _rel(m, t)
        assert ef < 1.5 * em + 1e-3, f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"


def test_shared_conditioning_matches_the_materialised_one():
    """The same block on the expanded conditioning (tables made once) and on its materialised copy (tables per sample row): the outputs are
    equal bit for bit (the same table values)."""
    _, eng = _blocks("parallel", "slide", seed=3)
    eng = eng.to(BF)
    s, c, z, mask = _inputs(4, 512, 480, seed=9)
    s, c0, z = s.to(BF), c[:1].to(BF), z.to(BF)
    with torch.no_grad():
        a = eng(s, c0.expand(4, -1, -1, -1), z, mask)
        b = eng(s, c0.expand(4, -1, -1, -1).contiguous(), z, mask)
    assert torch.equal(a, b)
