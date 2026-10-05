"""The A100 path of ``AugmentedAttentionPairBias`` (integrations/augattn_sm80.py): hand CUDA and cuBLAS from the AdaLN's output to the module's result -- the attention core for head dim
32 (the atom width, with the fused pair bias) and 48 (the token widths: d_pair 128 and 256), bf16 and fp32 (TF32 tensor cores), any A, B and L (padded to a multiple of 128 inside), no mask,
a key mask shared by the samples or one per sample -- its output and every input / parameter gradient no worse against an fp64 PyTorch module than the engine's own path (the switch off),
the compiled module equal to eager, a captured CUDA graph replaying to eager, and every call it does not serve keeping the module path."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import augattn_sm80
from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80)"),
]

ATOM = (128, 128, 16, 4)             # d_single, d_cond, d_pair, heads: head dim 32
TOKEN = (768, 384, 128, 16)          # head dim 48
TOKEN2 = (768, 768, 256, 16)         # the ESMFold2 token row: d_cond 768, d_pair 256
BF, F32 = torch.bfloat16, torch.float32


@pytest.fixture(autouse=True)
def policy():
    old = settings.configure(engine_backend="auto")
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        settings.configure(**vars(old))


def _modules(shape, seed=0, dtype=BF):
    torch.manual_seed(seed)
    ref = AugmentedAttentionPairBias(*shape, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():  # the zero inits (to_out, to_bias) would make the module an identity
        for name, p in ref.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = AugmentedAttentionPairBias(*shape, implementation=ImplementationType.MINIWORLD)
    eng.load_state_dict(ref.state_dict())
    return ref.cuda().double(), eng.cuda().to(dtype)


def _inputs(shape, a, n, seed=1, b=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ds, dc, dp, _ = shape
    return (torch.randn(a, b, n, ds, device="cuda", generator=g), torch.randn(a, b, n, dc, device="cuda", generator=g),
            torch.randn(b, n, n, dp, device="cuda", generator=g))


def _mask(n, seed, b=1, a=None):
    """A [B, N] key mask with scattered masked atoms and a masked tail (as a padded structure has); ``a``: one per sample [A, B, N] instead."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = torch.rand((b, n) if a is None else (a, b, n), device="cuda", generator=g) > 0.15
    m[..., n - n // 8:] = False
    m[..., :8] = True
    return m


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _train(m, ins, dtype, w, mask=None):
    leaves = [t.detach().clone().to(dtype).requires_grad_() for t in ins]
    out = m(*leaves, mask)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "dsingle": leaves[0].grad, "dcond": leaves[1].grad, "dpair": leaves[2].grad}
    res.update({n.removeprefix("_orig_mod."): p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


def _spy(monkeypatch):
    """Record every call of the sm_80 entry points (the module calls ``forward`` / ``delta`` of ``augattn_sm80``)."""
    calls = []
    for name in ("forward", "delta"):
        orig = getattr(augattn_sm80, name)
        monkeypatch.setattr(augattn_sm80, name, (lambda o: lambda *a, **k: calls.append(1) or o(*a, **k))(orig))
    return calls


def _tol(dtype):
    """(slack factor, absolute floor) of the comparison with the engine's own path: the bf16 module's rounding, or TF32's."""
    return (1.5, 3e-3) if dtype is BF else (1.5, 2e-4)


# ---------------------------------------------------------------------------------------------------------------------------------- numerics
@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("shape", "a", "b", "n", "mask_kind"), [
    (ATOM, 4, 1, 256, "none"), (ATOM, 3, 1, 384, "shared"), (ATOM, 2, 1, 200, "shared"), (ATOM, 5, 1, 128, "none"), (ATOM, 3, 1, 300, "sample"), (ATOM, 2, 2, 256, "shared"),
    (TOKEN, 2, 1, 256, "none"), (TOKEN, 3, 1, 384, "shared"), (TOKEN, 2, 1, 200, "sample"), (TOKEN, 2, 2, 128, "shared"), (TOKEN2, 2, 1, 256, "none"), (TOKEN2, 2, 1, 200, "shared"),
])
def test_training_matches_the_module_path(shape, a, b, n, mask_kind, dtype, monkeypatch):
    if dtype is F32:
        torch.backends.cuda.matmul.allow_tf32 = True
    ref, eng = _modules(shape, dtype=dtype)
    ins = _inputs(shape, a, n, b=b)
    mask = {"none": None, "shared": _mask(n, 11, b), "sample": _mask(n, 12, b, a)}[mask_kind]
    w = torch.randn(a, b, n, shape[0], device="cuda", dtype=torch.float64)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = _spy(monkeypatch)
    fused = _train(eng, ins, dtype, w, mask)
    assert calls, "the call did not take the sm_80 path"
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
    n_calls = len(calls)
    module = _train(eng, ins, dtype, w, mask)
    assert len(calls) == n_calls, "the switch did not keep the module path"
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    slack, floor = _tol(dtype)
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(fused[k]).all(), k
        assert fused[k].dtype == module[k].dtype, k
        assert ef < slack * em + floor, f"{k}: sm_80 {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("shape", "a", "b", "n", "mask_kind"), [
    (ATOM, 5, 1, 1024, "none"), (ATOM, 5, 1, 300, "shared"), (ATOM, 1, 1, 1000, "shared"), (ATOM, 3, 1, 256, "sample"), (TOKEN, 5, 1, 384, "shared"), (TOKEN, 5, 1, 200, "none"),
    (TOKEN, 3, 2, 256, "sample"), (TOKEN2, 5, 1, 384, "shared"),
])
def test_inference_matches_the_module_path(shape, a, b, n, mask_kind, dtype, monkeypatch):
    if dtype is F32:
        torch.backends.cuda.matmul.allow_tf32 = True
    ref, eng = _modules(shape, 2, dtype)
    s, c, z = _inputs(shape, a, n, 3, b)
    mask = {"none": None, "shared": _mask(n, 13, b), "sample": _mask(n, 14, b, a)}[mask_kind]
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        calls = _spy(monkeypatch)
        fused = eng(s.to(dtype), c.to(dtype), z.to(dtype), mask)
        assert calls, "the call did not take the sm_80 path"
        monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
        module = eng(s.to(dtype), c.to(dtype), z.to(dtype), mask)
    assert fused.dtype == dtype
    assert fused.shape == s.shape
    ef, em = _rel(fused, truth), _rel(module, truth)
    slack, floor = _tol(dtype)
    assert ef < slack * em + floor, f"sm_80 {ef:.2e} vs module path {em:.2e}"


def test_the_delta_entry_point_is_the_update_without_the_residual():
    """``delta`` is ``forward - single`` (the residual is added inside the last pass of ``forward``)."""
    _, eng = _modules(ATOM, 9)
    s, c, z = (t.bfloat16() for t in _inputs(ATOM, 3, 256, 4))
    mask = _mask(256, 5)
    with torch.no_grad():
        full, upd = eng(s, c, z, mask), eng.delta(s, c, z, mask)
    assert _rel(full - s, upd) < 2e-2
    assert upd.shape == s.shape
    assert upd.dtype is BF


def test_a_fully_masked_sample_gets_a_zero_output_with_a_per_sample_mask_and_a_finite_one_shared():
    _, eng = _modules(ATOM, 3)
    s, c, z = (t.bfloat16() for t in _inputs(ATOM, 3, 256, 6))
    per_sample = _mask(256, 7, 1, 3)
    per_sample[1] = False
    shared = _mask(256, 8, 1)
    shared[:] = False
    with torch.no_grad():
        out = eng(s, c, z, per_sample)
        out_shared = eng(s, c, z, shared)
        update = eng.delta(s, c, z, per_sample)
    assert torch.isfinite(out).all()
    assert torch.isfinite(out_shared).all()
    assert not update[1].any(), "a sample with no valid key: the attention contributes nothing"


# ---------------------------------------------------------------------------------------------------------------------------------- determinism, compile, graphs
def test_a_replay_is_bit_identical():
    """The kernels sum the bias gradient from bf16 partials in a fixed order and the rest is cuBLAS GEMMs and elementwise passes: the same inputs give the same bits
    -- except the AdaLN parameters' gradients, which the module's Triton AdaLN backward accumulates with atomics (they agree to fp32 rounding)."""
    _, eng = _modules(ATOM, 4)
    ins = _inputs(ATOM, 4, 384, 5)
    w = torch.randn(4, 1, 384, 128, device="cuda", dtype=torch.float64)
    mask = _mask(384, 17)
    first = _train(eng, ins, BF, w, mask)
    second = _train(eng, ins, BF, w, mask)
    for k in first:
        if k.startswith("ada_ln_in."):
            assert _rel(first[k], second[k]) < 1e-4, k
        else:
            assert torch.equal(first[k], second[k]), k


@pytest.mark.parametrize("dtype", [BF, F32])
def test_under_torch_compile(dtype, monkeypatch):
    """torch.compile(fullgraph=True) keeps the kernels as opaque ops inside the autograd Functions: the compiled module takes the path and matches eager."""
    if dtype is F32:
        torch.backends.cuda.matmul.allow_tf32 = True
    _, eng = _modules(ATOM, 6, dtype)
    ins = _inputs(ATOM, 2, 256, 7)
    w = torch.randn(2, 1, 256, 128, device="cuda", dtype=torch.float64)
    mask = _mask(256, 19)
    eager = _train(eng, ins, dtype, w, mask)
    calls = _spy(monkeypatch)
    compiled = _train(torch.compile(eng, fullgraph=True), ins, dtype, w, mask)
    assert calls, "the compiled module did not take the sm_80 path"
    for k in eager:
        assert _rel(compiled[k], eager[k]) < (2e-2 if dtype is BF else 2e-3), k


@pytest.mark.parametrize("dtype", [BF, F32])
def test_a_captured_cuda_graph_replays_to_eager(dtype):
    """Inference and a training step (forward + backward) are graph-capturable (no host syncs): the replay with the static buffers refilled gives the eager results."""
    if dtype is F32:
        torch.backends.cuda.matmul.allow_tf32 = True
    _, eng = _modules(TOKEN, 8, dtype)
    s, c, z = (t.to(dtype) for t in _inputs(TOKEN, 2, 256, 9))
    mask = _mask(256, 21)
    with torch.no_grad():
        want = eng(s, c, z, mask)
    ss, cs, zs = s.clone(), c.clone(), z.clone()
    with torch.no_grad():
        eng(ss, cs, zs, mask)                                              # warm-up outside the capture (kernels build, allocator)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        out = eng(ss, cs, zs, mask)
    ss.copy_(s)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, want, atol=0, rtol=0)
    # a training step
    dy = torch.randn_like(s)
    xs, xc, xz = (t.clone().requires_grad_() for t in (s, c, z))

    def step():
        for t in (xs, xc, xz):
            t.grad = None
        eng.zero_grad(set_to_none=True)
        y = eng(xs, xc, xz, mask)
        y.backward(dy)
        return y

    step()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    eager_grads = [t.grad.clone() for t in (xs, xc, xz)]
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        step()
    g2.replay()
    torch.cuda.synchronize()
    for got, ref in zip((xs.grad, xc.grad, xz.grad), eager_grads, strict=True):
        assert _rel(got, ref) < (1e-3 if dtype is BF else 1e-5)


def test_the_dit_block_at_atom_widths_takes_it(monkeypatch):
    """The whole atom DiT block (attention + conditioned transition) runs the core inside its attention part and matches the module path."""
    torch.manual_seed(8)
    ref = DiTBlock(128, 128, 16, 4, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = DiTBlock(128, 128, 16, 4, implementation=ImplementationType.MINIWORLD)
    eng.load_state_dict(ref.state_dict())
    ref, eng = ref.cuda().double(), eng.cuda().bfloat16()
    s, c, z = _inputs(ATOM, 3, 384, 9)
    mask = _mask(384, 21)
    calls = _spy(monkeypatch)
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        fused = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
        assert calls
        monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
        module = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
    assert _rel(fused, truth) < 1.5 * _rel(module, truth) + 3e-3


def test_the_pair_bias_cache_is_explicit_and_never_stale_in_eager_mode(monkeypatch):
    """``cache_pair_bias`` keeps the pair bias for no-grad calls with the same tensors: the output is the uncached one's, to the bit, without running the pair-bias
    pass; an in-place change of ``pair``, of a weight or of the mask (anything PyTorch sees) makes the next call recompute; ``clear_pair_bias`` drops it; and a CUDA graph
    that captured a cached call reads the buffer ``cache_pair_bias`` refreshes in place."""
    _, eng = _modules(ATOM, 10)
    s, c, z = _inputs(ATOM, 3, 384, 23)
    sb, cb, zb = s.bfloat16(), c.bfloat16(), z.bfloat16()
    mask = _mask(384, 29)
    runs = []
    orig = augattn_sm80._bias_into
    monkeypatch.setattr(augattn_sm80, "_bias_into", lambda *a, **k: runs.append(1) or orig(*a, **k))

    def call(zz, mm=mask):
        with torch.no_grad():
            return eng(sb, cb, zz, mm)

    base = call(zb)
    assert len(runs) == 1
    assert eng.cache_pair_bias(zb, mask)
    assert torch.equal(call(zb), base), "a cached call changed the result"
    assert len(runs) == 1, "a cached call recomputed the bias"
    zz = zb.clone()
    assert eng.cache_pair_bias(zz, mask)
    zz.add_(0.25)                                                       # PyTorch sees the change: the cache is stale
    fresh = call(zz)
    assert len(runs) == 2
    assert not torch.equal(fresh, base)
    eng.clear_pair_bias()
    assert torch.equal(call(zz), fresh)
    assert len(runs) == 3
    assert eng.cache_pair_bias(zz, mask)
    with torch.no_grad():
        eng.to_bias.weight.mul_(1.5)
    after = call(zz)
    assert len(runs) == 4, "a changed weight was not seen"
    assert not torch.equal(after, fresh)
    assert eng.cache_pair_bias(zz, mask)
    mask2 = mask.clone()
    mask2[:, :40] = False
    other = call(zz, mask2)
    assert len(runs) == 5, "a different mask was not seen"
    assert not torch.equal(other, after)
    # a CUDA graph: the captured call reads the module's buffer; after the static pair changes, one eager refresh makes the replay right
    static = zb.clone()
    eng.clear_pair_bias()
    assert eng.cache_pair_bias(static, mask)
    with torch.no_grad():
        warm = eng(sb, cb, static, mask)                                # warm-up outside the capture (kernels build, allocator)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        out = eng(sb, cb, static, mask)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, warm)
    static.copy_(zz)
    assert eng.cache_pair_bias(static, mask)
    graph.replay()
    torch.cuda.synchronize()
    eng.clear_pair_bias()
    assert torch.equal(out, call(static)), "the replay after a refresh differs from an uncached call"


# ---------------------------------------------------------------------------------------------------------------------------------- the gate
def test_the_gate_serves_what_the_core_was_built_for_and_keeps_the_module_path_for_the_rest(monkeypatch):
    """Served: bf16 or fp32 (all tensors and the weights alike), any A / B / L, a key mask [B, L] or [A, B, L], d_pair 16 / 128 / 256.  Not served: mixed dtypes, a non-bool mask, QK-norm, another head
    dim, a core precision other than the module's, the switches."""
    _, eng = _modules(ATOM)
    s, c, z = _inputs(ATOM, 2, 256)
    sb, cb, zb = s.bfloat16(), c.bfloat16(), z.bfloat16()
    mask = _mask(256, 1)
    assert augattn_sm80.serves(eng, sb, zb, None, cond=cb)
    assert augattn_sm80.serves(eng, sb, zb, mask, cond=cb)
    assert augattn_sm80.serves(eng, sb, zb, mask.expand(2, 1, 256), cond=cb), "the same mask for every sample"
    assert augattn_sm80.serves(eng, sb, zb, _mask(256, 2, 1, 2), cond=cb), "a mask per sample"
    assert augattn_sm80.serves(eng, sb.expand(2, 2, 256, 128), zb.expand(2, 256, 256, 16), None, cond=cb.expand(2, 2, 256, 128)), "B > 1"
    assert not augattn_sm80.serves(eng, s, z, None, cond=c), "fp32 inputs on bf16 weights"
    assert not augattn_sm80.serves(eng, sb, zb, mask.to(torch.uint8), cond=cb), "a non-bool mask"
    assert not augattn_sm80.serves(eng, sb, zb, None, compute_dtype=torch.float32, cond=cb), "a core precision other than the module's"
    assert not augattn_sm80.serves(eng, sb[..., :96], zb, None), "another width"
    assert not augattn_sm80.serves(eng, sb, z, None, cond=cb), "mixed pair dtype"
    qk = AugmentedAttentionPairBias(*ATOM, use_qk_norm=True, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    assert not augattn_sm80.serves(qk, sb, zb, None), "QK-norm"
    odd = AugmentedAttentionPairBias(96, 128, 16, 4, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()      # head dim 24
    assert not augattn_sm80.serves(odd, sb[..., :96], zb, None), "head dim 24"
    _, eng32 = _modules(TOKEN2, dtype=F32)
    s2, c2, z2 = (t for t in _inputs(TOKEN2, 2, 256))
    assert augattn_sm80.serves(eng32, s2, z2, None, cond=c2), "fp32 module and inputs"
    assert augattn_sm80.serves(eng32, s2.bfloat16(), z2.bfloat16(), None, cond=c2.bfloat16()), "bf16 inputs on fp32 master weights (AMP bf16-mixed)"
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80_FUSED", "0")
    assert not augattn_sm80.serves(eng32, s2, z2, None, cond=c2), "the earlier composition is bf16 only"
    assert augattn_sm80.serves(eng, sb, zb, None)
    monkeypatch.delenv("MINIWORLD_AUGATTN_SM80_FUSED")
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
    assert not augattn_sm80.serves(eng, sb, zb, None, cond=cb)


def test_the_pair_bias_cache_serves_fp32_too(monkeypatch):
    """The opt-in cache of the atom width keeps the fp32 bias (natural units) like the bf16 one: a cached no-grad call is the uncached one's, to the bit, without making the bias again."""
    torch.backends.cuda.matmul.allow_tf32 = True
    _, eng = _modules(ATOM, 12, F32)
    s, c, z = _inputs(ATOM, 3, 300, 31)
    mask = _mask(300, 33)
    runs = []
    orig = augattn_sm80._bias_into
    monkeypatch.setattr(augattn_sm80, "_bias_into", lambda *a, **k: runs.append(1) or orig(*a, **k))
    with torch.no_grad():
        base = eng(s, c, z, mask)
        assert len(runs) == 1
        assert eng.cache_pair_bias(z, mask)
        assert torch.equal(eng(s, c, z, mask), base)
        assert len(runs) == 1
        assert not eng.cache_pair_bias(z.double(), mask), "fp64 is not served"
    eng.clear_pair_bias()


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("dp", "heads", "length", "masked"), [(128, 16, 256, False), (128, 16, 200, True), (256, 16, 384, True), (64, 8, 130, True), (512, 12, 128, False), (256, 24, 200, False), (128, 4, 128, True)])
def test_the_generic_pair_bias_kernels_match_autograd(dp, heads, length, masked, dtype):
    """``pair_bias_generic`` (any width, the pair read once) and its backward (``dz`` and the folded weight's gradient in one pass) against fp64 autograd of ``LayerNorm(z) W'^T``: the bias,
    its keys masked / padded as the module path needs them, dz, dW'."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm80

    g = torch.Generator(device="cuda").manual_seed(dp + heads + length)
    z = (torch.randn(length, length, dp, device="cuda", generator=g) * 1.5 + 0.3).to(dtype)
    w = torch.randn(heads, dp, device="cuda", generator=g) * dp ** -0.5
    mask = (torch.rand(length, device="cuda", generator=g) > 0.2) if masked else None
    padded = -(-length // 128) * 128
    zd, wd = z.double().requires_grad_(), w.double().requires_grad_()
    truth = torch.einsum("ijc,hc->hij", torch.nn.functional.layer_norm(zd, (dp,), eps=1e-5), wd)
    db = torch.randn(heads, padded, padded, device="cuda", generator=g)
    keep = torch.ones(length, dtype=torch.bool, device="cuda") if mask is None else mask
    truth.backward(db[:, :length, :length].double() * keep[None, None, :])               # a masked key's gradient is dropped
    assert sm80.pair_bias_generic_supported(dp, heads, dtype) == (heads in sm80.GEN_HEADS)
    oscale = 1.0 if dtype is F32 else 2.0
    out = sm80.pair_bias_generic(z, w.t().contiguous(), mask, padded, oscale=oscale)
    got = out[:, :length, :length].double() / oscale
    tol = 4e-3 if dtype is BF else 1e-5
    assert _rel(got[:, :, keep], truth.detach()[:, :, keep]) < tol
    assert padded == length or (out[:, :length, length:].double() / oscale < -9000).all()                    # padded keys: the fill
    assert not out[:, length:, :length][:, :, keep].any(), "padded query rows are zero on the valid keys"
    if masked:
        assert (out[:, :length, :length][:, :, ~keep].double() / oscale < -9000).all(), "masked keys carry the fill"
    if sm80.pair_bias_generic_bwd_supported(dp, heads, dtype):
        dz, dw = sm80.pair_bias_generic_backward(z, w.t().contiguous(), db, mask, padded)
        assert dz.dtype is dtype
        assert dz.shape == z.shape
        assert _rel(dz, zd.grad) < (6e-3 if dtype is BF else 2e-5)
        assert _rel(dw, wd.grad) < 1e-4
        again = sm80.pair_bias_generic_backward(z, w.t().contiguous(), db, mask, padded)
        assert torch.equal(again[1], dw), "a replay is bit-identical"
        assert torch.equal(again[0], dz), "a replay is bit-identical"
