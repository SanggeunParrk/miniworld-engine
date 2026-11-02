"""The fused H100 token DiT with a 256-wide pair (Protenix-v2's diffusion pair), bf16 and fp32, inference and training: the bias hoist
(inference) and the training block's pair kernels take the 256-wide pair, the rest of the step is the d_pair = 128 block's. Before,
any pair but 128 ran the composed path (cuBLAS projections + PyTorch statements around the attention core).

Every call must (a) take the fused route (a silent fallback to the composed path is a failure) and (b) hold the repo's yardstick:
inference against an fp64 PyTorch run, error no worse than 1.5x the PyTorch module's in the same precision; training as
``test_token_dit_train_gpu`` (every gradient no worse than the engine's module path against the fp32 PyTorch block)."""
import pytest
import torch
from portable_helpers import randomize_half_std as _randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize(("A", "L", "masked", "qk"), [(5, 128, False, False), (5, 256, True, False), (3, 384, True, True)])
def test_pair256_takes_the_fused_route_and_holds_the_fp64_yardstick(A, L, masked, qk):
    from miniworld_engine.kernels.conditioned_transition import cuda as ct_cuda

    torch.manual_seed(5)
    kw = {"d_pair": 256, "use_qk_norm": qk}
    eng = _randomize(DiTBlock(**kw, implementation=ImplementationType.MINIWORLD).cuda()).eval()
    pt = DiTBlock(**kw, implementation=ImplementationType.PYTORCH).cuda().eval()
    pt.load_state_dict(eng.state_dict())
    ref = DiTBlock(**kw, implementation=ImplementationType.PYTORCH).cuda().double().eval()
    ref.load_state_dict({k: v.double() for k, v in eng.state_dict().items()})

    bf = torch.bfloat16
    single = torch.randn(A, 1, L, 768, device="cuda").to(bf)
    cond = torch.randn(1, 1, L, 384, device="cuda").to(bf).expand(A, -1, -1, -1)
    pair = torch.randn(1, L, L, 256, device="cuda").to(bf)
    mask = (torch.rand(1, L, device="cuda") > 0.15) if masked else None

    taken = []
    orig = ct_cuda.token_inference
    ct_cuda.token_inference = lambda s, *a, **k: (taken.append(tuple(s.shape)), orig(s, *a, **k))[1]
    try:
        with torch.no_grad():
            got = eng(single, cond, pair, mask)
    finally:
        ct_cuda.token_inference = orig
    assert taken, "the 256-wide pair did not reach the fused core"
    with torch.no_grad():
        r = ref(single.double(), cond.double(), pair.double(), mask)
        with torch.autocast("cuda", dtype=bf):
            p = pt(single, cond, pair, mask)
    assert torch.isfinite(got).all()
    assert _rel(got, r) <= 1.5 * _rel(p, r), (_rel(got, r), _rel(p, r))


@pytest.mark.parametrize(("heads", "A", "L", "masked"), [(16, 5, 128, False), (16, 5, 256, True), (16, 3, 384, True), (24, 3, 256, True)])
def test_fp32_pair256_takes_the_fused_route_and_holds_the_tf32_yardstick(heads, A, L, masked):
    """fp32 (TF32) inference at pair 256: the native fp32 family takes the call (16 heads: the one-pass LN + projection + key
    permutation of ``augmented_attention/cuda/pair_rows.cu``; other heads: the LayerNorm rows in key order + cuBLAS) and the error
    against an fp64 run is no worse than 1.5x the fp32 PyTorch module's in the same TF32 regime (or the 1.5e-3 the 128-wide layouts
    are held to)."""
    from miniworld_engine.integrations import token_dit as TD
    from miniworld_engine.kernels.conditioned_transition import cuda as ct_cuda

    torch.manual_seed(6)
    kw = {"d_pair": 256, "n_head": heads}
    eng = _randomize(DiTBlock(**kw, implementation=ImplementationType.MINIWORLD).cuda()).eval()
    pt = DiTBlock(**kw, implementation=ImplementationType.PYTORCH).cuda().eval()
    pt.load_state_dict(eng.state_dict())
    ref = DiTBlock(**kw, implementation=ImplementationType.PYTORCH).cuda().double().eval()
    ref.load_state_dict({k: v.double() for k, v in eng.state_dict().items()})

    single = torch.randn(A, 1, L, 768, device="cuda")
    cond = torch.randn(1, 1, L, 384, device="cuda").expand(A, -1, -1, -1)
    pair = torch.randn(1, L, L, 256, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.15) if masked else None

    with torch.no_grad():
        assert TD._native_fp32_inference(eng, single), "the native fp32 family must take the 256-wide pair"
    taken = []
    orig = ct_cuda.token_inference
    ct_cuda.token_inference = lambda s, *a, **k: (taken.append(tuple(s.shape)), orig(s, *a, **k))[1]
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            got = eng(single, cond, pair, mask)
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.no_grad():
            p = pt(single, cond, pair, mask)
    finally:
        ct_cuda.token_inference = orig
        torch.backends.cuda.matmul.allow_tf32 = old
    assert taken, "the 256-wide fp32 pair did not reach the fused core"
    with torch.no_grad():
        r = ref(single.double(), cond.double(), pair.double(), mask)
    assert torch.isfinite(got).all()
    assert _rel(got, r) <= max(1.5 * _rel(p, r), 1.5e-3), (_rel(got, r), _rel(p, r))


def _train_run(m, single, cond, pair, mask, w, tf32=None):
    """Output and every input / parameter gradient of one forward + backward (the narrowest parameter dtype is the activations')."""
    old = torch.backends.cuda.matmul.allow_tf32
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        dtype = min((p.dtype for p in m.parameters()), key=lambda d: d.itemsize)
        ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
        out = m(*ins, mask)
        (out.float() * w).sum().backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    res.update({n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("qk", [False, True])
@pytest.mark.parametrize("masked", [False, True])
# (3, 200): the block pads L to a multiple of 128
@pytest.mark.parametrize(("A", "L"), [(4, 256), (3, 200), (3, 384)])
def test_pair256_training_takes_the_fused_block_and_matches_the_module_path(dtype, qk, masked, A, L, monkeypatch):
    """Training at pair 256 (bf16, and fp32 on TF32): the fused H100 block takes the call -- bf16 with the one-pass pair bias and its
    one-pass backward, as at pair 128 -- and the output and every input / parameter gradient are no worse than the engine's own
    module path in the same regime, against the fp32 PyTorch block (IEEE): the yardstick of ``test_token_dit_train_gpu``."""
    from settings_helpers import override_setting

    from miniworld_engine.integrations import token_dit_train

    torch.manual_seed(7)
    ref = _randomize(DiTBlock(d_pair=256, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(d_pair=256, use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    eng = eng.to(dtype)
    g = torch.Generator(device="cuda").manual_seed(8)
    single = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 256, device="cuda", generator=g)
    mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15) if masked else None
    w = torch.randn(A, 1, L, 768, device="cuda", generator=g)

    truth = _train_run(ref, single, cond, pair, mask, w, tf32=False)
    calls, fused_pair = [], []
    orig_apply, orig_fused = token_dit_train._Block.apply, token_dit_train._pair_fused
    monkeypatch.setattr(token_dit_train._Block, "apply", lambda *a: calls.append(1) or orig_apply(*a))
    monkeypatch.setattr(token_dit_train, "_pair_fused", lambda *a: fused_pair.append(orig_fused(*a)) or fused_pair[-1])
    tf32 = True if dtype is torch.float32 else None
    fused = _train_run(eng, single, cond, pair, mask, w, tf32=tf32)
    assert calls, f"the {dtype} pair-256 training call did not take the fused block"
    if dtype is torch.bfloat16:
        assert any(fused_pair), "the bf16 block must build its pair-256 bias with the one-pass pair kernel"
    override_setting(monkeypatch, "token_dit_train", "0")
    module = _train_run(eng, single, cond, pair, mask, w, tf32=tf32)
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    worst = []
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        worst.append((ef / max(em, 1e-6), k, ef, em))
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"
    print("worst ratios:", sorted(worst, reverse=True)[:4])
