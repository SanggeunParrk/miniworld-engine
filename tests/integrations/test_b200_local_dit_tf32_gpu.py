"""B200 block-local AF3 atom block, fp32 path (``integrations.local_dit`` on the TF32 kernels): fp32 LocalDiTBlock calls run the sm_100a
kind::tf32 kernels for inference and training -- with or without a [B, N] key mask, cross mode or not, at N not a multiple of 32 / 128,
eagerly, under torch.compile and in CUDA graphs --; the output and every gradient are no worse against an fp64 PyTorch block than the fp32
module path in the same regime; the TF32 CUDA kernels are what ran (no Triton, none of the bf16 kernels); no kernel spills.

Tolerance. The fp32 path is the TF32 recipe: every product reads its operands at tf32 precision (10 mantissa bits), the rest is fp32. The
bar is therefore the engine's fp32 module path with allow_tf32 on (cuBLAS / einsum on TF32 tensor cores, its Triton AdaLN / transition
kernels on tf32 dots): ``ef < 1.5 em + 1e-3``, the bf16 test's form with a third of its additive slack (1e-3 ~ two tf32 ulps, 2^-10,
covers gradients whose module-path error is tiny). The module path with TF32 off (IEEE fp32) is measured and printed, and bounds the
fused path from above only loosely (``ef < 30 em_ieee + 1e-3``): a TF32 product is ~2^-11 relative where IEEE is ~2^-24, so no tighter
claim holds by construction."""

import pytest
import torch

from miniworld_engine.integrations import local_dit
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock, to_windows

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]

DS, DC, DP = 128, 128, 16
F32 = torch.float32
TF32_KERNELS = ("local_attn_fwd_tf32", "atom_gemm_tf32", "f32_adaln", "f32_gate")      # every call (inference hoists the tables)
BF16_KERNELS = ("local_attn_fwd", "local_attn_dq", "local_attn_dkv", "atom_cond_fwd_sm100", "atom_pre_fwd_sm100", "atom_post_fwd_sm100",
                "local_bias_fwd")


def _block(seed=0, cross=False):
    torch.manual_seed(seed)
    ref = LocalDiTBlock(cross_attention=cross, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():  # the zero inits (to_out, squeeze, to_bias) would make the block an identity
        for name, p in ref.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = LocalDiTBlock(cross_attention=cross, implementation=ImplementationType.MINIWORLD)
    eng.load_state_dict(ref.state_dict())
    return ref.cuda().double(), eng.cuda().float()


def _inputs(A, N, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dense = torch.randn(1, N, N, DP, device="cuda", generator=g)
    return (torch.randn(A, 1, N, DS, device="cuda", generator=g), torch.randn(A, 1, N, DC, device="cuda", generator=g), to_windows(dense))


def _mask(N, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = torch.rand(1, N, device="cuda", generator=g) > 0.15
    m[:, N - N // 8:] = False
    return m


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


class _tf32:
    """allow_tf32 for the module path's cuBLAS products (the fused path forces it on for its own weight gradients)."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.on

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.old


def _train(m, ins, dtype, w, mask=None):
    leaves = [t.detach().clone().to(dtype).requires_grad_() for t in ins]
    out = m(*leaves, mask)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "dsingle": leaves[0].grad, "dcond": leaves[1].grad, "dpair": leaves[2].grad}
    res.update({n: p.grad.clone() for n, p in m.named_parameters()})
    m.zero_grad(set_to_none=True)
    return res


def _spy(monkeypatch):
    calls = []
    orig_apply, orig_infer = local_dit._LocalBlock32.apply, local_dit._infer32
    monkeypatch.setattr(local_dit._LocalBlock32, "apply", lambda *a: calls.append("train32") or orig_apply(*a))
    monkeypatch.setattr(local_dit, "_infer32", lambda *a: calls.append("infer32") or orig_infer(*a))
    return calls


def _check(fused, truth, module_tf32, module_ieee):
    worst = []
    for k in truth:
        ef, em, ei = _rel(fused[k], truth[k]), _rel(module_tf32[k], truth[k]), _rel(module_ieee[k], truth[k])
        worst.append((ef / max(em, 1e-9), k, ef, em, ei))
        assert torch.isfinite(fused[k]).all(), k
        assert fused[k].dtype == F32, (k, fused[k].dtype)
        assert ef < 1.5 * em + 1e-3, f"{k}: TF32 kernels {ef:.2e} vs module path (TF32) {em:.2e} (IEEE {ei:.2e})"
        assert ef < 30 * ei + 1e-3, f"{k}: TF32 kernels {ef:.2e} vs module path (IEEE fp32) {ei:.2e}"
    print("worst ratios to the TF32 module path:", sorted(worst, reverse=True)[:4])


@pytest.mark.parametrize(("A", "N", "masked", "cross"), [(4, 256, False, False), (3, 300, False, False), (3, 384, True, False),
                                                         (2, 200, True, False), (5, 130, False, False), (3, 300, False, True),
                                                         (2, 384, True, True), (4, 130, True, True), (1, 100, True, False)])
def test_training_matches_the_module_path(A, N, masked, cross, monkeypatch):
    ref, eng = _block(cross=cross)
    ins = _inputs(A, N)
    mask = _mask(N, 11) if masked else None
    w = torch.randn(A, 1, N, DS, device="cuda", dtype=torch.float64)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = _spy(monkeypatch)
    fused = _train(eng, ins, F32, w, mask)
    assert calls == ["train32"], calls
    monkeypatch.setenv("MINIWORLD_LOCAL_DIT_SM100", "0")
    with _tf32(True):
        module_tf32 = _train(eng, ins, F32, w, mask)
    with _tf32(False):
        module_ieee = _train(eng, ins, F32, w, mask)
    assert calls == ["train32"], "the switch did not keep the module path"
    assert set(fused) == set(truth)
    _check(fused, truth, module_tf32, module_ieee)


@pytest.mark.parametrize(("A", "N", "masked", "cross"), [(5, 128, False, False), (1, 384, False, False), (2, 1024, False, False),
                                                         (5, 1024, True, False), (5, 300, True, False), (1, 1000, True, False),
                                                         (2, 100, False, False), (2, 300, True, True), (5, 1024, False, True),
                                                         (1, 130, False, True)])
def test_inference_matches_the_module_path(A, N, masked, cross, monkeypatch):
    ref, eng = _block(2, cross)
    s, c, z = _inputs(A, N, 3)
    mask = _mask(N, 13) if masked else None
    with torch.no_grad():
        truth = {"out": ref(s.double(), c.double(), z.double(), mask)}
        calls = _spy(monkeypatch)
        fused = {"out": eng(s, c, z, mask)}
        assert calls == ["infer32"], calls
        monkeypatch.setenv("MINIWORLD_LOCAL_DIT_SM100", "0")
        with _tf32(True):
            module_tf32 = {"out": eng(s, c, z, mask)}
        with _tf32(False):
            module_ieee = {"out": eng(s, c, z, mask)}
    assert fused["out"].shape == s.shape
    _check(fused, truth, module_tf32, module_ieee)


@pytest.mark.parametrize(("N", "masked", "cross"), [(256, False, False), (300, True, False), (300, True, True)])
def test_inference_is_cuda_graph_capturable(N, masked, cross):
    _, eng = _block(4, cross)
    s, c, z = _inputs(2, N, 5)
    mask = _mask(N, 15) if masked else None
    with torch.no_grad():
        eager = eng(s, c, z, mask)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            eng(s, c, z, mask)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = eng(s, c, z, mask)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


@pytest.mark.parametrize("cross", [False, True])
def test_training_step_is_cuda_graph_capturable(cross, monkeypatch):
    """Forward + backward of the fp32 path captured in one graph; a replay gives the eager gradients (up to the order of the dbias /
    dgamma reductions, which use atomics)."""
    _, eng = _block(9, cross)
    ins = _inputs(3, 300, 6)
    mask = _mask(300, 16)
    w = torch.randn(3, 1, 300, DS, device="cuda")
    leaves = [t.detach().clone().requires_grad_() for t in ins]
    params = list(eng.parameters())

    def step():
        out = eng(*leaves, mask)
        return torch.autograd.grad((out * w).sum(), [*leaves, *params])

    calls = _spy(monkeypatch)
    eager = [g.clone() for g in step()]
    assert calls == ["train32"], calls
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grads = step()
    graph.replay()
    torch.cuda.synchronize()
    for g, e in zip(grads, eager, strict=True):
        assert _rel(g, e) < 1e-5


def test_compiled_training_and_inference_serve_the_kernels(monkeypatch):
    _, eng = _block(7)
    ins = list(_inputs(3, 300, 8))
    mask = _mask(300, 17)
    w = torch.randn(3, 1, 300, DS, device="cuda", dtype=torch.float64)
    eager = _train(eng, ins, F32, w, mask)
    compiled = LocalDiTBlock(implementation=ImplementationType.MINIWORLD).cuda().float()
    compiled.load_state_dict(eng.state_dict())
    fn = torch.compile(compiled, dynamic=False)
    leaves = [t.detach().clone().requires_grad_() for t in ins]
    out = fn(*leaves, mask)
    (out.double() * w).sum().backward()
    assert _rel(out, eager["out"]) < 1e-5
    for name, leaf in zip(("dsingle", "dcond", "dpair"), leaves, strict=True):
        assert _rel(leaf.grad, eager[name]) < 1e-5, name
    for name, p in compiled.named_parameters():
        assert _rel(p.grad, eager[name]) < 1e-5, name
    with torch.no_grad():
        assert _rel(fn(*ins, mask), eng(*ins, mask)) < 1e-5

    def infer():
        with torch.no_grad():
            fn(*ins, mask)

    names = _kernel_names(infer, ("local_attn_fwd_tf32", "atom_gemm_tf32"))
    assert all(_count(names, k) for k in ("local_attn_fwd_tf32", "atom_gemm_tf32")), names


# ------------------------------------------------------------------------------------------------------ the kernels that ran
def _kernel_names(fn, expect=()):
    """The CUDA kernels one call of ``fn`` launches (after a warm-up call). In long processes the profiler can drop the first kernel record
    of a window (an 'Activity Buffer Request' entry instead): a trivial torch kernel goes first, and the window is retried (3 times) while a
    name in ``expect`` is missing."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    pad = torch.zeros(1, device="cuda")
    names = []
    for _ in range(3):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            pad.add_(1)                                              # absorbs a dropped first record
            fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "Activity Buffer" not in e.name]
        if all(_count(names, k) for k in expect):
            break
    return names


def _count(names, *needles):
    return sum(any(n in name for n in needles) for name in names)


@pytest.mark.parametrize("cross", [False, True])
def test_the_tf32_cuda_kernels_run_and_no_triton(cross):
    _, eng = _block(3, cross)
    ins = _inputs(2, 300, 4)
    mask = _mask(300, 14)

    def train():
        with torch.enable_grad():
            leaves = [t.detach().clone().requires_grad_() for t in ins]
            eng(*leaves, mask).square().sum().backward()

    def infer():
        with torch.no_grad():
            eng(*ins, mask)

    for fn, extra in ((train, ("local_attn_dq_tf32", "local_attn_dkv_tf32", "f32_ln_aff", "local_bias_fwd_f32", "f32_adaln_bwd",
                               "f32_gate_bwd", "f32_ln_bwd", "f32_swiglu_bwd", "f32_tail_bwd", "local_bias_bwd_f32")), (infer, ())):
        names = _kernel_names(fn, (*TF32_KERNELS, *extra))
        for k in (*TF32_KERNELS, *extra):
            assert _count(names, k), f"{k} did not run: {sorted(set(names))}"
        assert not any("triton" in n.lower() for n in names), sorted(set(names))
        assert not any(n in BF16_KERNELS for n in names), sorted(set(names))
    eng.zero_grad(set_to_none=True)


def test_the_tf32_kernels_do_not_spill():
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom,
        sm100_atom_local,
    )

    index = torch.cuda.current_device()
    kernels = {**{f"atom.{n}": sm100_atom.atom_kernel32(n, index) for n in sm100_atom.KERNELS_TF32},
               **{f"local.{n}": sm100_atom_local._load32(n, index) for n in sm100_atom_local.KERNELS_TF32}}
    for name, k in kernels.items():
        assert k.lmem == 0, f"{name}: {k.lmem} B of local memory (spills)"
        assert k.regs <= 128, f"{name}: {k.regs} registers"


def test_calls_it_does_not_serve_keep_the_module_path(monkeypatch):
    _, eng = _block(6)
    s, c, z = _inputs(2, 256, 7)
    ok = torch.ones(1, 256, dtype=torch.bool, device="cuda")
    assert local_dit.serves(eng, s, c, z, ok)
    assert not local_dit.serves(eng, s, c.bfloat16(), z, ok)                               # mixed activation dtypes
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert not local_dit.serves(eng, s, c, z, ok)                                       # autocast: the module path's dtype rules
    bf = LocalDiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(torch.bfloat16)
    assert not local_dit.serves(bf, s, c, z, ok)                                            # a bf16 model fed fp32 activations
    monkeypatch.setenv("MINIWORLD_LOCAL_DIT_TF32", "0")
    assert not local_dit.serves(eng, s, c, z, ok)
    assert local_dit.serves(bf, s.bfloat16(), c.bfloat16(), z.bfloat16(), ok)              # the bf16 path is untouched by the switch
    calls = _spy(monkeypatch)
    with torch.no_grad():
        eng(s, c, z, ok)
    assert calls == []


# ------------------------------------------------------------------------------------------------------ hoisted per-item tables
@pytest.mark.parametrize("cross", [False, True])
def test_hoisted_fp32_tables_are_made_once_per_conditioning_and_pair(cross):
    """The second fp32 inference call with the same conditioning / pair launches neither the conditioning LayerNorm and GEMMs nor the pair
    bias, and returns the first call's result bit for bit."""
    _, eng = _block(cross=cross)
    single, cond, pair = _inputs(5, 1024)
    local_dit._HOIST_COND.clear()
    local_dit._HOIST_BIAS.clear()
    with torch.no_grad():
        first = eng(single, cond, pair)
        names = _kernel_names(lambda: eng(single, cond, pair), ("local_attn_fwd_tf32", "atom_gemm_tf32", "f32_adaln"))
        second = eng(single, cond, pair)
    assert torch.equal(first, second)
    assert _count(names, "f32_ln_aff", "local_bias_fwd_f32") == 0, names
    assert _count(names, "atom_gemm_tf32") == (5 if cross else 4), names      # the projections only: q|k|v|g (or q|g, k|v), Wo, Wu, Ws
    assert _count(names, "local_attn_fwd_tf32") == 1


def test_fp32_hoist_off_recomputes_every_call(monkeypatch):
    monkeypatch.setenv("MINIWORLD_LOCAL_DIT_HOIST", "0")
    _, eng = _block()
    single, cond, pair = _inputs(3, 256)
    with torch.no_grad():
        names = _kernel_names(lambda: eng(single, cond, pair), ("f32_ln_aff", "local_bias_fwd_f32", "atom_gemm_tf32", "local_attn_fwd_tf32"))
    assert _count(names, "f32_ln_aff") == 1, names
    assert _count(names, "local_bias_fwd_f32") == 1, names
    assert _count(names, "atom_gemm_tf32") == 7, names                      # + the three conditioning GEMMs
