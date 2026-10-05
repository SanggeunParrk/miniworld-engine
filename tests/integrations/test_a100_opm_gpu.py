"""OuterProductMean on A100 (integrations/opm_sm80.py) against the fp32 PyTorch module: inference and training (every gradient), the registry widths, masks, the pair
residual, both normalisation orders, eager / compiled / CUDA graph, the env switch and the gate. Errors are held to the bf16 PyTorch module's own error in the same regime."""

import copy
import os

import pytest
import torch

from miniworld_engine.modules import OuterProductMean
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


WIDTHS = [(64, 128), (128, 256), (128, 384)]                      # (d_msa, d_pair): the registry's OuterProductMean rows (d_hidden 32)


def _modules(d_msa, d_pair, norm_first=True, seed=0):
    torch.manual_seed(seed)
    ref = OuterProductMean(d_msa, d_pair, 32, normalize_before_proj=norm_first, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():                   # non-default parameters: the zero-initialised to_out, unit LayerNorm
        for n, p in ref.named_parameters():
            if "ln_msa.weight" in n:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif p.ndim == 1:
                p.copy_(0.2 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
    ours = copy.deepcopy(ref)
    ours.implementation = ImplementationType.MINIWORLD
    twin = copy.deepcopy(ref)
    return ref.cuda(), ours.cuda().to(torch.bfloat16), twin.cuda().to(torch.bfloat16)


def _inputs(S, L, d_msa, d_pair, mask_kind="ragged", residual=True, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    msa = torch.randn(1, S, L, d_msa, device="cuda", generator=g)
    mask = None
    if mask_kind != "none":
        mask = torch.rand(1, S, L, device="cuda", generator=g) > 0.3
        if mask_kind == "empty_rows":
            mask[..., ::7] = False
    res = torch.randn(1, L, L, d_pair, device="cuda", generator=g) if residual else None
    dy = torch.randn(1, L, L, d_pair, device="cuda", generator=g)
    return msa, mask, res, dy


def _run(mod, dtype, msa, mask, res, dy, grad):
    mod.zero_grad(set_to_none=True)
    x = msa.to(dtype).clone().requires_grad_(grad)
    r = None if res is None else res.to(dtype).clone().requires_grad_(grad)
    with torch.set_grad_enabled(grad):
        y = mod(x, mask, residual=r)
    got = {"out": y.detach().float()}
    if grad:
        y.backward(dy.to(dtype))
        got["dmsa"] = x.grad.float()
        if r is not None:
            got["dres"] = r.grad.float()
        got.update({n: p.grad.float() for n, p in mod.named_parameters()})
    return y, got


def _check(d_msa, d_pair, S, L, mask_kind, norm_first, residual, grad, strict=1.15):
    from miniworld_engine.integrations import opm_sm80

    ref, ours, twin = _modules(d_msa, d_pair, norm_first)
    msa, mask, res, dy = _inputs(S, L, d_msa, d_pair, mask_kind, residual)
    with torch.set_grad_enabled(grad):
        assert (opm_sm80.serves_train if grad else opm_sm80.serves_inference)(ours, msa.bfloat16(), mask, None, None if res is None else res.bfloat16())
    _, want = _run(ref, torch.float32, msa, mask, res, dy, grad)
    _, base = _run(twin, torch.bfloat16, msa, mask, res, dy, grad)
    y, got = _run(ours, torch.bfloat16, msa, mask, res, dy, grad)
    assert y.dtype == torch.bfloat16
    assert tuple(y.shape) == (1, L, L, d_pair)
    for n in want:
        ea, eb = _rel(got[n], want[n]), _rel(base[n], want[n])
        assert ea <= max(2e-3, (strict if n in ("out", "dmsa", "dres") else 1.25) * eb), (n, ea, eb)     # parameter gradients are sums over every token: a noisier statistic


@pytest.mark.parametrize(("d_msa", "d_pair"), WIDTHS)
@pytest.mark.parametrize("mask_kind", ["none", "ragged", "empty_rows"])
def test_opm_a100_inference(d_msa, d_pair, mask_kind):
    _check(d_msa, d_pair, S=200, L=136, mask_kind=mask_kind, norm_first=True, residual=True, grad=False)


@pytest.mark.parametrize(("d_msa", "d_pair"), WIDTHS)
@pytest.mark.parametrize("mask_kind", ["none", "ragged"])
@pytest.mark.parametrize("residual", [False, True])
def test_opm_a100_training(d_msa, d_pair, mask_kind, residual):
    _check(d_msa, d_pair, S=200, L=136, mask_kind=mask_kind, norm_first=True, residual=residual, grad=True)


@pytest.mark.parametrize(("d_msa", "d_pair"), WIDTHS)
@pytest.mark.parametrize("grad", [False, True])
def test_opm_a100_esmfold2_order(d_msa, d_pair, grad):
    """normalize_before_proj = False divides after the projection, bias included."""
    _check(d_msa, d_pair, S=130, L=48, mask_kind="empty_rows", norm_first=False, residual=True, grad=grad)


@pytest.mark.parametrize("L", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize(("d_msa", "d_pair"), WIDTHS)
def test_opm_a100_registry_lengths_inference(d_msa, d_pair, L):
    """Every registry length at the three registry widths (a shallow MSA keeps the fp32 reference cheap)."""
    _check(d_msa, d_pair, S=64, L=L, mask_kind="ragged", norm_first=True, residual=True, grad=False)


@pytest.mark.parametrize("S", [1, 7, 64, 65, 1024, 1100])
def test_opm_a100_msa_depth(S):
    """Any MSA depth (the operands of the outer product are [S, 32 L]: no padding of the contraction)."""
    _check(64, 128, S=S, L=64, mask_kind="ragged", norm_first=True, residual=True, grad=True)


@pytest.mark.parametrize("L", [1, 5, 33, 70, 100])
@pytest.mark.parametrize(("d_msa", "d_pair"), WIDTHS)
def test_opm_a100_tiny_and_odd_lengths(d_msa, d_pair, L):
    """L that is not a multiple of the 4 x 32 pair tile of any kernel (tails are zero-filled / masked), at every registry width."""
    _check(d_msa, d_pair, S=96, L=L, mask_kind="none", norm_first=True, residual=False, grad=True)


@pytest.mark.parametrize("d_pair", [128, 256, 384])
def test_opm_a100_epilogue_schedules_agree(d_pair):
    """The schedules of the epilogue (CTA shape, ring depth; `MINIWORLD_OPM_SM80_EPI`) accumulate in the same order: bit-identical outputs, at an L that is no multiple of the tile."""
    from miniworld_engine.kernels.outer_product_mean.cuda import sm80

    L = 70
    g = torch.Generator(device="cuda").manual_seed(3)
    o = torch.randn(32 * L, 32 * L, device="cuda", generator=g).bfloat16()
    wo = (torch.randn(d_pair, 1024, device="cuda", generator=g) * 0.03).bfloat16()
    bias = 0.1 * torch.randn(d_pair, device="cuda", generator=g)
    res = torch.randn(L, L, d_pair, device="cuda", generator=g).bfloat16()
    norm = torch.rand(L, L, device="cuda", generator=g) * 900 + 1
    want = sm80.epilogue(o, norm, 1.0, wo, bias, res, True, variant=0)
    for variant in (1, 2):
        assert torch.equal(sm80.epilogue(o, norm, 1.0, wo, bias, res, True, variant=variant), want), variant


def test_opm_a100_gate_conditions():
    """The path declines what it does not implement; the module path then serves."""
    from miniworld_engine.integrations import opm_sm80

    _, ours, _ = _modules(64, 128)
    msa, mask, res, _ = _inputs(64, 64, 64, 128)
    mb, rb = msa.bfloat16(), res.bfloat16()
    with torch.no_grad():
        assert opm_sm80.serves_inference(ours, mb, mask, None, rb)
        assert not opm_sm80.serves_inference(ours, msa, mask, None, rb)                          # fp32 MSA
        assert not opm_sm80.serves_inference(ours, mb.expand(2, -1, -1, -1), mask.expand(2, -1, -1), None, rb.expand(2, -1, -1, -1))   # B = 2
        assert not opm_sm80.serves_inference(ours, mb, mask, None, res)                           # fp32 residual
        assert not opm_sm80.serves_inference(ours, mb, mask.float(), None, rb)                    # a float mask
        ours.mask_interchain = True
        assert not opm_sm80.serves_inference(ours, mb, mask, torch.zeros(1, 64, dtype=torch.long, device="cuda"), rb)   # interchain masking
        ours.mask_interchain = False
        assert not opm_sm80.serves_train(ours, mb, mask, None, rb)                                # grad disabled: the inference gate
    assert not opm_sm80.serves_inference(ours, mb, mask, None, rb)                                # grad enabled
    assert opm_sm80.serves_train(ours, mb, mask, None, rb)
    wide = OuterProductMean(64, 128, 16, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    with torch.no_grad():
        assert not opm_sm80.serves_inference(wide, mb, mask, None, rb)                            # d_hidden 16
    other = OuterProductMean(96, 128, 32, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    with torch.no_grad():
        assert not opm_sm80.serves_inference(other, torch.randn(1, 8, 16, 96, device="cuda", dtype=torch.bfloat16), None, None, None)   # d_msa 96
    pt = copy.deepcopy(ours)
    pt.implementation = ImplementationType.PYTORCH
    with torch.no_grad():
        assert not opm_sm80.serves_inference(pt, mb, mask, None, rb)                              # not the miniworld implementation


def test_opm_a100_env_switch_and_triton_agreement():
    """MINIWORLD_OPM_SM80=0 runs the module's previous (Triton) path: the two agree to bf16 accuracy."""
    from miniworld_engine.integrations import opm_sm80

    _, ours, _ = _modules(64, 128)
    msa, mask, res, _ = _inputs(128, 96, 64, 128)
    mb, rb = msa.bfloat16(), res.bfloat16()
    with torch.no_grad():
        got = ours(mb, mask, residual=rb)
        os.environ["MINIWORLD_OPM_SM80"] = "0"
        try:
            assert not opm_sm80.serves_inference(ours, mb, mask, None, rb)
            base = ours(mb, mask, residual=rb)
        finally:
            del os.environ["MINIWORLD_OPM_SM80"]
    assert _rel(got - rb, base - rb) < 1e-2                         # the update alone


def test_opm_a100_training_is_bit_reproducible():
    """Every reduction is a fixed-order sum of per-CTA partials (no atomics): two runs agree bit for bit, gradients included."""
    _, ours, _ = _modules(64, 128)
    msa, mask, res, dy = _inputs(200, 136, 64, 128)
    _, g1 = _run(ours, torch.bfloat16, msa, mask, res, dy, True)
    _, g2 = _run(ours, torch.bfloat16, msa, mask, res, dy, True)
    for n in g1:
        assert torch.equal(g1[n], g2[n]), n


@pytest.mark.parametrize(("d_msa", "d_pair"), WIDTHS)
def test_opm_a100_compiled_equals_eager(d_msa, d_pair):
    """torch.compile(fullgraph=True) of inference and of a training step reproduces eager."""
    torch._dynamo.reset()
    _, ours, _ = _modules(d_msa, d_pair)
    msa, mask, res, dy = _inputs(96, 64, d_msa, d_pair)
    mb, rb = msa.bfloat16(), res.bfloat16()
    comp = torch.compile(ours, fullgraph=True)
    with torch.no_grad():
        want = ours(mb, mask, residual=rb)
        got = comp(mb, mask, residual=rb)
    assert torch.equal(got, want)
    _, ge = _run(ours, torch.bfloat16, msa, mask, res, dy, True)
    _, gc = _run(comp, torch.bfloat16, msa, mask, res, dy, True)
    for n in ge:
        assert torch.equal(gc.get(n, gc.get("_orig_mod." + n)), ge[n]), n


def test_opm_a100_cuda_graph_replay():
    """A captured call replays bit-identically (the extension is resolved and every buffer allocated by the graph's pool)."""
    _, ours, _ = _modules(64, 128)
    msa, mask, res, _ = _inputs(128, 96, 64, 128)
    mb, rb = msa.bfloat16(), res.bfloat16()
    with torch.no_grad():
        want = ours(mb, mask, residual=rb)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ours(mb, mask, residual=rb)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ours(mb, mask, residual=rb)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, want)


def test_opm_a100_training_cuda_graph_replay():
    """A forward + backward captured in a graph replays to the same gradients."""
    _, ours, _ = _modules(64, 128)
    msa, mask, res, dy = _inputs(128, 96, 64, 128)
    x = msa.bfloat16().clone().requires_grad_()
    r = res.bfloat16().clone().requires_grad_()
    dyb = dy.bfloat16()

    def step():
        for p in ours.parameters():
            p.grad = None
        x.grad = r.grad = None
        ours(x, mask, residual=r).backward(dyb)

    step()
    want = {n: p.grad.clone() for n, p in ours.named_parameters()}
    want["x"] = x.grad.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    for p in ours.parameters():
        p.grad = None
    x.grad = r.grad = None
    with torch.cuda.graph(graph):
        ours(x, mask, residual=r).backward(dyb)
    graph.replay()
    torch.cuda.synchronize()
    for n, p in ours.named_parameters():
        assert torch.equal(p.grad, want[n]), n
    assert torch.equal(x.grad, want["x"])
