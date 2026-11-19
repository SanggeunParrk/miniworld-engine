"""B200 (sm_100a, card ``b200_sm100``): the ``"cuda"`` backends of the sampler glue, smooth lDDT, the La-Proteina pair builders, the parallel
residual and ProteinMPNN route on the card (no refusal on their requirement inputs), and the pair builders' tcgen05 table gradient
(``kernels/pair_embed_geom/cuda/sm100/pe_tbl_sm100.cuh``) is bitwise reproducible and CUDA-graph capturable."""

import pytest
import torch

from miniworld_engine import modules as M

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only"),
]

_CASES = [
    (M.AugmentNoise, {}, 1024, 5, torch.float32),
    (M.AlignEuler, {}, 1024, 5, torch.float32),
    (M.PoolAtoms, {}, 1024, 5, torch.bfloat16),
    (M.SmoothLddtLoss, {}, 1024, 5, torch.float32),
    (M.ParallelResidual, {"d": 256, "branches": 3}, 128, 1, torch.bfloat16),
    (M.PairEmbedGeom, {}, 128, 1, torch.float32),
    (M.PairEmbedCA, {"ln_eps": 1e-5}, 128, 1, torch.float32),
    (M.PairEmbedAdaLN, {}, 128, 1, torch.float32),
]


@pytest.mark.parametrize(("cls", "kwargs", "length", "samples", "dtype"), _CASES, ids=lambda v: getattr(v, "__name__", None))
def test_cuda_routes_on_b200(cls, kwargs, length, samples, dtype) -> None:
    m = cls(**kwargs, backend="cuda").cuda()
    g = torch.Generator(device="cuda").manual_seed(0)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=dtype, device="cuda", generator=g)
    assert m.unsupported(**inp) is None


def _weight_grad(m, inp) -> torch.Tensor:
    m.weight.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(**inp)
    out.float().square().sum().backward()
    assert m.weight.grad is not None
    return m.weight.grad.clone()


@pytest.mark.parametrize("cls", [M.PairEmbedGeom, M.PairEmbedCA, M.PairEmbedAdaLN], ids=lambda c: c.__name__)
@pytest.mark.parametrize("samples", [1, 5])
def test_pair_embed_tcgen05_table_gradient(cls, samples: int) -> None:
    m = cls(backend="cuda").cuda().train()
    g = torch.Generator(device="cuda").manual_seed(0)
    inp = m.make_inputs(length=256, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=g)
    a = _weight_grad(m, inp)
    b = _weight_grad(m, inp)
    assert torch.equal(a, b)  # fixed-order partials, no atomics
    ref = cls(backend="triton").cuda().train()
    ref.load_state_dict(m.state_dict())
    r = _weight_grad(ref, inp)
    assert ((a - r).abs().max() / r.abs().max()).item() < 2e-2  # bf16 pair gradient on both sides
    # captured: the same bits as eager
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _weight_grad(m, inp)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    m.weight.grad = None
    with torch.cuda.graph(graph):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(**inp)
        out.float().square().sum().backward()
    graph.replay()
    torch.cuda.synchronize()
    assert m.weight.grad is not None
    assert torch.equal(m.weight.grad, a)


@pytest.mark.parametrize("ln_eps", [None, 1e-5])
@pytest.mark.parametrize(("length", "samples"), [(128, 1), (384, 1), (256, 5), (384, 5)])
def test_pair_embed_ca_inference_b200(ln_eps, length: int, samples: int) -> None:
    """The B200 decoder-pair inference forward (``pe_ca_sm100.cuh``) serves every size, repeats bitwise, captures, and matches the Triton
    kernel to the bf16 rounding of the LayerNorm."""
    m = M.PairEmbedCA(ln_eps=ln_eps, backend="cuda").cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(0)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=g)
    with torch.no_grad():
        assert m.unsupported(**inp) is None
        a = m(**inp)
        b = m(**inp)
        assert torch.equal(a, b)
        ref = M.PairEmbedCA(ln_eps=ln_eps, backend="triton").cuda().eval()
        ref.load_state_dict(m.state_dict())
        r = ref(**inp)
        assert (a.float() - r.float()).abs().max().item() <= (0.0 if ln_eps is None else 3e-2)  # without the LayerNorm: the same bf16 sum
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(**inp)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(graph):
            c = m(**inp)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(c, a)


@pytest.mark.parametrize(("length", "samples"), [(24, 5), (1024, 1), (4100, 5), (8192, 48), (16384, 5), (40000, 1)])
def test_sampler_cluster_kernels_b200(length: int, samples: int) -> None:
    """Round 3 (``sampler_ops_sm100.cu``): AugmentNoise / AlignEuler as one thread-block-cluster kernel match the PyTorch path, repeat bitwise;
    sizes past the cluster contract (40000 atoms) take the H100 chain."""
    for cls, tol in ((M.AugmentNoise, 1e-3), (M.AlignEuler, 1e-2)):
        m = cls(backend="cuda").cuda()
        g = torch.Generator(device="cuda").manual_seed(length + samples)
        inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=g)
        a, b = m(**inp), m(**inp)
        m.set_backend("pytorch")
        r = m(**inp)
        for x, y, z in zip(*(t if isinstance(t, (tuple, list)) else (t,) for t in (a, b, r)), strict=True):
            assert torch.equal(x, y)
            assert (x.double() - z.double()).abs().max().item() < tol


@pytest.mark.parametrize(("length", "samples", "dup"), [(1024, 1, 0), (8192, 5, 0), (512, 2, 5), (512, 2, 40)])
def test_pool_atoms_sort_free_backward_b200(length: int, samples: int, dup: int) -> None:
    """Round 3: the pooling backward without the sort (row buckets, slots added in slot order) gives the bits of the sorted H100 kernel -- also
    for rows with more slots than a bucket (``dup``: every index folded onto ``dup`` rows) -- and the forward the bits of the Triton kernel."""
    from miniworld_engine.kernels.sampler_ops.cuda import ext_sm100

    ext = ext_sm100()
    m = M.PoolAtoms(backend="cuda").cuda()
    g = torch.Generator(device="cuda").manual_seed(length * 7 + samples)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    x, idx, gm, am = inp["x"], inp["idx"], inp["gather_mask"], inp["atom_mask"]
    if dup:
        idx = idx % dup
    out, count = ext.pool_atoms_train(x, idx, gm, am, True)
    m.set_backend("triton")
    assert torch.equal(out, m(x=x, idx=idx, gather_mask=gm, atom_mask=am))
    gout = torch.randn(out.shape, device="cuda", generator=g).to(torch.bfloat16)
    gx = ext.pool_atoms_bwd_b200(gout, x, idx, gm, am, count)
    key, w = ext.pool_atoms_prep(idx, gm, am, x.shape[1])
    skey, perm = torch.sort(key, stable=True)
    assert torch.equal(gx, ext.pool_atoms_bwd(gout, x, skey, perm, w, idx.shape[1]))
    assert torch.equal(gx, ext.pool_atoms_bwd_b200(gout, x, idx, gm, am, torch.zeros_like(count)))


@pytest.mark.parametrize("cls", [M.PairEmbedGeom, M.PairEmbedCA, M.PairEmbedAdaLN], ids=lambda c: c.__name__)
def test_pair_embed_cached_inference_table_b200(cls) -> None:
    """Round 3: the inference table cached per weight version follows an in-place weight update, eager and re-captured."""
    m = cls(backend="cuda").cuda().eval()
    ref = cls(backend="triton").cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(0)
    inp = m.make_inputs(length=128, samples=2, depth=None, dtype=torch.float32, device="cuda", generator=g)
    with torch.no_grad():
        a = m(**inp).clone()
        m.weight.add_(0.1 * torch.randn(m.weight.shape, device="cuda", generator=g))
        b = m(**inp).clone()
        assert not torch.equal(a, b)
        ref.load_state_dict(m.state_dict())
        assert (b.float() - ref(**inp).float()).abs().max().item() <= 3e-2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            c = m(**inp)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(c, b)


def _pair_step(m, inp, up):
    """(pair, gradients of the parameters and of the floating leaves that require one) of one training step with the upstream gradient ``up``."""
    leaves = [*m.parameters(), *(v for v in inp.values() if isinstance(v, torch.Tensor) and v.requires_grad)]
    for t in leaves:
        t.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(**inp)
    torch.autograd.backward(out, up)
    return out.detach().clone(), [t.grad.clone() for t in leaves]


def _earlier_path(monkeypatch, on: bool) -> None:
    """``on``: the pair builders' route before the round-3 B200 kernels (the H100 / RTX 5090 sources built for sm_100a), the reference."""
    from miniworld_engine.modules import pair_embed as pe

    if on:
        monkeypatch.setattr(pe, "_b200", lambda device: False)
    else:
        monkeypatch.undo()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize(("length", "samples"), [(128, 1), (384, 1), (100, 3), (33, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("sc", [True, False], ids=["sc", "nosc"])
def test_pair_embed_adaln_b200_kernels(monkeypatch, length: int, samples: int, dtype, sc: bool) -> None:
    """Round 3 (``pe_adaln_sm100.cuh``): the B200 forward / row backward (int64 indices and fp32 / bf16 gamma / beta read as they are, the
    weight gradient in the weight's layout) against the H100 source's build: the pair to bf16 rounding, every gradient to rounding, in the
    same dtypes; a step repeats bitwise."""
    m = M.PairEmbedAdaLN(backend="cuda").cuda().train()
    g = torch.Generator(device="cuda").manual_seed(length + samples)
    with torch.no_grad():
        m.ln_weight.add_(0.1 * torch.randn(256, device="cuda", generator=g))
        m.ln_bias.add_(0.1 * torch.randn(256, device="cuda", generator=g))
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=dtype, device="cuda", generator=g)
    if not sc:
        inp["sc"] = None
    inp["gamma"].requires_grad_(True)
    inp["beta"].requires_grad_(True)
    up = torch.randn((samples, length, length, 256), device="cuda", generator=g).to(torch.bfloat16)
    assert m.unsupported(**inp) is None
    _earlier_path(monkeypatch, True)
    o0, g0 = _pair_step(m, inp, up)
    _earlier_path(monkeypatch, False)
    o1, g1 = _pair_step(m, inp, up)
    o2, g2 = _pair_step(m, inp, up)
    assert torch.equal(o1, o2)
    assert all(torch.equal(a, b) for a, b in zip(g1, g2, strict=True))
    diff = (o0.float() - o1.float()).abs()
    assert diff.max().item() <= o0.float().abs().max().item() / 64  # a bf16 ulp or two
    assert (diff > 0).float().mean().item() < 0.02
    for a, b in zip(g1, g0, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert _rel(a, b) < (2e-2 if a.dtype == torch.bfloat16 else 1e-2)


@pytest.mark.parametrize("cls", [M.PairEmbedGeom, M.PairEmbedCA], ids=lambda c: c.__name__)
@pytest.mark.parametrize("ln_eps", [None, 1e-5])
@pytest.mark.parametrize(("length", "samples"), [(128, 1), (384, 1), (100, 3), (33, 2)])
def test_pair_embed_geom_ca_b200_kernels(monkeypatch, cls, ln_eps, length: int, samples: int) -> None:
    """Round 3 (``pe_geom_sm100.cuh`` / ``pe_ln_sm100.cuh`` / the decoder forward with codes): against the earlier route, without the LayerNorm
    the pair and the weight gradient bitwise (the same rows in the same order, the same tcgen05 partials), with it to bf16 rounding; a step
    repeats bitwise; chains other than 0 exercise the chain row."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm100

    monkeypatch.setattr(sm100, "SMALL_PAIRS", 0)  # the reference's table gradient on tcgen05 at every size
    m = cls(ln_eps=ln_eps, backend="cuda").cuda().train()
    g = torch.Generator(device="cuda").manual_seed(length + samples)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=g)
    if "chains" in inp:
        inp["chains"] = torch.randint(0, 3, inp["chains"].shape, device="cuda", generator=g)
    up = torch.randn((samples, length, length, 256), device="cuda", generator=g).to(torch.bfloat16)
    assert m.unsupported(**inp) is None
    from miniworld_engine.modules import pair_embed as pe

    monkeypatch.setattr(pe, "_b200", lambda device: False)
    o0, (w0,) = _pair_step(m, inp, up)
    with torch.no_grad():
        e0 = m(**inp).clone()
    monkeypatch.undo()
    o1, (w1,) = _pair_step(m, inp, up)
    o2, (w2,) = _pair_step(m, inp, up)
    with torch.no_grad():
        e1 = m(**inp).clone()
    assert torch.equal(o1, o2) and torch.equal(w1, w2) and torch.equal(e1, o1)
    assert w0.dtype == w1.dtype and w0.shape == w1.shape
    if ln_eps is None:
        assert torch.equal(o0, o1) and torch.equal(e0, e1) and torch.equal(w0, w1)
    else:
        for a, b in ((o0, o1), (e0, e1)):
            diff = (a.float() - b.float()).abs()
            assert diff.max().item() <= a.float().abs().max().item() / 64
            assert (diff > 0).float().mean().item() < 0.02
        assert _rel(w1, w0) < 2e-2


@pytest.mark.parametrize("cls", [M.PairEmbedGeom, M.PairEmbedCA], ids=lambda c: c.__name__)
@pytest.mark.parametrize("ln_eps", [None, 1e-5])
@pytest.mark.parametrize(("length", "samples"), [(128, 1), (256, 1), (100, 3)])
def test_pair_embed_fused_reduce_graph_b200(monkeypatch, cls, ln_eps, length: int, samples: int) -> None:
    """Round 3, second part: the table gradient with its reduce inside the launch (cooperative grid barrier, ``RED_V`` 2) and the training
    table made by pg120_table behind the forward (PDL): the weight gradient bitwise equal to the separate reduce launch on a separate table
    kernel, and a captured training step replays to the eager bits (twice: the barrier words are reused)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm100

    m = cls(ln_eps=ln_eps, backend="cuda").cuda().train()
    g = torch.Generator(device="cuda").manual_seed(7 * length + samples)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=g)
    up = torch.randn((samples, length, length, 256), device="cuda", generator=g).to(torch.bfloat16)
    monkeypatch.setattr(sm100, "RED_V", 0)
    monkeypatch.setattr(sm100, "TBL_IN_LAUNCH", False)
    o0, (w0,) = _pair_step(m, inp, up)
    monkeypatch.undo()
    assert sm100.RED_V == 2 and sm100.TBL_IN_LAUNCH
    o1, (w1,) = _pair_step(m, inp, up)
    assert torch.equal(o0, o1) and torch.equal(w0, w1)

    def step():
        m.weight.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(**inp)
        torch.autograd.backward(out, up)
        return out

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
        gr = torch.cuda.CUDAGraph()
        m.weight.grad = None
        with torch.cuda.graph(gr):
            og = step()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        gr.replay()
        torch.cuda.synchronize()
        wg = m.weight.grad
        assert wg is not None and torch.equal(og, o1) and torch.equal(wg, w1)


@pytest.mark.parametrize(("length", "samples"), [(128, 1), (384, 1), (100, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("sc", [True, False], ids=["sc", "nosc"])
def test_pair_embed_adaln_stats_inference_b200(monkeypatch, length: int, samples: int, dtype, sc: bool) -> None:
    """Round 3, second part: PairEmbedAdaLN inference from the LayerNorm statistics tables (``fwd_s``) against the forward that reduces per
    pair: within a bf16 step or two; repeats bitwise; the tables follow a weight / LayerNorm-affine update."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm100

    m = M.PairEmbedAdaLN(backend="cuda").cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(length + 11 * samples)
    with torch.no_grad():
        m.ln_weight.add_(0.1 * torch.randn(256, device="cuda", generator=g))
        m.ln_bias.add_(0.1 * torch.randn(256, device="cuda", generator=g))
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=dtype, device="cuda", generator=g)
    if not sc:
        inp["sc"] = None

    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(**inp).clone()

    for update in (False, True):
        if update:
            with torch.no_grad():
                m.weight.mul_(1.25)
                m.ln_bias.add_(0.05)
        monkeypatch.setattr(sm100, "STATS", False)
        ref = run()
        monkeypatch.undo()
        a, b = run(), run()
        assert torch.equal(a, b)
        diff = (a.float() - ref.float()).abs()
        assert diff.max().item() <= ref.float().abs().max().item() / 64
        assert (diff > 0).float().mean().item() < 0.05
