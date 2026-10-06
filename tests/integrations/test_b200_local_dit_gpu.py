"""B200 block-local AF3 atom block (``integrations.local_dit``): LocalDiTBlock runs the sm_100a kernels for inference and training --
with or without a [B, N] key mask, at any N --, its output and every gradient no worse against an fp64 PyTorch block than the bf16
module path's, and every call it does not serve keeps the module path."""

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
    return ref.cuda().double(), eng.cuda().to(torch.bfloat16)


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
    orig_apply, orig_infer = local_dit._LocalBlock.apply, local_dit._infer
    monkeypatch.setattr(local_dit._LocalBlock, "apply", lambda *a: calls.append("train") or orig_apply(*a))
    monkeypatch.setattr(local_dit, "_infer", lambda *a: calls.append("infer") or orig_infer(*a))
    return calls


@pytest.mark.parametrize(("A", "N", "masked", "cross"), [(4, 256, False, False), (3, 300, False, False), (3, 384, True, False), (2, 200, True, False),
                                                         (5, 130, False, False), (3, 300, False, True), (2, 384, True, True), (4, 130, True, True)])
def test_training_matches_the_module_path(A, N, masked, cross, monkeypatch):
    ref, eng = _block(cross=cross)
    ins = _inputs(A, N)
    mask = _mask(N, 11) if masked else None
    w = torch.randn(A, 1, N, DS, device="cuda", dtype=torch.float64)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = _spy(monkeypatch)
    fused = _train(eng, ins, torch.bfloat16, w, mask)
    assert calls == ["train"], calls
    monkeypatch.setenv("MINIWORLD_LOCAL_DIT_SM100", "0")
    module = _train(eng, ins, torch.bfloat16, w, mask)
    assert calls == ["train"], "the switch did not keep the module path"
    assert set(fused) == set(truth)
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        assert torch.isfinite(fused[k]).all(), k
        assert fused[k].dtype == module[k].dtype, k
        assert ef < 1.5 * em + 3e-3, f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize(("A", "N", "masked", "cross"), [(5, 128, False, False), (1, 384, False, False), (2, 1024, False, False), (5, 1024, True, False),
                                                         (5, 300, True, False), (1, 1000, True, False), (2, 100, False, False),
                                                         (2, 300, True, True), (5, 1024, False, True), (1, 130, False, True)])
def test_inference_matches_the_module_path(A, N, masked, cross, monkeypatch):
    ref, eng = _block(2, cross)
    s, c, z = _inputs(A, N, 3)
    mask = _mask(N, 13) if masked else None
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        calls = _spy(monkeypatch)
        fused = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
        assert calls == ["infer"], calls
        monkeypatch.setenv("MINIWORLD_LOCAL_DIT_SM100", "0")
        module = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
    assert fused.dtype == torch.bfloat16
    assert fused.shape == s.shape
    ef, em = _rel(fused, truth), _rel(module, truth)
    assert ef < 1.5 * em + 3e-3, f"sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize(("N", "masked"), [(256, False), (300, True)])
def test_inference_is_cuda_graph_capturable(N, masked):
    _, eng = _block(4)
    s, c, z = (t.bfloat16() for t in _inputs(2, N, 5))
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


def test_compiled_training_and_inference_serve_the_kernels(monkeypatch):
    _, eng = _block(7)
    ins = [t.bfloat16() for t in _inputs(3, 300, 8)]
    mask = _mask(300, 17)
    w = torch.randn(3, 1, 300, DS, device="cuda", dtype=torch.float64)
    eager = _train(eng, ins, torch.bfloat16, w, mask)
    compiled = LocalDiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(torch.bfloat16)
    compiled.load_state_dict(eng.state_dict())
    fn = torch.compile(compiled, dynamic=False)
    leaves = [t.detach().clone().requires_grad_() for t in ins]
    out = fn(*leaves, mask)
    (out.double() * w).sum().backward()
    assert _rel(out, eager["out"]) < 1e-2
    for name, leaf in zip(("dsingle", "dcond", "dpair"), leaves, strict=True):
        assert _rel(leaf.grad, eager[name]) < 2e-2, name
    with torch.no_grad():
        assert _rel(fn(*ins, mask), eng(*ins, mask)) < 1e-2


def test_calls_it_does_not_serve_keep_the_module_path(monkeypatch):
    _, eng = _block(6)
    s, c, z = (t.bfloat16() for t in _inputs(2, 256, 7))
    calls = _spy(monkeypatch)
    ok = torch.ones(1, 256, dtype=torch.bool, device="cuda")
    assert local_dit.serves(eng, s, c, z, ok)
    assert not local_dit.serves(eng, s.float(), c.float(), z.float(), ok)                 # fp32
    assert not local_dit.serves(eng, s, c, z, torch.ones(1, 255, dtype=torch.bool, device="cuda"))   # wrong mask shape
    assert not local_dit.serves(eng, s, c, z[:, :-1], ok)                                  # not the trunked pair of this N
    assert not local_dit.serves(eng, s.expand(2, 1, 256, DS)[:, :, :, :64], c, z, ok)      # width
    with torch.no_grad():
        eng(s.float().bfloat16(), c, z, ok)
    assert calls == ["infer"]


# ------------------------------------------------------------------------------------------------------ hoisted per-item tables
def _kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    for _ in range(3):  # the profiler can drop the first records of a window in a long process: open it with a trivial kernel
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.ones(1, device="cuda").add_(1)
            fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        if any("local_attn_fwd" in n for n in names):                # the block's own kernel: the window recorded the call
            return names
    return names


def _bf(*ts):
    return [t.to(torch.bfloat16).contiguous() for t in ts]


# the per-item kernels of a cross-attention inference call: conditioning modulation, the LN of the conditioning, the windowed pair bias
TABLE_KERNELS = ("atom_cond_fwd", "local_cond_ln", "local_bias_fwd")


def _count(names, *needles):
    return sum(any(n in name for n in needles) for name in names)


@pytest.mark.parametrize("cross", [False, True])
def test_hoisted_tables_are_made_once_per_conditioning_and_pair(cross):
    """The second inference call with the same conditioning / pair launches neither the conditioning tables nor the pair bias, and returns
    the first call's result bit for bit."""
    _, eng = _block(cross=cross)
    single, cond, pair = _bf(*_inputs(5, 1024))
    local_dit._HOIST_COND.clear()
    local_dit._HOIST_BIAS.clear()
    with torch.no_grad():
        first = eng(single, cond, pair)
        names = _kernel_names(lambda: eng(single, cond, pair))
        second = eng(single, cond, pair)
    assert torch.equal(first, second)
    assert _count(names, *TABLE_KERNELS) == 0, names
    assert _count(names, "local_attn_fwd") == 1 and _count(names, "atom_pre_fwd", "atom_post_fwd") == 2, "the block's own kernels must still run"


def test_hoist_off_recomputes_every_call(monkeypatch):
    monkeypatch.setenv("MINIWORLD_LOCAL_DIT_HOIST", "0")
    _, eng = _block()
    single, cond, pair = _bf(*_inputs(3, 256))
    with torch.no_grad():
        names = _kernel_names(lambda: eng(single, cond, pair))
    assert _count(names, "atom_cond_fwd") == 1 and _count(names, "local_bias_fwd") == 1, names


@pytest.mark.parametrize("cross", [False, True])
def test_in_place_update_of_conditioning_or_pair_is_seen(cross):
    """An in-place write bumps the tensor's version: the next call remakes the tables and agrees with a call on a fresh copy."""
    _, eng = _block(cross=cross)
    single, cond, pair = _bf(*_inputs(3, 384))
    with torch.no_grad():
        eng(single, cond, pair)
        cond.mul_(1.7).add_(0.3)
        pair.mul_(0.6).sub_(0.1)
        got = eng(single, cond, pair)
        want = eng(single.clone(), cond.clone(), pair.clone())          # new tensors: nothing cached for them
    assert torch.equal(got, want)
    other = _bf(*_inputs(3, 384, seed=7))
    with torch.no_grad():
        assert not torch.equal(eng(single, *other[1:]), got), "different conditioning / pair must change the result"


@pytest.mark.parametrize("static", [False, True])
def test_graph_replays_and_static_inputs(static):
    """Default: a capture packs the tables once and every replay remakes them (so an in-place update between replays is seen).
    ``static_inputs()``: the capture serves from the eager entries and a replay launches none of those kernels."""
    from miniworld_engine.kernels import _capture

    _, eng = _block(cross=True)
    single, cond, pair = _bf(*_inputs(4, 384))
    local_dit._HOIST_COND.clear()
    local_dit._HOIST_BIAS.clear()
    with torch.no_grad():
        eng(single, cond, pair)
        eng(single, cond, pair)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), _capture.static_inputs(static):
            eng(single, cond, pair)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph), _capture.static_inputs(static):
            out = eng(single, cond, pair)
        graph.replay()
        torch.cuda.synchronize()
        from tests.cuda_graph_nodes import graph_kernels

        names = graph_kernels(graph)                     # the replay's launches, from the graph (profiling replays is unreliable)
        n_tables = _count(names, *TABLE_KERNELS)
        if static:
            assert n_tables == 0, names
        else:
            assert n_tables == 3, names
            cond.mul_(1.4)                                   # between replays: the scoped hoist remakes the tables
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, eng(single.clone(), cond.clone(), pair.clone()))


def test_a_new_tensor_at_a_freed_address_is_not_served_the_old_tables(monkeypatch):
    """Activations' addresses are reused as soon as they are freed. A new conditioning / pair at the address of a dead one, with the same shape
    and the same in-place version, must get its own tables -- not the dead tensor's."""
    import gc

    _, eng = _block(cross=True)
    single, cond, pair = _bf(*_inputs(3, 256))
    other = _bf(*_inputs(3, 256, seed=9))
    local_dit._HOIST_COND.clear()
    local_dit._HOIST_BIAS.clear()
    with torch.no_grad():
        eng(single, cond, pair)
        addresses = (cond.data_ptr(), pair.data_ptr())
        del cond, pair
        gc.collect()
        cond2, pair2 = other[1].clone(), other[2].clone()           # clone: version 0, as the dead tensors
        if (cond2.data_ptr(), pair2.data_ptr()) != addresses:
            pytest.skip("the allocator did not hand the freed addresses to the new tensors")
        assert (cond2._version, pair2._version) == (0, 0)
        got = eng(single, cond2, pair2)
        with monkeypatch.context() as m:
            m.setenv("MINIWORLD_LOCAL_DIT_HOIST", "0")
            want = eng(single, cond2, pair2)                         # no hoist: the tables made from these tensors
    assert torch.equal(got, want)


def test_entries_of_freed_tensors_are_dropped():
    import gc

    _, eng = _block(cross=True)
    local_dit._HOIST_COND.clear()
    local_dit._HOIST_BIAS.clear()
    with torch.no_grad():
        for seed in range(8):
            single, cond, pair = _bf(*_inputs(3, 256, seed=seed))
            eng(single, cond, pair)
            del single, cond, pair
            gc.collect()
    assert len(local_dit._HOIST_COND) <= 1 and len(local_dit._HOIST_BIAS) <= 1, (len(local_dit._HOIST_COND), len(local_dit._HOIST_BIAS))
