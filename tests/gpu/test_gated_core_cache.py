"""B200 (sm_100a): ``GatedInferenceCore.runs`` (bound launches keyed by buffer addresses) is bounded and does not own the caller's tensors.

A diffusion sampler allocates fresh qkvg / bias tensors at every step. The cache used to keep each run's tensors alive (TMA-map ``keep`` and the
run's closure), so it grew by the step's tensors (~1.1 GiB per step at 1029 tokens) until the card was full. Now a run holds only descriptors and
its own scratch, the keys carry the layout, and the cache is an LRU: fresh tensors leave memory flat; the same static tensors still hit; a CUDA
graph that captured a run keeps working (bit for bit) after the run was evicted.
"""

import gc

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only"),
]

H, DH = 16, 48
D = H * DH
STEPS = 60


def _sm100():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    return sm100


def _inputs(fp32: bool, S: int, L: int, nb: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = torch.float32 if fp32 else torch.bfloat16
    qkvg = (torch.randn(S * L, (3 if fp32 else 4) * D, device="cuda", generator=g) * 0.5).to(dt)
    bias = (torch.randn(nb * H, L, L, device="cuda", generator=g) * 0.5).to(dt)
    vt = (torch.randn(D, S * L, device="cuda", generator=g) * 0.5).to(dt) if fp32 else None
    return qkvg, bias, vt


def _core(fp32: bool):
    return _sm100().GatedInferenceCore(torch.cuda.current_device(), torch.float32 if fp32 else torch.bfloat16, H, DH)


#: (fp32, samples, length): bf16 batch, TF32 batch, TF32 one sample (L 640: the key-split core, which owns scratch)
CASES = [(False, 5, 512), (True, 5, 512), (True, 1, 640), (False, 1, 640)]


@pytest.mark.parametrize(("fp32", "S", "L"), CASES)
def test_fresh_tensors_every_step_keep_memory_flat(fp32: bool, S: int, L: int) -> None:
    core = _core(fp32)
    used = []
    for step in range(STEPS):
        qkvg, bias, vt = _inputs(fp32, S, L, 2, 1000 + step)
        core(qkvg, bias, step % 2, S, vt=vt)
        del qkvg, bias, vt
        if step in (9, STEPS - 1):
            torch.cuda.synchronize()
            gc.collect()
            used.append(torch.cuda.memory_allocated())
    assert len(core.runs) <= core.runs.cap
    assert used[1] - used[0] <= (1 << 20), f"allocated memory grew {(used[1] - used[0]) / 2**20:.1f} MiB over {STEPS - 10} steps"


@pytest.mark.parametrize(("fp32", "S", "L"), CASES)
def test_cache_is_bounded_when_every_step_has_new_addresses(fp32: bool, S: int, L: int, monkeypatch) -> None:
    """Live tensors with distinct addresses (an allocator that never recycles): the LRU holds at most ``cap`` runs, and the runs own none of the
    caller's buffers (dropping the tensors frees them even though their runs are cached)."""
    sm = _sm100()
    monkeypatch.setattr(sm.BoundedRuns, "cap", 4)
    core = _core(fp32)
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    live = []
    for step in range(STEPS):
        qkvg, bias, vt = _inputs(fp32, S, L, 1, step)
        core(qkvg, bias, 0, S, vt=vt)
        live.append((qkvg, bias, vt))
        assert len(core.runs) <= 4
    torch.cuda.synchronize()
    live.clear()
    del qkvg, bias, vt  # the last step's locals
    gc.collect()
    scratch = sum(getattr(r, "nbytes", 0) for r in core.runs.values())
    assert torch.cuda.memory_allocated() - base <= scratch + (1 << 20)


@pytest.mark.parametrize(("fp32", "S", "L"), CASES)
def test_static_tensors_reuse_the_run_and_stay_bitwise(fp32: bool, S: int, L: int) -> None:
    core = _core(fp32)
    qkvg0, bias, vt = _inputs(fp32, S, L, 2, 7)
    outs = []
    qkvg = qkvg0.clone()
    for _ in range(3):
        qkvg.copy_(qkvg0)
        core(qkvg, bias, 1, S, vt=vt)
        outs.append(qkvg[:, :D].clone())
    assert len(core.runs) == 1, "the same static tensors must hit the cached run"
    (run,) = core.runs.values()
    qkvg.copy_(qkvg0)
    core(qkvg, bias, 1, S, vt=vt)
    assert next(iter(core.runs.values())) is run
    assert all(torch.equal(outs[0], o) for o in outs[1:])
    # a different block or a different layout is a different run
    qkvg.copy_(qkvg0)
    core(qkvg, bias, 0, S, vt=vt)
    assert len(core.runs) == 2
    # the same bits as a fresh core without any cache history
    ref = qkvg0.clone()
    _core(fp32)(ref, bias, 1, S, vt=vt)
    assert torch.equal(ref[:, :D], outs[0])


@pytest.mark.parametrize(("fp32", "S", "L"), CASES)
def test_captured_graph_survives_eviction_and_matches_eager(fp32: bool, S: int, L: int, monkeypatch) -> None:
    sm = _sm100()
    monkeypatch.setattr(sm.BoundedRuns, "cap", 2)
    core = _core(fp32)
    qkvg0, bias, vt = _inputs(fp32, S, L, 2, 11)
    qkvg = qkvg0.clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        core(qkvg, bias, 1, S, vt=vt)  # warm-up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        core(qkvg, bias, 1, S, vt=vt)
    qkvg.copy_(qkvg0)
    graph.replay()
    torch.cuda.synchronize()
    first = qkvg[:, :D].clone()
    # thrash the LRU (the captured run is evicted), recycle the freed memory, replay again
    for step in range(8):
        a, b, v = _inputs(fp32, S, L, 2, 50 + step)
        core(a, b, step % 2, S, vt=v)
        junk = torch.empty(1 << 26, device="cuda")  # noqa: F841
    assert len(core.runs) <= 2
    qkvg.copy_(qkvg0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(qkvg[:, :D], first)
    ref = qkvg0.clone()
    _core(fp32)(ref, bias, 1, S, vt=vt)
    assert torch.equal(ref[:, :D], first), "captured replay differs from the eager core"


def test_gemm_swiglu_runs_do_not_pin_fresh_tensors() -> None:
    """The token DiT inference's expand GEMM (``GemmSwiglu.runs``) had the same unbounded, tensor-owning cache."""
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

    K, Hh, M = 768, 1536, 640
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(2 * Hh, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    gemm = gemm_swiglu.GemmSwiglu(torch.cuda.current_device(), K=K, H=Hh)
    x0 = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
    first = torch.empty(M, Hh, device="cuda", dtype=torch.bfloat16)
    gemm(x0, w, first)
    used = []
    for step in range(STEPS):
        x = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
        out = torch.empty(M, Hh, device="cuda", dtype=torch.bfloat16)
        gemm(x, w, out)
        del x, out
        if step in (9, STEPS - 1):
            torch.cuda.synchronize()
            gc.collect()
            used.append(torch.cuda.memory_allocated())
    assert len(gemm.runs) <= gemm.runs.cap
    assert used[1] - used[0] <= (1 << 20)
    again = torch.empty_like(first)
    gemm(x0, w, again)
    assert torch.equal(first, again)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "tf32"])
def test_token_dit_sampler_steps_keep_memory_flat(dtype) -> None:
    """The module level: a token DiT inference block called for 50 'diffusion steps' with fresh inputs (B200 CUDA path, both precisions) keeps
    allocated memory flat, and its output at the last step for the first step's inputs is the first step's output, bit for bit."""
    from miniworld_engine.measure import config as C
    from miniworld_engine.modules.token_dit import TokenDiTProtenix

    m0 = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(m0, torch.Generator(device="cuda").manual_seed(0))
    m = C.cast(m0, "master").eval().set_backend("cuda")  # type: ignore[attr-defined]

    def call(seed: int):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        inp = m.make_inputs(length=256, samples=2, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
        with torch.no_grad():
            if dtype is torch.bfloat16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return m(**inp).float()
            inp = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inp.items()}
            return m(**inp).float()

    first = call(0)
    used = []
    for step in range(1, 51):
        call(step)
        if step in (10, 50):
            torch.cuda.synchronize()
            gc.collect()
            used.append(torch.cuda.memory_allocated())
    assert used[1] - used[0] <= (1 << 20), f"allocated memory grew {(used[1] - used[0]) / 2**20:.1f} MiB over 40 steps"
    assert torch.equal(call(0), first)
