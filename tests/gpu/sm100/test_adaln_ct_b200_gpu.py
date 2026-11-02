"""B200: the hand-CUDA AdaptiveLayerNorm and ConditionedTransition behind the standalone modules (FoldForge imports both): the AdaLN
inference as one row pass and one tcgen05 GEMM with the gate in its epilogue (``kernels/adaln/cuda/sm100``), its training as the row
kernels around cuBLAS (``kernels/adaln/cuda/adaln_rows.cu``, built for sm_100a too), the inference tail on two tcgen05 GEMMs
(``kernels/conditioned_transition/cuda/ct_sm100``) and the training tail's row kernels around cuBLAS (``ct_train``).

Pinned as on H100 (``tests/gpu/sm90/test_adaln_ct_cuda_gpu.py``): the route (CUDA on B200, the portable family under
``force_portable_kernels``); that these kernels launch, eager and captured in a CUDA graph, and no Triton kernel does; parity with an fp64
reference within 1.5x the error of the same-dtype PyTorch statements (inference, and every training gradient); a bitwise-identical repeat.
Shapes: the token transformer (d 768, cond 384 -- 768 for Boltz-2 / ESMFold2), the atom transformer (128 / 128), at a row count off the
128-row tiles too."""
from __future__ import annotations

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition.module import ConditionedTransition
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import ImplementationType as IT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 only"),
]

# (rows layout [A, 1, L], d, d_cond)
SHAPES = [((2, 1, 128), 768, 384), ((1, 1, 333), 768, 768), ((4, 1, 96), 128, 128), ((1, 1, 1000), 128, 128)]


@pytest.fixture(autouse=True)
def _settings():
    previous = settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    settings.configure(**{f: getattr(previous, f) for f in ("force_portable_kernels", "portable_on_hopper_blackwell")})


def _kernels(fn):
    from tests.cuda_graph_nodes import launched_kernels

    return set(launched_kernels(fn))


def _is_triton(name):
    return name.startswith("_") and not name.startswith("_Z") and "_kernel" in name


def _has(names, *needles):
    return all(any(n in k for k in names) for n in needles)


def _build(op, d, dc, impl, dtype=torch.float32):
    torch.manual_seed(0)
    m = (AdaptiveLayerNorm(d, dc, implementation=impl, dtype=dtype) if op == "adaln" else
         ConditionedTransition(d, dc, implementation=impl, dtype=dtype)).cuda()
    for p in m.parameters():
        p.data.normal_(0, 0.05) if p.ndim > 1 else p.data.normal_(0, 0.5)
    return m


def _inputs(rows, d, dc, dtype, grad=False):
    torch.manual_seed(1)
    x = torch.randn(*rows, d, device="cuda", dtype=dtype, requires_grad=grad)
    c = torch.randn(*rows, dc, device="cuda", dtype=dtype, requires_grad=grad)
    return x, c


def _call(m, x, c, bf16):
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, c)
    return m(x, c)


def _ref64(op, m, d, dc):
    ref = _build(op, d, dc, IT.PYTORCH, torch.float64)
    ref.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    return ref


def _err(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("op", ["adaln", "ct"])
def test_the_route_is_cuda_on_b200_and_portable_under_the_ab_switch(op):
    assert _build(op, 768, 384, IT.MINIWORLD)._backend is KernelBackend.CUDA
    settings.configure(force_portable_kernels=True)
    assert _build(op, 768, 384, IT.MINIWORLD)._backend is KernelBackend.PORTABLE


@pytest.mark.parametrize("op", ["adaln", "ct"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{'x'.join(map(str, r))}-d{d}-c{c}" for r, d, c in SHAPES])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_inference_launches_the_cuda_kernels_and_matches_the_reference(op, bf16, shape, mode):
    rows, d, dc = shape
    dtype = torch.bfloat16 if bf16 else torch.float32
    m = _build(op, d, dc, IT.MINIWORLD)
    ref, pt = _ref64(op, m, d, dc), _build(op, d, dc, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    x, c = _inputs(rows, d, dc, dtype)
    from tests.cuda_graph_nodes import graph_kernels

    with torch.no_grad():
        run = lambda: _call(m, x, c, bf16)  # noqa: E731
        y = run()
        if mode == "graph":
            run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g):
                out = run()
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, y)
            y = out
            names = set(graph_kernels(g))
        else:
            names = _kernels(run)
        yr = ref(x.double(), c.double())
        yp = _call(pt, x, c, bf16)
        assert _err(y, yr) <= 1.5 * _err(yp, yr) + 1e-6, (_err(y, yr), _err(yp, yr))
        again = run() if mode != "graph" else (g.replay(), out.clone())[1]
        assert torch.equal(again, y)
    # inference (kernels/adaln/cuda/sm100): bf16 at the atom width (cond 128) one tcgen05 kernel (LN(cond) built in shared memory, the gate and
    # LN(x) in its epilogue), elsewhere the row pass + the tcgen05 GEMM with the gate in its epilogue; fp32 the row kernels around one cuBLAS
    # TF32 GEMM (the weights packed once). The tail at the token width (bf16 up to 4096 rows) cuBLAS between the CUDA SwiGLU / gate row kernels,
    # else the two tcgen05 GEMMs. The ConditionedTransition's bf16 call on the cuBLAS tail takes its AdaLN and gate projection as one head
    # (ct_sm100.head_inference: the row-statistics pass, then ct_head_sm100 / ct_head_split_sm100).
    rows_ = 1
    for r_ in rows:
        rows_ *= r_
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_sm100

    ct_head = op == "ct" and bf16 and d > 128 and rows_ <= ct_sm100.BF16_CUBLAS_MAX_ROWS and ct_sm100.HEAD_FUSED
    # The fp32 ConditionedTransition at the token widths: the whole module as one op with the AdaLN and the gate projection one GEMM on the centred
    # conditioning (ct_sm100.ct_inference_full_f32: cond_center, adaln_fold_fwd, sigmul_fwd_ld).
    fold = op == "ct" and not bf16 and d > 128
    if fold:
        assert _has(names, "adaln_fold_fwd", "swiglu_fwd", "sigmul_fwd_ld"), names
    elif not bf16:
        assert _has(names, "ln_cond_fwd", "adaln_fwd_k"), names
    elif ct_head:
        assert _has(names, "head_stats", "ct_head"), names
    elif dc == 128:
        assert _has(names, "adaln_fused_sm100"), names
    else:
        assert _has(names, "ln_cond_xstats", "adaln_gemm_sm100"), names
    if op == "ct":
        cublas_tail = d > 128 and (not bf16 or rows_ <= ct_sm100.BF16_CUBLAS_MAX_ROWS)
        assert _has(names, "swiglu_fwd", "sigmul_fwd") if cublas_tail else _has(names, "tg_gemm_sm100"), names
    assert not any(_is_triton(k) for k in names), names


@pytest.mark.parametrize("op", ["adaln", "ct"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES[:3], ids=[f"{'x'.join(map(str, r))}-d{d}-c{c}" for r, d, c in SHAPES[:3]])
def test_training_gradients_match_the_reference(op, bf16, shape):
    rows, d, dc = shape
    dtype = torch.bfloat16 if bf16 else torch.float32
    m = _build(op, d, dc, IT.MINIWORLD)
    ref, pt = _ref64(op, m, d, dc), _build(op, d, dc, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    x, c = _inputs(rows, d, dc, dtype, grad=True)
    dy = torch.randn(*rows, d, device="cuda", dtype=dtype)

    def run(mod, xx, cc, ac):
        for t in (xx, cc, *mod.parameters()):
            t.grad = None
        y = _call(mod, xx, cc, ac)
        y.backward(dy.to(y.dtype))
        return [y.detach()] + [t.grad.detach().clone() for t in (xx, cc, *mod.parameters())]

    mine = run(m, x, c, bf16)
    x64, c64 = x.detach().double().requires_grad_(), c.detach().double().requires_grad_()
    y64 = ref(x64, c64)
    y64.backward(dy.double())
    r64 = [y64.detach()] + [t.grad for t in (x64, c64, *ref.parameters())]
    theirs = run(pt, x, c, bf16)
    for i, (a, b, p) in enumerate(zip(mine, r64, theirs)):
        assert _err(a, b) <= 1.5 * _err(p, b) + 1e-6, (i, _err(a, b), _err(p, b))
    again = run(m, x, c, bf16)
    assert all(torch.equal(u, v) for u, v in zip(mine, again))
    names = _kernels(lambda: run(m, x, c, bf16))
    assert _has(names, "ln_cond_fwd") or _has(names, "ln_cond_fwd_t"), names
    assert _has(names, "adaln_bwd"), names
    if op == "ct":
        assert _has(names, "sigmul_bwd"), names
    assert not any(_is_triton(k) for k in names), names


# fp32 training at the row counts where the tail takes its TF32 tcgen05 GEMMs (``ct_train.tc_route``: the expand GEMM with the SwiGLU in its
# epilogue, the dh GEMM with the SwiGLU backward in its epilogue, ``ct_tc_sm100.cu``), forced down to these small shapes; and the small-row side
# stream (``ct_train.SIDE_MAX_ROWS``) on and off.
@pytest.mark.parametrize("shape", [((2, 1, 300), 768, 384), ((1, 1, 333), 768, 768), ((3, 1, 700), 128, 128)], ids=["d768-c384", "d768-c768", "d128-c128"])
@pytest.mark.parametrize("side", [True, False], ids=["side", "noside"])
def test_fp32_training_tcgen05_tail_matches_the_reference(shape, side, monkeypatch):
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_train

    monkeypatch.setattr(ct_train, "TC_MIN_ROWS", {"token": 0, "atom": 0})
    monkeypatch.setattr(ct_train, "SIDE_MAX_ROWS", 1 << 30 if side else 0)
    rows, d, dc = shape
    m = _build("ct", d, dc, IT.MINIWORLD)
    ref, pt = _ref64("ct", m, d, dc), _build("ct", d, dc, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    x, c = _inputs(rows, d, dc, torch.float32, grad=True)
    dy = torch.randn(*rows, d, device="cuda")

    def run(mod, xx, cc):
        for t in (xx, cc, *mod.parameters()):
            t.grad = None
        y = mod(xx, cc)
        y.backward(dy)
        return [y.detach()] + [t.grad.detach().clone() for t in (xx, cc, *mod.parameters())]

    mine = run(m, x, c)
    x64, c64 = x.detach().double().requires_grad_(), c.detach().double().requires_grad_()
    y64 = ref(x64, c64)
    y64.backward(dy.double())
    r64 = [y64.detach()] + [t.grad for t in (x64, c64, *ref.parameters())]
    theirs = run(pt, x, c)
    for i, (a, b, p) in enumerate(zip(mine, r64, theirs)):
        assert _err(a, b) <= 1.5 * _err(p, b) + 1e-6, (i, _err(a, b), _err(p, b))
    again = run(m, x, c)
    assert all(torch.equal(u, v) for u, v in zip(mine, again))
    names = _kernels(lambda: run(m, x, c))
    assert _has(names, "ctc_sm100"), names
    assert not any(_is_triton(k) for k in names), names
