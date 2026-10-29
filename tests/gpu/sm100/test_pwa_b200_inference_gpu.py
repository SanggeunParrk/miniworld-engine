"""MSAPairWeightedAveraging inference on B200 beyond the MiniWorld shape (integrations.pwa_train / pwa_wide, csrc/sm100):

* the widths FoldForge's predictors use: AF3 / Protenix-v1 (d_msa 64, d_pair 128, 8 x 8), Protenix-v2 (128 / 256 / 8 x 8), OpenDDE
  (128 / 384 / 8 x 8), ESMFold2 (128 / 256 / 8 x 16), Chai-1 (64 / 256 / 8 x 32): pair_fwd at d_pair 128 / 256 / 384, ln_vg_w / gate_out_w
  of pwa_wide_sm100.cu;
* token counts 1024 < N <= 2048 (pair_fwd's 2048-logit build), MiniWorld's width and the wide ones.

Each against an fp32 copy of the module: the update within 1.15x of the module's own bf16 PyTorch statements' error (the bar of
tests/gpu/any/test_pwa_train_gpu.py), the call served by the kernels (not the PyTorch fallback), CUDA-graph replay, and the refusals
(training at the wide widths, N > 2048, training N > 1024)."""
import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import pwa_train as pt
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.msa_pair_weighted_averaging import MSAPairWeightedAveraging

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) required"),
]

DEV, BF = "cuda", torch.bfloat16
FOLDFORGE = {  # (d_msa, d_pair, d_hidden): DenseSpec msa_channel / pair_channel / msa_value_dim, 8 heads
    "af3": (64, 128, 8),
    "protenix2": (128, 256, 8),
    "opendde": (128, 384, 8),
    "esmfold2": (128, 256, 16),
    "chai1": (64, 256, 32),
}


@pytest.fixture(autouse=True)
def native_on():
    saved = settings.current().pwa_train
    settings.configure(pwa_train=True)
    yield
    settings.configure(pwa_train=saved)


def rel(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float())).item()


def make(d_msa, d_pair, d_hidden, seed=0):
    torch.manual_seed(seed)
    m = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden, implementation=ImplementationType.MINIWORLD).to(DEV).to(BF)
    with torch.no_grad():   # zero / unit initialisations hide kernel bugs: randomise everything
        for n, p in m.named_parameters():
            p.normal_(0, 0.3) if "ln_" in n and n.endswith("bias") else p.normal_(1.0 if "ln_" in n else 0.0, 0.3 if "ln_" in n else 0.15)
    return m.eval()


def inputs(d_msa, d_pair, n, s, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(1, s, n, d_msa, device=DEV, dtype=BF) * 0.5
    z = torch.randn(1, n, n, d_pair, device=DEV, dtype=BF) * 0.5
    mask = torch.rand(1, n, device=DEV) < 0.85
    return x, z, mask


def check_update(m, x, z, mask):
    """our update vs fp32, against the module's own bf16 statements vs fp32; the call is served by the kernels"""
    ref = copy.deepcopy(m).float()
    ref.implementation = ImplementationType.PYTORCH
    pyt = copy.deepcopy(m)
    pyt.implementation = ImplementationType.PYTORCH
    before = pt.STATS["served"]
    with torch.no_grad():
        ours = m(x, z, mask).float() - x.float()
        served = pt.STATS["served"] - before
        r = ref(x.float(), z.float(), mask).float() - x.float()
        p = pyt(x, z, mask).float() - x.float()
    assert served == 1, pt.STATS["refused"]
    assert ours.shape == p.shape
    assert torch.isfinite(ours).all()
    assert rel(ours, r) <= 1.15 * rel(p, r), (rel(ours, r), rel(p, r))
    return ours


@pytest.mark.parametrize("family", list(FOLDFORGE))
@pytest.mark.parametrize(("n", "s"), [(203, 129), (384, 256)])
def test_foldforge_widths(family, n, s):
    d_msa, d_pair, d_hidden = FOLDFORGE[family]
    m = make(d_msa, d_pair, d_hidden)
    check_update(m, *inputs(d_msa, d_pair, n, s))


@pytest.mark.parametrize(("family", "n"), [("miniworld", 1536), ("miniworld", 2048), ("af3", 1100), ("opendde", 2048), ("esmfold2", 1536)])
def test_long_inference(family, n):
    d_msa, d_pair, d_hidden = (64, 128, 32) if family == "miniworld" else FOLDFORGE[family]
    m = make(d_msa, d_pair, d_hidden)
    check_update(m, *inputs(d_msa, d_pair, n, 128))


def test_no_mask_and_a_fully_masked_row_free_call():
    m = make(*FOLDFORGE["protenix2"])
    x, z, _ = inputs(128, 256, 256, 128)
    check_update(m, x, z, None)


@pytest.mark.parametrize("family", ["af3", "esmfold2"])
def test_cuda_graph_replay(family):
    d_msa, d_pair, d_hidden = FOLDFORGE[family]
    m = make(d_msa, d_pair, d_hidden)
    x, z, mask = inputs(d_msa, d_pair, 300, 200)
    with torch.no_grad():
        eager = m(x, z, mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                m(x, z, mask)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = m(x, z, mask)
        stream.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)
        x.add_(0.25)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, m(x, z, mask))


def test_refusals():
    x, z, _ = inputs(128, 256, 256, 128)
    dims = (128, 256, 8, 16)
    with torch.no_grad():
        assert pt.refusal(x, z, *dims) is None
        assert pt.refusal(x, z, *dims, dropout=True) is not None
        assert pt.refusal(x[:, :, :1], z, 128, 256, 8, 32) is not None              # d_msa 128 x 32: not built
        big = torch.empty(1, 1, 2049, 128, device=DEV, dtype=BF)
        assert "2048" in pt.refusal(big, z, *dims)
        assert pt.refusal(torch.empty(1, 1, 2048, 128, device=DEV, dtype=BF), z, *dims) is None
    assert "inference only" in pt.refusal(x, z, *dims)                              # grad enabled: the wide kernels have no backward on B200
    mw = torch.empty(1, 128, 1100, 64, device=DEV, dtype=BF)
    assert "1024" in pt.refusal(mw, z, 64, 128, 8, 32)                               # MiniWorld width, training: N <= 1024
    with torch.no_grad():
        assert pt.refusal(mw, z, 64, 128, 8, 32) is None
