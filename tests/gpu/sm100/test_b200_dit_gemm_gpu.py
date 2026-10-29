"""The DiT variants' GEMM-epilogue kernels on B200 (``kernels/dit_variants/cuda/dit_gemm.py``, ``dit_gemm_sm100.cu``):

* ``glu_gemm``: silu(a) b [u] of [a | b | u] = x W^T (SwiGLU and Boltz-2's up-gated GLU) against the fp32 math of the same bf16 operands,
  within 1.5 x the error of the unfused path (bf16 GEMM output + the GLU row pass);
* ``res_gemm``: x += g1 (a1 W1^T) [+ g2 (a2 W2^T)] and the LayerNorm + AdaLN outputs / the bf16 stack output, against fp32 math of the
  same operands: x to 1e-5, the bf16 outputs within 1.5 x the unfused path's error (cuBLAS + ``res_rows`` / ``ln_mod_rows``); raw-logit
  and pre-sigmoided tables, one and two outputs, M not a multiple of 128 and tables whose rows wrap around L (the cp.async path);
* replay of a captured chain of both kernels (programmatic dependent launch) many times, bitwise equal to the eager call.
"""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.dit_variants import cuda as rows
from miniworld_engine.kernels.dit_variants.cuda import dit_gemm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100) required",
    ),
]

BF = torch.bfloat16
ERR_RATIO = 1.5


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("nops", [2, 3])
@pytest.mark.parametrize("m", [200, 1920, 3840, 1000])
def test_glu_gemm(nops, m):
    torch.manual_seed(0)
    h = 1536
    x = torch.randn(m, 768, device="cuda", dtype=BF)
    w = (torch.randn(nops * h, 768, device="cuda") * 768 ** -0.5).to(BF)
    out = torch.empty(m, h, device="cuda", dtype=BF)
    dit_gemm.glu_gemm(x, w, out, nops)
    ab = x.float() @ w.float().t()
    ref = F.silu(ab[:, :h]) * ab[:, h:2 * h] * (ab[:, 2 * h:] if nops == 3 else 1)
    unfused = torch.empty_like(out)
    rows.glu_rows(x @ w.t(), unfused, nops == 3)
    assert torch.isfinite(out).all()
    assert _rel(out, ref) <= ERR_RATIO * _rel(unfused, ref), (_rel(out, ref), _rel(unfused, ref))


def _tables(L, presig):
    t = torch.randn(L, 6 * 768, device="cuda", dtype=BF)
    raw = t.clone()
    if presig:   # gates g1 | g2 and the scales ms0 / ms1 as sigmoid values
        for lo, hi in ((0, 1536), (1536, 2304), (3072, 3840)):
            t[:, lo:hi] = torch.sigmoid(raw[:, lo:hi].float()).to(BF)
    cols = lambda tt, j: tt[:, 768 * j:768 * (j + 1)]
    return t, raw, cols


@pytest.mark.parametrize("presig", [False, True])
@pytest.mark.parametrize("mode", ["one", "dual", "final", "dual_final"])
@pytest.mark.parametrize(("m", "L"), [(1920, 384), (1000, 200), (200, 200), (3840, 768)])
def test_res_gemm(mode, presig, m, L):
    torch.manual_seed(1)
    dual = mode.startswith("dual")
    final = mode.endswith("final")
    a1 = torch.randn(m, 3072, device="cuda", dtype=BF)[:, :768]                  # a column view, like the attention output over q
    w1 = (torch.randn(768, 768, device="cuda") * 768 ** -0.5).to(BF)
    a2 = torch.randn(m, 1536, device="cuda", dtype=BF)
    w2 = (torch.randn(768, 1536, device="cuda") * 1536 ** -0.5).to(BF)
    t, raw, cols = _tables(L, presig)
    x0 = torch.randn(m, 768, device="cuda")
    tok = torch.arange(m, device="cuda") % L
    # the gates / scales the kernel applies: the given sigmoid values (bf16) or the sigmoid of the given logits
    sg = lambda j: (cols(t, j).float() if presig else torch.sigmoid(cols(raw, j).float()))[tok]
    xr = x0 + sg(0) * (a1.float() @ w1.float().t())
    if dual:
        xr = xr + sg(1) * (a2.float() @ w2.float().t())
    nout = 0 if final else (2 if dual else 1)
    xh = F.layer_norm(xr, (768,), eps=1e-5)
    refs = [xh * sg(2 + 2 * o) + cols(raw, 3 + 2 * o).float()[tok] for o in range(nout)]

    x = x0.clone()
    outs = [torch.empty(m, 768, device="cuda", dtype=BF) for _ in range(nout)]
    fin = torch.empty(m, 768, device="cuda", dtype=BF) if final else None
    kw = dict(a2=a2, w2=w2, g2=cols(t, 1)) if dual else {}
    mods = [(cols(t, 2 + 2 * o), cols(t, 3 + 2 * o), outs[o]) for o in range(nout)]
    dit_gemm.res_gemm(x, a1, w1, cols(t, 0), L, mods=mods, final=fin, sig=not presig, sig_g=not presig, **kw)

    # the unfused path: cuBLAS bf16 products + the row passes (raw logits)
    xu = x0.clone()
    ya, yt = a1 @ w1.t(), (a2 @ w2.t() if dual else None)
    uouts = [torch.empty(m, 768, device="cuda", dtype=BF) for _ in range(nout)]
    ufin = torch.empty(m, 768, device="cuda", dtype=BF) if final else None
    rows.res_rows(xu, ya, cols(raw, 0), L, yt=yt, gt=cols(raw, 1) if dual else None, final=ufin)
    if nout:
        rows.ln_mod_rows(xu, [(cols(raw, 2 + 2 * o), cols(raw, 3 + 2 * o), uouts[o]) for o in range(nout)], L, 1e-5, True)
    if final:
        assert torch.equal(x, x0)                                                   # not written back
        assert _rel(fin, xr) <= ERR_RATIO * _rel(ufin, xr), (_rel(fin, xr), _rel(ufin, xr))
    else:
        assert _rel(x, xr) <= 1e-5, _rel(x, xr)
        for o in range(nout):
            assert _rel(outs[o], refs[o]) <= ERR_RATIO * _rel(uouts[o], refs[o]), (o, _rel(outs[o], refs[o]), _rel(uouts[o], refs[o]))


def test_chain_graph_replay():
    """glu_gemm -> res_gemm (dual, two outputs) -> res_gemm (one output) -> res_gemm (final), captured once (programmatic dependent
    launches between them) and replayed 3000 times: no hang, every replay bitwise equal to the eager chain."""
    torch.manual_seed(2)
    m, L = 1920, 384
    xt = torch.randn(m, 768, device="cuda", dtype=BF)
    wexp = (torch.randn(3 * 1536, 768, device="cuda") * 768 ** -0.5).to(BF)
    wo = (torch.randn(768, 768, device="cuda") * 768 ** -0.5).to(BF)
    wsq = (torch.randn(768, 1536, device="cuda") * 1536 ** -0.5).to(BF)
    o = torch.randn(m, 768, device="cuda", dtype=BF)
    t, _, cols = _tables(L, True)
    x0 = torch.randn(m, 768, device="cuda")
    x = x0.clone()
    h = torch.empty(m, 1536, device="cuda", dtype=BF)
    xa, xb, out = (torch.empty(m, 768, device="cuda", dtype=BF) for _ in range(3))

    def chain():
        x.copy_(x0)
        dit_gemm.glu_gemm(xt, wexp, h, 3)
        dit_gemm.res_gemm(x, o, wo, cols(t, 0), L, a2=h, w2=wsq, g2=cols(t, 1), mods=[(cols(t, 2), cols(t, 3), xa),
                          (cols(t, 4), cols(t, 5), xb)], sig=False, sig_g=False)
        dit_gemm.res_gemm(x, xa, wo, cols(t, 1), L, mods=[(cols(t, 2), cols(t, 3), xb)], sig=False, sig_g=False)
        dit_gemm.res_gemm(x, xb, wo, cols(t, 0), L, final=out, sig_g=False)

    chain()
    torch.cuda.synchronize()
    ref = out.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        chain()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    for i in range(3000):
        graph.replay()
        if i % 500 == 0:
            torch.cuda.synchronize()
            assert torch.equal(out, ref), i
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
