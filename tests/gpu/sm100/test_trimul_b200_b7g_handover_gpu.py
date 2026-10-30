"""b7g (D128 input-side backward) hand-over: source CTAs pass dgp tiles to consumer CTAs through an L2 ring guarded by global flags
(TMA store -> release -> acquire -> TMA load; slot return: TMA load landed -> release -> acquire -> TMA store). These tests replay the
kernel in a CUDA graph many times and require every replay bitwise equal (a missing proxy / release fence shows up as a rare stale
tile), at L768 and at L256 (each source group's 12 ring slots recycle every few tiles), and check odd tile counts (consumers then
run one CTA per tile instead of in pairs) against the paired schedule."""

import pytest
import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI
from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) kernels"),
]

D = 128


def _inputs(tiles, nch, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, p = tiles * 128, 64 * nch

    def bf(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)

    return dict(x=bf(m, D), xn=bf(m, D), dy=bf(m, D), dpl=bf(p, m, scale=0.1), dg=bf(m, D, scale=0.01),
                w1=bf(2 * p, D, scale=D ** -0.5), wg=bf(D, D, scale=D ** -0.5),
                gi=(1 + 0.1 * torch.randn(D, device="cuda", generator=g)).contiguous())


def _buffers(inp, length, nch):
    sg = BT._b7_source_groups(length, nch)
    nc = num_sms(inp["x"].device) - nch * sg
    p = 64 * nch
    return dict(sg=sg, flags=torch.zeros(sg * BT.B7_RD * nch, dtype=torch.int32, device="cuda"),
                ring=inp["x"].new_empty((sg * BT.B7_RD * nch * 128, D)), dx=torch.empty_like(inp["x"]),
                part=torch.empty(sg, 2 * p * D, device="cuda"), lnp=torch.empty(nc, 2 * D, device="cuda"))


def _launch(inp, buf, tmask, length):
    BI._ext().b7g_backward(inp["x"], inp["xn"], inp["dy"], inp["dpl"], inp["dg"], tmask, inp["w1"], inp["wg"], inp["gi"], buf["dx"],
                           buf["part"], buf["lnp"], buf["ring"], buf["flags"], 1e-5, buf["sg"], length)


@pytest.mark.parametrize("length", [256, 768])
def test_b7g_graph_replays_bitwise(length):
    """500 CUDA-graph replays of b7g alone: dx, the dW1 partials and the LN partials bitwise equal to the first replay, the ring
    flags 0 after every replay."""
    nch = 8
    inp = _inputs(length * length // 128, nch, seed=length)
    mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
    mask[:, length - length // 8:] = False
    tmask = mask.view(-1)
    buf = _buffers(inp, length, nch)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _launch(inp, buf, tmask, length)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(inp, buf, tmask, length)
    graph.replay()
    ref = {k: buf[k].clone() for k in ("dx", "part", "lnp")}
    bad = 0
    for _ in range(500):
        graph.replay()
        bad += sum(int(not torch.equal(buf[k], ref[k])) for k in ref) + int(buf["flags"].any())
    torch.cuda.synchronize()
    assert bad == 0


@pytest.mark.parametrize("nch", [8, 4])
@pytest.mark.parametrize("tiles", [3, 129])
def test_b7g_odd_tile_count(tiles, nch):
    """An odd tile count (not produced by the module, whose L is a multiple of 128) runs the consumers one CTA per tile: dx bitwise
    equal to the paired schedule on the same tiles plus one zero tile, the parameter partial sums equal to fp32 summation order."""
    inp = _inputs(tiles, nch, seed=tiles + nch)
    pad = {k: (torch.cat((v, v.new_zeros(128, *v.shape[1:]))) if k in ("x", "xn", "dy", "dg") else v) for k, v in inp.items()}
    pad["dpl"] = torch.cat((inp["dpl"], inp["dpl"].new_zeros(inp["dpl"].shape[0], 128)), 1).contiguous()
    length = 8                                              # any L with L^2 | M: no token mask, one sample of M tokens
    odd, even = _buffers(inp, length, nch), _buffers(pad, length, nch)
    _launch(inp, odd, None, length)
    _launch(pad, even, None, length)
    torch.cuda.synchronize()
    assert torch.equal(odd["dx"], even["dx"][: tiles * 128])
    for k in ("part", "lnp"):
        a, b = odd[k].sum(0), even[k].sum(0)
        assert ((a - b).norm() / b.norm()).item() < 1e-5, k
    assert not odd["flags"].any() and not even["flags"].any()
