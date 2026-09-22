"""Premise check for a dWab + d_xn co-scheduled kernel: per-row cost of the dxn_lnbwd kernel and of cuBLAS dWab when dAB is
L2-resident (just written) vs cold (L2 flushed).  python l2_premise.py"""
import sys, statistics
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
D, H = 256, 1024; bf = torch.bfloat16
k = drv.Kernel(str(HERE / "build/dl_c2_w0.cubin"), "dxn_lnbwd", 231424)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
def case(M):
    x = torch.randn(M, D, device="cuda").to(bf); dy = torch.randn_like(x); dab = torch.randn(M, 2 * H, device="cuda").to(bf)
    xn = torch.randn_like(x); wab = (torch.randn(2 * H, D, device="cuda") * 0.02).to(bf); wabT = wab.t().contiguous()
    g = torch.ones(D, device="cuda"); rs = torch.ones(M, device="cuda"); c1 = torch.zeros(M, device="cuda")
    out = torch.empty_like(x); pdg = torch.empty(132, D, device="cuda"); pdb = torch.empty_like(pdg)
    mA = drv.TensorMap(dab, dims=[2 * H, M], stride_bytes=4 * H, box=[64, 64]); mB = drv.TensorMap(wabT, dims=[2 * H, D], stride_bytes=4 * H, box=[64, D // 2])
    mX = drv.TensorMap(x, dims=[D, M], stride_bytes=2 * D, box=[64, 64])
    dx = lambda: k((132, 1, 1), (384, 1, 1), mA, mB, mX, x, dy, out, g, rs, c1, pdg, pdb, int(M))
    dw = lambda: torch.mm(dab.t(), xn, out_dtype=torch.float32)
    touch = lambda: dab.add_(0)                         # re-writes dAB: it is then L2-resident (as right after the gate kernel)
    res = {}
    for name, fn in (("dxn_lnbwd", dx), ("dWab cuBLAS", dw)):
        for warm in (False, True):
            o = []
            for _ in range(15):
                flush.zero_()
                if warm: touch()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); fn(); e.record(); torch.cuda.synchronize()
                o.append(s.elapsed_time(e) * 1e3)
            res[(name, warm)] = statistics.median(o[3:])
    mb = M * 2 * H * 2 / 1e6
    for name in ("dxn_lnbwd", "dWab cuBLAS"):
        c, w = res[(name, False)], res[(name, True)]
        print(f"M {M:6d} (dAB {mb:5.1f} MB) {name:12s} cold {c:7.1f} us  warm {w:7.1f} us  -> x{c / w:.2f}   per 1k rows cold {1e3 * c / M:.2f} us")
for M in (4096, 8192, 147456):
    case(M)
