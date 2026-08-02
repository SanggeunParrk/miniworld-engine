import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/mw-bidir/src")
import torch
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication as TMB)
from miniworld_kernels.modules import ImplementationType

torch.manual_seed(0)
d = 128
dev = "cuda"
Ls = [int(x) for x in sys.argv[1:]] or [384, 768, 1024]


def rnd(m):
    for n, p in m.named_parameters():
        if n.endswith("ln_pair.weight") or n.endswith("ln_out.weight"):
            p.data.normal_(1.0, 0.05)
        elif n.endswith(".bias"):
            p.data.normal_(0.0, 0.02)
        else:
            p.data.normal_(0.0, 0.05)
    return m


def score(y, r):
    e = (y - r).abs()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), r.flatten(), dim=0).item()
    return cos, (e.mean() / r.abs().mean()).item(), e.max().item()


def graph_score(m, pair):
    with torch.no_grad():
        eager = m(pair).clone()
        si = pair.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(8): _ = m(si)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = m(si)
        si.copy_(pair); g.replay(); torch.cuda.synchronize()
        return score(out.float(), eager.float())


for L in Ls:
    print(f"=== BIDIR L={L} ===", flush=True)
    pair = torch.randn(1, L, L, d, device=dev)
    bref = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("pytorch")).to(dev))
    with torch.inference_mode():
        ybref = bref(pair.float()).float()
    bm = TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("cute")).to(dev)
    bm.load_state_dict(bref.state_dict())
    bm = bm.to(torch.bfloat16).eval()
    pb = pair.to(torch.bfloat16)
    with torch.inference_mode():
        yb = bm(pb).float()
    c, r, mx = score(yb, ybref)
    print(f"  eager cute vs fp32ref: cos={c:.6f} relmean={r:.3e} maxabs={mx:.3e}", flush=True)
    gc, gr, gmx = graph_score(bm, pb)
    print(f"  graph  vs eager cute : cos={gc:.6f} relmean={gr:.3e} maxabs={gmx:.3e}", flush=True)
    # bf16 floor reference: pytorch-ref run in bf16 vs fp32 ref
    bf = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("pytorch")).to(dev))
    bf.load_state_dict(bref.state_dict()); bf = bf.to(torch.bfloat16).eval()
    with torch.inference_mode():
        ybf = bf(pb).float()
    c2, r2, mx2 = score(ybf, ybref)
    print(f"  bf16-ref floor       : cos={c2:.6f} relmean={r2:.3e} maxabs={mx2:.3e}", flush=True)
