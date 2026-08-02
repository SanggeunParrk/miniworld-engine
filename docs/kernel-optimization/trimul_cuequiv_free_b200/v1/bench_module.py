"""Module-level bench: cuequiv-free (default sm100) vs cuequiv-reusing _forward_cute
vs cuequiv, through TriangleMultiplication. Honest: eager cos/relmean/maxabs vs fp32
ref + graph time with replay-cos check."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, "/home/snu_hwle/psk/miniworld-kernels/src")
import torch, triton
from miniworld_engine.modules.triangle_multiplication.module import TriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1 as _dtv1,
)


def dtv1_call(m, pair):
    return _dtv1(
        pair, "outgoing" if m.outgoing else "incoming", None,
        m.ln_pair.weight, m.ln_pair.bias,
        torch.cat([m.to_left.weight, m.to_right.weight], dim=0),
        torch.cat([m.to_left_gate.weight, m.to_right_gate.weight], dim=0),
        m.ln_out.weight, m.ln_out.bias, m.to_out.weight, m.to_gate.weight,
    )


def make_module(D=128, seed=0):
    torch.manual_seed(seed)
    m = TriangleMultiplication(d_pair=D, outgoing=True,
                              implementation=ImplementationType.CUTE).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for lin in (m.to_left, m.to_left_gate, m.to_right, m.to_right_gate, m.to_gate, m.to_out):
            lin.weight.normal_(0, 0.05)
        m.ln_pair.weight.normal_(1.0, 0.02); m.ln_pair.bias.normal_(0, 0.02)
        m.ln_out.weight.normal_(1.0, 0.02); m.ln_out.bias.normal_(0, 0.02)
    return m

def fp32_ref(m, pair):
    m32 = TriangleMultiplication(d_pair=pair.shape[-1], outgoing=True,
                                 implementation=ImplementationType.PYTORCH).cuda().float()
    with torch.no_grad(): m32.load_state_dict(m.state_dict())
    return m32(pair.float())

def metrics(y, ref):
    y=y.float(); ref=ref.float()
    return (torch.nn.functional.cosine_similarity(y.flatten(),ref.flatten(),dim=0).item(),
            (y-ref).abs().mean().item()/(ref.abs().mean().item()+1e-12),
            (y-ref).abs().max().item())

def bench(fn): return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

def graph_time(fn):
    eager=fn().clone()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    g.replay(); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(out.float().flatten(),eager.float().flatten(),dim=0).item()
    return bench(lambda: g.replay()), cos

def main():
    Ls=[int(x) for x in os.environ.get("LS","384,768,1024").split(",")]
    D=int(os.environ.get("D","128"))
    for L in Ls:
        m=make_module(D)
        pair=torch.randn(1,L,L,D,device="cuda",dtype=torch.bfloat16)
        ref=fp32_ref(m,pair)
        m.implementation=ImplementationType.CUTE
        # free (default on sm100)
        os.environ.pop("MINIWORLD_TRIMUL_CUEQUIV_FREE",None)
        y=m(pair); fm=metrics(y,ref); t_free,c_free=graph_time(lambda: m(pair))
        # cuequiv-reusing old cute
        os.environ["MINIWORLD_TRIMUL_CUEQUIV_FREE"]="0"
        y2=m(pair); om=metrics(y2,ref); t_old,c_old=graph_time(lambda: m(pair))
        os.environ.pop("MINIWORLD_TRIMUL_CUEQUIV_FREE",None)
        # cuequiv
        m.implementation=ImplementationType.CUEQUIVARIANCE
        t_cq,c_cq=graph_time(lambda: m(pair))
        # dtv1 (triton baseline, cuequiv weight API)
        dm=metrics(dtv1_call(m,pair),ref); t_dt,c_dt=graph_time(lambda: dtv1_call(m,pair))
        print(f"L={L} D={D}")
        print(f"  free   : graph {t_free:.4f}ms replaycos {c_free:.5f} | eager cos {fm[0]:.5f} relmean {fm[1]:.2e} maxabs {fm[2]:.2e}")
        print(f"  old-cute:graph {t_old:.4f}ms replaycos {c_old:.5f} | eager cos {om[0]:.5f} relmean {om[1]:.2e}")
        print(f"  cuequiv: graph {t_cq:.4f}ms replaycos {c_cq:.5f}")
        print(f"  dtv1   : graph {t_dt:.4f}ms replaycos {c_dt:.5f} | eager cos {dm[0]:.5f} relmean {dm[1]:.2e}")
        print(f"  speedup free vs cuequiv {t_cq/t_free:.3f}x  vs old-cute {t_old/t_free:.3f}x  vs dtv1 {t_dt/t_free:.3f}x", flush=True)

if __name__=="__main__": main()
