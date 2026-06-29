"""CUDA-graph per-stage micro: fused-triton dgrad (gate/swiglu folded) vs cuBLAS+separate elem.

Isolates whether the FUSION (fewer kernels, no elem HBM round-trip) wins per stage once the
launch floor is removed by graph capture.
"""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import (
    _dh_gatebwd, _dx_swiglubwd, _dgemm, _gate_bwd, _swiglu_bwd_pack,
)


def gbench(fn, it=100, wu=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/it*1e3


def run():
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'stage':>14} | {'fused_tri':>9} {'cublas+elem':>11} {'ratio':>6}")
    for stream, d, Ms in [("atom",128,(2048,8192)),("token",768,(384,1024))]:
        for M in Ms:
            dev="cuda"; ND=2*d; D=d; DC=384; K=d
            Ws=torch.randn(D,ND,device=dev); Wsc=torch.randn(D,DC,device=dev)
            Wa=torch.randn(ND,K,device=dev); Wb=torch.randn(ND,K,device=dev)
            wcat=torch.cat([Wa,Wb],0).contiguous()
            out=torch.randn(M,D,device=dev); scale=torch.randn(M,D,device=dev); dy=torch.randn(M,D,device=dev)
            ab=torch.randn(M,2*ND,device=dev); dh=torch.randn(M,ND,device=dev)
            # stage dh+gatebwd: fused vs (gate_bwd elem -> cuBLAS dh)
            def fused_dh():
                return _dh_gatebwd(out,scale,dy,Ws,ND)
            def cub_dh():
                dout,dsc=_gate_bwd(out,scale,dy); return dout@Ws, dout, dsc
            tf=gbench(fused_dh); tc=gbench(cub_dh)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dh+gatebwd':>14} | {tf:9.1f} {tc:11.1f} {tf/tc:5.2f}x")
            # stage dx+swiglubwd: fused vs (swiglu pack elem -> cuBLAS dab@wcat)
            dsc2=torch.randn(M,D,device=dev)
            def fused_dx():
                return _dx_swiglubwd(dh,ab,wcat)
            def cub_dx():
                dab=_swiglu_bwd_pack(dh,ab); return dab@wcat, dab
            tf=gbench(fused_dx); tc=gbench(cub_dx)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dx+swiglubwd':>14} | {tf:9.1f} {tc:11.1f} {tf/tc:5.2f}x")
            # stage dcond: triton _dgemm vs cuBLAS
            dscale=torch.randn(M,D,device=dev)
            tf=gbench(lambda: _dgemm(dscale,Wsc,M,DC,D,Wsc.stride(0),Wsc.stride(1)))
            tc=gbench(lambda: dscale@Wsc)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dcond':>14} | {tf:9.1f} {tc:11.1f} {tf/tc:5.2f}x")
    print("DONE")


if __name__=="__main__":
    print(torch.cuda.get_device_name()); run()
