"""Quick: improved triton dgrad GEMM vs cuBLAS only (no full e2e)."""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import _dgemm, _dx_fused


def bench(fn, it=60, wu=25):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3


def run():
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'gemm':>6} | {'triton':>8} {'cublas':>8} {'tri/cub':>7}")
    for stream, d, Ms in [("atom",128,(2048,8192)),("token",768,(384,1024))]:
        for M in Ms:
            dev="cuda"; ND=2*d; D=d; DC=384
            Ws=torch.randn(D,ND,device=dev); Wsc=torch.randn(D,DC,device=dev)
            Wa=torch.randn(ND,d,device=dev); Wb=torch.randn(ND,d,device=dev)
            dout=torch.randn(M,D,device=dev); dscale=torch.randn(M,D,device=dev)
            dh=torch.randn(M,ND,device=dev); ab=torch.randn(M,2*ND,device=dev)
            wcat=torch.cat([Wa,Wb],0); dab=torch.randn(M,2*ND,device=dev)
            for name,tri,cub in [
                ("dh", lambda: _dgemm(dout,Ws,M,ND,D,Ws.stride(0),Ws.stride(1)), lambda: dout@Ws),
                ("dcond", lambda: _dgemm(dscale,Wsc,M,DC,D,Wsc.stride(0),Wsc.stride(1)), lambda: dscale@Wsc),
                ("dx", lambda: _dx_fused(dh,ab,Wa,Wb), lambda: dab@wcat),
            ]:
                tt=bench(tri); tc=bench(cub)
                print(f"{stream:>6} {M:>6} {d:>4} | {name:>6} | {tt:8.1f} {tc:8.1f} {tt/tc:6.2f}x")
    print("DONE")


if __name__=="__main__":
    print(torch.cuda.get_device_name()); run()
