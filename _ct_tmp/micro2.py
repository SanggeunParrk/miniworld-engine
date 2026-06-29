"""dgrad GEMM tuning probe: triton _dgemm vs cuBLAS for dh, dcond, dx(=dab@wcat one GEMM).

Also tries dx both ways (2-dot fused kernel vs 1 concatenated _dgemm) to prove the fix.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import _dgemm, _dx_fused


def bench(fn, it=80, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3


def run():
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'gemm':>9} | {'triton':>8} {'cublas':>8} {'tri/cub':>7}")
    for stream, d, Ms in [("atom",128,(2048,4096,8192)),("token",768,(384,512,768,1024))]:
        for M in Ms:
            dev="cuda"; ND=2*d; D=d; DC=384
            Ws=torch.randn(D,ND,device=dev); Wsc=torch.randn(D,DC,device=dev)
            Wa=torch.randn(ND,d,device=dev); Wb=torch.randn(ND,d,device=dev)
            wcat=torch.cat([Wa,Wb],0).contiguous()   # (2ND, d)
            dout=torch.randn(M,D,device=dev); dscale=torch.randn(M,D,device=dev)
            dh=torch.randn(M,ND,device=dev); ab=torch.randn(M,2*ND,device=dev)
            dab=torch.randn(M,2*ND,device=dev)
            # dh = dout @ Ws
            tt=bench(lambda: _dgemm(dout,Ws,M,ND,D,Ws.stride(0),Ws.stride(1))); tc=bench(lambda: dout@Ws)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dh':>9} | {tt:8.1f} {tc:8.1f} {tt/tc:6.2f}x")
            # dcond = dscale @ Wsc
            tt=bench(lambda: _dgemm(dscale,Wsc,M,DC,D,Wsc.stride(0),Wsc.stride(1))); tc=bench(lambda: dscale@Wsc)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dcond':>9} | {tt:8.1f} {tc:8.1f} {tt/tc:6.2f}x")
            # dx one concatenated GEMM: dab @ wcat
            tt=bench(lambda: _dgemm(dab,wcat,M,d,2*ND,wcat.stride(0),wcat.stride(1))); tc=bench(lambda: dab@wcat)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dx_1gemm':>9} | {tt:8.1f} {tc:8.1f} {tt/tc:6.2f}x")
            # dx old 2-dot fused (for contrast)
            tt=bench(lambda: _dx_fused(dh,ab,Wa,Wb))
            print(f"{stream:>6} {M:>6} {d:>4} | {'dx_2dot':>9} | {tt:8.1f} {tc:8.1f} {tt/tc:6.2f}x")
    print("DONE")


def overhead_probe():
    print("=== launch-overhead probe: tiny GEMM (1 tile) triton vs cublas ===")
    dev="cuda"
    for M,N,K in [(64,64,64),(256,128,128)]:
        a=torch.randn(M,K,device=dev); w=torch.randn(K,N,device=dev)
        tt=bench(lambda: _dgemm(a,w,M,N,K,w.stride(0),w.stride(1))); tc=bench(lambda: a@w)
        print(f"  tiny {M}x{N}x{K}: triton={tt:.1f}us cublas={tc:.1f}us  (floor)")


if __name__=="__main__":
    print(torch.cuda.get_device_name()); overhead_probe(); print(); run()
