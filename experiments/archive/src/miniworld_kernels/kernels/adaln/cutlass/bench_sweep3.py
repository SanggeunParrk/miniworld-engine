import os, torch
from torch.utils.cpp_extension import load
HERE=os.path.dirname(os.path.abspath(__file__)); CUTLASS="/home/psk6950/miniworld-kernels/_ct_cutlass/cutlass"
BUILD="/home/psk6950/.cache/adaln_cutlass_sweep3"; os.makedirs(BUILD,exist_ok=True)
print("building sweep3 (K=128 tiles)...",flush=True)
mod=load(name="adaln_cutlass_sweep3",sources=[os.path.join(HERE,"gemm_sweep3.cu")],
  extra_include_paths=[os.path.join(CUTLASS,"include"),os.path.join(CUTLASS,"tools","util","include")],
  extra_cuda_cflags=["-O3","-std=c++17","-arch=sm_90a","--expt-relaxed-constexpr","--expt-extended-lambda","-DNDEBUG"],
  build_directory=BUILD,verbose=False)
print("build OK",flush=True)
torch.backends.cuda.matmul.allow_tf32=True; import torch.cuda as tc
def tcub(A,W):
    for _ in range(20): A@W.t()
    tc.synchronize(); b=tc.Event(enable_timing=True);e=tc.Event(enable_timing=True);b.record()
    for _ in range(100): A@W.t()
    e.record();tc.synchronize();return b.elapsed_time(e)/100*1000
for (M,K,N) in [(32768,768,768),(32768,768,1536),(262144,128,256)]:
    A=torch.randn(M,K,device="cuda",dtype=torch.float32); W=torch.randn(N,K,device="cuda",dtype=torch.float32)*K**-0.5
    gf=2*M*K*N/1e9; tc_=tcub(A,W)
    c=torch.nn.functional.cosine_similarity((A@W.t()).flatten(), mod.gemm_ref(A,W).flatten(),dim=0).item()
    print(f"\n## M={M} K={K} N={N} ({gf:.0f}GF) cuBLAS={tc_:.1f}us ({gf/(tc_/1e6)/1e3:.0f}TF/s) cos={c:.5f}",flush=True)
    best=(1e9,-1)
    for cfg in range(mod.n_cfg()):
        tm=mod.bench(A,W,cfg,20,100)
        if tm<0: print(f"   cfg{cfg}: unsupported({tm:.0f})",flush=True); continue
        if tm<best[0]: best=(tm,cfg)
        print(f"   cfg{cfg}: {tm:7.1f}us ({gf/(tm/1e6)/1e3:3.0f}TF/s) cuBLAS/ct={tc_/tm:.2f}x",flush=True)
    print(f"   => BEST cfg{best[1]} {best[0]:.1f}us  cuBLAS/ours={tc_/best[0]:.2f}x",flush=True)
print("DONE",flush=True)
