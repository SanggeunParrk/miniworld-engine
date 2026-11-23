import torch, time, sys
from torch.utils.cpp_extension import load_inline
src = r'''
#include <cstdint>
template<int KIND> __global__ void k(float* out, int iters){
  uint32_t a0=threadIdx.x,a1=a0*3,a2=a0*5,a3=a0*7,b0=a0*11,b1=a0*13;
  float c[8][4]={};
  for(int i=0;i<iters;i++){
    #pragma unroll
    for(int j=0;j<8;j++){
      if(KIND==0) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
      else asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
  }
  float s=0; for(int j=0;j<8;j++)for(int q=0;q<4;q++)s+=c[j][q]; out[blockIdx.x*blockDim.x+threadIdx.x]=s;
}
void run(torch::Tensor out,int kind,int blocks,int threads,int iters){
  if(kind==0) k<0><<<blocks,threads>>>(out.data_ptr<float>(),iters); else k<1><<<blocks,threads>>>(out.data_ptr<float>(),iters);
}
'''
m = load_inline("p0mma", cpp_sources="#include <torch/extension.h>\nvoid run(torch::Tensor,int,int,int,int);", cuda_sources="#include <torch/extension.h>\n"+src,
   functions=["run"], extra_cuda_cflags=["-gencode","arch=compute_120a,code=sm_120a","-O3"], verbose=False)
sm = torch.cuda.get_device_properties(0).multi_processor_count
print("SMs", sm, torch.cuda.get_device_name(0))
out = torch.empty(sm*8*256, device="cuda")
for kind,name,flop in ((0,"bf16 m16n8k16",16*8*16*2),(1,"e4m3 m16n8k32",16*8*32*2)):
    blocks, thr, it = sm*8, 256, 20000
    m.run(out,kind,blocks,thr,100); torch.cuda.synchronize()
    ts=[]
    for _ in range(5):
        e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();m.run(out,kind,blocks,thr,it);e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1)/1e3)
    t=min(ts); warps=blocks*thr/32
    print(f"{name}: {warps*it*8*flop/t/1e12:.1f} TFLOPS (mma.sync, fp32 acc)")
import torch
def bench(f, flush=torch.empty(128<<20,dtype=torch.uint8,device="cuda"), n=20):
    f(); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): flush.zero_(); f()
    gf=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n): flush.zero_()
    def t(gr):
        gr.replay(); torch.cuda.synchronize(); ts=[]
        for _ in range(5):
            e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();gr.replay();e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1))
        return min(ts)
    return (t(g)-t(gf))/n*1e3  # us
fp8=torch.float8_e4m3fn
print("shape M,K,N | bf16 us | fp8 GEMM us (prequant, bf16 out, tensorwise) | rowwise us | cast x (naive) us | amax+cast us | speedup gemm | speedup incl naive cast")
cases=[]
for L in (256,512,1024,2048):
    M=L*L
    for K,N,name in ((128,512,"TM/Tr d128 in(N=4d)"),(128,1024,"Tr d128 n4 up N=2*4d"),(256,1024,"TM d256 in N=4d"),(256,2048,"Tr d256 n4 up"),(128,640,"TM d128 5d")):
        if M*N*2>4e9 or M*K*2>4e9: continue
        cases.append((M,K,N,f"L{L} {name}"))
cases+= [(1024*1024,256,128,"Tr d128 n2 down"),(1024*1024,512,128,"Tr d128 n4 down"),(512*512,512,256,"d256 n2 down"),(2048,384*2,384*4,"single d384 n2 up S2048")]
for M,K,N,name in cases:
    a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16); w=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)
    ab=a.to(fp8); wb=w.to(fp8); s=torch.ones((),device="cuda"); 
    sr=torch.ones(M,1,device="cuda"); sc=torch.ones(1,N,device="cuda")
    tb=bench(lambda: a@w.t())
    try:
        tf=bench(lambda: torch._scaled_mm(ab,wb.t(),scale_a=s,scale_b=s,out_dtype=torch.bfloat16))
    except Exception as e: tf=float('nan'); print(e)
    try:
        tr=bench(lambda: torch._scaled_mm(ab,wb.t(),scale_a=sr,scale_b=sc,out_dtype=torch.bfloat16))
    except Exception as e: tr=float('nan')
    tc=bench(lambda: a.to(fp8))
    ta=bench(lambda: (a/ (a.abs().amax()/448)).to(fp8))
    print(f"{name:28s} M={M:8d} K={K} N={N} | {tb:8.1f} | {tf:8.1f} | {tr:8.1f} | {tc:7.1f} | {ta:7.1f} | {tb/tf:.2f}x | {tb/(tf+tc):.2f}x")
