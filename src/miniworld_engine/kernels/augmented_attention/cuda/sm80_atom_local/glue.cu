// A100 attention backward row dot, one warp per head of one query.
#include "local.cuh"
#include <math.h>
extern "C" __global__ void local_attn_delta(const __nv_bfloat16 *o,const __nv_bfloat16 *dout,float *delta,int n,int samples) {
  const int lane=threadIdx.x&31;
  const int unit=blockIdx.x*8+(threadIdx.x>>5);
  if(unit>=samples*n*NH)return;
  const int head=unit%NH,row=unit/NH,sample=row/n,query=row-sample*n;
  float value=__bfloat162float(o[(size_t)row*DM+head*DH+lane])*__bfloat162float(dout[(size_t)row*DM+head*DH+lane]);
#pragma unroll
  for(int shift=16;shift>0;shift>>=1)value+=__shfl_xor_sync(0xffffffffu,value,shift);
  if(lane==0)delta[((size_t)sample*NH+head)*n+query]=value;
}

// Preserve the BF16 sigmoid and product rounding points of the module, also under compile.
extern "C" __global__ void local_gate_fwd(const __nv_bfloat16 *gate,const __nv_bfloat16 *x,
    __nv_bfloat16 *out,__nv_bfloat16 *sigmoid,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if(i>=count)return;
  const __nv_bfloat16 s=__float2bfloat16_rn(1.f/(1.f+expf(-__bfloat162float(gate[i]))));
  sigmoid[i]=s;
  out[i]=__float2bfloat16_rn(__bfloat162float(s)*__bfloat162float(x[i]));
}
extern "C" __global__ void local_gate_bwd(const __nv_bfloat16 *dy,const __nv_bfloat16 *sigmoid,const __nv_bfloat16 *x,
    __nv_bfloat16 *dgate,__nv_bfloat16 *dx,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if(i>=count)return;
  const float s=__bfloat162float(sigmoid[i]),d=__bfloat162float(dy[i]);
  const float ds=__bfloat162float(__float2bfloat16_rn(d*__bfloat162float(x[i])));
  dx[i]=__float2bfloat16_rn(d*s);
  dgate[i]=__float2bfloat16_rn(ds*(1.f-s)*s);
}
