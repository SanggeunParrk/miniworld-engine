// A100 attention backward helpers: the row dot D, the dbias partial sum, the BF16 gate.
#include "local.cuh"
#include <math.h>
// D[sample][head][query] = sum_d O dO over the head's 32 channels. A quad of lanes owns one (row, head): 16-byte loads, 8 channels a lane.
extern "C" __global__ void local_attn_delta(const __nv_bfloat16 *o,const __nv_bfloat16 *dout,float *delta,int n,int samples) {
  const long long t=(long long)blockIdx.x*256+threadIdx.x;       // (row, head, quad lane): 16 lanes a row
  const long long row=t>>4;
  const int part=(int)(t&15),head=part>>2;
  const bool live=row<(long long)samples*n;
  const long long rr=live?row:0;
  const uint4 a=__ldg(reinterpret_cast<const uint4*>(o+rr*DM+part*8)),b=__ldg(reinterpret_cast<const uint4*>(dout+rr*DM+part*8));
  const uint32_t av[4]={a.x,a.y,a.z,a.w},bv[4]={b.x,b.y,b.z,b.w};
  float value=0.f;
#pragma unroll
  for(int i=0;i<4;++i){
    value=fmaf(__uint_as_float(av[i]<<16),__uint_as_float(bv[i]<<16),value);
    value=fmaf(__uint_as_float(av[i]&0xffff0000u),__uint_as_float(bv[i]&0xffff0000u),value);
  }
  value+=__shfl_xor_sync(0xffffffffu,value,1);
  value+=__shfl_xor_sync(0xffffffffu,value,2);
  if(live&&(threadIdx.x&3)==0){
    const long long sample=row/n,query=row-sample*n;
    delta[((size_t)sample*NH+head)*n+query]=value;
  }
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

// dbias = the sum of the split partials (fixed order), 4 floats a thread
extern "C" __global__ void local_dbias_sum(const float *parts, float *out, int nparts, int total) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (4 * i >= total) return;
  float4 acc = *reinterpret_cast<const float4 *>(parts + (size_t)4 * i);
  for (int z = 1; z < nparts; ++z) {
    const float4 v = *reinterpret_cast<const float4 *>(parts + (size_t)z * total + 4 * i);
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  *reinterpret_cast<float4 *>(out + 4 * i) = acc;
}
