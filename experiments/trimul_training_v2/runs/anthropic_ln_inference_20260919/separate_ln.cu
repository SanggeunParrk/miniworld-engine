// SPDX-License-Identifier: Apache-2.0
// Wrapper around Anthropic's unchanged ln_fragment, revision f4f62fa.
// Same MMA fragment layout, reduction, affine scheduling and BF16 rounding.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
struct Params { const __nv_bfloat16 *x; const float *gamma,*beta; __nv_bfloat16 *y; float *mean,*rstd; int m; float eps; };
extern "C" __global__ __launch_bounds__(MW_THREADS)
void mw_ln(__grid_constant__ const Params p) {
  __shared__ float sg[MW_C],sb[MW_C];
  for(int c=threadIdx.x;c<MW_C;c+=blockDim.x) {sg[c]=p.gamma[c];sb[c]=p.beta[c];}
  __syncthreads();
  const int lane=threadIdx.x%32,q=lane%4;
  const int ra=(blockIdx.x*blockDim.x/32+threadIdx.x/32)*16+lane/4,rb=ra+8;
  uint32_t fa[MW_C/16][4];
#pragma unroll
  for(int k=0;k<MW_C/16;++k) {
    int c=k*16+2*q;
    if(MW_TRANSPOSE) {
      fa[k][0]=ra<p.m?pack_bf16(__bfloat162float(p.x[(size_t)c*p.m+ra]),__bfloat162float(p.x[(size_t)(c+1)*p.m+ra])):0;
      fa[k][1]=rb<p.m?pack_bf16(__bfloat162float(p.x[(size_t)c*p.m+rb]),__bfloat162float(p.x[(size_t)(c+1)*p.m+rb])):0;
      fa[k][2]=ra<p.m?pack_bf16(__bfloat162float(p.x[(size_t)(c+8)*p.m+ra]),__bfloat162float(p.x[(size_t)(c+9)*p.m+ra])):0;
      fa[k][3]=rb<p.m?pack_bf16(__bfloat162float(p.x[(size_t)(c+8)*p.m+rb]),__bfloat162float(p.x[(size_t)(c+9)*p.m+rb])):0;
    } else {
      fa[k][0]=ra<p.m?*(const uint32_t*)(p.x+(size_t)ra*MW_C+c):0;
      fa[k][1]=rb<p.m?*(const uint32_t*)(p.x+(size_t)rb*MW_C+c):0;
      fa[k][2]=ra<p.m?*(const uint32_t*)(p.x+(size_t)ra*MW_C+c+8):0;
      fa[k][3]=rb<p.m?*(const uint32_t*)(p.x+(size_t)rb*MW_C+c+8):0;
    }
  }
  LnStats st=ln_fragment<MW_C/16,MW_SERIAL>(fa,sg,sb,lane,p.eps);
#pragma unroll
  for(int k=0;k<MW_C/16;++k) {
    int c=k*16+2*q;
    if(ra<p.m){stg32(p.y+(size_t)ra*MW_C+c,fa[k][0]);stg32(p.y+(size_t)ra*MW_C+c+8,fa[k][2]);}
    if(rb<p.m){stg32(p.y+(size_t)rb*MW_C+c,fa[k][1]);stg32(p.y+(size_t)rb*MW_C+c+8,fa[k][3]);}
  }
}
