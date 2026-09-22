// Diagnostic only: streaming bandwidth calibration, no model-kernel dispatch.
#include <cuda_runtime.h>
extern "C" __global__ void bandwidth3(const uint4* __restrict__ a,const uint4* __restrict__ b,const uint4* __restrict__ c,uint4* __restrict__ out,int n){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  uint4 x=a[i],y=b[i],z=c[i];
  out[i]=make_uint4(x.x^y.x^z.x,x.y^y.y^z.y,x.z^y.z^z.z,x.w^y.w^z.w);
 }
}
extern "C" __global__ void bandwidth1(const uint4* __restrict__ a,uint4* __restrict__ out,int n){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)out[i]=a[i];
}
