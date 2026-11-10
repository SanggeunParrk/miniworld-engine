// E210 private BFgroup reduce plus unchanged original pair-bias backward arithmetic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aux_sm80.cuh"
namespace atom_pair_reduce210 {
using namespace aa80;
template <typename T>
__global__ void __launch_bounds__(PB_NT) pair_bias_bwd_from_partials_kernel(const T* __restrict__ z, const float* __restrict__ wp, const __nv_bfloat16* __restrict__ dbp,
                                                              T* __restrict__ dz, float* __restrict__ pw, int N, int NZ, int groups,
                                                              const unsigned char* __restrict__ kv, float eps) {
  __shared__ float w[PB_H * PB_C];
  __shared__ float red[PB_NT / 32][PB_H * PB_C];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < PB_H * PB_C) w[tid] = wp[tid];
  __syncthreads();
  const int i0 = blockIdx.y * PB_TI, j = blockIdx.x * PB_TJ + tx;
  float acc[PB_H][PB_C];
#pragma unroll
  for (int h = 0; h < PB_H; ++h)
#pragma unroll
    for (int c = 0; c < PB_C; ++c) acc[h][c] = 0.f;
  const bool valid = j < NZ && (kv == nullptr || kv[j]);
  for (int k = 0; k < PB_TI / 4; ++k) {
    const int i = i0 + ty + 4 * k;
    if (i >= NZ || j >= NZ) continue;                                      // padding: no z there
    const size_t e = (size_t)i * NZ + j;
    float x[PB_C];
    pb_load_row(z, e, x);
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < PB_C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
    const float mean = s * (1.f / PB_C), rstd = rsqrtf(fmaxf(s2 * (1.f / PB_C) - mean * mean, 0.f) + eps);
    float g[PB_H];
#pragma unroll
    for (int h = 0; h < PB_H; ++h) {
      g[h]=0.f;
      if(valid) {
        // Preserve each original BF four-sample partial and original ascending
        // FP32 group sum; original db_reduce scale is exactly 1.0 (natural units).
#pragma unroll 8
        for(int group=0;group<groups;++group) {
          const size_t at=(((size_t)group*PB_H+h)*N+i)*N+j;
          g[h]+=__bfloat162float(__ldg(dbp+at));
        }
      }
    }
    float m1 = 0.f, m2 = 0.f, dxh[PB_C];
#pragma unroll
    for (int c = 0; c < PB_C; ++c) {
      const float xh = (x[c] - mean) * rstd;
      float d = 0.f;
#pragma unroll
      for (int h = 0; h < PB_H; ++h) { d = fmaf(g[h], w[h * PB_C + c], d); acc[h][c] = fmaf(g[h], xh, acc[h][c]); }
      dxh[c] = d; m1 += d; m2 += d * xh; x[c] = xh;
    }
    m1 *= 1.f / PB_C; m2 *= 1.f / PB_C;
    float out[PB_C];
#pragma unroll
    for (int c = 0; c < PB_C; ++c) out[c] = (dxh[c] - m1 - x[c] * m2) * rstd;
    pb_store_row(dz + e * PB_C, out);
  }
  // block partial of d(W'): warp shuffle, then across the 8 warps
#pragma unroll
  for (int h = 0; h < PB_H; ++h)
#pragma unroll
    for (int c = 0; c < PB_C; ++c) {
      float v = acc[h][c];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if ((tid & 31) == 0) red[tid >> 5][h * PB_C + c] = v;
    }
  __syncthreads();
  if (tid < PB_H * PB_C) {
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < PB_NT / 32; ++r) v += red[r][tid];
    pw[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * PB_H * PB_C + tid] = v;
  }
}


void run(torch::Tensor z,torch::Tensor w,torch::Tensor dbp,torch::Tensor dz,torch::Tensor pw,torch::Tensor mask,int64_t padded,int64_t length,double eps) {
  TORCH_CHECK(z.is_cuda()&&(z.scalar_type()==at::kBFloat16||z.scalar_type()==at::kFloat)&&z.is_contiguous()&&z.numel()==length*length*16,"physical CUDA BF/F32 z[L,L,16]");
  TORCH_CHECK(padded>=length&&padded%128==0&&length>0,"original padded128 matrix");
  TORCH_CHECK(w.device()==z.device()&&w.scalar_type()==at::kFloat&&w.is_contiguous()&&w.sizes()==at::IntArrayRef({4,16}),"F32 foldedW[4,16]");
  TORCH_CHECK(dbp.device()==z.device()&&dbp.scalar_type()==at::kBFloat16&&dbp.is_contiguous()&&dbp.dim()==4&&dbp.size(0)>0&&dbp.sizes().slice(1)==at::IntArrayRef({4,padded,padded}),"original BF partial[G,4,N,N]");
  TORCH_CHECK(dz.device()==z.device()&&dz.scalar_type()==z.scalar_type()&&dz.is_contiguous()&&dz.sizes()==z.sizes(),"same original dz dtype/shape");
  const int64_t blocks=(padded/64)*(padded/32);TORCH_CHECK(pw.device()==z.device()&&pw.scalar_type()==at::kFloat&&pw.is_contiguous()&&pw.sizes()==at::IntArrayRef({blocks,4,16}),"original F32 parameterpartials[blocks,4,16]");
  TORCH_CHECK(mask.device()==z.device()&&mask.scalar_type()==at::kBool&&mask.is_contiguous()&&(mask.numel()==0||mask.numel()==length),"original empty/bool[L] key mask");
  const c10::cuda::CUDAGuard guard(z.device());const dim3 grid(padded/64,padded/32);auto stream=at::cuda::getCurrentCUDAStream();const auto *kv=mask.numel()?reinterpret_cast<const unsigned char*>(mask.data_ptr<bool>()):nullptr;
  if(z.scalar_type()==at::kBFloat16)pair_bias_bwd_from_partials_kernel<__nv_bfloat16><<<grid,256,0,stream>>>((const __nv_bfloat16*)z.data_ptr(),w.data_ptr<float>(),(const __nv_bfloat16*)dbp.data_ptr(),(__nv_bfloat16*)dz.data_ptr(),pw.data_ptr<float>(),padded,length,dbp.size(0),kv,eps);
  else pair_bias_bwd_from_partials_kernel<float><<<grid,256,0,stream>>>(z.data_ptr<float>(),w.data_ptr<float>(),(const __nv_bfloat16*)dbp.data_ptr(),dz.data_ptr<float>(),pw.data_ptr<float>(),padded,length,dbp.size(0),kv,eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<typename T>std::vector<int64_t> resource() {
 cudaFuncAttributes a;int blocks;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,pair_bias_bwd_from_partials_kernel<T>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,pair_bias_bwd_from_partials_kernel<T>,256,0));const auto*d=at::cuda::getCurrentDeviceProperties();return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,256,blocks,blocks*8,d->maxThreadsPerMultiProcessor,d->multiProcessorCount,a.binaryVersion,a.ptxVersion};
}
std::vector<int64_t> resources(bool fp32) {return fp32?resource<float>():resource<__nv_bfloat16>();}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_pair_reduce210::run);m.def("resources",&atom_pair_reduce210::resources);}
