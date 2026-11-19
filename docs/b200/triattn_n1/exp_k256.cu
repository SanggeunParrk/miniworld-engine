// experiment: Bidirectional K=256 tail / gate-backward kernels with ablation + hoist macros (timing only for ABL_*)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) { f[0]=bf_lo(v.x); f[1]=bf_hi(v.x); f[2]=bf_lo(v.y); f[3]=bf_hi(v.y); f[4]=bf_lo(v.z); f[5]=bf_hi(v.z); f[6]=bf_lo(v.w); f[7]=bf_hi(v.w); }
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) { return make_uint4(pk(f[0],f[1]), pk(f[2],f[3]), pk(f[4],f[5]), pk(f[6],f[7])); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* p) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p))); }
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* p) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p))); }
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)); }

template <int K, bool SU>
__global__ void __launch_bounds__(256, 2) tail_x(const bf16* __restrict__ G, const bf16* __restrict__ O, const bf16* __restrict__ Wo, const bf16* __restrict__ X,
                                                 bf16* __restrict__ out, bf16* __restrict__ U, long LL, long ldg, long ldo) {
  constexpr int TM = 128, SP = 136, WS = K + 8, NK = K / 16, NS = 4, STG = 2 * 16 * SP;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sWo = reinterpret_cast<bf16*>(smraw); bf16* sT = sWo + 128 * WS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, ur = tid >> 4, uc = tid & 15;
  for (int i = tid; i < 128 * (K / 8); i += 256) { const int n = i / (K / 8), c = i % (K / 8); cp16(sWo + n * WS + c * 8, Wo + (long)n * K + c * 8); }
  cp_commit();
  const long ntiles = LL / TM;
  for (long tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const long tok0 = tile * TM;
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    auto issue = [&](int kc) {
#ifndef ABL_LD
      if (kc < NK) {
        bf16* g = sT + (kc % NS) * STG;
        cp16(g + ur * SP + uc * 8, G + (long)(kc * 16 + ur) * ldg + tok0 + uc * 8);
        cp16(g + 16 * SP + ur * SP + uc * 8, O + (long)(kc * 16 + ur) * ldo + tok0 + uc * 8);
      }
#endif
      cp_commit();
    };
    issue(0); issue(1); issue(2);
    for (int kc = 0; kc < NK; ++kc) {
      cp_wait<NS - 2>();
      bf16* g = sT + (kc % NS) * STG; bf16* o = g + 16 * SP;
      {
        float gv[8], ov[8], rv[8];
        unpack8(*reinterpret_cast<const uint4*>(g + ur * SP + uc * 8), gv);
        unpack8(*reinterpret_cast<const uint4*>(o + ur * SP + uc * 8), ov);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#ifdef ABL_MUFU
          rv[e] = gv[e] * ov[e];
#elif defined(TANH_SIG)
          { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * gv[e])); rv[e] = fmaf(0.5f, t, 0.5f) * ov[e]; }
#else
          rv[e] = (1.f / (1.f + __expf(-gv[e]))) * ov[e];
#endif
        }
        const uint4 pu = pack8(rv);
        *reinterpret_cast<uint4*>(o + ur * SP + uc * 8) = pu;
#ifndef ABL_ST
        if (SU) *reinterpret_cast<uint4*>(U + (long)(kc * 16 + ur) * LL + tok0 + uc * 8) = pu;
#endif
      }
      __syncthreads();
      issue(kc + NS - 1);
#ifndef ABL_MMA
      unsigned a[4];
      { const int mi = lane >> 3; ldsm4t(a, o + ((mi >> 1) * 8 + (lane & 7)) * SP + w * 16 + (mi & 1) * 8); }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        unsigned b[4]; const int mi = lane >> 3;
        ldsm4(b, sWo + (j * 16 + (mi >> 1) * 8 + (lane & 7)) * WS + kc * 16 + (mi & 1) * 8);
        mma16816(acc[2 * j], a, b[0], b[1]); mma16816(acc[2 * j + 1], a, b[2], b[3]);
      }
#endif
    }
    cp_wait<0>(); __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int c = j * 8 + 2 * (lane & 3), r0 = w * 16 + (lane >> 2);
      *reinterpret_cast<__nv_bfloat162*>(sT + r0 * SP + c) = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
      *reinterpret_cast<__nv_bfloat162*>(sT + (r0 + 8) * SP + c) = __floats2bfloat162_rn(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    for (int i = tid; i < TM * 16; i += 256) {
      const int r = i >> 4, ch = i & 15; float p[8], xf[8], q[8];
      unpack8(*reinterpret_cast<const uint4*>(sT + r * SP + ch * 8), p);
#ifndef ABL_LD
      unpack8(*reinterpret_cast<const uint4*>(X + (tok0 + r) * 128 + ch * 8), xf);
#else
      for (int e = 0; e < 8; ++e) xf[e] = 0.f;
#endif
#pragma unroll
      for (int e = 0; e < 8; ++e) q[e] = p[e] + xf[e];
#ifndef ABL_ST
      *reinterpret_cast<uint4*>(out + (tok0 + r) * 128 + ch * 8) = pack8(q);
#else
      if (q[0] == 123456.f) *reinterpret_cast<uint4*>(out + (tok0 + r) * 128 + ch * 8) = pack8(q);
#endif
    }
    __syncthreads();
  }
}

template <int K>
__global__ void __launch_bounds__(256, 2) gb_x(const bf16* __restrict__ DY, const bf16* __restrict__ Wo, const bf16* __restrict__ G, const bf16* __restrict__ O,
                                               bf16* __restrict__ dout, bf16* __restrict__ dg, long LL, long ldg, long ldo, long lddg) {
  constexpr int TMB = 64, SD = 136, WS = K + 8, MT = K / 128, WR = K / 8, BUF = 9216;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sWo = reinterpret_cast<bf16*>(smraw); bf16* sDy = sWo + 128 * WS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, q4 = lane & 3, g8 = lane >> 2, mi = lane >> 3;
  for (int i = tid; i < 128 * (K / 8); i += 256) { const int n = i / (K / 8), c = i % (K / 8); cp16(sWo + n * WS + c * 8, Wo + (long)n * K + c * 8); }
  cp_commit();
  const long ntiles = LL / TMB;
  auto issue = [&](long tile, int s) { for (int i = tid; i < TMB * 16; i += 256) { const int r = i >> 4, ch = i & 15; cp16(sDy + s * BUF + r * SD + ch * 8, DY + (tile * TMB + r) * 128 + ch * 8); } cp_commit(); };
  if ((long)blockIdx.x < ntiles) issue(blockIdx.x, 0); else cp_commit();
  int it = 0;
  for (long tile = blockIdx.x; tile < ntiles; tile += gridDim.x, ++it) {
    if (tile + gridDim.x < ntiles) issue(tile + gridDim.x, (it + 1) & 1); else cp_commit();
    const long tok0 = tile * TMB;
#ifdef HOIST
    uint4 gg[MT][4], oo[MT][4];
    {
      const long tk = tok0 + (lane & 7) * 8;
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) { const long row = w * WR + m * 16 + i * 4 + (lane >> 3); gg[m][i] = __ldcs(reinterpret_cast<const uint4*>(G + row * ldg + tk)); oo[m][i] = __ldcs(reinterpret_cast<const uint4*>(O + row * ldo + tk)); }
    }
#endif
    cp_wait<1>(); __syncthreads();
    const bf16* dy = sDy + (it & 1) * BUF;
    float acc[MT][8][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
#ifndef ABL_MMA
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      unsigned a[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) ldsm4t(a[m], sWo + (ks * 16 + (mi >> 1) * 8 + (lane & 7)) * WS + w * WR + m * 16 + (mi & 1) * 8);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned b[4]; ldsm4(b, dy + (j * 16 + (mi >> 1) * 8 + (lane & 7)) * SD + ks * 16 + (mi & 1) * 8);
#pragma unroll
        for (int m = 0; m < MT; ++m) { mma16816(acc[m][2 * j], a[m], b[0], b[1]); mma16816(acc[m][2 * j + 1], a[m], b[2], b[3]); }
      }
    }
#endif
    __syncthreads();
    bf16* stg = sDy + (it & 1) * BUF + w * 16 * 72;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = j * 8 + 2 * q4;
        *reinterpret_cast<__nv_bfloat162*>(stg + g8 * 72 + c) = __floats2bfloat162_rn(acc[m][j][0], acc[m][j][1]);
        *reinterpret_cast<__nv_bfloat162*>(stg + (g8 + 8) * 72 + c) = __floats2bfloat162_rn(acc[m][j][2], acc[m][j][3]);
      }
      __syncwarp();
      uint4 du[4];
#ifndef HOIST
      uint4 gg_[4], oo_[4];
#endif
      const long tk = tok0 + (lane & 7) * 8;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int rl = i * 4 + (lane >> 3); const long row = w * WR + m * 16 + rl;
#ifndef HOIST
#ifndef ABL_LD
        gg_[i] = *reinterpret_cast<const uint4*>(G + row * ldg + tk); oo_[i] = *reinterpret_cast<const uint4*>(O + row * ldo + tk);
#else
        gg_[i] = make_uint4(0,0,0,0); oo_[i] = gg_[i];
#endif
#endif
        du[i] = *reinterpret_cast<const uint4*>(stg + rl * 72 + (lane & 7) * 8);
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float a[8], gv[8], ovv[8], r[8], q[8];
        unpack8(du[i], a);
#ifdef HOIST
        unpack8(gg[m][i], gv); unpack8(oo[m][i], ovv);
#else
        unpack8(gg_[i], gv); unpack8(oo_[i], ovv);
#endif
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#ifdef ABL_MUFU
          const float s = gv[e];
#elif defined(TANH_SIG)
          float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * gv[e])); const float s = fmaf(0.5f, t, 0.5f);
#else
          const float s = 1.f / (1.f + __expf(-gv[e]));
#endif
          r[e] = a[e] * s; q[e] = a[e] * ovv[e] * s * (1.f - s);
        }
        const long row = w * WR + m * 16 + i * 4 + (lane >> 3);
#ifndef ABL_ST
        *reinterpret_cast<uint4*>(dout + row * LL + tk) = pack8(r);
        *reinterpret_cast<uint4*>(dg + row * lddg + tk) = pack8(q);
#else
        if (r[0] == 123456.f) { *reinterpret_cast<uint4*>(dout + row * LL + tk) = pack8(r); *reinterpret_cast<uint4*>(dg + row * lddg + tk) = pack8(q); }
#endif
      }
    }
    __syncthreads();
  }
}
}  // namespace

torch::Tensor tail_x_run(torch::Tensor g, torch::Tensor o, torch::Tensor wo, torch::Tensor x, torch::Tensor out, c10::optional<torch::Tensor> u) {
  const long K = wo.size(1), LL = g.size(1); const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = (int)std::min<long>(LL / 128, 2L * sms); bf16* up = u.has_value() ? reinterpret_cast<bf16*>(u->data_ptr<at::BFloat16>()) : nullptr;
  auto go = [&](auto kern, size_t smem) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kern<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>((const bf16*)g.data_ptr(), (const bf16*)o.data_ptr(), (const bf16*)wo.data_ptr(), (const bf16*)x.data_ptr(), (bf16*)out.data_ptr(), up, LL, g.stride(0), o.stride(0)); };
  const size_t sm = (size_t)(128 * (K + 8) + 4 * 2 * 16 * 136) * 2;
  if (K == 256) { if (up) go(tail_x<256, true>, sm); else go(tail_x<256, false>, sm); } else { if (up) go(tail_x<128, true>, sm); else go(tail_x<128, false>, sm); }
  return out;
}
torch::Tensor gb_x_run(torch::Tensor dy, torch::Tensor wo, torch::Tensor g, torch::Tensor o, torch::Tensor dg) {
  const long K = wo.size(1), LL = dy.size(0); const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto dout = torch::empty({K, LL}, dy.options()); const int grid = (int)std::min<long>(LL / 64, 2L * sms);
  auto go = [&](auto kern, size_t smem) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kern<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>((const bf16*)dy.data_ptr(), (const bf16*)wo.data_ptr(), (const bf16*)g.data_ptr(), (const bf16*)o.data_ptr(), (bf16*)dout.data_ptr(), (bf16*)dg.data_ptr(), LL, g.stride(0), o.stride(0), dg.stride(0)); };
  if (K == 256) go(gb_x<256>, (size_t)(128 * 264 + 2 * 9216) * 2); else go(gb_x<128>, (size_t)(128 * 136 + 2 * 9216) * 2);
  return dout;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tail_x", &tail_x_run); m.def("gb_x", &gb_x_run); }
