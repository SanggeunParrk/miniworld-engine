// gemm_sm120.cuh -- RTX 5090 (sm_120a build of this extension only, -DMWE_SM120): C = A W^T (+ bias) for the bias-only token DiT block's
// inference GEMMs, A [M, K] and W [N, K] row-major (K contiguous: a Linear's activation and weight), C [M, N] row-major with any leading
// dimension). bf16 operands (mma.sync m16n8k16, fp32 accumulation, one rounding to bf16; the template's fp32 / TF32 form, m16n8k8 on the
// operands' bits as the hardware truncates them, is not instantiated: it was less accurate than cuBLAS's TF32). cuBLAS picks 32 x 32 wmma
// kernels for several of these shapes on sm_120; bias_only_dit_sm80.GEMM_PICK routes the measured ones here.
//
// CTA tile BM x BN, K-blocks of 128 bytes per row (64 bf16 / 32 fp32), STAGES-deep cp.async ring; each 128-byte row's 16-byte chunks
// XOR-swizzled by (row & 7) (conflict-free ldmatrix). Warp grid WM x WN; a k-step reads 32 bytes of every row (A: ldmatrix.x4 per 16-row
// m-tile; B: ldmatrix.x4 per two 8-column n-tiles, W rows being the B operand's columns). Requires M % BM == N % BN == 0, K % (K-block) == 0.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace g120 {

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
template <typename T> __device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1);
template <> __device__ __forceinline__ void mma<__nv_bfloat16>(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <> __device__ __forceinline__ void mma<float>(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void store2(__nv_bfloat16* p, float x, float y) { *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(x, y); }
__device__ __forceinline__ void store2(float* p, float x, float y) { *reinterpret_cast<float2*>(p) = make_float2(x, y); }
__device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float tof(float v) { return v; }

template <int BM, int BN, int ST> constexpr int smem_bytes() { return ST * (BM + BN) * 128; }

template <typename T, int BM, int BN, int WM, int WN, int ST>
__global__ void __launch_bounds__(WM * WN * 32)
gemm(const T* __restrict__ A, const T* __restrict__ W, const T* __restrict__ bias, T* __restrict__ Cp, int M, int N, int K, long long ldc) {
  constexpr int NTH = WM * WN * 32, MT = BM / WM / 16, NT = BN / WN / 8, EPR = 128 / sizeof(T);   // EPR: elements per 128-byte row
  static_assert(NT % 2 == 0, "n-tiles in pairs (ldmatrix.x4)");
  extern __shared__ __align__(128) unsigned char sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int wm = warp % WM, wn = warp / WM;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int nkb = K / EPR;
  auto load = [&](int kb, int s) {
    unsigned char* as = sm + s * (BM + BN) * 128;
    unsigned char* bs = as + BM * 128;
    const int k0 = kb * EPR;
#pragma unroll
    for (int c = tid; c < BM * 8; c += NTH) {
      const int r = c >> 3, ch = c & 7;
      cp16(as + r * 128 + ((ch ^ (r & 7)) << 4), A + (size_t)(m0 + r) * K + k0 + ch * (16 / sizeof(T)));
    }
#pragma unroll
    for (int c = tid; c < BN * 8; c += NTH) {
      const int r = c >> 3, ch = c & 7;
      cp16(bs + r * 128 + ((ch ^ (r & 7)) << 4), W + (size_t)(n0 + r) * K + k0 + ch * (16 / sizeof(T)));
    }
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < nkb) load(s, s);
    commit();
  }
  const unsigned base = (unsigned)__cvta_generic_to_shared(sm);
  for (int kb = 0; kb < nkb; kb++) {
    wait<ST - 2>();
    __syncthreads();
    if (kb + ST - 1 < nkb) load(kb + ST - 1, (kb + ST - 1) % ST);
    commit();
    const unsigned as = base + (kb % ST) * (BM + BN) * 128, bs = as + BM * 128;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {                          // 32 bytes of every row a k-step
      unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int r = wm * (BM / WM) + i * 16 + (lane & 15), ch = 2 * ks + (lane >> 4);
        ldsm4(af[i], as + r * 128 + ((ch ^ (r & 7)) << 4));
      }
#pragma unroll
      for (int j = 0; j < NT / 2; j++) {
        const int r = wn * (BN / WN) + j * 16 + (lane & 7) + ((lane >> 4) << 3), ch = 2 * ks + ((lane >> 3) & 1);
        ldsm4(bf[j], bs + r * 128 + ((ch ^ (r & 7)) << 4));
      }
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) mma<T>(acc[i][j], af[i], bf[j / 2][(j & 1) * 2], bf[j / 2][(j & 1) * 2 + 1]);
    }
  }
  wait<0>();
#pragma unroll
  for (int j = 0; j < NT; j++) {
    const int col = n0 + wn * (BN / WN) + j * 8 + 2 * tig;
    const float b0 = bias != nullptr ? tof(bias[col]) : 0.f, b1 = bias != nullptr ? tof(bias[col + 1]) : 0.f;
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const int row = m0 + wm * (BM / WM) + i * 16 + gid;
      store2(Cp + (size_t)row * ldc + col, acc[i][j][0] + b0, acc[i][j][1] + b1);
      store2(Cp + (size_t)(row + 8) * ldc + col, acc[i][j][2] + b0, acc[i][j][3] + b1);
    }
  }
}

}  // namespace g120
