// gemm_sm120.cuh -- RTX 5090 (sm_120a build of this extension only, -DMWE_SM120): the bias-only token DiT block's inference GEMMs with
// their row passes fused into the epilogue. acc = A W^T, A [M, K] and W [N, K] row-major (K contiguous: a Linear's activation and weight);
// W is read as the virtual concatenation [W1; W2] (rows < N1 from W1), so no packed weight is copied.
//
//   EPI_PLAIN   C[m, n] = acc (+ bias[n])                                   (the [v | g] projection, the gate logits)
//   EPI_GATERES C[m, n] = R[m, n] + sigmoid(G[m % P, n]) acc                 (the residual gates: attention out, transition out)
//   EPI_ADALN   C[m, c] = sigmoid(S + sb[c]) (X[m, c] - mean) rstd + B       (the AdaLN: S, B = the scale / bias columns c of aff [Ws; Wb]^T)
//   EPI_SWIGLU  C[m, c] = silu(a) b                                          (the transition's expansion: a, b = columns c of xa [Wa; Wb]^T)
//
// The paired epilogues (ADALN, SWIGLU) give a CTA the 64 output columns c0 .. c0 + 63 as B-tile rows 0 .. 63 = W1[c0 ..] and 64 .. 127 =
// W2[c0 ..]; warp wn of the 4 along N takes rows wn*16 .. of each half, so a thread holds both operands of its output columns.
// The products and the epilogue in fp32, one rounding to the output dtype (the unfused path rounds the GEMM output first).
//
// T = bf16: mma.sync m16n8k16. T = float: TF32 m16n8k8 with the operands rounded to TF32 (cvt.rna) as they reach the registers.
// CTA tile BM x BN, K-blocks of RB = 128 or 64 bytes per row, an ST-deep cp.async ring, 16-byte chunks XOR-swizzled (swz).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef G120_RNA
#define G120_RNA 1  // TF32: round the operands to nearest (0: the hardware's truncation; A/B only, less accurate than cuBLAS)
#endif

namespace g120 {

enum { EPI_PLAIN = 0, EPI_GATERES = 1, EPI_ADALN = 2, EPI_SWIGLU = 3, EPI_SWIGLU_BWD = 4 };

template <typename T> struct Params {
  const T* A; const T* W1; const T* W2; int N1;
  T* C; long long ldc; int M, N, K;                         // N: the columns of C (paired: the output columns)
  const T* bias;                                            // PLAIN
  const T* R; const T* G; long long P;                      // GATERES (R may be null)
  const T* X; const float2* xst; const T* sb;               // ADALN
  const T* AB; T* H;                                        // SWIGLU_BWD: [a | b] [M, 2 N]; C = [da | db] (ldc 2 N), H = silu(a) b [M, N]
};

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ unsigned rna(unsigned x) {
  unsigned y;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(__uint_as_float(x)));
  return y;
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
__device__ __forceinline__ float2 load2(const __nv_bfloat16* p) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p)); }
__device__ __forceinline__ float2 load2(const float* p) { return *reinterpret_cast<const float2*>(p); }
__device__ __forceinline__ float sig(float x) { return 1.f / (1.f + __expf(-x)); }

template <int BM, int BN, int ST, int RB> constexpr int smem_bytes() { return ST * (BM + BN) * RB; }

// the 16-byte chunk ch of row r of a K-block tile with RB-byte rows, XOR-swizzled so that the 8 rows an ldmatrix reads hit distinct banks
template <int RB> __device__ __forceinline__ int swz(int r, int ch) { return RB == 128 ? ch ^ (r & 7) : ch ^ ((r >> 1) & 3); }

template <typename T, int EPI, int BM, int BN, int WM, int WN, int ST, int RB, int MINB>
__global__ void __launch_bounds__(WM * WN * 32, MINB)
gemm(const Params<T> p) {
  constexpr bool PAIRED = EPI == EPI_ADALN || EPI == EPI_SWIGLU;
  constexpr int NTH = WM * WN * 32, MT = BM / WM / 16, NT = BN / WN / 8, CPR = 16 / sizeof(T);   // CPR: elements per 16-byte chunk
  constexpr int CH = RB / 16;                                   // 16-byte chunks per tile row (RB: 128 or 64 bytes of K a K-block)
  static_assert(RB == 128 || RB == 64, "K-block rows of 128 or 64 bytes");
  static_assert(NT % 2 == 0, "n-tiles in pairs (ldmatrix.x4)");
  static_assert(!PAIRED || (BN == 128 && WN == 4), "the paired epilogues: 64 output columns a CTA over 4 warps");
  extern __shared__ __align__(128) unsigned char sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int wm = warp % WM, wn = warp / WM;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * (PAIRED ? 64 : BN);
  const int K = p.K, nkb = K / (CH * CPR);
  auto wrow = [&](int r) -> const T* {                      // B-tile row r: a row of [W1; W2]
    if (PAIRED) return r < 64 ? p.W1 + (size_t)(n0 + r) * K : p.W2 + (size_t)(n0 + r - 64) * K;
    const int n = n0 + r;
    return n < p.N1 ? p.W1 + (size_t)n * K : p.W2 + (size_t)(n - p.N1) * K;
  };
  auto load = [&](int kb, int s) {
    unsigned char* as = sm + s * (BM + BN) * RB;
    unsigned char* bs = as + BM * RB;
    const int k0 = kb * CH * CPR;
#pragma unroll
    for (int c = tid; c < BM * CH; c += NTH) {
      const int r = c / CH, ch = c % CH;
      cp16(as + r * RB + (swz<RB>(r, ch) << 4), p.A + (size_t)(m0 + r) * K + k0 + ch * CPR);
    }
#pragma unroll
    for (int c = tid; c < BN * CH; c += NTH) {
      const int r = c / CH, ch = c % CH;
      cp16(bs + r * RB + (swz<RB>(r, ch) << 4), wrow(r) + k0 + ch * CPR);
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
    const unsigned as = base + (kb % ST) * (BM + BN) * RB, bs = as + BM * RB;
#pragma unroll
    for (int ks = 0; ks < RB / 32; ks++) {                    // 32 bytes of every row a k-step
      unsigned af[MT][4], bf[NT / 2][4];
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int r = wm * (BM / WM) + i * 16 + (lane & 15), ch = 2 * ks + (lane >> 4);
        ldsm4(af[i], as + r * RB + (swz<RB>(r, ch) << 4));
      }
#pragma unroll
      for (int jp = 0; jp < NT / 2; jp++) {
        const int rb = PAIRED ? jp * 64 + wn * 16 : wn * (BN / WN) + jp * 16;
        const int r = rb + (lane & 7) + ((lane >> 4) << 3), ch = 2 * ks + ((lane >> 3) & 1);
        ldsm4(bf[jp], bs + r * RB + (swz<RB>(r, ch) << 4));
      }
      if constexpr (sizeof(T) == 4 && G120_RNA) {
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 4; e++) af[i][e] = rna(af[i][e]);
#pragma unroll
        for (int jp = 0; jp < NT / 2; jp++)
#pragma unroll
          for (int e = 0; e < 4; e++) bf[jp][e] = rna(bf[jp][e]);
      }
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) mma<T>(acc[i][j], af[i], bf[j / 2][(j & 1) * 2], bf[j / 2][(j & 1) * 2 + 1]);
    }
  }
  wait<0>();
  if constexpr (PAIRED) {
    // n-tiles 0, 1: the W1 columns (scale / a), 2, 3: the same W2 columns (bias / b)
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      const int col = n0 + wn * 16 + jj * 8 + 2 * tig;
      float2 sbv = make_float2(0.f, 0.f);
      if constexpr (EPI == EPI_ADALN) sbv = load2(p.sb + col);
#pragma unroll
      for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int row = m0 + wm * (BM / WM) + i * 16 + gid + 8 * h;
          const float s0 = acc[i][jj][2 * h], s1 = acc[i][jj][2 * h + 1], b0 = acc[i][2 + jj][2 * h], b1 = acc[i][2 + jj][2 * h + 1];
          float y0, y1;
          if constexpr (EPI == EPI_ADALN) {
            const float2 st = p.xst[row];
            const float2 xv = load2(p.X + (size_t)row * p.N + col);
            y0 = sig(s0 + sbv.x) * ((xv.x - st.x) * st.y) + b0;
            y1 = sig(s1 + sbv.y) * ((xv.y - st.x) * st.y) + b1;
          } else {
            y0 = s0 * sig(s0) * b0;
            y1 = s1 * sig(s1) * b1;
          }
          store2(p.C + (size_t)row * p.ldc + col, y0, y1);
        }
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < NT; j++) {
      const int col = n0 + wn * (BN / WN) + j * 8 + 2 * tig;
      float2 bv = make_float2(0.f, 0.f);
      if constexpr (EPI == EPI_PLAIN) if (p.bias != nullptr) bv = load2(p.bias + col);
#pragma unroll
      for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int row = m0 + wm * (BM / WM) + i * 16 + gid + 8 * h;
          float y0 = acc[i][j][2 * h], y1 = acc[i][j][2 * h + 1];
          if constexpr (EPI == EPI_GATERES) {
            const float2 g = load2(p.G + (size_t)(row % p.P) * p.N + col);
            const float2 r = p.R != nullptr ? load2(p.R + (size_t)row * p.N + col) : make_float2(0.f, 0.f);
            y0 = fmaf(sig(g.x), y0, r.x);
            y1 = fmaf(sig(g.y), y1, r.y);
          } else if constexpr (EPI == EPI_SWIGLU_BWD) {   // acc = dh rounded to T first (the unfused path's dh, so the gradients are its values): da = dh b s (1 + a (1 - s)), db = dh a s, h = a s b
            if constexpr (sizeof(T) == 2) { y0 = __bfloat162float(__float2bfloat16_rn(y0)); y1 = __bfloat162float(__float2bfloat16_rn(y1)); }
            const float2 av = load2(p.AB + (size_t)row * 2 * p.N + col), bv2 = load2(p.AB + (size_t)row * 2 * p.N + p.N + col);
            const float sa0 = sig(av.x), sa1 = sig(av.y);
            store2(p.C + (size_t)row * p.ldc + col, y0 * bv2.x * sa0 * (1.f + av.x * (1.f - sa0)), y1 * bv2.y * sa1 * (1.f + av.y * (1.f - sa1)));
            store2(p.C + (size_t)row * p.ldc + p.N + col, y0 * (av.x * sa0), y1 * (av.y * sa1));
            store2(p.H + (size_t)row * p.N + col, (av.x * sa0) * bv2.x, (av.y * sa1) * bv2.y);
            continue;
          } else {
            y0 += bv.x;
            y1 += bv.y;
          }
          store2(p.C + (size_t)row * p.ldc + col, y0, y1);
        }
      }
    }
  }
}

// mean / rstd of every row of X [M, D] (the AdaLN's LayerNorm, no affine): a warp a row
template <typename T, int D>
__global__ void __launch_bounds__(256) row_stats(const T* __restrict__ X, float2* __restrict__ st, int M, float eps) {
  constexpr int PER = D / 64;                               // element pairs per lane
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= M) return;
  float2 v[PER];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < PER; k++) { v[k] = load2(X + (size_t)row * D + 2 * (k * 32 + lane)); s += v[k].x + v[k].y; }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / D;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < PER; k++) { const float a = v[k].x - mean, b = v[k].y - mean; q += a * a + b * b; }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  if (lane == 0) st[row] = make_float2(mean, rsqrtf(q / D + eps));
}

}  // namespace g120
