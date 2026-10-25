// Pair-representation builder of La-Proteina's denoiser on sm_90a (H100), forward and backward; 256 channels.
//
//   pair[b,i,j,:] = mask_ij * ( LN(h) * gamma[b] + beta[b] ),   h = LN_affine( T[sep] + T[nS + dist_xt] + T[nS + nX + dist_sc] )
//
// The reference builds three binned one-hots per pair (127 + 30 + 30 channels, fp32), a bias-free Linear over their concatenation,
// an affine LayerNorm, a per-pair conditioning path and an adaptive LayerNorm.  A one-hot times a matrix is a row gather, and the
// conditioning (diffusion times) is constant over (i, j), so gamma / beta are per sample [B][256].  tbl [NT][256] = Linear weight^T.
//
//   forward   one warp per (b, i) row, 32 pairs at a time: lane l bins pair (i, j0 + l) (sequence separation, CA distances),
//             the packed bin word goes to `bins` (saved for the backward) and to the warp by shuffle; per pair the lane gathers
//             8 channels of the three table rows, runs both LayerNorms with warp shuffles and stores 16 B of bf16.
//   backward  BITWISE DETERMINISTIC (no atomics).  rows kernel: recomputes the pair, writes dx (grad of the table sum, bf16) and
//             per-CTA partials of dgamma / dbeta / dln_w / dln_b; table kernel: a CTA owns a quarter of the channels of a slice
//             of the pairs, its four warps accumulate into private shared-memory bin rows and are added in fixed order;
//             reduce kernels add the partials in fixed order.  The batch is processed in chunks so dx stays small.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int C = 256;
constexpr int NW = 8;                       // warps per rows-kernel CTA = rows per CTA
constexpr int NBLK = 33;                    // table kernel: CTAs per channel quarter
#ifndef TU
#define TU 4
#endif
constexpr uint32_t MASKED = 1u << 31;
constexpr uint32_t HAS_SC = 1u << 24;

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void ld8(const float* p, float* o) {
  float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void st8(__nv_bfloat16* p, const float* v) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; k++) h[k] = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void ldb8(const __nv_bfloat16* p, float* o) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; k++) { float2 f = __bfloat1622float2(h[k]); o[2 * k] = f.x; o[2 * k + 1] = f.y; }
}

// bins of pair (i, j) as torch.bucketize would give them: the number of limits strictly below the value
__device__ __forceinline__ int bucket(float d, const float* lim, int nlim) {
  int c = 0;
  for (int k = 0; k < nlim; k++) c += (__ldg(lim + k) < d);
  return c;
}
__device__ __forceinline__ float dist3(const float* x, int b, int n, int i, int j) {
  const float* p = x + ((size_t)b * n + i) * 3; const float* q = x + ((size_t)b * n + j) * 3;
  float a = p[0] - q[0], bb = p[1] - q[1], c = p[2] - q[2];
  return sqrtf(a * a + bb * bb + c * c);
}
__device__ __forceinline__ uint32_t bin_word(int b, int i, int j, int n, const int* idx, const uint8_t* mask, const float* xt,
                                              const float* sc, int has_sc, const float* xl, int nxl, const float* cl, int ncl, int nS) {
  if (!(mask[b * n + i] && mask[b * n + j])) return MASKED;
  int s = idx[b * n + i] - idx[b * n + j] + (nS - 1) / 2;
  s = min(max(s, 0), nS - 1);
  uint32_t w = (uint32_t)s | ((uint32_t)bucket(dist3(xt, b, n, i, j), xl, nxl) << 8);
  if (has_sc) w |= HAS_SC | ((uint32_t)bucket(dist3(sc, b, n, i, j), cl, ncl) << 16);
  return w;
}

struct Fw { float yh1[8], yh2[8], r1, r2; };

__device__ __forceinline__ void fw(uint32_t w, const float* tbl, int c0, const float* lw, const float* lb, int nS, int nX, float e1, float e2, Fw& f) {
  float x[8], t[8];
  ld8(tbl + (size_t)(w & 255) * C + c0, x);
  ld8(tbl + (size_t)(nS + ((w >> 8) & 255)) * C + c0, t);
#pragma unroll
  for (int e = 0; e < 8; e++) x[e] += t[e];
  if (w & HAS_SC) {
    ld8(tbl + (size_t)(nS + nX + ((w >> 16) & 255)) * C + c0, t);
#pragma unroll
    for (int e = 0; e < 8; e++) x[e] += t[e];
  }
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) s += x[e];
  float mean = wsum(s) * (1.f / C), q = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) { x[e] -= mean; q += x[e] * x[e]; }
  f.r1 = rsqrtf(wsum(q) * (1.f / C) + e1);
  float y[8]; s = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) { f.yh1[e] = x[e] * f.r1; y[e] = f.yh1[e] * lw[e] + lb[e]; s += y[e]; }
  mean = wsum(s) * (1.f / C); q = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) { y[e] -= mean; q += y[e] * y[e]; }
  f.r2 = rsqrtf(wsum(q) * (1.f / C) + e2);
#pragma unroll
  for (int e = 0; e < 8; e++) f.yh2[e] = y[e] * f.r2;
}

__global__ void __launch_bounds__(NW * 32)
pe_fwd(const float* tbl, const float* lnw, const float* lnb, const float* gamma, const float* beta, const float* xt, const float* sc,
       int has_sc, const int* idx, const uint8_t* mask, const float* xl, int nxl, const float* cl, int ncl, __nv_bfloat16* out,
       int* bins, int B, int n, int nS, int nX, float e1, float e2) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * NW + warp;
  if (row >= B * n) return;
  int b = row / n, i = row % n, c0 = lane * 8;
  float lw[8], lb[8], gm[8], bt[8];
  ld8(lnw + c0, lw); ld8(lnb + c0, lb); ld8(gamma + (size_t)b * C + c0, gm); ld8(beta + (size_t)b * C + c0, bt);
  __nv_bfloat16* orow = out + (size_t)row * n * C;
  for (int j0 = 0; j0 < n; j0 += 32) {
    int j = j0 + lane;
    uint32_t w = j < n ? bin_word(b, i, j, n, idx, mask, xt, sc, has_sc, xl, nxl, cl, ncl, nS) : MASKED;
    if (j < n) bins[(size_t)row * n + j] = w;
    int cnt = min(32, n - j0);
    for (int t = 0; t < cnt; t++) {
      uint32_t ww = __shfl_sync(0xffffffffu, w, t);
      float o[8];
      if (ww & MASKED) {
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = 0.f;
      } else {
        Fw f;
        fw(ww, tbl, c0, lw, lb, nS, nX, e1, e2, f);
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = f.yh2[e] * gm[e] + bt[e];
      }
      st8(orow + (size_t)(j0 + t) * C + c0, o);
    }
  }
}

// grad of the table sum per pair + per-CTA partials [dgamma | dbeta | dln_w | dln_b] (4 x 256 floats)
__global__ void __launch_bounds__(NW * 32)
pe_bwd_rows(const float* tbl, const float* lnw, const float* lnb, const float* gamma, const int* bins, const __nv_bfloat16* dout,
            __nv_bfloat16* dx, float* part, int b0, int n, int nrb, int nS, int nX, float e1, float e2) {
  __shared__ float sm[NW * 4 * C];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  int b = b0 + blockIdx.x / nrb, rb = blockIdx.x % nrb, i = rb * NW + warp;
  float lw[8], lb[8], gm[8], dg[8], dbt[8], dw[8], dl[8];
  ld8(lnw + c0, lw); ld8(lnb + c0, lb); ld8(gamma + (size_t)b * C + c0, gm);
#pragma unroll
  for (int e = 0; e < 8; e++) { dg[e] = dbt[e] = dw[e] = dl[e] = 0.f; }
  if (i < n) {
    size_t prow = ((size_t)b * n + i) * n;
    for (int j = 0; j < n; j++) {
      uint32_t w = bins[prow + j];
      float o[8];
      if (w & MASKED) {
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = 0.f;
      } else {
        Fw f; float g[8];
        fw(w, tbl, c0, lw, lb, nS, nX, e1, e2, f);
        ldb8(dout + (prow + j) * C + c0, g);
        float d2[8], p2 = 0.f, a2 = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          dbt[e] += g[e]; dg[e] += g[e] * f.yh2[e];
          d2[e] = g[e] * gm[e]; p2 += d2[e]; a2 += d2[e] * f.yh2[e];
        }
        p2 = wsum(p2) * (1.f / C); a2 = wsum(a2) * (1.f / C);
        float d1[8], p1 = 0.f, a1 = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          float dy = f.r2 * (d2[e] - p2 - f.yh2[e] * a2);
          dl[e] += dy; dw[e] += dy * f.yh1[e];
          d1[e] = dy * lw[e]; p1 += d1[e]; a1 += d1[e] * f.yh1[e];
        }
        p1 = wsum(p1) * (1.f / C); a1 = wsum(a1) * (1.f / C);
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = f.r1 * (d1[e] - p1 - f.yh1[e] * a1);
      }
      st8(dx + (((size_t)(blockIdx.x / nrb) * n + i) * n + j) * C + c0, o);   // dx [Bc][n][n][C], chunk-relative
    }
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    sm[(warp * 4 + 0) * C + c0 + e] = dg[e]; sm[(warp * 4 + 1) * C + c0 + e] = dbt[e];
    sm[(warp * 4 + 2) * C + c0 + e] = dw[e]; sm[(warp * 4 + 3) * C + c0 + e] = dl[e];
  }
  __syncthreads();
  size_t cta = blockIdx.x;                                   // the host offsets `part` to this chunk's first CTA
  for (int k = threadIdx.x; k < 4 * C; k += NW * 32) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) s += sm[w * 4 * C + k];
    part[cta * 4 * C + k] = s;
  }
}

// table gradient: CTA (q, t) = channel quarter q of pair slice t; warps w = 0..3 take pairs p0 + w, p0 + w + 4, ...
__global__ void __launch_bounds__(128)
pe_bwd_tbl(const __nv_bfloat16* dx, const int* bins, long long P, int nS, int nX, int NT, float* tpart) {
  extern __shared__ float acc[];                            // [4 warps][NT][64]
  int q = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int k = threadIdx.x; k < 4 * NT * 64; k += 128) acc[k] = 0.f;
  __syncthreads();
  float* my = acc + (size_t)w * NT * 64 + lane * 2;
  long long per = (P + NBLK - 1) / NBLK, p0 = (long long)t * per, p1 = min(P, p0 + per);
  for (long long p = p0 + w; p < p1; p += 4 * TU) {            // TU pairs' loads in flight, then added in pair order (same sums as one at a time)
    uint32_t wd[TU];
    float2 f[TU];
#pragma unroll
    for (int u = 0; u < TU; u++) {
      long long pp = p + 4 * u;
      wd[u] = pp < p1 ? (uint32_t)bins[pp] : MASKED;
      f[u] = make_float2(0.f, 0.f);
      if (!(wd[u] & MASKED)) f[u] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(dx + (size_t)pp * C + q * 64 + lane * 2));
    }
#pragma unroll
    for (int u = 0; u < TU; u++) {
      if (wd[u] & MASKED) continue;
      float2* a0 = reinterpret_cast<float2*>(my + (size_t)(wd[u] & 255) * 64);
      float2* a1 = reinterpret_cast<float2*>(my + (size_t)(nS + ((wd[u] >> 8) & 255)) * 64);
      a0->x += f[u].x; a0->y += f[u].y; a1->x += f[u].x; a1->y += f[u].y;
      if (wd[u] & HAS_SC) {
        float2* a2 = reinterpret_cast<float2*>(my + (size_t)(nS + nX + ((wd[u] >> 16) & 255)) * 64);
        a2->x += f[u].x; a2->y += f[u].y;
      }
    }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < NT * 64; k += 128) {
    float s = acc[k] + acc[NT * 64 + k] + acc[2 * NT * 64 + k] + acc[3 * NT * 64 + k];
    tpart[((size_t)t * NT + k / 64) * C + q * 64 + k % 64] = s;
  }
}


// ---- table gradient on the tensor cores (wgmma) --------------------------------------------------------------------------------------------------
// dT[row] = sum over the pairs that use the row of dx[pair]  =  onehot^T @ dx.  The one-hot operand (rows x 64 pairs, bf16, 128 B rows, 16-byte chunks XOR-swizzled by
// the row) lives in shared memory and is written ("scattered") by two extra warps, 3 entries per pair; the dx tile arrives by cp.async; one warpgroup per 64 table rows
// issues wgmma.m64n64k16 on the two swizzled operands.  CTA (q, t) = channel quarter q of the FIXED pair slice t, tiles in order: bitwise deterministic, no atomics.
constexpr int TPB = 64, ST4 = 5, NWG4 = 3, SC4 = NWG4 * 128, NTH4 = SC4 + 64, NR4 = NWG4 * 64;
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  int sz = pred ? 16 : 0;                                   // src-size 0: zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint64_t wg_desc(unsigned smem_addr, unsigned lbo_bytes, unsigned sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo_bytes & 0x3FFFFu) >> 4) << 16;
  d |= (uint64_t)((sbo_bytes & 0x3FFFFu) >> 4) << 32;
  d |= (uint64_t)1 << 62;                                   // 128-byte swizzle
  return d;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_m64n64k16(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(da), "l"(db));
}
__device__ __forceinline__ void wg_fence_acc(float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; i++) asm volatile("" : "+f"(d[i])::"memory");
}

__device__ __forceinline__ void st_u16(unsigned addr, unsigned v) { asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(addr), "r"(v)); }

__global__ void __launch_bounds__(NTH4, 1)
pe_bwd_tbl4(const __nv_bfloat16* dx, const int* bins, long long P, int nS, int nX, int NT, float* tpart) {
  extern __shared__ __align__(1024) unsigned char dsm0[];
  unsigned char* dsm = dsm0 + ((1024u - ((unsigned)__cvta_generic_to_shared(dsm0) & 1023u)) & 1023u);
  unsigned char* Bs = dsm;
  unsigned char* Ab = dsm + ST4 * TPB * 128;
  const unsigned as_base = (unsigned)__cvta_generic_to_shared(Ab), bs_base = (unsigned)__cvta_generic_to_shared(Bs);
  const int q = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const long long per = (P + NBLK - 1) / NBLK, p0 = (long long)t * per, p1 = min(P, p0 + per);
  const int ntiles = p1 > p0 ? (int)((p1 - p0 + TPB - 1) / TPB) : 0;
  const int wuni = __shfl_sync(0xffffffffu, tid >> 5, 0);        // uniform warp index: keeps the wgmma out of a 'divergent' path for ptxas
  const bool scat = wuni >= NWG4 * 4;
  const int wg = wuni >> 2;
  for (int k = tid; k < 2 * NR4 * 8; k += NTH4) reinterpret_cast<uint4*>(Ab)[k] = make_uint4(0u, 0u, 0u, 0u);
  int prev[2][3];
#pragma unroll
  for (int b = 0; b < 2; b++)
#pragma unroll
    for (int k = 0; k < 3; k++) prev[b][k] = -1;
  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; i++) acc[i] = 0.f;
  auto load_tile = [&](int tile, int stage) {
    const long long pb = p0 + (long long)tile * TPB;
    for (int c = tid; c < TPB * 8; c += NTH4) {
      int kp = c >> 3, ch8 = c & 7;
      long long pair = pb + kp;
      bool pred = pair < p1;
      cp16(Bs + stage * (TPB * 128) + kp * 128 + ((ch8 ^ (kp & 7)) << 4), dx + (size_t)(pred ? pair : 0) * C + q * 64 + ch8 * 8, pred);
    }
  };
  auto word = [&](int tile) -> uint32_t {                    // scatter threads: the bin word of pair (tid - SC4) of `tile` (MASKED outside the slice)
    const long long pair = p0 + (long long)tile * TPB + (tid - SC4);
    return pair < p1 ? (uint32_t)bins[pair] : MASKED;
  };
  auto scatter = [&](int tile, uint32_t w) {
    if (!scat) return;
    const int p = tid - SC4, b = tile & 1;
    const unsigned buf = as_base + b * (NR4 * 128);
#pragma unroll
    for (int k = 0; k < 3; k++) {
      if (prev[b][k] >= 0) { const int r = prev[b][k]; st_u16(buf + r * 128 + (((p >> 3) ^ (r & 7)) << 4) + ((p & 7) << 1), 0u); }
    }
    int rows[3] = {-1, -1, -1};
    if (!(w & MASKED)) {
      rows[0] = (int)(w & 255);
      rows[1] = nS + (int)((w >> 8) & 255);
      if (w & HAS_SC) rows[2] = nS + nX + (int)((w >> 16) & 255);
    }
#pragma unroll
    for (int k = 0; k < 3; k++) {
      prev[b][k] = rows[k];
      if (rows[k] >= 0) { const int r = rows[k]; st_u16(buf + r * 128 + (((p >> 3) ^ (r & 7)) << 4) + ((p & 7) << 1), 0x3F80u); }
    }
    fence_async();
  };
#pragma unroll
  for (int s = 0; s < ST4 - 1; s++) {
    if (s < ntiles) load_tile(s, s);
    cp_commit();
  }
  uint32_t wn = scat ? word(0) : 0u;
  cp_wait<ST4 - 2>();
  fence_async();
  __syncthreads();
  if (ntiles > 0) scatter(0, wn);
  wn = scat ? word(1) : 0u;
  for (int kt = 0; kt < ntiles; kt++) {
    cp_wait<ST4 - 3>();
    fence_async();
    if (!scat) wg_wait<0>();
    __syncthreads();
    if (kt + 1 < ntiles) scatter(kt + 1, wn);
    if (scat) wn = word(kt + 2);
    if (kt + ST4 - 1 < ntiles) load_tile(kt + ST4 - 1, (kt + ST4 - 1) % ST4);
    cp_commit();
    if (scat) continue;
    const int stage = kt % ST4;
    const unsigned abase = as_base + (kt & 1) * (NR4 * 128) + wg * (64 * 128);
    const unsigned bbase = bs_base + stage * (TPB * 128);
    wg_fence_acc(acc);
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ks++) wgmma_m64n64k16(acc, wg_desc(abase + ks * 32, 16, 1024), wg_desc(bbase + ks * 16 * 128, 8192, 1024));
    wg_commit();
    wg_fence_acc(acc);
  }
  cp_wait<0>();
  if (scat) return;
  wg_wait<0>();
  wg_fence_acc(acc);
#pragma unroll
  for (int nt = 0; nt < 8; nt++) {
#pragma unroll
    for (int hi = 0; hi < 2; hi++) {
      const int r = 64 * wg + 16 * (warp & 3) + gid + 8 * hi;
      if (r >= NT) continue;
      *reinterpret_cast<float2*>(tpart + ((size_t)t * NT + r) * C + q * 64 + nt * 8 + 2 * tig) = make_float2(acc[4 * nt + 2 * hi], acc[4 * nt + 2 * hi + 1]);
    }
  }
}

// dgamma / dbeta [B][256] (sum over the CTAs of the sample), dln_w / dln_b [256] (sum over all CTAs), fixed order
__global__ void pe_reduce_rows(const float* part, float* dgamma, float* dbeta, float* dlw, float* dlb, int B, int nrb) {
  int c = threadIdx.x, b = blockIdx.x;
  float s0 = 0.f, s1 = 0.f;
  if (b < B) {
    for (int r = 0; r < nrb; r++) { const float* p = part + ((size_t)b * nrb + r) * 4 * C; s0 += p[c]; s1 += p[C + c]; }
    dgamma[(size_t)b * C + c] = s0; dbeta[(size_t)b * C + c] = s1;
  } else {
    for (size_t r = 0; r < (size_t)B * nrb; r++) { const float* p = part + r * 4 * C; s0 += p[2 * C + c]; s1 += p[3 * C + c]; }
    dlw[c] = s0; dlb[c] = s1;
  }
}

// dtbl [NT][256] = sum over the (chunk, CTA) partials in order
__global__ void pe_reduce_tbl(const float* tpart, float* dtbl, int nparts, int NT) {
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (size_t)NT * C) return;
  float s = 0.f;
  for (int r = 0; r < nparts; r++) s += tpart[(size_t)r * NT * C + k];
  dtbl[k] = s;
}

void forward(torch::Tensor tbl, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor gamma, torch::Tensor beta, torch::Tensor xt,
             torch::Tensor sc, int64_t has_sc, torch::Tensor idx, torch::Tensor mask, torch::Tensor xl, torch::Tensor cl,
             torch::Tensor out, torch::Tensor bins, int64_t nS, int64_t nX, double e1, double e2) {
  const at::cuda::CUDAGuard guard(tbl.device());
  int B = idx.size(0), n = idx.size(1);
  int grid = (B * n + NW - 1) / NW;
  pe_fwd<<<grid, NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      tbl.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      xt.data_ptr<float>(), sc.data_ptr<float>(), (int)has_sc, idx.data_ptr<int>(), mask.data_ptr<uint8_t>(), xl.data_ptr<float>(),
      (int)xl.numel(), cl.data_ptr<float>(), (int)cl.numel(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      bins.data_ptr<int>(), B, n, (int)nS, (int)nX, (float)e1, (float)e2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward(torch::Tensor tbl, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor gamma, torch::Tensor bins, torch::Tensor dout,
              torch::Tensor dx, torch::Tensor part, torch::Tensor tpart, torch::Tensor dtbl, torch::Tensor dgamma,
              torch::Tensor dbeta, torch::Tensor dlw, torch::Tensor dlb, int64_t impl, int64_t chunk_b, int64_t nS, int64_t nX, double e1, double e2) {
  const at::cuda::CUDAGuard guard(tbl.device());
  int B = bins.size(0), n = bins.size(1), NT = tbl.size(0);
  int nrb = (n + NW - 1) / NW;
  int nchunk = (B + chunk_b - 1) / chunk_b;
  auto st = at::cuda::getCurrentCUDAStream();
  size_t smem = (size_t)4 * NT * 64 * sizeof(float);
  cudaFuncSetAttribute(pe_bwd_tbl, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  auto* dxp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>());
  auto* dop = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>());
  for (int ch = 0; ch < nchunk; ch++) {
    int b0 = ch * chunk_b, Bc = std::min<int>(chunk_b, B - b0);
    pe_bwd_rows<<<Bc * nrb, NW * 32, 0, st>>>(tbl.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(), gamma.data_ptr<float>(),
                                              bins.data_ptr<int>(), dop, dxp, part.data_ptr<float>() + (size_t)b0 * nrb * 4 * C, b0, n, nrb,
                                              (int)nS, (int)nX, (float)e1, (float)e2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (impl == 1 && NT <= NR4) {
      const size_t sm4 = (size_t)ST4 * TPB * 128 + (size_t)2 * NR4 * 128 + 1024;
      cudaFuncSetAttribute(pe_bwd_tbl4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm4);
      pe_bwd_tbl4<<<dim3(4, NBLK), NTH4, sm4, st>>>(dxp, bins.data_ptr<int>() + (size_t)b0 * n * n, (long long)Bc * n * n, (int)nS, (int)nX, NT,
                                                    tpart.data_ptr<float>() + (size_t)ch * NBLK * NT * C);
    } else {
    pe_bwd_tbl<<<dim3(4, NBLK), 128, smem, st>>>(dxp, bins.data_ptr<int>() + (size_t)b0 * n * n, (long long)Bc * n * n, (int)nS, (int)nX, NT,
                                                 tpart.data_ptr<float>() + (size_t)ch * NBLK * NT * C);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  pe_reduce_rows<<<B + 1, C, 0, st>>>(part.data_ptr<float>(), dgamma.data_ptr<float>(), dbeta.data_ptr<float>(), dlw.data_ptr<float>(),
                                      dlb.data_ptr<float>(), B, nrb);
  pe_reduce_tbl<<<(NT * C + 255) / 256, 256, 0, st>>>(tpart.data_ptr<float>(), dtbl.data_ptr<float>(), nchunk * NBLK, NT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
}
