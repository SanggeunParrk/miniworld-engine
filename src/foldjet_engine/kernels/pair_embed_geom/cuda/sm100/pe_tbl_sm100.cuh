// pe_tbl_sm100.cuh -- the table gradient of the La-Proteina pair builders on sm_100a (B200): tcgen05 / TMEM / TMA.
//
//   dT[r][c] = sum over the pairs p that use table row r of  v_pr * G[p][c]      (v = 1, or the chain product for the chain row)
//
// is the product of the one-hot matrix F^T [rows x pairs] and the gradient G [pairs x 256]. On B200 it is issued as D = G^T F:
// M = 128 channels (TMEM lane = channel), N = the table rows (TMEM column = row: 317 rows are two N = 160 MMAs, 157 one, the
// denoiser's <= 224 one), K = 64 pairs per tile.
//
//   warp 0      TMA producer: the [64 pairs x 128 channels] G tile of this CTA's channel half (two 64-channel boxes, 128-byte swizzle:
//               exactly the MN-major operand layout), NS stages
//   warp 1      MMA issuer (one elected lane): A = G^T (MN-major), B = the one-hot tile (K-major), fp32 accumulators in TMEM
//   warps 2-3   scatter: one pair per thread, the pair's table rows written into a [rows x 64 pairs] bf16 tile (128-byte swizzle) and the
//               previous use of that buffer cleared entry by entry (NF buffers); the codes arrive by TMA with the G tile (same stage)
//   warps 4-7   epilogue: TMEM -> the slice's fp32 partial [rows][256] (a warp writes 32 consecutive channels of a row)
//
// Measured (B200, B 16 x L 512, CUDA graph, 10 % dead pairs, another process's matmul filler on the card): 737 us encoder table,
// 426 us decoder; the sm_90 mma.sync kernel built for sm_100a 5868 / 2662 us. With every pair live (plain launches) 474 / 422 us, and
// without the MMAs and the scatter (data movement alone) 339 / 326 us: what is left is mostly the tensor core reading the mostly-zero
// one-hot operand from shared memory (640 B a pair).
//
// CTA (h, t) = channel half h of the FIXED pair slice t, tiles in order; the slice partials are added in slice order by the reduce kernel:
// bitwise deterministic, no atomics. The tile beyond a slice's end reads real G rows of the next slice (or TMA zero fill past P) against
// zero one-hot columns.
#pragma once
#include "../../../integrations/csrc/sm100/sm100.cuh"

namespace pe100 {
constexpr int TP = 64;            // pairs per tile (K)
constexpr int MC = 128;           // channels per CTA (M)
#ifndef PE_NS
#define PE_NS 6
#endif
#ifndef PE_NF
#define PE_NF 2
#endif
constexpr int NS = PE_NS;         // G (+ codes) stages
constexpr int NF = PE_NF;         // one-hot buffers
constexpr int NTHR = 256;
constexpr int GST = TP * MC * 2;  // 16 KiB per G stage
constexpr int NBLK = 74;          // pair slices: 2 channel halves x 74 = 148 CTAs (one per SM)

// R: the row decoder of one builder
//   R::NPADMAX  rows of the one-hot buffer (a multiple of 16), R::TCOLS TMEM columns (power of two >= NPADMAX), R::MAXR rows a pair can use
//   R::Code, R::CB (its bytes: 16 = a [64][4] int32 TMA box of a 2-D map; 4 = one int32 word, copied by the producer warp's lanes, two a lane,
//   one tile ahead in registers -- a rank-1 tensor map faulted), R::DEAD (the code of no pair)
//   int rows(const Code&, int (&r)[MAXR], uint32_t (&v)[MAXR]) const   its rows and bf16 values (bits)
// the reduce of the slice partials inside the launch (FR, 2026-11-17): every CTA, after a grid barrier, sums its share of the columns over the
// slices in slice order into dw [256][NTAB] (each sum rounded to bf16): the adds and the rounding of reduce_parts_t<true>, the same bits, without
// the reduce launch (~2 us of graph node) and with all SMs reading the partials (the reduce kernel: 7.8 us for 12 MB on 79-157 CTAs). The launch
// is cooperative (every CTA resident: one per SM); bar [2] = (arrivals, generation), zero once, reusable by every launch on the device.
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned nct) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* gen = bar + 1;
    const unsigned g0 = *gen;
    __threadfence();
    if (atomicAdd(bar, 1u) == nct - 1) {
      atomicExch(bar, 0u);
      __threadfence();
      atomicAdd(bar + 1, 1u);
    } else {
      while (*gen == g0) __nanosleep(64);
    }
    __threadfence();
  }
  __syncthreads();
}
constexpr int FR_IN = 80;  // partials requested at once per column pair in the fused reduce (74 slices: one round trip)
__device__ __forceinline__ void reduce_cols(const float* __restrict__ part, float* __restrict__ dw, int n, int NTAB, unsigned cta, unsigned nct) {
  const long width = (long)NTAB * 256;
  for (long k = (long)cta * NTHR + threadIdx.x; 2 * k < width; k += (long)nct * NTHR) {   // columns 2 k, 2 k + 1
    float2 s = make_float2(0.f, 0.f);
    for (int r0 = 0; r0 < n; r0 += FR_IN) {
      float2 q[FR_IN];
#pragma unroll
      for (int u = 0; u < FR_IN; u++) q[u] = r0 + u < n ? __ldcg(reinterpret_cast<const float2*>(part + (size_t)(r0 + u) * width) + k) : make_float2(0.f, 0.f);
#pragma unroll
      for (int u = 0; u < FR_IN; u++) { s.x += q[u].x; s.y += q[u].y; }
    }
    const long row = 2 * k / 256, ch = 2 * k % 256;
    dw[ch * NTAB + row] = __bfloat162float(__float2bfloat16_rn(s.x));
    dw[(ch + 1) * NTAB + row] = __bfloat162float(__float2bfloat16_rn(s.y));
  }
}

template <class R, bool FR = false>
__global__ void __launch_bounds__(NTHR, 1)
tbl_grad_sm100(const __grid_constant__ CUtensorMap gmap, const __grid_constant__ CUtensorMap cmap, const typename R::Code* __restrict__ codes, long long ncodes,
               R rd, long long P, int coff, int NTAB, int nchunk, int ncw, float* __restrict__ tpart, float* __restrict__ dw = nullptr,
               unsigned* __restrict__ bar = nullptr) {
  using namespace sm100;
  constexpr int FB = R::NPADMAX * 128;   // one one-hot buffer
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* Gs = smb;                // [NS][2 x 64-channel blocks][64 pairs][128 B]
  unsigned char* Fs = Gs + NS * GST;      // [NF][NPADMAX rows][128 B]
  unsigned char* Cs = Fs + NF * FB;       // [NS][64 codes]
  uint64_t* bars = reinterpret_cast<uint64_t*>(Cs + NS * TP * R::CB);
  uint64_t* full = bars;                  // [NS] TMA bytes
  uint64_t* empty = full + NS;            // [NS] MMA commit
  uint64_t* fready = empty + NS;          // [NF] 64 scatter threads
  uint64_t* fempty = fready + NF;         // [NF] MMA commit
  uint64_t* accd = fempty + NF;           // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(accd + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int half = blockIdx.x, t = blockIdx.y;
  const long long per = (P + gridDim.y - 1) / gridDim.y, p0 = (long long)t * per, p1 = min(P, p0 + per);
  const int ntiles = p1 > p0 ? (int)((p1 - p0 + TP - 1) / TP) : 0;

  for (int k = tid; k < NF * FB / 16; k += NTHR) reinterpret_cast<uint4*>(Fs)[k] = make_uint4(0u, 0u, 0u, 0u);
  if (tid == 0) {
    for (int s = 0; s < NS; s++) { bar_init(&full[s], R::CB == 16 ? 1 : 32); bar_init(&empty[s], 1); }
    for (int b = 0; b < NF; b++) { bar_init(&fready[b], 64); bar_init(&fempty[b], 1); }
    bar_init(accd, 1);
    bar_init_fence();
  }
  if (warp == 0) tmem_alloc(tslot, R::TCOLS);
  fence_proxy_async();                    // the zeroed one-hot buffers, seen by the tensor core
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  if constexpr (FR) asm volatile("griddepcontrol.wait;" ::: "memory");   // G from the previous launch (PDL: the setup above overlapped it)

  if (warp == 0) {
    if constexpr (R::CB == 16) {
      if (elect_one()) {
        prefetch_map(&gmap);
        prefetch_map(&cmap);
        for (int kt = 0; kt < ntiles; kt++) {
          const int s = kt % NS, u = kt / NS;
          if (u > 0) wait(&empty[s], (u - 1) & 1);
          expect_tx(&full[s], GST + TP * R::CB);
          const int pb = (int)(p0 + (long long)kt * TP);
          load_2d(&gmap, Gs + s * GST, &full[s], half * MC, pb);
          load_2d(&gmap, Gs + s * GST + GST / 2, &full[s], half * MC + 64, pb);
          load_2d(&cmap, Cs + s * TP * R::CB, &full[s], 0, coff + pb);
        }
      }
    } else {
      if (lane == 0) prefetch_map(&gmap);
      auto ld = [&](long long q) { return coff + q < ncodes ? __ldg(codes + coff + q) : R::DEAD; };
      typename R::Code n0 = R::DEAD, n1 = R::DEAD;
      if (ntiles > 0) { n0 = ld(p0 + lane); n1 = ld(p0 + 32 + lane); }
      for (int kt = 0; kt < ntiles; kt++) {
        const int s = kt % NS, u = kt / NS;
        if (u > 0) wait(&empty[s], (u - 1) & 1);
        const long long pb = p0 + (long long)kt * TP;
        typename R::Code* cs = reinterpret_cast<typename R::Code*>(Cs + s * TP * R::CB);
        cs[lane] = n0;
        cs[32 + lane] = n1;
        if (kt + 1 < ntiles) { n0 = ld(pb + TP + lane); n1 = ld(pb + TP + 32 + lane); }
        if (lane == 0) {
          expect_tx(&full[s], GST);
          load_2d(&gmap, Gs + s * GST, &full[s], half * MC, (int)pb);
          load_2d(&gmap, Gs + s * GST + GST / 2, &full[s], half * MC + 64, (int)pb);
        } else {
          arrive(&full[s]);
        }
      }
    }
  } else if (warp == 1) {
    const uint32_t leader = elect_one() ? 1u : 0u;
    const uint32_t idesc = idesc_bf16(MC, ncw, 1, 0);
    const uint32_t gbase = sa(Gs), fbase = sa(Fs);
    for (int kt = 0; kt < ntiles; kt++) {
      const int s = kt % NS, b = kt % NF;
      wait(&full[s], (kt / NS) & 1);
      wait(&fready[b], (kt / NF) & 1);
      tc_fence_after();
#pragma unroll
      for (int ks = 0; ks < TP / 16; ks++) {
        const uint64_t ad = sdesc(gbase + s * GST + ks * 2048, GST / 2, 1024, 2);      // G^T: MN-major, two 64-channel blocks
        for (int c = 0; c < nchunk; c++) {
          const uint64_t bd = sdesc(fbase + b * FB + c * ncw * 128 + ks * 32, 16, 1024, 2);   // one-hot: K-major
          mma_ss_if(leader, tmem + c * ncw, ad, bd, idesc, (kt | ks) ? 1u : 0u);
        }
      }
      mma_commit_if(leader, sa(&empty[s]));
      mma_commit_if(leader, sa(&fempty[b]));
      __syncwarp();
    }
    mma_commit_if(leader, sa(accd));
  } else if (warp < 4) {
    const int p = tid - 64;
    const uint32_t fbase = sa(Fs);
    int prev[NF][R::MAXR];
#pragma unroll
    for (int b = 0; b < NF; b++)
#pragma unroll
      for (int k = 0; k < R::MAXR; k++) prev[b][k] = -1;
    for (int kt = 0; kt < ntiles; kt++) {
      const int s = kt % NS;
      wait(&full[s], (kt / NS) & 1);
      const typename R::Code c = p0 + (long long)kt * TP + p < p1 ? reinterpret_cast<const typename R::Code*>(Cs + s * TP * R::CB)[p] : R::DEAD;
      const int b = kt % NF;
      if (kt >= NF) wait(&fempty[b], ((kt / NF) - 1) & 1);
      const uint32_t buf = fbase + b * FB;
      const uint32_t col = ((p & 7) << 1);
#pragma unroll
      for (int k = 0; k < R::MAXR; k++) {
        const int r = prev[b][k];
        if (r >= 0) asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(buf + r * 128 + ((((p >> 3) ^ (r & 7))) << 4) + col), "h"((unsigned short)0));
      }
      int rows[R::MAXR];
      uint32_t vals[R::MAXR];
      const int n = rd.rows(c, rows, vals);
#pragma unroll
      for (int k = 0; k < R::MAXR; k++) {
        prev[b][k] = k < n ? rows[k] : -1;
        if (k < n) {
          const int r = rows[k];
          asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(buf + r * 128 + ((((p >> 3) ^ (r & 7))) << 4) + col), "h"((unsigned short)vals[k]));
        }
      }
      fence_proxy_async();
      arrive(&fready[b]);
    }
  } else {
    const int sp = warp & 3, ch = 32 * sp + lane;
    float* dst = tpart + (size_t)t * NTAB * 256 + half * MC + ch;
    if (ntiles > 0) {
      wait(accd, 0);
      tc_fence_after();
      for (int c0 = 0; c0 < NTAB; c0 += 32) {
        float v[32];
        tmem_ld32(tmem + ((uint32_t)(32 * sp) << 16) + c0, v);
        tmem_wait_ld();
#pragma unroll
        for (int r = 0; r < 32; r++)
          if (c0 + r < NTAB) dst[(size_t)(c0 + r) * 256] = v[r];
      }
    } else {
      for (int r = 0; r < NTAB; r++) dst[(size_t)r * 256] = 0.f;
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    tmem_dealloc(tmem, R::TCOLS);
  }
  if constexpr (FR) {
    const unsigned nct = gridDim.x * gridDim.y, cta = blockIdx.y * gridDim.x + blockIdx.x;
    grid_barrier(bar, nct);
    reduce_cols(tpart, dw, (int)gridDim.y, NTAB, cta, nct);
  }
}

template <class R>
inline size_t smem_bytes() {
  return 1024 + (size_t)NS * GST + (size_t)NF * R::NPADMAX * 128 + (size_t)NS * TP * R::CB + (2 * NS + 2 * NF + 1) * 8 + 16;
}

// dtbl[k] = sum over the n partials [n][NTAB * 256] in order (fp32 adds, one rounding to OutT). 16 partials are requested at once (2026-11-17:
// with 4 in flight the 74 strided loads of a thread were 19 dependent round trips, ~11 us whatever the table size); the adds keep their order
// (a missing partial adds +0), so the sums are the same bits.
template <typename OutT>
__global__ void __launch_bounds__(256) reduce_parts(const float* __restrict__ part, OutT* __restrict__ out, int n, long width) {
  const long k4 = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (k4 * 4 >= width) return;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r0 = 0; r0 < n; r0 += 16) {
    float4 q[16];
#pragma unroll
    for (int u = 0; u < 16; u++)
      q[u] = r0 + u < n ? __ldg(reinterpret_cast<const float4*>(part + (size_t)(r0 + u) * width) + k4) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 16; u++) { s.x += q[u].x; s.y += q[u].y; s.z += q[u].z; s.w += q[u].w; }
  }
  OutT* o = out + k4 * 4;
  o[0] = sm100::from_f<OutT>(s.x); o[1] = sm100::from_f<OutT>(s.y); o[2] = sm100::from_f<OutT>(s.z); o[3] = sm100::from_f<OutT>(s.w);
}

// reduce_parts written in the weight's layout: dw [256][NTAB] fp32 (no transposed copy afterwards); RB: every sum rounded to bf16 first (the
// value of the bf16 table gradient reduce_parts<__nv_bfloat16> writes), else the fp32 sum. The same adds in the same order.
template <bool RB>
__global__ void __launch_bounds__(256) reduce_parts_t(const float* __restrict__ part, float* __restrict__ dw, int n, int NTAB) {
  const long k4 = (long)blockIdx.x * blockDim.x + threadIdx.x, width = (long)NTAB * 256;
  if (k4 * 4 >= width) return;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r0 = 0; r0 < n; r0 += 16) {
    float4 q[16];
#pragma unroll
    for (int u = 0; u < 16; u++)
      q[u] = r0 + u < n ? __ldg(reinterpret_cast<const float4*>(part + (size_t)(r0 + u) * width) + k4) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 16; u++) { s.x += q[u].x; s.y += q[u].y; s.z += q[u].z; s.w += q[u].w; }
  }
  if constexpr (RB) {
    s.x = __bfloat162float(__float2bfloat16_rn(s.x)); s.y = __bfloat162float(__float2bfloat16_rn(s.y));
    s.z = __bfloat162float(__float2bfloat16_rn(s.z)); s.w = __bfloat162float(__float2bfloat16_rn(s.w));
  }
  const long row = k4 * 4 / 256, ch = k4 * 4 % 256;
  dw[(ch + 0) * NTAB + row] = s.x; dw[(ch + 1) * NTAB + row] = s.y; dw[(ch + 2) * NTAB + row] = s.z; dw[(ch + 3) * NTAB + row] = s.w;
}

// reduce_parts_t with two columns a thread and up to 80 partials requested at once (one round trip for the 74 slices instead of five dependent
// ones); the same adds in the same order (a missing partial adds +0)
template <bool RB>
__global__ void __launch_bounds__(256) reduce_parts_t2(const float* __restrict__ part, float* __restrict__ dw, int n, int NTAB) {
  const long k2 = (long)blockIdx.x * blockDim.x + threadIdx.x, width = (long)NTAB * 256;
  if (k2 * 2 >= width) return;
  float2 s = make_float2(0.f, 0.f);
  for (int r0 = 0; r0 < n; r0 += 80) {
    float2 q[80];
#pragma unroll
    for (int u = 0; u < 80; u++)
      q[u] = r0 + u < n ? __ldg(reinterpret_cast<const float2*>(part + (size_t)(r0 + u) * width) + k2) : make_float2(0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 80; u++) { s.x += q[u].x; s.y += q[u].y; }
  }
  if constexpr (RB) {
    s.x = __bfloat162float(__float2bfloat16_rn(s.x));
    s.y = __bfloat162float(__float2bfloat16_rn(s.y));
  }
  const long row = k2 * 2 / 256, ch = k2 * 2 % 256;
  dw[(ch + 0) * NTAB + row] = s.x; dw[(ch + 1) * NTAB + row] = s.y;
}

// G [P][256] bf16 as a 2-D map: (256 channels, P pairs), box (64, 64), 128-byte swizzle
inline CUtensorMap g_map(const void* g, long long P) {
  return sm100::make_map<2>(g, {256ull, (uint64_t)P}, {256ull}, {64u, 64u}, CU_TENSOR_MAP_SWIZZLE_128B, "pair-embed gradient");
}

// the 16-byte codes [ncodes][4] int32 as (4, ncodes), box (4, 64); past the end: zero fill (no live bit). A 4-byte code needs no map (G's is passed).
template <class R>
inline CUtensorMap c_map(const void* codes, long long ncodes) {
  return sm100::make_map<2>(codes, {4ull, (uint64_t)ncodes}, {4ull}, {4u, 64u}, CU_TENSOR_MAP_SWIZZLE_NONE, "pair-embed codes",
                            CU_TENSOR_MAP_DATA_TYPE_INT32, 4);
}

// g: [P][256] bf16; codes: the whole code tensor ([ncodes] codes), this call's pairs start at code `coff`
template <class R>
inline void launch(const void* g, long long P, const void* codes, long long ncodes, long long coff, const R& rd, int NTAB, int nchunk, int ncw,
                   float* tpart, cudaStream_t st, int nblk = NBLK) {
  const size_t sm = smem_bytes<R>();
  cudaFuncSetAttribute(tbl_grad_sm100<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  const CUtensorMap m = g_map(g, P);
  const CUtensorMap cm = R::CB == 16 ? c_map<R>(codes, ncodes) : m;
  tbl_grad_sm100<R><<<dim3(2, nblk), NTHR, sm, st>>>(m, cm, reinterpret_cast<const typename R::Code*>(codes), ncodes, rd, P, (int)coff, NTAB,
                                                     nchunk, ncw, tpart);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// launch + the fused reduce (FR) into dw [256][NTAB] fp32: one cooperative launch of 2 x nblk CTAs (at most one per SM); bar: 2 zeroed words.
// Programmatic stream serialization: the CTAs set up (barriers, TMEM, the zeroed one-hot buffers) while the previous kernel ends
template <class R>
inline void launch_fr(const void* g, long long P, const void* codes, long long ncodes, const R& rd, int NTAB, int nchunk, int ncw, float* tpart,
                      float* dw, unsigned* bar, cudaStream_t st, int nblk) {
  const size_t sm = smem_bytes<R>();
  cudaFuncSetAttribute(tbl_grad_sm100<R, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  const CUtensorMap m = g_map(g, P);
  const CUtensorMap cm = R::CB == 16 ? c_map<R>(codes, ncodes) : m;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2, nblk);
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeCooperative;
  at[0].val.cooperative = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tbl_grad_sm100<R, true>, m, cm, reinterpret_cast<const typename R::Code*>(codes), ncodes, rd, P, 0, NTAB,
                                    nchunk, ncw, tpart, dw, bar));
}
}  // namespace pe100
