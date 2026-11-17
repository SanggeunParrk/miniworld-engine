// transition_fp8_d256_sm90a.cu -- OPT-IN FP8 (e4m3) inference forward of the D256 Transition (engine 3.0.0 line, docs/fp8.md).
// SPDX-License-Identifier: Apache-2.0
// The 2.5.12 D256 forward (wide/kernels/transition_fwd_d256.cu: producer warpgroup + two ping-pong consumer warpgroups, 128-row tile,
// 64 x 256 fp32 output accumulator per warpgroup held across the hidden chunks of 32 units) with the GEMMs on the e4m3 tensor-core path:
//
//   xn8  = e4m3(LN(x) sx)                          F8G1: written to its own 32 KB buffer (x stays resident for the residual: no x reload)
//   per hidden chunk j (32 units):
//     [A|B] = xn8 [Wa8_j; Wb8_j]^T                  SS m64n64k32 e4m3 x 8          (F8G1 = 0: the 2.5.12 bf16 SS m64n64k16 x 16 over bf16 xn)
//     h8_j  = e4m3(silu(A ca) B cb)                 ca = 1 / (sx s1), cb = ca sh: the C fragment IS the k32 e4m3 A fragment (K permutation)
//     acc  += h8_j Ws8_j^T                          RS m64n256k32 e4m3 x 1         (Ws8 K-major [256 d][32 hs] per chunk, 32B swizzle)
//   out = bf16(x + acc co)                         co = 1 / (sh s2)
//
// Scales, guard and the K permutation are those of transition_fp8_fwd_sm90a.cu (D128): powers of two in a device float[8]
// {sx, ca, cb, co}, max |h sh| (pre-cast) per CTA to `peak` by atomicMax. The packed weights carry K in the e4m3 fragment order (host).
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "transition_fp8_d256_wgmma.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <array>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef NCTA
#define NCTA 132
#endif
#ifndef MW_N
#define MW_N 4
#endif
#ifndef F8G1
#define F8G1 1     // 1: expand in e4m3 (xn8, Wa8 / Wb8); 0: the 2.5.12 bf16 expand (h and the squeeze stay e4m3: the "h + squeeze" variant)
#endif
#ifndef WSLOTS
#define WSLOTS (F8G1 ? 4 : 3)
#endif
#ifndef PP
#define PP 1       // ping-pong the two consumer warpgroups (the 2.5.12 D256 schedule)
#endif

constexpr int D_ = 256, H_ = MW_N * 256, HS = 32, NCH = H_ / HS, ROWS = 128, WGR = 64;
// slot: [Wa_j; Wb_j] (64 rows; e4m3: 2 K-slabs x [64][128 B] = 16 KB, bf16: 4 K-quarters x [64][128 B] = 32 KB), then Ws8_j [256 d][32 B] 8 KB
constexpr int F_W2 = F8G1 ? 16384 : 32768;
constexpr int F_SLOT = F_W2 + 8192;
constexpr int NSLOT = WSLOTS;
constexpr int F_XB = NSLOT * F_SLOT;                 // x tile, 4 quarters x [128 rows][128 B] (bf16)
constexpr int F_XN = F_XB + 65536;                   // F8G1: xn8, 2 slabs x [128 rows][128 B]
constexpr int F_BAR = F_XN + (F8G1 ? 32768 : 0);
constexpr int SMEM_BYTES = F_BAR + 256;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");
static_assert(NSLOT <= NCH, "weight ring deeper than one tile");

constexpr int PP_T0 = 3, PP_T1 = 4;
TMN_DEVI void named_bar_arrive_(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t lay) { return smem_desc(addr, lbo, sbo, lay); }
TMN_DEVI uint32_t f8x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return (uint32_t)r;
}
TMN_DEVI uint32_t f8x4(float v0, float v1, float v2, float v3) { return f8x2(v0, v1) | (f8x2(v2, v3) << 16); }
#if !F8G1
TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate));
}
#endif
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
transition_fp8_d256(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw1, const __grid_constant__ CUtensorMap mw2,
                    const __grid_constant__ CUtensorMap mout, const float* __restrict__ gamma, const float* __restrict__ beta,
                    const float* __restrict__ scales, unsigned int* __restrict__ peak, int tiles, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + NSLOT;
  uint64_t* x_load = bars + 2 * NSLOT;     // x(i) has landed
  uint64_t* out_done = x_load + 1;         // both warpgroups' output stores have read the tile buffer
  uint64_t* xn_free = x_load + 2;          // (F8G1 = 0) both warpgroups' last expand of the tile retired: the buffer may take x(i) again
  uint64_t* x_res = x_load + 3;            // (F8G1 = 0) x(i) landed again, for the residual
  const int n_local = (tiles > cta) ? (tiles - cta + NCTA - 1) / NCTA : 0;
  if (tid == 0) {
    for (int s = 0; s < NSLOT; ++s) { mbar_init(w_full + s, 1); mbar_init(w_free + s, 2); }
    mbar_init(x_load, 1); mbar_init(out_done, 2); mbar_init(xn_free, 2); mbar_init(x_res, 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mx); tma_prefetch_desc(&mw1); tma_prefetch_desc(&mw2); tma_prefetch_desc(&mout);
    auto load_x = [&](int i, uint64_t* bar) {
      const int row = (cta + i * NCTA) * ROWS;
      mbar_arrive_expect_tx(bar, 65536);
      for (int q = 0; q < 4; ++q)
        for (int h = 0; h < 2; ++h) tma_load_2d(sm + F_XB + q * 16384 + h * 8192, &mx, bar, 64 * q, row + 64 * h);
    };
    auto load_w = [&](uint32_t seq) {
      const int s = (int)(seq % NSLOT), j = (int)(seq % NCH);
      if (seq >= NSLOT) mbar_wait(w_free + s, ((seq / NSLOT) - 1) & 1u);
      uint8_t* b = sm + s * F_SLOT;
      mbar_arrive_expect_tx(w_full + s, F_SLOT);
#if F8G1
      for (int q = 0; q < 2; ++q) tma_load_2d(b + q * 8192, &mw1, w_full + s, 128 * q, j * 64);   // [Wa8_j; Wb8_j], K-slab q
#else
      for (int q = 0; q < 4; ++q) tma_load_2d(b + q * 8192, &mw1, w_full + s, 64 * q, j * 64);    // [Wa_j; Wb_j] bf16, K-quarter q
#endif
      tma_load_2d(b + F_W2, &mw2, w_full + s, j * HS, 0);                                          // Ws8[:, 32 j .. 32 j + 31]
    };
    if (n_local == 0) return;
    load_x(0, x_load);
    for (int j = 0; j < NSLOT; ++j) load_w((uint32_t)j);
    for (int i = 0; i < n_local; ++i) {
      for (int j = NSLOT; j < NCH; ++j) load_w((uint32_t)(i * NCH + j));
#if !F8G1
      mbar_wait(xn_free, i & 1);
      load_x(i, x_res);
#endif
      if (i + 1 < n_local) for (int j = 0; j < NSLOT; ++j) load_w((uint32_t)((i + 1) * NCH + j));
      mbar_wait(out_done, i & 1);
      if (i + 1 < n_local) load_x(i + 1, x_load);
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  const uint32_t xb = su + F_XB;
#if F8G1
  const uint32_t xnb = su + F_XN;
#endif
  const float sx = scales[0], ca = scales[1], cb = scales[2], co = scales[3];
  const float hca = 0.5f * ca;
  float pk = 0.f;
  const int q_ = lane >> 3, gr_ = lane & 7;
  uint32_t wseq = 0;
  for (int i = 0; i < n_local; ++i) {
    const int trow = (cta + i * NCTA) * ROWS;
    mbar_wait(x_load, i & 1);
    // ---------------------------------------------------------------- LayerNorm of this warpgroup's 64 rows (the 2.5.12 code)
    {
      float g8[8], b8[8];
      { const float4* gp = reinterpret_cast<const float4*>(gamma + 8 * lane); const float4* bp = reinterpret_cast<const float4*>(beta + 8 * lane);
        const float4 g0 = __ldg(gp), g1 = __ldg(gp + 1), b0 = __ldg(bp), b1 = __ldg(bp + 1);
        g8[0] = g0.x; g8[1] = g0.y; g8[2] = g0.z; g8[3] = g0.w; g8[4] = g1.x; g8[5] = g1.y; g8[6] = g1.z; g8[7] = g1.w;
        b8[0] = b0.x; b8[1] = b0.y; b8[2] = b0.z; b8[3] = b0.w; b8[4] = b1.x; b8[5] = b1.y; b8[6] = b1.z; b8[7] = b1.w; }
#if F8G1
#pragma unroll
      for (int e = 0; e < 8; ++e) b8[e] *= sx;
#endif
#pragma unroll 1
      for (int half = 0; half < 2; ++half) {
        uint4 v[8]; uint32_t ad[8]; float s8[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int r = WGR * cw + 16 * warp + 8 * half + u;
          ad[u] = xb + q_ * 16384 + swz128((uint32_t)r, (uint32_t)(gr_ * 16));
          v[u] = lds128(ad[u]);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u)
          s8[u] = ((bf16lo(v[u].x) + bf16hi(v[u].x)) + (bf16lo(v[u].y) + bf16hi(v[u].y))) +
                  ((bf16lo(v[u].z) + bf16hi(v[u].z)) + (bf16lo(v[u].w) + bf16hi(v[u].w)));
#pragma unroll
        for (int k = 16; k; k >>= 1) {
#pragma unroll
          for (int u = 0; u < 8; ++u) s8[u] += __shfl_xor_sync(0xffffffffu, s8[u], k);
        }
        float mean8[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          mean8[u] = s8[u] * (1.f / D_);
          const uint32_t w4[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
          float a = 0.f;
#pragma unroll
          for (int e = 0; e < 4; ++e) { float d = bf16lo(w4[e]) - mean8[u]; a += d * d; d = bf16hi(w4[e]) - mean8[u]; a += d * d; }
          s8[u] = a;
        }
#pragma unroll
        for (int k = 16; k; k >>= 1) {
#pragma unroll
          for (int u = 0; u < 8; ++u) s8[u] += __shfl_xor_sync(0xffffffffu, s8[u], k);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const float mean = mean8[u];
          const uint32_t w4[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#if F8G1
          const int r = WGR * cw + 16 * warp + 8 * half + u;
          const float rs = rsqrtf(s8[u] * (1.f / D_) + eps) * sx;
          float y[8];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            y[2 * e] = (bf16lo(w4[e]) - mean) * rs * g8[2 * e] + b8[2 * e];
            y[2 * e + 1] = (bf16hi(w4[e]) - mean) * rs * g8[2 * e + 1] + b8[2 * e + 1];
          }
          const uint32_t lo = f8x4(y[0], y[1], y[2], y[3]), hi = f8x4(y[4], y[5], y[6], y[7]);
          const uint32_t a8 = xnb + (q_ >> 1) * 16384 + swz128((uint32_t)r, (uint32_t)((q_ & 1) * 64 + gr_ * 8));
          asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a8), "r"(lo), "r"(hi) : "memory");
#else
          const float rs = rsqrtf(s8[u] * (1.f / D_) + eps);
          uint4 o;
          uint32_t* op = &o.x;
#pragma unroll
          for (int e = 0; e < 4; ++e)
            op[e] = pack_bf16((bf16lo(w4[e]) - mean) * rs * g8[2 * e] + b8[2 * e], (bf16hi(w4[e]) - mean) * rs * g8[2 * e + 1] + b8[2 * e + 1]);
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ad[u]), "r"(o.x), "r"(o.y), "r"(o.z), "r"(o.w) : "memory");
#endif
        }
      }
    }
    fence_proxy_async();                                  // xn (generic stores) -> the wgmma operand reads
    named_bar_sync(1 + cw, 128);
    // ---------------------------------------------------------------- the hidden chunks (ping-pong batches [G2(j-1), G1(j)])
    float acc[128];
#pragma unroll
    for (int e = 0; e < 128; ++e) acc[e] = 0.f;
    uint32_t fh[4] = {0u, 0u, 0u, 0u};
    uint32_t pslot = 0;
    const bool last_tile = (i == n_local - 1);
#pragma unroll 1
    for (int j = 0; j <= NCH; ++j, ++wseq) {
      const int s = (int)(wseq % NSLOT);
      const uint32_t slot = su + s * F_SLOT;
      if (j < NCH) mbar_wait(w_full + s, (wseq / NSLOT) & 1u);
      float AB[32];
#pragma unroll
      for (int e = 0; e < 32; ++e) AB[e] = 0.f;
#if PP
      if (!(cw == 0 && i == 0 && j == 0)) named_bar_sync(cw ? PP_T1 : PP_T0, 256);
#endif
      fence_regs(AB); fence_regs(acc); fence_regs(fh); wgmma_fence();
      if (j > 0) mma8_256_rs(acc, fh, dsc_(pslot + F_W2, 16, 256, 3));
      if (j < NCH) {
#if F8G1
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          mma8_64_ss(AB, dsc_(xnb + (ks >> 2) * 16384 + cw * 8192 + (ks & 3) * 32, 16, 1024, 1), dsc_(slot + (ks >> 2) * 8192 + (ks & 3) * 32, 16, 1024, 1), ks > 0);
#else
#pragma unroll
        for (int ks = 0; ks < 16; ++ks)
          mma64(AB, dsc_(xb + (ks >> 2) * 16384 + cw * 8192 + (ks & 3) * 32, 16, 1024, 1), dsc_(slot + (ks >> 2) * 8192 + (ks & 3) * 32, 16, 1024, 1), ks > 0);
#endif
      }
      wgmma_commit();
#if PP
      if (!(cw == 1 && last_tile && j == NCH)) named_bar_arrive_(cw ? PP_T0 : PP_T1, 256);
#endif
      wgmma_wait<0>(); fence_regs(acc); fence_regs(AB);
      if (j > 0 && wtid == 0) mbar_arrive(w_free + (int)((wseq - 1) % NSLOT));   // G2(j-1) has read chunk j-1's slot
      if (j == NCH) break;
#if !F8G1
      if (j == NCH - 1 && wtid == 0) mbar_arrive(xn_free);
#endif
      // h8 = e4m3(silu(a) b sh): C groups 0..3 hold a, 4..7 hold b; k32 A register 2 hf + R = C groups {2 hf, 2 hf + 1}, row half R
#pragma unroll
      for (int hf = 0; hf < 2; ++hf)
#pragma unroll
        for (int R = 0; R < 2; ++R) {
          const int g0 = 2 * hf, g1 = g0 + 1;
          float v[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int e = 4 * (u < 2 ? g0 : g1) + 2 * R + (u & 1);
            const float hu = AB[e] * hca;                   // silu(a) = (a / 2)(1 + tanh(a / 2))
            float th;
            asm("tanh.approx.f32 %0, %1;\n" : "=f"(th) : "f"(hu));
            v[u] = fmaf(hu, th, hu) * (AB[e + 16] * cb);
            pk = fmaxf(pk, fabsf(v[u]));
          }
          fh[2 * hf + R] = f8x4(v[0], v[1], v[2], v[3]);
        }
      pslot = slot;
    }
    // ---------------------------------------------------------------- out = bf16(x + acc co), in place over the x tile
#if !F8G1
    mbar_wait(x_res, i & 1);
#endif
    {
      const int mi = lane >> 3, mrow = WGR * cw + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 16; ++gp) {
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t ad = xb + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        uint32_t xr[4];
        ldsm_x4(xr, ad);
        const int g0 = 2 * gp, g1 = 2 * gp + 1;
        stsm_x4(ad, pack_bf16(fmaf(acc[4 * g0 + 0], co, bf16lo(xr[0])), fmaf(acc[4 * g0 + 1], co, bf16hi(xr[0]))),
                    pack_bf16(fmaf(acc[4 * g0 + 2], co, bf16lo(xr[1])), fmaf(acc[4 * g0 + 3], co, bf16hi(xr[1]))),
                    pack_bf16(fmaf(acc[4 * g1 + 0], co, bf16lo(xr[2])), fmaf(acc[4 * g1 + 1], co, bf16hi(xr[2]))),
                    pack_bf16(fmaf(acc[4 * g1 + 2], co, bf16lo(xr[3])), fmaf(acc[4 * g1 + 3], co, bf16hi(xr[3]))));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      for (int q = 0; q < 4; ++q) tma_store_2d(&mout, sm + F_XB + q * 16384 + cw * 8192, 64 * q, trow + WGR * cw);
      tma_store_commit();
      tma_store_wait_read<0>();
      mbar_arrive(out_done);
    }
  }
  if (wtid == 0) tma_store_wait_all();
  unsigned int pb = __reduce_max_sync(0xffffffffu, __float_as_uint(pk));
  if (lane == 0 && pb) atomicMax(peak, pb);
}

// ================================================================================================ host
namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr) throw std::runtime_error("cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
using Key = std::array<uint64_t, 7>;
// 2-D descriptor of a contiguous [rows][cols] tensor (bf16 or uint8), swizzle 32 / 64 / 128 B
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t box_inner, uint32_t box_outer, int swz) {
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const int esz = (int)t.element_size();
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), (uint64_t)t.size(1), (uint64_t)t.size(0), box_inner, box_outer, (uint64_t)esz, (uint64_t)swz};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.size(1) * esz};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed");
  const CUtensorMapSwizzle sw = swz == 32 ? CU_TENSOR_MAP_SWIZZLE_32B : swz == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
  CUresult r = encoder()(&map, esz == 1 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}
void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition fp8 d256 sm90a: ") + what);
}
}  // namespace

// x [M, 256] bf16 (M a multiple of 128); w1 [H/32*64, 256] (chunk j = [Wa_j (32 rows); Wb_j (32 rows)], K natural): F8G1 uint8 (e4m3) | bf16;
// w2 Ws8 [256, H] uint8 (e4m3, K permuted); scales float[>= 4] = {sx, ca, cb, co}; peak float[1] (atomicMax of max |h sh| as bits).
torch::Tensor transition_fp8_d256_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor w1, torch::Tensor w2,
                                      torch::Tensor scales, torch::Tensor peak, double eps) {
  const int64_t M = x.size(0);
  expect(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == D_, "x must be contiguous cuda bf16 [M, 256]");
  expect(M % ROWS == 0, "row count must be a whole 128-row tile");
  expect(w1.scalar_type() == (F8G1 ? torch::kUInt8 : torch::kBFloat16) && w1.is_contiguous() && w1.size(0) == NCH * 64 && w1.size(1) == D_,
         F8G1 ? "w1 must be uint8 [H/32*64, 256]" : "w1 must be bf16 [H/32*64, 256]");
  expect(w2.scalar_type() == torch::kUInt8 && w2.is_contiguous() && w2.size(0) == D_ && w2.size(1) == H_, "w2 must be uint8 [256, H]");
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32, "gamma/beta must be fp32");
  expect(scales.scalar_type() == torch::kFloat32 && scales.numel() >= 4, "scales float[>=4]");
  expect(peak.scalar_type() == torch::kFloat32 && peak.numel() >= 1, "peak float[1]");
  auto out = torch::empty_like(x);
  const int tiles = (int)(M / ROWS);
  static const bool ready = [] {
    cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(transition_fp8_d256), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) throw std::runtime_error(std::string("transition_fp8_d256 smem opt-in: ") + cudaGetErrorString(e));
    return true;
  }();
  (void)ready;
  if (tiles == 0) return out;
  const CUtensorMap& mx = tile_map(x, 64, 64, 128);
  const CUtensorMap& m1 = F8G1 ? tile_map(w1, 128, 64, 128) : tile_map(w1, 64, 64, 128);
  const CUtensorMap& m2 = tile_map(w2, 32, 256, 32);
  const CUtensorMap& mo = tile_map(out, 64, 64, 128);
  transition_fp8_d256<<<(tiles < NCTA ? tiles : NCTA), 384, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
      mx, m1, m2, mo, gamma.data_ptr<float>(), beta.data_ptr<float>(), scales.data_ptr<float>(),
      reinterpret_cast<unsigned int*>(peak.data_ptr<float>()), tiles, (float)eps);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transition_fp8_d256_fwd", &transition_fp8_d256_fwd, "FP8 (e4m3) D256 Transition inference forward (opt-in)");
}
