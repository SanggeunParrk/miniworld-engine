// fp32_fwd_sm120.cuh -- GENERATED from fp32_fwd_sm80.cuh for the RTX 5090 (sm_120a): the same two fused TF32 inference kernels with each warp's weight ring two
// chunks deep instead of 4 / 3 (the A100's ring is 128 KB a CTA, over the 5090's 99 KB a block): qkvg 74 / 83 / 100 KB at MT 1 / 2 / 4, the FFN 99.8 KB at MT 1
// (its only shape here).  The statements, operands and summation order are the A100 kernels'; only the cp.async prefetch depth differs.  Namespace sw120f.
#pragma once
#include "sm80_common.cuh"

namespace sw120f {
using namespace sw80;

constexpr int C = 128, NHID = 256, MODW = 768;
constexpr int NW = 8, NTHR = NW * 32;
constexpr int XS = C + 8, HS = NHID + 8;   // shared row strides (floats): 544 / 1056 B, = 32 mod 128 -> the float2 A loads are conflict free

DEVI uint32_t tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float tf32f(float x) { return __uint_as_float(tf32(x)); }

DEVI void mma1688(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI float4 ldg4(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }
DEVI float2 ldg2(const float* p) { return __ldg(reinterpret_cast<const float2*>(p)); }
DEVI float sig(float v) { return 1.f / (1.f + expf(-v)); }      // rows_sm80.cuh's sigmoid

// The B operand stream of one warp: its slices of the packed weights, 4 KB chunks copied by cp.async into a private ring of NS slots
// in shared memory, NS - 1 chunks ahead of the products (no registers held, no CTA barrier: a chunk's 32 lanes meet at __syncwarp).
// Every tile consumes the same chunk sequence (``chunk``), so the stream simply runs on into the next tile of the CTA.
//   KIND 0 (qkvg): 8 chunks = the warp's [Wq; Wk; Wv; Wg] block (32 KB, 1 k-step pair of 8 n tiles each)
//   KIND 1 (ffn):  2 chunks of Wo (4 pairs of 2 n tiles each), 8 of [Wa; Wb] (1 pair of 8 n tiles), 4 of Wd (4 pairs of 2 n tiles)
constexpr int CHUNK = 1024;                 // floats per chunk (4 KB)
template <int KIND> constexpr int chunks_per_tile() { return KIND == 0 ? 8 : 14; }

template <int NS, int KIND>
struct Ring {
  float* base;                              // the warp's NS slots
  const float* w0;                          // the warp's packed block of: qkvg W1 | Wo
  const float* w1;                          // Wu
  const float* w2;                          // Wd
  int q, total, lane;
  DEVI const float* src(int j) const {
    if (KIND == 0) return w0 + (j % 8) * CHUNK;
    j %= 14;
    return j < 2 ? w0 + j * CHUNK : j < 10 ? w1 + (j - 2) * CHUNK : w2 + (j - 10) * CHUNK;
  }
  DEVI void issue(int j) {
    if (j < total) {
      const float* s = src(j);
      const uint32_t d = smem_u32(base + (j % NS) * CHUNK);
#pragma unroll
      for (int i = 0; i < 8; ++i) cp_async16(d + (i * 32 + lane) * 16, s + (i * 32 + lane) * 4);
    }
    cp_async_commit();                      // a group per chunk, empty past the end: the wait counts stay uniform
  }
  DEVI void start() {
#pragma unroll
    for (int j = 0; j < NS - 1; ++j) issue(j);
  }
  // the next chunk, complete and visible to the whole warp; the slot of the chunk before it is refilled
  DEVI const float* acquire() {
    cp_async_wait<NS - 2>();
    __syncwarp();
    issue(q + NS - 1);
    return base + (q++ % NS) * CHUNK;
  }
};

// One warp's product over the R = 16 MT rows of an A tile in shared memory (row stride AS floats, tf32 values) and NT n tiles of packed
// B ([k-step pair][NT][32 lanes] float4 = {W[n][16 kp + 2t], [.. + 1], [16 kp + 8 + 2t], [.. + 1]} of row n = 8 nt + g), from NSL ring
// chunks of KPS k-step pairs each (K = 16 KPS NSL): acc[mt][nt] += A W^T.
template <int MT, int NT, int KPS, int NSL, int AS, class RingT>
DEVI void warp_gemm(float (&acc)[MT][NT][4], const float* sa, RingT& ring, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll 1
  for (int sl = 0; sl < NSL; ++sl) {
    const float* b = ring.acquire();
#pragma unroll
    for (int kl = 0; kl < KPS; ++kl) {
      const int kp = sl * KPS + kl;
      uint32_t a[MT][2][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const float2 lo = *reinterpret_cast<const float2*>(sa + (16 * mt + g) * AS + 16 * kp + 8 * s + 2 * t);
          const float2 hi = *reinterpret_cast<const float2*>(sa + (16 * mt + g + 8) * AS + 16 * kp + 8 * s + 2 * t);
          a[mt][s][0] = __float_as_uint(lo.x); a[mt][s][2] = __float_as_uint(lo.y);
          a[mt][s][1] = __float_as_uint(hi.x); a[mt][s][3] = __float_as_uint(hi.y);
        }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const float4 bb = *reinterpret_cast<const float4*>(b + ((kl * NT + nt) * 32 + lane) * 4);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma1688(acc[mt][nt], a[mt][0], __float_as_uint(bb.x), __float_as_uint(bb.y));
          mma1688(acc[mt][nt], a[mt][1], __float_as_uint(bb.z), __float_as_uint(bb.w));
        }
      }
    }
  }
}

template <int MT, int NT>
DEVI void zero(float (&acc)[MT][NT][4]) {
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
}

// the modulation / RoPE row of flattened row r (N = M / S sequences, one modulation row per (n mod B, s))
DEVI long mrow_of(long r, int S, int B) { const long n = r / S; return (n % B) * S + (r - n * S); }

// ================================================================ qkvg
struct QkvgParams {
  const float* q;                 // [M][128] the block input
  const float* mod;               // [B S][768]
  const float* cos;               // [B S][16]
  const float* sin;
  const float* w;                 // packed [8 warps][8 kp][8 nt][32][4]: warp w = output channels 64 w .. (q | k | v | g), the f1 order
  __nv_bfloat16 *qh, *kh, *vh;    // head-major [N][4][S][32] (hm) or token-major [M][4][32]
  float* g;                       // [M][128]
  int M, S, B, hm;
  float eps, qk_eps;
};

template <int MT>
__global__ void __launch_bounds__(NTHR, 1) qkvg_f32_kernel(const QkvgParams p) {
  constexpr int R = 16 * MT, NS = 2;
  extern __shared__ __align__(16) float smf[];
  float* sx = smf + NW * NS * CHUNK;                            // [R][XS] x, tf32 (after the 8 warps' rings)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int ntile = (p.M + R - 1) / R;
  const int proj = warp >> 1;                                   // 0 q, 1 k, 2 v, 3 g
  const int mine = blockIdx.x < ntile ? (ntile - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  Ring<NS, 0> ring{smf + warp * NS * CHUNK, p.w + (size_t)warp * 8 * CHUNK, nullptr, nullptr, 0, mine * chunks_per_tile<0>(), lane};
  ring.start();
  for (int tile = blockIdx.x; tile < ntile; tile += gridDim.x) {
    const long t0 = (long)tile * R;
    // ---- x = RMS(q) (1 + scale_a) + shift_a, a warp per row (lane: channels 4 lane .. 4 lane + 3)
    for (int rr = warp; rr < R; rr += NW) {
      const long r = t0 + rr;
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < p.M) {
        const float4 z = ldg4(p.q + r * C + 4 * lane);
        const float rs = rsqrtf(warp_sum(z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w) / 128 + p.eps);
        const float* mp = p.mod + mrow_of(r, p.S, p.B) * MODW;
        const float4 sh = ldg4(mp + 4 * lane), sc = ldg4(mp + C + 4 * lane);
        x = make_float4(tf32f(z.x * rs * (1 + sc.x) + sh.x), tf32f(z.y * rs * (1 + sc.y) + sh.y),
                        tf32f(z.z * rs * (1 + sc.z) + sh.z), tf32f(z.w * rs * (1 + sc.w) + sh.w));
      }
      *reinterpret_cast<float4*>(sx + rr * XS + 4 * lane) = x;
    }
    __syncthreads();
    float acc[MT][8][4];
    zero(acc);
    warp_gemm<MT, 8, 1, 8, XS>(acc, sx, ring, lane);
    // ---- epilogue: n tile (og, j) = (nt >> 2, nt & 3) holds channels 32 og + 8 q4 + 2 j + {0, 1} of head 2 (warp & 1) + og
#pragma unroll
    for (int og = 0; og < 2; ++og)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const long r = t0 + 16 * mt + g8 + 8 * hh;
          const bool ok = r < p.M;
          const long rc = ok ? r : 0;
          float v[8];
#pragma unroll
          for (int j = 0; j < 4; ++j) { v[2 * j] = acc[mt][4 * og + j][2 * hh]; v[2 * j + 1] = acc[mt][4 * og + j][2 * hh + 1]; }
          const int head = 2 * (warp & 1) + og;
          const long n = rc / p.S, s = rc - n * p.S;
          const size_t dst = p.hm ? (((size_t)n * 4 + head) * p.S + s) * 32 + 8 * q4 : ((size_t)rc * 4 + head) * 32 + 8 * q4;
          if (proj == 3) {
            if (ok) {
              float* gp = p.g + rc * C + 32 * head + 8 * q4;
              *reinterpret_cast<float4*>(gp) = make_float4(v[0], v[1], v[2], v[3]);
              *reinterpret_cast<float4*>(gp + 4) = make_float4(v[4], v[5], v[6], v[7]);
            }
          } else {
            if (proj < 2) {                                     // head RMSNorm (fp32), then the RoPE with the partner d +- 16 (lane q4 ^ 2)
              float ss = 0.f;
#pragma unroll
              for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
              const float rh = rsqrtf(quad_sum(ss) / 32 + p.qk_eps);
#pragma unroll
              for (int i = 0; i < 8; ++i) v[i] *= rh;
              const long mr = mrow_of(rc, p.S, p.B);
              const float4 c0 = ldg4(p.cos + mr * 16 + 8 * (q4 & 1)), c1 = ldg4(p.cos + mr * 16 + 8 * (q4 & 1) + 4);
              const float4 s0 = ldg4(p.sin + mr * 16 + 8 * (q4 & 1)), s1 = ldg4(p.sin + mr * 16 + 8 * (q4 & 1) + 4);
              const float cs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
              const float sn[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
              float o[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) o[i] = __shfl_xor_sync(0xffffffffu, v[i], 2);
#pragma unroll
              for (int i = 0; i < 8; ++i) v[i] = v[i] * cs[i] + (q4 < 2 ? -o[i] : o[i]) * sn[i];
            }
            if (ok) {
              __nv_bfloat16* base = proj == 0 ? p.qh : proj == 1 ? p.kh : p.vh;
              stg128(base + dst, make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7])));
            }
          }
        }
    __syncthreads();                                            // the tile's x is read by every warp before the next one overwrites it
  }
}

template <int MT> constexpr int qkvg_smem() { return (NW * 2 * CHUNK + 16 * MT * XS) * 4; }
static_assert(qkvg_smem<4>() <= 101376, "99 KB");

// ================================================================ out-projection + FFN
// ring slots per warp: as many as fit beside the R-row tiles (163 KB per CTA)
template <int MT> constexpr int ffn_ns() { return 2; }

struct FfnParams {
  const float* q;                 // [M][128] the block input (residual)
  const float* g;                 // [M][128]
  const void* o;                  // [M][128] attention output, fp32 or bf16 (obf)
  const float* mod;               // [B S][768]
  const float* wo;                // packed [8][8 kp][2 nt][32][4]: warp w = att columns 16 w ..
  const float* wu;                // packed [8][8 kp][8 nt][32][4]: warp w = a units 32 w .. (nt 0..3), b units 32 w .. (nt 4..7)
  const float* wd;                // packed [8][16 kp][2 nt][32][4]: warp w = out columns 16 w ..
  float* out;                     // [M][128]
  int M, S, B, obf;
  float eps;
};

template <int MT>
__global__ void __launch_bounds__(NTHR, 1) ffn_f32_kernel(const FfnParams p) {
  constexpr int R = 16 * MT, NS = ffn_ns<MT>();
  extern __shared__ __align__(16) float smf[];
  float* sx = smf + NW * NS * CHUNK;                                              // [R][XS] gated, then y (tf32)
  float* sq = sx + R * XS;                                      // [R][XS] q1 (fp32)
  float* sh = sq + R * XS;                                      // [R][HS] h (tf32)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int ntile = (p.M + R - 1) / R;
  const int mine = blockIdx.x < ntile ? (ntile - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  Ring<NS, 1> ring{smf + warp * NS * CHUNK, p.wo + (size_t)warp * 2 * CHUNK, p.wu + (size_t)warp * 8 * CHUNK, p.wd + (size_t)warp * 4 * CHUNK, 0,
                   mine * chunks_per_tile<1>(), lane};
  ring.start();
  for (int tile = blockIdx.x; tile < ntile; tile += gridDim.x) {
    const long t0 = (long)tile * R;
    // ---- gated = O sigmoid(G)
    for (int rr = warp; rr < R; rr += NW) {
      const long r = t0 + rr;
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < p.M) {
        const float4 gv = ldg4(p.g + r * C + 4 * lane);
        float4 ov;
        if (p.obf) {
          const uint2 u = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(p.o) + r * C + 4 * lane));
          ov = make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y));
        } else {
          ov = ldg4(reinterpret_cast<const float*>(p.o) + r * C + 4 * lane);
        }
        x = make_float4(tf32f(ov.x * sig(gv.x)), tf32f(ov.y * sig(gv.y)), tf32f(ov.z * sig(gv.z)), tf32f(ov.w * sig(gv.w)));
      }
      *reinterpret_cast<float4*>(sx + rr * XS + 4 * lane) = x;
    }
    __syncthreads();
    // ---- att = gated Wo^T (columns 16 warp ..); q1 = q + gate_a att
    {
      float acc[MT][2][4];
      zero(acc);
      warp_gemm<MT, 2, 4, 2, XS>(acc, sx, ring, lane);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int rr = 16 * mt + g8 + 8 * hh;
          const long r = t0 + rr;
          if (r >= p.M) continue;
          const float* mp = p.mod + mrow_of(r, p.S, p.B) * MODW;
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const int c = 16 * warp + 8 * nt + 2 * q4;
            const float2 qv = ldg2(p.q + r * C + c), ga = ldg2(mp + 256 + c);
            *reinterpret_cast<float2*>(sq + rr * XS + c) =
                make_float2(qv.x + ga.x * acc[mt][nt][2 * hh], qv.y + ga.y * acc[mt][nt][2 * hh + 1]);
          }
        }
    }
    __syncthreads();
    // ---- y = RMS(q1) (1 + scale_f) + shift_f
    for (int rr = warp; rr < R; rr += NW) {
      const long r = t0 + rr;
      float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < p.M) {
        const float4 z = *reinterpret_cast<const float4*>(sq + rr * XS + 4 * lane);
        const float rs = rsqrtf(warp_sum(z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w) / 128 + p.eps);
        const float* mp = p.mod + mrow_of(r, p.S, p.B) * MODW;
        const float4 shf = ldg4(mp + 384 + 4 * lane), scf = ldg4(mp + 512 + 4 * lane);
        y = make_float4(tf32f(z.x * rs * (1 + scf.x) + shf.x), tf32f(z.y * rs * (1 + scf.y) + shf.y),
                        tf32f(z.z * rs * (1 + scf.z) + shf.z), tf32f(z.w * rs * (1 + scf.w) + shf.w));
      }
      *reinterpret_cast<float4*>(sx + rr * XS + 4 * lane) = y;
    }
    __syncthreads();
    // ---- [a | b] = y [Wa; Wb]^T for hidden units 32 warp ..; h = silu(a) b
    {
      float acc[MT][8][4];
      zero(acc);
      warp_gemm<MT, 8, 1, 8, XS>(acc, sx, ring, lane);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int rr = 16 * mt + g8 + 8 * hh, u = 32 * warp + 8 * j + 2 * q4;
            const float a0 = acc[mt][j][2 * hh], a1 = acc[mt][j][2 * hh + 1], b0 = acc[mt][4 + j][2 * hh], b1 = acc[mt][4 + j][2 * hh + 1];
            *reinterpret_cast<float2*>(sh + rr * HS + u) = make_float2(tf32f(a0 * sig(a0) * b0), tf32f(a1 * sig(a1) * b1));
          }
    }
    __syncthreads();
    // ---- ffn = h Wd^T (columns 16 warp ..); out = q1 + gate_f ffn
    {
      float acc[MT][2][4];
      zero(acc);
      warp_gemm<MT, 2, 4, 4, HS>(acc, sh, ring, lane);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int rr = 16 * mt + g8 + 8 * hh;
          const long r = t0 + rr;
          if (r >= p.M) continue;
          const float* mp = p.mod + mrow_of(r, p.S, p.B) * MODW;
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const int c = 16 * warp + 8 * nt + 2 * q4;
            const float2 q1 = *reinterpret_cast<const float2*>(sq + rr * XS + c), gf = ldg2(mp + 640 + c);
            *reinterpret_cast<float2*>(p.out + r * C + c) = make_float2(q1.x + gf.x * acc[mt][nt][2 * hh], q1.y + gf.y * acc[mt][nt][2 * hh + 1]);
          }
        }
    }
    __syncthreads();                                            // sx / sq / sh are free for the next tile
  }
}

template <int MT> constexpr int ffn_smem() { return (NW * ffn_ns<MT>() * CHUNK + 16 * MT * (2 * XS + HS)) * 4; }
static_assert(ffn_smem<1>() <= 101376, "99 KB");

}  // namespace sw120f
