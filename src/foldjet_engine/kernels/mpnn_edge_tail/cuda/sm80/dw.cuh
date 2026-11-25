// dw.cuh -- the weight gradients of the edge tail (and of the edge MLP) on the A100 (sm_80): dW_l[o][i] = sum_r G_l[r][o] A_l[r][i] over all rows, three matrices in one launch, plus the
// column sums of G_l (the bias gradients) from a ones column of the same product.
//
//   job 0: G = G3 (gradient of the third product), A = rn(gelu(hid))        -> dW3, db3
//   job 1: G = G2,                                  A = rn(gelu(pre))        -> dW2, db2
//   job 2: G = G1,                                  A = edge                 -> dW1e
//
// A CTA = 8 warps owns (job, row slab) and walks its slab in stages of 64 rows through a two-stage shared-memory ring (G tile by cp.async, A tile by cp.async, or by registers when
// it needs the GELU).  Each warp owns a 32 x 64 block of the 128 x 128 result (A operand = the transposed G tile, B operand = the A tile, both by ldmatrix.trans) and the warps of the
// first column half also run one extra n tile against a constant ones fragment, whose accumulator is the column sum of G.  Partial results of every (job, slab) go to a fp32 buffer,
// summed in a fixed order by ``dw_finalize_kernel`` (no atomics: a replay is bit-identical).
#pragma once
#include "common.cuh"

namespace me80 {

struct DwParams {
  const __nv_bfloat16* g[3];       // G3, G2, G1: [rows][128]
  const __nv_bfloat16* a[3];       // hid, pre, edge
  int njob;                        // 3 (tail) or 2 (MLP: jobs 0 and 1)
  int rows, slabs, slab_rows;      // slab_rows: a multiple of 64
  float* part;                     // [njob][slabs][128][128]
  float* csum;                     // [njob][slabs][128]
};

constexpr int DW_STAGE = 2 * 64 * 256;     // G tile + A tile of 64 rows (256 B each)
// GELU = the A operand is rn(gelu(a)) (the edge MLP: a = hid, x): two stages, the next A tile in registers until it is activated.  Otherwise (the edge tail: the chain kernel saved the
// GELU outputs) the A tiles are plain bf16 and every tile comes through a three-stage cp.async ring.
constexpr int DW_SMEM_GELU = 2 * DW_STAGE, DW_SMEM_PLAIN = 3 * DW_STAGE;

template <bool GELU>
__global__ void __launch_bounds__(256, 1) dw_kernel(const DwParams p) {
  constexpr int NS = GELU ? 2 : 3;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int job = blockIdx.x / p.slabs, slab = blockIdx.x - job * p.slabs;
  const __nv_bfloat16* __restrict__ G = p.g[job];
  const __nv_bfloat16* __restrict__ A = p.a[job];
  const int row_lo = slab * p.slab_rows;
  const int row_hi = min(p.rows, row_lo + p.slab_rows);
  const int nstage = row_hi > row_lo ? (row_hi - row_lo + 63) >> 6 : 0;

  const int wm = warp >> 1, wn = warp & 1;                         // 4 x 2 warps: o rows 32 wm .., i columns 64 wn ..
  const int ar = (((lane >> 4) & 1) << 3) + (lane & 7), ag = (lane >> 3) & 1;      // A (transposed G tile) ldmatrix lane -> (k row, 8-column granule)
  const int br = (((lane >> 3) & 1) << 3) + (lane & 7), bg = lane >> 4;            // B (A tile)

  float acc[2][8][4], accs[2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
    accs[mt][0] = accs[mt][1] = accs[mt][2] = accs[mt][3] = 0.f;
  }

  uint4 areg[4];                                                   // the A tile of the next stage, in registers until it is activated (GELU jobs)
  auto issue = [&](int stage, int buf) {
    const int row0 = row_lo + stage * 64;
    const uint32_t gt = sb + buf * DW_STAGE, at = gt + 64 * 256;
    if (stage < nstage) {                                          // past the slab: an empty group, so the wait counts stay uniform
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + 256 * k, r = i >> 4, c = i & 15;
      const bool v = row0 + r < p.rows;
      const size_t off = (size_t)(v ? row0 + r : 0) * D + 8 * c;
      cp_async16z(gt + sw256(r, c), G + off, v ? 16u : 0u);
      if (GELU) areg[k] = v ? ldg128(A + off) : make_uint4(0u, 0u, 0u, 0u);
      else cp_async16z(at + sw256(r, c), A + off, v ? 16u : 0u);
    }
    }
    cp_async_commit();
  };
  auto activate = [&](int buf) {
    const uint32_t at = sb + buf * DW_STAGE + 64 * 256;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + 256 * k, r = i >> 4, c = i & 15;
      uint4 o;
      o.x = pack_bf16(gelu(bf16lo(areg[k].x)), gelu(bf16hi(areg[k].x)));
      o.y = pack_bf16(gelu(bf16lo(areg[k].y)), gelu(bf16hi(areg[k].y)));
      o.z = pack_bf16(gelu(bf16lo(areg[k].z)), gelu(bf16hi(areg[k].z)));
      o.w = pack_bf16(gelu(bf16lo(areg[k].w)), gelu(bf16hi(areg[k].w)));
      sts128(at + sw256(r, c), o);
    }
  };

  if (GELU) {
    if (nstage > 0) { issue(0, 0); activate(0); }
  } else {
    issue(0, 0);                                                   // the ring holds 2 stages in flight: a group per stage, empty ones past the end
    issue(1, 1);
  }
  for (int s = 0; s < nstage; ++s) {
    if (GELU) cp_async_wait<0>(); else cp_async_wait<1>();
    __syncthreads();                                               // stage s is in shared memory; every warp is past stage s - 1
    if (GELU) { if (s + 1 < nstage) issue(s + 1, (s + 1) & 1); }
    else issue(s + 2, (s + 2) % 3);
    const uint32_t gt = sb + (s % NS) * DW_STAGE, at = gt + 64 * 256;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[2][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4_t(af[mt], gt + sw256(16 * kk + ar, ((32 * wm + 16 * mt) >> 3) + ag));
#pragma unroll
      for (int n2 = 0; n2 < 4; ++n2) ldsm_x4_t(bf[n2], at + sw256(16 * kk + br, ((64 * wn + 16 * n2) >> 3) + bg));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int n2 = 0; n2 < 4; ++n2) {
          mma16816(acc[mt][2 * n2], af[mt], bf[n2][0], bf[n2][1]);
          mma16816(acc[mt][2 * n2 + 1], af[mt], bf[n2][2], bf[n2][3]);
        }
        if (wn == 0) mma16816(accs[mt], af[mt], ONE2, ONE2);
      }
    }
    if (GELU && s + 1 < nstage) activate((s + 1) & 1);
  }

  // ---- partial results of (job, slab): acc[mt][nt] = rows o = 32 wm + 16 mt + g8 (+ 8), columns i = 64 wn + 8 nt + 2 q4 (+ 1)
  float* part = p.part + (size_t)blockIdx.x * (D * D);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int o = 32 * wm + 16 * mt + g8, i = 64 * wn + 8 * nt + 2 * q4;
      *reinterpret_cast<float2*>(part + (size_t)o * D + i) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<float2*>(part + (size_t)(o + 8) * D + i) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
  if (wn == 0 && q4 == 0) {
    float* cs = p.csum + (size_t)blockIdx.x * D;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) { cs[32 * wm + 16 * mt + g8] = accs[mt][0]; cs[32 * wm + 16 * mt + g8 + 8] = accs[mt][2]; }
  }
}

// ---- the fixed-order sums of every partial buffer, straight into the parameters' dtypes
struct FinParams {
  const float* part;               // [njob][slabs][128][128]
  const float* csum;               // [njob][slabs][128]
  const float* ln_part;            // [nwarps][2][128] or nullptr
  int njob, slabs, nwarps;
  void* dw[3]; int dw_fp32[3];     // per job: dW3, dW2, dW1e
  void* db[2]; int db_fp32[2];     // db3 (job 0), db2 (job 1)
  void* dn[2]; int dn_fp32[2];     // dgamma, dbeta
};

DEVI void store_cvt4(void* base, int fp32, size_t i, float4 v) {
  if (fp32) *reinterpret_cast<float4*>(reinterpret_cast<float*>(base) + i) = v;
  else *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(base) + i) = make_uint2(pack_bf16(v.x, v.y), pack_bf16(v.z, v.w));
}

// The sums over the slabs / CTAs, in a fixed order.  Region A (the weight gradients and the bias column sums): one thread per 4 consecutive outputs (coalesced), the slabs added in
// increasing order, the loads unrolled.  Region B (the LayerNorm partials: 2 x 128 outputs, one partial per CTA of the chain kernel): one warp per 4 outputs, lane l adds the partials
// l, l + 32, ... in increasing order and the 32 sums are combined by a butterfly (a fixed order), so no thread walks the ~100 partials alone.
DEVI float4 butterfly_sum4(float4 s) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s.x += __shfl_xor_sync(0xffffffffu, s.x, o); s.y += __shfl_xor_sync(0xffffffffu, s.y, o); s.z += __shfl_xor_sync(0xffffffffu, s.z, o); s.w += __shfl_xor_sync(0xffffffffu, s.w, o);
  }
  return s;
}

__global__ void __launch_bounds__(256) dw_finalize_kernel(const FinParams p) {
  const int nmat = p.njob * D * D;
  const int nA = (nmat + 2 * D) / 4;                              // float4 outputs of region A: the matrices and the two bias vectors
  const int t = blockIdx.x * 256 + threadIdx.x;
  if (t < nA) {
    const int i = t * 4;
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < nmat) {
      const int job = i / (D * D), e = i - job * D * D;
      const float* base = p.part + (size_t)job * p.slabs * (D * D) + e;
#pragma unroll 12
      for (int k = 0; k < p.slabs; ++k) {
        const float4 v = *reinterpret_cast<const float4*>(base + (size_t)k * (D * D));
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      store_cvt4(p.dw[job], p.dw_fp32[job], e, s);
    } else {
      const int j = i - nmat, v = j >> 7, c = j & 127;
      if (v < p.njob) {
        const float* base = p.csum + (size_t)v * p.slabs * D + c;
#pragma unroll 12
        for (int k = 0; k < p.slabs; ++k) {
          const float4 x = *reinterpret_cast<const float4*>(base + (size_t)k * D);
          s.x += x.x; s.y += x.y; s.z += x.z; s.w += x.w;
        }
        // the bias gradient is rounded to bf16 before the parameter dtype (autocast Linear's boundary)
        store_cvt4(p.db[v], p.db_fp32[v], c, make_float4(round_bf16f(s.x), round_bf16f(s.y), round_bf16f(s.z), round_bf16f(s.w)));
      }
    }
  } else if (p.ln_part != nullptr) {
    const int w = (t - ((nA + 255) / 256) * 256) >> 5, lane = threadIdx.x & 31;      // region B starts at the first block past region A
    if (blockIdx.x >= (nA + 255) / 256 && w < 2 * D / 4) {
      const int v = w >> 5, c = (w & 31) * 4;                                         // 32 float4 outputs per vector
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int k = lane; k < p.nwarps; k += 32) {
        const float4 x = *reinterpret_cast<const float4*>(p.ln_part + (size_t)k * 256 + v * 128 + c);
        s.x += x.x; s.y += x.y; s.z += x.z; s.w += x.w;
      }
      s = butterfly_sum4(s);
      if (lane == 0) store_cvt4(p.dn[v], p.dn_fp32[v], c, s);
    }
  }
}

}  // namespace me80
