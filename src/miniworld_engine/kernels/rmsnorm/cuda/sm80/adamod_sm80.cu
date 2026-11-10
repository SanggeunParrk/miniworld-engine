// A100 (sm_80) RMSNorm + adaLN modulation ("rms_norm_modulation": the atom stream, d_hidden = d_cond = 128, bf16), forward and backward.
//
//   forward:   y = rmsnorm(q) * (1 + c Wsc^T) + c Wsh^T,   gate = c Wg^T          (rmsnorm(q) = q rstd (w), rstd = 1 / sqrt(mean(q^2) + eps), all in fp32)
//   backward:  scale = c Wsc^T is recomputed;  dscale = dy * normed, dshift = dy, dnormed = dy (1 + scale), dq = rstd (w dnormed - xhat mean(xhat w dnormed)), dw = sum dnormed xhat
//              and [dscale | dy | dgate] is written as ONE [M, 3 N] buffer: the weight gradients and dc are then two cuBLAS GEMMs over it (the caller's).
//
// Forward: a persistent CTA keeps the three weight matrices (96 KB, XOR-swizzled 256-byte rows) in shared memory (one cp.async group each, behind its first tile: the scale GEMM starts when Wsc has
// landed and Wsh / Wg arrive under it -- every CTA fetches all 96 KB, which a small M waits for), streams 64-row tiles of c and q through a two-stage cp.async ring,
// and runs three [64 x 128] x [128 x 128] products on the tensor cores (mma.sync m16n8k16, ldmatrix fragments, fp32 accumulate).  Warp (mi, nq): rows 32 mi .. +32, columns 32 nq .. +32,
// so a B fragment feeds two m-tiles.  The row statistic comes from a pass over the q tile in shared memory; the epilogue works on the accumulator fragments and writes y over the q tile
// and gate over the c tile (the same element positions: no staging buffer), which the CTA then copies out with coalesced 16-byte stores.
#include "norm_common.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

using namespace norms;

namespace {

constexpr int D = 128;                      // normalised width = conditioning width
constexpr int BM = 64;                      // rows of a tile
constexpr int NTA = 256;                    // threads of a CTA (8 warps)
constexpr int W_BYTES = D * D * 2;          // one weight matrix
constexpr int TILE_BYTES = BM * D * 2;      // one [BM, D] bf16 tile
constexpr int FWD_SMEM = 3 * W_BYTES + 4 * TILE_BYTES + (D + BM) * 4;   // weights | two stages of (c, q) | w, rstd

// byte offset of 16-byte chunk `chunk` of row `row` of a [rows, 128] bf16 tile: the low three chunk bits XOR the row (ldmatrix of 8 rows x 16 bytes is then conflict free)
__device__ __forceinline__ unsigned swz(int row, int chunk) { return (unsigned)(row * 256 + (((chunk & 8) | ((chunk & 7) ^ (row & 7))) << 4)); }

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// acc[mt][nt] += A(rows 32 mi + 16 mt) x W^T(columns 32 nq + 8 nt) over the 128 channels: `afr` the warp's A fragments (two m-tiles x eight k-steps), `wbase` the shared-memory matrix
__device__ __forceinline__ void gemm_tile(float (&acc)[2][4][4], const unsigned (&afr)[2][8][4], unsigned wbase, int nq, int lane) {
  const int g8 = lane >> 3, r8 = lane & 7;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {                                   // n-tile pair 2p, 2p + 1
      const int n = 32 * nq + 16 * p + r8 + 8 * (g8 >> 1);
      unsigned b[4];
      ldsm_x4(b, wbase + swz(n, 2 * kk + (g8 & 1)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mma16816(acc[mt][2 * p], afr[mt][kk], b[0], b[1]);
        mma16816(acc[mt][2 * p + 1], afr[mt][kk], b[2], b[3]);
      }
    }
  }
}

template <bool HAS_W, bool SAVE>
__global__ void __launch_bounds__(NTA, 1) adamod_fwd(const bf* __restrict__ Q, const bf* __restrict__ C, const bf* __restrict__ WSC, const bf* __restrict__ WSH,
                                                     const bf* __restrict__ WG, const void* __restrict__ Wn, bool w_bf, bf* __restrict__ Y, bf* __restrict__ GATE,
                                                     float* __restrict__ RSTD, long M, float eps) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* Ws = smem;
  unsigned char* stages = smem + 3 * W_BYTES;
  float* wn = reinterpret_cast<float*>(smem + 3 * W_BYTES + 4 * TILE_BYTES);
  float* rstd_s = wn + D;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int mi = warp >> 2, nq = warp & 3;
  const unsigned smem_base = static_cast<unsigned>(__cvta_generic_to_shared(smem));

  if constexpr (HAS_W) {
    if (tid < D) wn[tid] = param_at(Wn, w_bf, tid);
  }
  const long ntiles = (M + BM - 1) / BM;
  auto issue = [&](long t, int stage) {
    const long row0 = t * BM;
    const int nr = (int)min((long)BM, M - row0);
    unsigned char* cs = stages + (long)stage * 2 * TILE_BYTES;
    unsigned char* qs = cs + TILE_BYTES;
    for (int c = tid; c < nr * 16; c += NTA) {
      const int row = c >> 4, ch = c & 15;
      cp_async16(cs + swz(row, ch), C + (row0 + row) * D + ch * 8);
      cp_async16(qs + swz(row, ch), Q + (row0 + row) * D + ch * 8);
    }
    cp_async_commit();
  };
  long tile = blockIdx.x;
  int s = 0;
  // copy groups, in order: this CTA's first tile, then the three weight matrices one group each -- a CTA with one tile (a small M) starts on the scale GEMM when the tile and Wsc
  // have landed, and Wsh / Wg arrive under it, instead of waiting for all 96 KB (every CTA fetches them: the L2 traffic is what a small M waits for)
  if (tile < ntiles) issue(tile, 0);
#pragma unroll
  for (int mat = 0; mat < 3; ++mat) {
    const bf* src = mat == 0 ? WSC : (mat == 1 ? WSH : WG);
    for (int c = tid; c < D * 16; c += NTA) cp_async16(Ws + mat * W_BYTES + swz(c >> 4, c & 15), src + (c >> 4) * D + (c & 15) * 8);
    cp_async_commit();
  }
  const int g8 = lane >> 3, r8 = lane & 7;
  bool first = true;
  for (; tile < ntiles; tile += gridDim.x, s ^= 1) {
    const long nxt = tile + gridDim.x;
    const bool has_next = nxt < ntiles;
    if (has_next) issue(nxt, s ^ 1);
    if (first) {                                                   // pending after the tile and Wsc: Wsh, Wg (and the next tile)
      if (has_next) cp_async_wait<3>(); else cp_async_wait<2>();
    } else {
      if (has_next) cp_async_wait<1>(); else cp_async_wait<0>();
    }
    __syncthreads();                                               // B1: Wsc (the first time) and this tile have landed
    const long row0 = tile * BM;
    const int nr = (int)min((long)BM, M - row0);
    unsigned char* cs = stages + (long)s * 2 * TILE_BYTES;
    unsigned char* qs = cs + TILE_BYTES;
    const unsigned cs_a = smem_base + (unsigned)(3 * W_BYTES + s * 2 * TILE_BYTES);

    // the A fragments of this warp's rows (c), and the row statistic from the q tile
    unsigned afr[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const int row = 32 * mi + 16 * mt + r8 + 8 * (g8 & 1);
        ldsm_x4(afr[mt][kk], cs_a + swz(row, 2 * kk + (g8 >> 1)));
      }
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int row = (tid >> 3) + 32 * pass, part = tid & 7;
      const uint4 u0 = *reinterpret_cast<const uint4*>(qs + swz(row, 2 * part));
      const uint4 u1 = *reinterpret_cast<const uint4*>(qs + swz(row, 2 * part + 1));
      float f0[8], f1[8];
      Chunk<bf>::unpack(u0, f0);
      Chunk<bf>::unpack(u1, f1);
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += f0[e] * f0[e] + f1[e] * f1[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      if (part == 0) {
        const float rs = rsqrtf(ss * (1.f / (float)D) + eps);
        rstd_s[row] = rs;
        if (SAVE && row < nr) RSTD[row0 + row] = rs;
      }
    }
    __syncthreads();                                               // B2: statistics visible, every warp has its A fragments

    float acc0[2][4][4], acc1[2][4][4];
    gemm_tile(acc0, afr, smem_base + 0 * W_BYTES, nq, lane);       // scale
    if (first) {                                                   // Wsh has landed (every thread's copies: wait, then barrier)
      if (has_next) cp_async_wait<2>(); else cp_async_wait<1>();
      __syncthreads();
    }
    gemm_tile(acc1, afr, smem_base + 1 * W_BYTES, nq, lane);       // shift
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int row = 32 * mi + 16 * mt + (lane >> 2) + 8 * i;
          const int col = 32 * nq + 8 * nt + 2 * (lane & 3);
          unsigned char* qa = qs + swz(row, col >> 3) + 2 * (col & 7);
          const uint32_t qq = *reinterpret_cast<const uint32_t*>(qa);
          const float rs = rstd_s[row];
          float w0 = 1.f, w1 = 1.f;
          if constexpr (HAS_W) { w0 = wn[col]; w1 = wn[col + 1]; }
          const float y0 = ((bf_lo(qq) * rs) * w0) * (1.f + acc0[mt][nt][2 * i]) + acc1[mt][nt][2 * i];
          const float y1 = ((bf_hi(qq) * rs) * w1) * (1.f + acc0[mt][nt][2 * i + 1]) + acc1[mt][nt][2 * i + 1];
          *reinterpret_cast<uint32_t*>(qa) = pack_bf2(y0, y1);
        }
    if (first) {                                                   // Wg has landed
      if (has_next) cp_async_wait<1>(); else cp_async_wait<0>();
      __syncthreads();
      first = false;
    }
    gemm_tile(acc0, afr, smem_base + 2 * W_BYTES, nq, lane);       // gate
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int row = 32 * mi + 16 * mt + (lane >> 2) + 8 * i;
          const int col = 32 * nq + 8 * nt + 2 * (lane & 3);
          *reinterpret_cast<uint32_t*>(cs + swz(row, col >> 3) + 2 * (col & 7)) = pack_bf2(acc0[mt][nt][2 * i], acc0[mt][nt][2 * i + 1]);
        }
    __syncthreads();                                               // B3: y over q, gate over c
    for (int c = tid; c < nr * 16; c += NTA) {
      const int row = c >> 4, ch = c & 15;
      *reinterpret_cast<uint4*>(Y + (row0 + row) * D + ch * 8) = *reinterpret_cast<const uint4*>(qs + swz(row, ch));
      *reinterpret_cast<uint4*>(GATE + (row0 + row) * D + ch * 8) = *reinterpret_cast<const uint4*>(cs + swz(row, ch));
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------ backward
// scale = c Wsc^T is recomputed on the tensor cores (the only GEMM of the stage; the weight gradients and dc are the caller's cuBLAS GEMMs over the stacked [dscale | dy | dgate]),
// then rows are processed by whole warps: lane l holds columns 4 l .. 4 l + 3 of q, dy, dgate (loaded one tile ahead, so their latency hides behind the GEMM), the row sum of
// xhat . w dnormed is a warp shuffle.  dscale / dq are the only new values; dy and dgate are copied into their slots of the stacked buffer.
constexpr int SC_LD = D + 4;                // fp32 row stride of the recomputed scale tile (4 banks a row: float2 accesses of 8 rows meet at most two-way)
constexpr int BWD_SMEM = W_BYTES + 2 * TILE_BYTES + BM * SC_LD * 4 + D * 4;
constexpr int RPWB = BM / 8;                // rows of a tile per warp

template <bool HAS_W>
__global__ void __launch_bounds__(NTA, 1) adamod_bwd(const bf* __restrict__ DY, const bf* __restrict__ DG, const bf* __restrict__ Q, const bf* __restrict__ C,
                                                     const bf* __restrict__ WSC, const void* __restrict__ Wn, bool w_bf, const float* __restrict__ RSTD, bf* __restrict__ DQ,
                                                     bf* __restrict__ DSD, float* __restrict__ DW, long M) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* Ws = smem;
  unsigned char* cst = smem + W_BYTES;                                  // two stages of the c tile
  float* sc_s = reinterpret_cast<float*>(smem + W_BYTES + 2 * TILE_BYTES);
  float* red = reinterpret_cast<float*>(smem + W_BYTES + 2 * TILE_BYTES + BM * SC_LD * 4);   // [128] dw partial
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int mi = warp >> 2, nq = warp & 3;
  const int g8 = lane >> 3, r8 = lane & 7;
  const unsigned smem_base = static_cast<unsigned>(__cvta_generic_to_shared(smem));

  for (int c = tid; c < D * 16; c += NTA) cp_async16(Ws + swz(c >> 4, c & 15), WSC + (c >> 4) * D + (c & 15) * 8);
  cp_async_commit();
  float wl[4] = {1.f, 1.f, 1.f, 1.f}, dwa[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (HAS_W) {
#pragma unroll
    for (int i = 0; i < 4; ++i) wl[i] = param_at(Wn, w_bf, 4 * lane + i);
  }
  const long ntiles = (M + BM - 1) / BM;
  auto issue = [&](long t, int stage) {
    const long row0 = t * BM;
    const int nr = (int)min((long)BM, M - row0);
    unsigned char* cs = cst + (long)stage * TILE_BYTES;
    for (int c = tid; c < nr * 16; c += NTA) cp_async16(cs + swz(c >> 4, c & 15), C + (row0 + (c >> 4)) * D + (c & 15) * 8);
    cp_async_commit();
  };
  // this warp's rows of tile t: q, dy, dgate (four bf16 = two registers each) and rstd
  uint2 pq[RPWB], pdy[RPWB], pdg[RPWB];
  float prs[RPWB];
  auto fetch = [&](long t) {
    const long row0 = t * BM;
#pragma unroll
    for (int j = 0; j < RPWB; ++j) {
      const long row = row0 + warp * RPWB + j;
      const bool ok = t * BM + warp * RPWB + j < M;
      pq[j] = ok ? *reinterpret_cast<const uint2*>(Q + row * D + 4 * lane) : make_uint2(0u, 0u);
      pdy[j] = ok ? *reinterpret_cast<const uint2*>(DY + row * D + 4 * lane) : make_uint2(0u, 0u);
      pdg[j] = ok ? *reinterpret_cast<const uint2*>(DG + row * D + 4 * lane) : make_uint2(0u, 0u);
      prs[j] = ok ? RSTD[row] : 0.f;
    }
  };
  long tile = blockIdx.x;
  int s = 0;
  if (tile < ntiles) { issue(tile, 0); fetch(tile); }
  for (; tile < ntiles; tile += gridDim.x, s ^= 1) {
    const long nxt = tile + gridDim.x;
    if (nxt < ntiles) {
      issue(nxt, s ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();                                                   // B1: the weights and this tile's c have landed
    const long row0 = tile * BM;
    const int nr = (int)min((long)BM, M - row0);
    const unsigned cs_a = smem_base + (unsigned)(W_BYTES + s * TILE_BYTES);
    unsigned afr[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const int row = 32 * mi + 16 * mt + r8 + 8 * (g8 & 1);
        ldsm_x4(afr[mt][kk], cs_a + swz(row, 2 * kk + (g8 >> 1)));
      }
    float acc[2][4][4];
    gemm_tile(acc, afr, smem_base, nq, lane);                          // scale
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int row = 32 * mi + 16 * mt + (lane >> 2) + 8 * i;
          const int col = 32 * nq + 8 * nt + 2 * (lane & 3);
          *reinterpret_cast<float2*>(sc_s + row * SC_LD + col) = make_float2(acc[mt][nt][2 * i], acc[mt][nt][2 * i + 1]);
        }
    __syncthreads();                                                   // B2: the scale tile is complete (and the c stage is free again)
    // the rows of this warp
#pragma unroll
    for (int j = 0; j < RPWB; ++j) {
      const int rl = warp * RPWB + j;
      const long row = row0 + rl;
      const float rs = prs[j];
      float q4[4], dy4[4], dg4[4];
      q4[0] = bf_lo(pq[j].x); q4[1] = bf_hi(pq[j].x); q4[2] = bf_lo(pq[j].y); q4[3] = bf_hi(pq[j].y);
      dy4[0] = bf_lo(pdy[j].x); dy4[1] = bf_hi(pdy[j].x); dy4[2] = bf_lo(pdy[j].y); dy4[3] = bf_hi(pdy[j].y);
      (void)dg4;
      const float4 sc = *reinterpret_cast<const float4*>(sc_s + rl * SC_LD + 4 * lane);
      const float sc4[4] = {sc.x, sc.y, sc.z, sc.w};
      float xh[4], wdy[4], dsc[4];
      float s1 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        xh[i] = q4[i] * rs;
        const float normed = xh[i] * wl[i];
        dsc[i] = dy4[i] * normed;
        const float dn = dy4[i] * (1.f + sc4[i]);
        if constexpr (HAS_W) dwa[i] = fmaf(dn, xh[i], dwa[i]);
        wdy[i] = dn * wl[i];
        s1 += xh[i] * wdy[i];
      }
      s1 = warp_sum(s1) * (1.f / (float)D);
      float dq4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) dq4[i] = (wdy[i] - xh[i] * s1) * rs;
      if (rl < nr) {
        *reinterpret_cast<uint2*>(DQ + row * D + 4 * lane) = make_uint2(pack_bf2(dq4[0], dq4[1]), pack_bf2(dq4[2], dq4[3]));
        bf* d = DSD + row * 3 * D;
        *reinterpret_cast<uint2*>(d + 4 * lane) = make_uint2(pack_bf2(dsc[0], dsc[1]), pack_bf2(dsc[2], dsc[3]));
        *reinterpret_cast<uint2*>(d + D + 4 * lane) = pdy[j];
        *reinterpret_cast<uint2*>(d + 2 * D + 4 * lane) = pdg[j];
      }
    }
    if (nxt < ntiles) fetch(nxt);
  }
  if constexpr (HAS_W) {
    // the lanes' column partials -> this CTA's row (a few atomics: the weight gradient of an RMSNorm weight is 128 floats)
    for (int i = tid; i < D; i += NTA) red[i] = 0.f;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) atomicAdd(red + 4 * lane + i, dwa[i]);
    __syncthreads();
    for (int i = tid; i < D; i += NTA) atomicAdd(DW + i, red[i]);
  }
}

bool aligned16(const void* p) { return p == nullptr || (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

template <bool HAS_W, bool SAVE>
void run_fwd(const at::Tensor& q, const at::Tensor& c, const at::Tensor& wsc, const at::Tensor& wsh, const at::Tensor& wg, const c10::optional<at::Tensor>& w, at::Tensor& y,
             at::Tensor& gate, const c10::optional<at::Tensor>& rstd, double eps) {
  auto k = adamod_fwd<HAS_W, SAVE>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, FWD_SMEM);
  const long M = q.size(0);
  const long ntiles = (M + BM - 1) / BM;
  const int grid = (int)std::min<long>(ntiles, sm_count());
  const bool w_bf = HAS_W && w->scalar_type() == at::kBFloat16;
  k<<<grid, NTA, FWD_SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf*>(q.data_ptr()), reinterpret_cast<const bf*>(c.data_ptr()), reinterpret_cast<const bf*>(wsc.data_ptr()), reinterpret_cast<const bf*>(wsh.data_ptr()),
      reinterpret_cast<const bf*>(wg.data_ptr()), HAS_W ? w->data_ptr() : nullptr, w_bf, reinterpret_cast<bf*>(y.data_ptr()), reinterpret_cast<bf*>(gate.data_ptr()),
      SAVE ? rstd->data_ptr<float>() : nullptr, M, (float)eps);
}

}  // namespace

// y, gate [M, 128] bf16 from q, c [M, 128] bf16 and the three [128, 128] bf16 weights; w: the RMSNorm weight [128] fp32 / bf16 or None; rstd: [M] fp32 (saved for the backward) or None
void adamod_fwd_op(at::Tensor q, at::Tensor c, at::Tensor wsc, at::Tensor wsh, at::Tensor wg, c10::optional<at::Tensor> w, at::Tensor y, at::Tensor gate, c10::optional<at::Tensor> rstd,
                   double eps) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.dim() == 2 && q.size(1) == D && q.scalar_type() == at::kBFloat16 && q.is_contiguous(), "adamod_fwd: q [M, 128] bf16 contiguous");
  TORCH_CHECK(c.sizes() == q.sizes() && c.scalar_type() == at::kBFloat16 && c.is_contiguous() && y.sizes() == q.sizes() && gate.sizes() == q.sizes(), "adamod_fwd: c, y, gate like q");
  for (const at::Tensor* t : {&wsc, &wsh, &wg})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->dim() == 2 && t->size(0) == D && t->size(1) == D && t->is_contiguous(), "adamod_fwd: weights [128, 128] bf16");
  TORCH_CHECK(aligned16(q.data_ptr()) && aligned16(c.data_ptr()) && aligned16(y.data_ptr()) && aligned16(gate.data_ptr()) && aligned16(wsc.data_ptr()) && aligned16(wsh.data_ptr()) &&
              aligned16(wg.data_ptr()), "adamod_fwd: 16-byte aligned operands");
  const bool has_w = w.has_value() && w->defined();
  if (has_w) TORCH_CHECK(w->is_cuda() && w->numel() == D && w->is_contiguous() && (w->scalar_type() == at::kFloat || w->scalar_type() == at::kBFloat16), "adamod_fwd: w [128] fp32 / bf16");
  const bool save = rstd.has_value() && rstd->defined();
  if (q.size(0) == 0) return;
  if (has_w) { if (save) run_fwd<true, true>(q, c, wsc, wsh, wg, w, y, gate, rstd, eps); else run_fwd<true, false>(q, c, wsc, wsh, wg, w, y, gate, rstd, eps); }
  else { if (save) run_fwd<false, true>(q, c, wsc, wsh, wg, w, y, gate, rstd, eps); else run_fwd<false, false>(q, c, wsc, wsh, wg, w, y, gate, rstd, eps); }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "adamod_fwd: launch failed");
}

// dq [M, 128] and dsd [M, 384] = [dscale | dy | dgate] (bf16) from dy, dgate, q, c [M, 128], wsc, the saved rstd [M] and the RMSNorm weight (dw [128] fp32, zeroed by the caller: accumulated)
void adamod_bwd_op(at::Tensor dy, at::Tensor dg, at::Tensor q, at::Tensor c, at::Tensor wsc, c10::optional<at::Tensor> w, at::Tensor rstd, at::Tensor dq, at::Tensor dsd,
                   c10::optional<at::Tensor> dw) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.dim() == 2 && q.size(1) == D && q.scalar_type() == at::kBFloat16 && q.is_contiguous(), "adamod_bwd: q [M, 128] bf16 contiguous");
  for (const at::Tensor* t : {&dy, &dg, &c, &dq})
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == at::kBFloat16 && t->is_contiguous(), "adamod_bwd: dy, dgate, c, dq like q");
  TORCH_CHECK(dsd.dim() == 2 && dsd.size(0) == q.size(0) && dsd.size(1) == 3 * D && dsd.scalar_type() == at::kBFloat16 && dsd.is_contiguous(), "adamod_bwd: dsd [M, 384] bf16");
  TORCH_CHECK(wsc.is_cuda() && wsc.scalar_type() == at::kBFloat16 && wsc.dim() == 2 && wsc.size(0) == D && wsc.size(1) == D && wsc.is_contiguous(), "adamod_bwd: wsc [128, 128] bf16");
  TORCH_CHECK(rstd.is_cuda() && rstd.scalar_type() == at::kFloat && rstd.numel() == q.size(0), "adamod_bwd: rstd [M] fp32");
  TORCH_CHECK(aligned16(q.data_ptr()) && aligned16(c.data_ptr()) && aligned16(dy.data_ptr()) && aligned16(dg.data_ptr()) && aligned16(dq.data_ptr()) && aligned16(dsd.data_ptr()) &&
              aligned16(wsc.data_ptr()), "adamod_bwd: 16-byte aligned operands");
  const bool has_w = w.has_value() && w->defined();
  if (has_w) {
    TORCH_CHECK(w->is_cuda() && w->numel() == D && w->is_contiguous() && (w->scalar_type() == at::kFloat || w->scalar_type() == at::kBFloat16), "adamod_bwd: w [128] fp32 / bf16");
    TORCH_CHECK(dw.has_value() && dw->defined() && dw->scalar_type() == at::kFloat && dw->numel() == D, "adamod_bwd: dw [128] fp32");
  }
  const long M = q.size(0);
  if (M == 0) return;
  const long ntiles = (M + BM - 1) / BM;
  const int grid = (int)std::min<long>(ntiles, sm_count());
  const bool w_bf = has_w && w->scalar_type() == at::kBFloat16;
  auto launch = [&](auto kern) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, BWD_SMEM);
    kern<<<grid, NTA, BWD_SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf*>(dy.data_ptr()), reinterpret_cast<const bf*>(dg.data_ptr()), reinterpret_cast<const bf*>(q.data_ptr()), reinterpret_cast<const bf*>(c.data_ptr()),
        reinterpret_cast<const bf*>(wsc.data_ptr()), has_w ? w->data_ptr() : nullptr, w_bf, rstd.data_ptr<float>(), reinterpret_cast<bf*>(dq.data_ptr()),
        reinterpret_cast<bf*>(dsd.data_ptr()), has_w ? dw->data_ptr<float>() : nullptr, M);
  };
  if (has_w) launch(adamod_bwd<true>);
  else launch(adamod_bwd<false>);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "adamod_bwd: launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adamod_fwd", &adamod_fwd_op);
  m.def("adamod_bwd", &adamod_bwd_op);
}
