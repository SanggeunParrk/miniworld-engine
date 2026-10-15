// WIDE pair-side kernels (d_pair a multiple of 64: 128 / 256 / 384).  pair3.cu with the c_z width a compile-time macro (PW_DZ),
// the softmax / softmax-backward looping over the heads when there are fewer warps than heads (PW_NWARP), the logits kept in
// bf16 (they are bf16-rounded values already: halves their shared footprint) and the d_pair-wide accumulators sized by it.
// Pair-side kernels of the PWA block on tensor cores (sm_90a): per pair row i, LayerNorm(DZ) -> proj_z (8 heads) ->
// mask -> softmax over j, and its backward.  A warp processes 16 j's at a time: the [16 j][128 d] z tile is
// cp.async-staged (swizzled), read as mma A fragments with ldmatrix, normalised in registers (the row statistics are
// quad shuffles), and the projection is eight m16n8k16 mma with Wb^T as the B operand.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace {
#ifndef PW_DZ
#define PW_DZ 128
#endif
#ifndef PW_NWARP
#define PW_NWARP 8
#endif
#ifndef PW_MINB
#define PW_MINB 1
#endif
constexpr int DZ = PW_DZ, H = 8, NWARP = PW_NWARP, THREADS = NWARP * 32, GJ = 16;
constexpr int KS_ = DZ / 16, CPR = DZ / 8;                // k steps / 16-byte chunks per row
static_assert(DZ % 64 == 0, "d_pair must be a multiple of 64");
constexpr int TILEB = GJ * DZ * 2;                        // one [16][DZ] bf16 tile

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint32_t pack2(float a, float b) { const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b)); return *reinterpret_cast<const uint32_t*>(&h); }
__device__ __forceinline__ float2 unpack2(uint32_t u) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u)); }
__device__ __forceinline__ float qsum(float v) { v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2); return v; }
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// swizzled [16][128] bf16 tile: row r at r*256 B, 16-byte chunk ch stored at chunk (ch ^ (r & 7))
__device__ __forceinline__ uint32_t tile_off(int r, int ch) { return (uint32_t)(r * (DZ * 2) + ((ch ^ (r & 7)) << 4)); }
// A fragments (a0..a3) of the 16x16 sub-tile at column 16*ks: lane l addresses matrix l/8, row l%8
__device__ __forceinline__ void ldm_a(uint32_t tile, int ks, int lane, uint32_t* a) {
  const int m = lane >> 3, r = (lane & 7) + ((m & 1) << 3), ch = 2 * ks + (m >> 1);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(tile + tile_off(r, ch)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_tile(uint32_t dst, const __nv_bfloat16* src_row0, int lane) {   // 16 rows x DZ*2 B
#pragma unroll
  for (int q = 0; q < KS_; ++q) {
    const int v = q * 32 + lane, r = v / CPR, ch = v % CPR;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst + tile_off(r, ch)), "l"(src_row0 + (long)r * DZ + ch * 8) : "memory");
  }
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// ---------------------------------------------------------------- forward ----------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, PW_MINB) pair_fwd_kernel(const __nv_bfloat16* __restrict__ Z, const __nv_bfloat16* __restrict__ MASK,
                                                              const float* __restrict__ LNW, const float* __restrict__ LNB, const __nv_bfloat16* __restrict__ WB,
                                                              __nv_bfloat16* __restrict__ W, int N, int NP, float eps) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* tiles = smem;                                          // [NWARP][2][TILEB]
  __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem + NWARP * 2 * TILEB);   // [H][NP] logits (bf16-rounded values)
  float* sg = reinterpret_cast<float*>(sb + H * NP);                    // gamma [DZ], beta [DZ]
  const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3, r = lane >> 2;
  for (int k = tid; k < DZ; k += THREADS) { sg[k] = LNW[k]; sg[DZ + k] = LNB[k]; }
  // B fragments of Wb^T [k = d][n = h]: b0 = (Wb[n][16ks + 2q], +1), b1 = (+8, +9), n = lane / 4
  uint32_t bw[KS_][2];
#pragma unroll
  for (int ks = 0; ks < KS_; ++ks) {
    bw[ks][0] = *reinterpret_cast<const uint32_t*>(WB + r * DZ + 16 * ks + 2 * q);
    bw[ks][1] = *reinterpret_cast<const uint32_t*>(WB + r * DZ + 16 * ks + 2 * q + 8);
  }
  __syncthreads();
  const uint32_t tbase = sa(tiles + warp * 2 * TILEB);
  const int ngroups = N / GJ;
  int gi = warp;
  if (gi < ngroups) cp_tile(tbase, Z + ((long)i * N + gi * GJ) * DZ, lane);
  cp_commit();
  for (int it = 0; gi < ngroups; ++it, gi += NWARP) {
    const int stage = it & 1;
    if (gi + NWARP < ngroups) cp_tile(tbase + (stage ^ 1) * TILEB, Z + ((long)i * N + (gi + NWARP) * GJ) * DZ, lane);
    cp_commit();
    cp_wait<1>();
    __syncwarp();
    const uint32_t tile = tbase + stage * TILEB;
    uint32_t a[KS_][4];
#pragma unroll
    for (int ks = 0; ks < KS_; ++ks) ldm_a(tile, ks, lane, a[ks]);
    // row statistics: this lane holds 32 values of row r (a0, a2) and of row r+8 (a1, a3); a quad completes a row
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS_; ++ks) {
      const float2 x0 = unpack2(a[ks][0]), x2 = unpack2(a[ks][2]), x1 = unpack2(a[ks][1]), x3 = unpack2(a[ks][3]);
      s0 += x0.x + x0.y + x2.x + x2.y; s1 += x1.x + x1.y + x3.x + x3.y;
    }
    const float m0 = qsum(s0) * (1.f / DZ), m1 = qsum(s1) * (1.f / DZ);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS_; ++ks) {
      const float2 x0 = unpack2(a[ks][0]), x2 = unpack2(a[ks][2]), x1 = unpack2(a[ks][1]), x3 = unpack2(a[ks][3]);
      v0 += (x0.x - m0) * (x0.x - m0) + (x0.y - m0) * (x0.y - m0) + (x2.x - m0) * (x2.x - m0) + (x2.y - m0) * (x2.y - m0);
      v1 += (x1.x - m1) * (x1.x - m1) + (x1.y - m1) * (x1.y - m1) + (x3.x - m1) * (x3.x - m1) + (x3.y - m1) * (x3.y - m1);
    }
    const float rs0 = 1.f / sqrtf(qsum(v0) * (1.f / DZ) + eps), rs1 = 1.f / sqrtf(qsum(v1) * (1.f / DZ) + eps);
    // zn = bf16((x - mean) * rstd * gamma + beta) into A fragments, then the projection
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < KS_; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sg + col), g1 = *reinterpret_cast<const float2*>(sg + col + 8);
      const float2 b0 = *reinterpret_cast<const float2*>(sg + DZ + col), b1 = *reinterpret_cast<const float2*>(sg + DZ + col + 8);
      const float2 x0 = unpack2(a[ks][0]), x1 = unpack2(a[ks][1]), x2 = unpack2(a[ks][2]), x3 = unpack2(a[ks][3]);
      uint32_t zn[4];
      zn[0] = pack2((x0.x - m0) * rs0 * g0.x + b0.x, (x0.y - m0) * rs0 * g0.y + b0.y);
      zn[1] = pack2((x1.x - m1) * rs1 * g0.x + b0.x, (x1.y - m1) * rs1 * g0.y + b0.y);
      zn[2] = pack2((x2.x - m0) * rs0 * g1.x + b1.x, (x2.y - m0) * rs0 * g1.y + b1.y);
      zn[3] = pack2((x3.x - m1) * rs1 * g1.x + b1.x, (x3.y - m1) * rs1 * g1.y + b1.y);
      mma_bf16(c, zn, bw[ks][0], bw[ks][1]);
    }
    // c0, c1: (row r, heads 2q, 2q+1); c2, c3: (row r+8)
    const int j0 = gi * GJ;
    const float mk0 = __bfloat162float(MASK[j0 + r]), mk1 = __bfloat162float(MASK[j0 + r + 8]);
    const __nv_bfloat16 neg = __float2bfloat16(-1e30f);
    sb[(2 * q) * NP + j0 + r] = mk0 > 0.5f ? __float2bfloat16(c[0]) : neg;
    sb[(2 * q + 1) * NP + j0 + r] = mk0 > 0.5f ? __float2bfloat16(c[1]) : neg;
    sb[(2 * q) * NP + j0 + r + 8] = mk1 > 0.5f ? __float2bfloat16(c[2]) : neg;
    sb[(2 * q + 1) * NP + j0 + r + 8] = mk1 > 0.5f ? __float2bfloat16(c[3]) : neg;
    __syncwarp();                                                       // this stage may be refilled next iteration
  }
  cp_wait<0>();
  __syncthreads();
  for (int h = warp; h < H; h += NWARP) {                               // softmax over j for head h
    float mx = -1e30f;
    for (int j = lane; j < N; j += 32) mx = fmaxf(mx, __bfloat162float(sb[h * NP + j]));
    mx = wmax(mx);
    float den = 0.f;
    for (int j = lane; j < N; j += 32) den += __expf(__bfloat162float(sb[h * NP + j]) - mx);
    den = wsum(den);
    const float inv = 1.f / den;
    for (int j = lane; j < N; j += 32) W[((long)h * N + i) * N + j] = __float2bfloat16(__expf(__bfloat162float(sb[h * NP + j]) - mx) * inv);
  }
}

// ---------------------------------------------------------------- backward ----------------------------------------------------------------
// db = bf16(w (dw - sum_j w dw)); dzn = db . Wb (mma, K = 8 heads padded to 16); dz = LN_bwd(dzn) (registers);
// dWb^T[d][h] += zn^T . db (mma, A = zn tile read transposed with ldmatrix.trans); dgamma / dbeta partials in registers.
// A fragments of the transposed 16x16 sub-tile (d rows 16*mt.., j cols) of a [16 j][128 d] tile
__device__ __forceinline__ void ldm_at(uint32_t tile, int mt, int lane, uint32_t* a) {
  const int m = lane >> 3, jrow = (lane & 7) + ((m >> 1) << 3), ch = 2 * mt + (m & 1);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(tile + tile_off(jrow, ch)));
}
__global__ void __launch_bounds__(THREADS, PW_MINB) pair_bwd_kernel(const __nv_bfloat16* __restrict__ Z, const __nv_bfloat16* __restrict__ W16, const __nv_bfloat16* __restrict__ DW,
                                                              const float* __restrict__ LNW, const float* __restrict__ LNB, const __nv_bfloat16* __restrict__ WB,
                                                              __nv_bfloat16* __restrict__ DZO, float* __restrict__ PWB, float* __restrict__ PLN, int N, float eps) {
  // Per row i.  db = bf16(w (dw - sum_j w dw)); dzn = db . Wb (mma, K = 8 heads padded to 16, computed twice: once for the
  // row sums, once for dz); dz = LN_bwd(dzn).  The parameter gradients come from two per-row reductions instead of
  // per-element accumulators:  M^T[d][h] = sum_j xhat[j][d] db[j][h] (mma, A = the xhat tile read transposed) and
  // S[h] = sum_j db[j][h]:  dWb[h][d] = gamma[d] M^T[d][h] + beta[d] S[h],  dgamma[d] = sum_h Wb[h][d] M^T[d][h],
  // dbeta[d] = sum_h Wb[h][d] S[h]  (dzn = db . Wb is linear in db, zn = xhat gamma + beta).
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* tiles = smem;                                          // [NWARP][3][TILEB]: two x stages + one aux (xhat / dz) tile per warp
  float* sg = reinterpret_cast<float*>(smem + NWARP * 2 * TILEB);       // gamma [DZ], beta [DZ]
  float* acc = sg + 2 * DZ;                                             // [H][DZ] M^T (as [h][d]), [DZ] unused, [H] S
  uint32_t* wbt2 = reinterpret_cast<uint32_t*>(acc + H * DZ + DZ + H);  // [DZ][4]: pack(Wb[2q][d], Wb[2q+1][d])
  __nv_bfloat16* sdb = reinterpret_cast<__nv_bfloat16*>(wbt2 + DZ * 4); // [N][H] db (bf16)
  uint2* sPt = reinterpret_cast<uint2*>(sdb + (size_t)N * H);           // [2][KS_][32]: B fragments of P = gamma . Wb, hi then lo bf16 parts
  float* sc = reinterpret_cast<float*>(sPt + 2 * KS_ * 32);             // [H]: c_h = sum_d P[h][d]
  float* sdw = reinterpret_cast<float*>(tiles);                         // staging of dw / w rows [H][N] fp32 each, aliased on the tile area
  float* sw = sdw + H * N;
  const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3, r = lane >> 2;
  for (int k = tid; k < DZ; k += THREADS) { sg[k] = LNW[k]; sg[DZ + k] = LNB[k]; }
  for (int k = tid; k < H * DZ + DZ + H; k += THREADS) acc[k] = 0.f;
  for (int k = tid; k < DZ * 4; k += THREADS) { const int d = k >> 2, qq = k & 3; wbt2[k] = pack2(__bfloat162float(WB[(2 * qq) * DZ + d]), __bfloat162float(WB[(2 * qq + 1) * DZ + d])); }
  for (int k0 = 0; k0 < H * N; k0 += 8 * THREADS) {                     // staged in chunks of 8 per thread: all 16 loads in flight before the stores
    float vd[8], vw[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const int k = k0 + m * THREADS + tid;
      if (k < H * N) { const int h = k / N, j = k - h * N; const long base = ((long)h * N + i) * N + j; vd[m] = __bfloat162float(DW[base]); vw[m] = __bfloat162float(W16[base]); }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) { const int k = k0 + m * THREADS + tid; if (k < H * N) { sdw[k] = vd[m]; sw[k] = vw[m]; } }
  }
  __syncthreads();
  for (int h = warp; h < H; h += NWARP) {                               // softmax backward, head h: sdot, db (bf16), S = sum_j db

    float sdot = 0.f;
    for (int j = lane; j < N; j += 32) sdot += sw[h * N + j] * sdw[h * N + j];
    sdot = wsum(sdot);
    float S = 0.f;
    for (int j = lane; j < N; j += 32) { const __nv_bfloat16 v = __float2bfloat16(sw[h * N + j] * (sdw[h * N + j] - sdot)); sdb[j * H + h] = v; S += __bfloat162float(v); }
    S = wsum(S);
    if (lane == 0) acc[H * DZ + DZ + h] = S;
  }
  // P[h][d] = gamma[d] Wb[h][d] as the two bf16 mma operands of sum_d x_d P[h][d] (B fragment: lane (n = h = lane / 4, q) holds
  // k = 2q, 2q + 1 and 2q + 8, 2q + 9 of each 16-wide d block), and c_h = sum_d P[h][d] in fp32 in a fixed order
  for (int k = tid; k < 2 * KS_ * 32; k += THREADS) {
    const int part = k / (KS_ * 32), ks = (k / 32) % KS_, ln = k & 31, hh = ln >> 2, qq = ln & 3;
    const int d = 16 * ks + 2 * qq;
    const float p0 = sg[d] * __bfloat162float(WB[hh * DZ + d]), p1 = sg[d + 1] * __bfloat162float(WB[hh * DZ + d + 1]);
    const float p2 = sg[d + 8] * __bfloat162float(WB[hh * DZ + d + 8]), p3 = sg[d + 9] * __bfloat162float(WB[hh * DZ + d + 9]);
    if (part == 0) sPt[k] = make_uint2(pack2(p0, p1), pack2(p2, p3));
    else {
      sPt[k] = make_uint2(pack2(p0 - bf16r(p0), p1 - bf16r(p1)), pack2(p2 - bf16r(p2), p3 - bf16r(p3)));
    }
  }
  if (tid < H) { float s = 0.f; for (int d = 0; d < DZ; ++d) s += sg[d] * __bfloat162float(WB[tid * DZ + d]); sc[tid] = s; }
  __syncthreads();                                                      // the staging area becomes the tile ring
  float cm[KS_][4];                                                      // M^T accumulators: rows d = 16 mt + r / + 8, cols h = 2q, 2q+1
#pragma unroll
  for (int mt = 0; mt < KS_; ++mt) { cm[mt][0] = cm[mt][1] = cm[mt][2] = cm[mt][3] = 0.f; }
  const uint32_t tbase = sa(tiles + warp * 2 * TILEB);               // two x stages per warp; xhat^T for the M^T mma comes from registers (movmatrix)
  const int ngroups = N / GJ;
  int gi = warp;
  if (gi < ngroups) cp_tile(tbase, Z + ((long)i * N + gi * GJ) * DZ, lane);
  cp_commit();
  for (int it = 0; gi < ngroups; ++it, gi += NWARP) {
    const int stage = it & 1, j0 = gi * GJ;
    if (gi + NWARP < ngroups) cp_tile(tbase + (stage ^ 1) * TILEB, Z + ((long)i * N + (gi + NWARP) * GJ) * DZ, lane);
    cp_commit();
    cp_wait<1>();
    __syncwarp();
    const uint32_t tile = tbase + stage * TILEB;
    // row statistics in one pass (sum and sum of squares)
    float s0 = 0.f, s1 = 0.f, v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS_; ++ks) {
      uint32_t a[4]; ldm_a(tile, ks, lane, a);
      const float2 x0 = unpack2(a[0]), x2 = unpack2(a[2]), x1 = unpack2(a[1]), x3 = unpack2(a[3]);
      s0 += x0.x + x0.y + x2.x + x2.y; s1 += x1.x + x1.y + x3.x + x3.y;
      v0 += x0.x * x0.x + x0.y * x0.y + x2.x * x2.x + x2.y * x2.y; v1 += x1.x * x1.x + x1.y * x1.y + x3.x * x3.x + x3.y * x3.y;
    }
    const float m0 = qsum(s0) * (1.f / DZ), m1 = qsum(s1) * (1.f / DZ);
    const float rs0 = 1.f / sqrtf(fmaxf(qsum(v0) * (1.f / DZ) - m0 * m0, 0.f) + eps), rs1 = 1.f / sqrtf(fmaxf(qsum(v1) * (1.f / DZ) - m1 * m1, 0.f) + eps);
    uint32_t adb[4];
    adb[0] = *reinterpret_cast<const uint32_t*>(sdb + (j0 + r) * H + 2 * q);
    adb[1] = *reinterpret_cast<const uint32_t*>(sdb + (j0 + r + 8) * H + 2 * q);
    adb[2] = 0u; adb[3] = 0u;
    float gs0 = 0.f, gx0 = 0.f, gs1 = 0.f, gx1 = 0.f;
    const __nv_bfloat16 e0 = sdb[(j0 + 2 * q) * H + r], e1 = sdb[(j0 + 2 * q + 1) * H + r], e2 = sdb[(j0 + 2 * q + 8) * H + r], e3 = sdb[(j0 + 2 * q + 9) * H + r];
    const uint32_t bdb0 = pack2(__bfloat162float(e0), __bfloat162float(e1)), bdb1 = pack2(__bfloat162float(e2), __bfloat162float(e3));
    // Row sums of g = dzn gamma without forming dzn: sum_d g = sum_h db_h c_h with c_h = sum_d gamma_d Wb[h][d] (constant), and
    // sum_d g xhat = rs sum_h db_h (a_h - mean c_h) with a_h = sum_d x_d P[h][d], P = gamma Wb (fp32, split hi + lo into two bf16
    // operands: the x operand is exact bf16, so the product error is ~2^-17).  a is a [16 rows x 8 heads] mma over the x tile.
    {
      float ca[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < KS_; ++ks) {
        uint32_t a[4]; ldm_a(tile, ks, lane, a);
        const uint2 ph = sPt[ks * 32 + lane], pl = sPt[(KS_ + ks) * 32 + lane];
        mma_bf16(ca, a, ph.x, ph.y);
        mma_bf16(ca, a, pl.x, pl.y);
      }
      const float2 u0 = unpack2(adb[0]), u1 = unpack2(adb[1]);          // db of rows r, r + 8 for heads 2q, 2q + 1
      const float c0 = sc[2 * q], c1 = sc[2 * q + 1];
      gs0 = u0.x * c0 + u0.y * c1;
      gx0 = rs0 * (u0.x * (ca[0] - m0 * c0) + u0.y * (ca[1] - m0 * c1));
      gs1 = u1.x * c0 + u1.y * c1;
      gx1 = rs1 * (u1.x * (ca[2] - m1 * c0) + u1.y * (ca[3] - m1 * c1));
    }
    __syncwarp();
    gs0 = qsum(gs0) * (1.f / DZ); gx0 = qsum(gx0) * (1.f / DZ); gs1 = qsum(gs1) * (1.f / DZ); gx1 = qsum(gx1) * (1.f / DZ);
    // pass B (per d-half): dz into the x tile
#pragma unroll
    for (int pass = 1; pass < 2; ++pass) {
#pragma unroll
      for (int hf = 0; hf < DZ / 64; ++hf) {
        float dzn[8][4];
#pragma unroll
        for (int t = 0; t < 8; ++t) { const int nt = hf * 8 + t; dzn[t][0] = dzn[t][1] = dzn[t][2] = dzn[t][3] = 0.f; mma_bf16(dzn[t], adb, wbt2[(nt * 8 + r) * 4 + q], 0u); }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int ks = hf * 4 + kk, col = 16 * ks + 2 * q;
          uint32_t a[4]; ldm_a(tile, ks, lane, a);
          const float2 x0 = unpack2(a[0]), x1 = unpack2(a[1]), x2 = unpack2(a[2]), x3 = unpack2(a[3]);
          const float2 g0 = *reinterpret_cast<const float2*>(sg + col), g1 = *reinterpret_cast<const float2*>(sg + col + 8);
          const float xa = (x0.x - m0) * rs0, xb = (x0.y - m0) * rs0, xc = (x2.x - m0) * rs0, xd = (x2.y - m0) * rs0;   // row r
          const float xe = (x1.x - m1) * rs1, xf = (x1.y - m1) * rs1, xg = (x3.x - m1) * rs1, xhh = (x3.y - m1) * rs1;  // row r+8
          {                                                             // M^T[d][h] += xhat^T . db: the bf16 xhat fragment, transposed in registers
            uint32_t at[4];
            const uint32_t p0 = pack2(xa, xb), p1 = pack2(xe, xf), p2 = pack2(xc, xd), p3 = pack2(xg, xhh);
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(at[0]) : "r"(p0));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(at[1]) : "r"(p2));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(at[2]) : "r"(p1));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(at[3]) : "r"(p3));
            mma_bf16(cm[ks], at, bdb0, bdb1);
          }
          const float* d0 = dzn[2 * kk]; const float* d1 = dzn[2 * kk + 1];
          const float ga = d0[0] * g0.x, gb = d0[1] * g0.y, gc = d1[0] * g1.x, gd = d1[1] * g1.y;
          const float ge = d0[2] * g0.x, gf = d0[3] * g0.y, gg = d1[2] * g1.x, gh = d1[3] * g1.y;
          {
            asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(tile + tile_off(r, 2 * ks) + 4 * q), "r"(pack2(rs0 * (ga - gs0 - xa * gx0), rs0 * (gb - gs0 - xb * gx0))) : "memory");
            asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(tile + tile_off(r + 8, 2 * ks) + 4 * q), "r"(pack2(rs1 * (ge - gs1 - xe * gx1), rs1 * (gf - gs1 - xf * gx1))) : "memory");
            asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(tile + tile_off(r, 2 * ks + 1) + 4 * q), "r"(pack2(rs0 * (gc - gs0 - xc * gx0), rs0 * (gd - gs0 - xd * gx0))) : "memory");
            asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(tile + tile_off(r + 8, 2 * ks + 1) + 4 * q), "r"(pack2(rs1 * (gg - gs1 - xg * gx1), rs1 * (gh - gs1 - xhh * gx1))) : "memory");
          }
        }
      }
    }
    __syncwarp();
#pragma unroll
    for (int qq = 0; qq < KS_; ++qq) {                                  // dz tile -> global (coalesced 16-byte chunks)
      const int v = qq * 32 + lane, rr = v / CPR, ch = v % CPR;
      uint4 u;
      asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(tile + tile_off(rr, ch)));
      *reinterpret_cast<uint4*>(DZO + ((long)i * N + j0 + rr) * DZ + ch * 8) = u;
    }
    __syncwarp();
  }
  cp_wait<0>();
  // ---- block reductions: M^T (as acc[h][d]) ----
  // Warp by warp in a fixed order (a float atomicAdd would make the sum depend on which warp arrives first): within a warp every
  // (h, d) element is held by exactly one lane, so the read-modify-write below is race-free.
  for (int w = 0; w < NWARP; ++w) {
    if (warp == w) {
#pragma unroll
      for (int mt = 0; mt < KS_; ++mt) {
        acc[(2 * q) * DZ + 16 * mt + r] += cm[mt][0]; acc[(2 * q + 1) * DZ + 16 * mt + r] += cm[mt][1];
        acc[(2 * q) * DZ + 16 * mt + r + 8] += cm[mt][2]; acc[(2 * q + 1) * DZ + 16 * mt + r + 8] += cm[mt][3];
      }
    }
    __syncthreads();
  }
  // dWb[h][d] = gamma[d] M[h][d] + beta[d] S[h];  dgamma[d] = sum_h Wb[h][d] M[h][d];  dbeta[d] = sum_h Wb[h][d] S[h]
  const float* S = acc + H * DZ + DZ;
  for (int k = tid; k < H * DZ; k += THREADS) { const int h = k / DZ, d = k - h * DZ; PWB[(long)i * H * DZ + k] = sg[d] * acc[k] + sg[DZ + d] * S[h]; }
  for (int d = tid; d < DZ; d += THREADS) {
    float pg = 0.f, pb = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) { const float wv = __bfloat162float(WB[h * DZ + d]); pg += wv * acc[h * DZ + d]; pb += wv * S[h]; }
    PLN[(long)i * 2 * DZ + d] = pg; PLN[(long)i * 2 * DZ + DZ + d] = pb;
  }
}
}  // namespace

torch::Tensor pair_fwd3(torch::Tensor z, torch::Tensor mask, torch::Tensor ln_w, torch::Tensor ln_b, double eps, torch::Tensor wb) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.is_contiguous() && z.dim() == 3 && z.size(2) == DZ && z.size(0) == z.size(1), "z: [N, N, DZ] bf16");
  const int N = (int)z.size(0);
  TORCH_CHECK(N % GJ == 0 && N <= 1024, "N must be a multiple of 16 (and <= 1024)");
  TORCH_CHECK(mask.scalar_type() == torch::kBFloat16 && mask.is_contiguous() && mask.numel() == (long)N, "mask: the [N] key mask, bf16 0/1");
  TORCH_CHECK(ln_w.scalar_type() == torch::kFloat && ln_w.is_contiguous() && ln_w.numel() == DZ && ln_b.scalar_type() == torch::kFloat && ln_b.is_contiguous() && ln_b.numel() == DZ, "LN params fp32[DZ]");
  auto wb16 = wb.to(torch::kBFloat16).contiguous();
  TORCH_CHECK(wb16.dim() == 2 && wb16.size(0) == H && wb16.size(1) == DZ, "wb: [8, DZ]");
  auto w = torch::empty({H, N, N}, z.options());
  const int NP = N + 4;
  const int smem = NWARP * 2 * TILEB + H * NP * 2 + 2 * DZ * 4;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NWARP * 2 * TILEB + H * 1028 * 2 + 2 * DZ * 4)); attr = true; }
  pair_fwd_kernel<<<N, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(mask.data_ptr<at::BFloat16>()),
      ln_w.data_ptr<float>(), ln_b.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(wb16.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>()), N, NP, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w;
}

std::vector<torch::Tensor> pair_bwd3(torch::Tensor z, torch::Tensor w16, torch::Tensor dw, torch::Tensor ln_w, torch::Tensor ln_b, double eps, torch::Tensor wb) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.is_contiguous() && z.dim() == 3 && z.size(2) == DZ && z.size(0) == z.size(1), "z: [N, N, DZ] bf16");
  const int N = (int)z.size(0);
  TORCH_CHECK(N % GJ == 0 && N <= 1024, "N must be a multiple of 16 (and <= 1024)");
  TORCH_CHECK(w16.scalar_type() == torch::kBFloat16 && w16.is_contiguous() && w16.sizes() == torch::IntArrayRef({H, N, N}), "w16: [8, N, N] bf16");
  auto dwc = dw.contiguous();
  TORCH_CHECK(dwc.scalar_type() == torch::kBFloat16 && dwc.sizes() == torch::IntArrayRef({H, N, N}), "dw: [8, N, N] bf16");
  auto wb16 = wb.to(torch::kBFloat16).contiguous();
  auto dz = torch::empty_like(z);
  auto pwb = torch::empty({N, H, DZ}, z.options().dtype(torch::kFloat));
  auto pln = torch::empty({N, 2 * DZ}, z.options().dtype(torch::kFloat));
  TORCH_CHECK(2 * H * N * 4 <= NWARP * 2 * TILEB, "the w / dw staging must fit in the tile area");
  const int smem = NWARP * 2 * TILEB + (2 * DZ + H * DZ + DZ + H) * 4 + DZ * 4 * 4 + N * H * 2 + 2 * KS_ * 32 * 8 + H * 4;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NWARP * 2 * TILEB + (2 * DZ + H * DZ + DZ + H) * 4 + DZ * 4 * 4 + 1024 * H * 2 + 2 * KS_ * 32 * 8 + H * 4)); attr = true; }
  pair_bwd_kernel<<<N, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(w16.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(dwc.data_ptr<at::BFloat16>()),
      ln_w.data_ptr<float>(), ln_b.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(wb16.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(dz.data_ptr<at::BFloat16>()), pwb.data_ptr<float>(), pln.data_ptr<float>(), N, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto ps = pln.sum(0);
  return {dz, pwb.sum(0), ps.narrow(0, 0, DZ), ps.narrow(0, DZ, DZ)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("pair_fwd3", &pair_fwd3, "pair LN -> proj_z -> mask -> softmax (tensor cores)", py::arg("z"), py::arg("mask"), py::arg("ln_w"), py::arg("ln_b"), py::arg("eps"), py::arg("wb"));
  mod.def("pair_bwd3", &pair_bwd3, "pair backward (tensor cores): (dz, dWb, dgamma, dbeta)", py::arg("z"), py::arg("w16"), py::arg("dw"), py::arg("ln_w"), py::arg("ln_b"), py::arg("eps"), py::arg("wb"));
}
