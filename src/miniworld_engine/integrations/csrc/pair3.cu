// Pair-side kernels of the PWA block on tensor cores (sm_90a): per pair row i, LayerNorm(128) -> proj_z (8 heads) ->
// mask -> softmax over j, and its backward.  A warp processes 16 j's at a time: the [16 j][128 d] z tile is
// cp.async-staged (swizzled), read as mma A fragments with ldmatrix, normalised in registers (the row statistics are
// quad shuffles), and the projection is eight m16n8k16 mma with Wb^T as the B operand.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace {
constexpr int DZ = 128, H = 8, NWARP = 8, THREADS = NWARP * 32, GJ = 16;
constexpr int TILEB = GJ * DZ * 2;                        // one [16][128] bf16 tile = 4 KiB

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ float bf16r(float x) {
#if !MW_PWA_ROUND_LOGITS
  // The fused projection feeds the FP32 softmax directly, like the engine's fused Triton path (whose logits are FP32 too).
  // Rounding the logits to BF16 (-DMW_PWA_ROUND_LOGITS=1, the old "stock proj_z output is bf16" statement) costs ~1e-3 of
  // output error at |logit| ~ 1-4 and nothing is saved by it; the opt-in build stays for the A/B.
  return x;
#else
  return __bfloat162float(__float2bfloat16(x));
#endif
}
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
__device__ __forceinline__ uint32_t tile_off(int r, int ch) { return (uint32_t)(r * 256 + ((ch ^ (r & 7)) << 4)); }
// A fragments (a0..a3) of the 16x16 sub-tile at column 16*ks: lane l addresses matrix l/8, row l%8
__device__ __forceinline__ void ldm_a(uint32_t tile, int ks, int lane, uint32_t* a) {
  const int m = lane >> 3, r = (lane & 7) + ((m & 1) << 3), ch = 2 * ks + (m >> 1);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(tile + tile_off(r, ch)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_tile(uint32_t dst, const __nv_bfloat16* src_row0, int lane) {   // 16 rows x 256 B, 8 chunks per lane
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int v = q * 32 + lane, r = v >> 4, ch = v & 15;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst + tile_off(r, ch)), "l"(src_row0 + (long)r * DZ + ch * 8) : "memory");
  }
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// ---------------------------------------------------------------- forward ----------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 2) pair_fwd_kernel(const __nv_bfloat16* __restrict__ Z, const __nv_bfloat16* __restrict__ MASK,
                                                              const float* __restrict__ LNW, const __nv_bfloat16* __restrict__ WB,
                                                              __nv_bfloat16* __restrict__ W, int N, int NP, float eps) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* tiles = smem;                                          // [NWARP][2][TILEB]
  float* sb = reinterpret_cast<float*>(smem + NWARP * 2 * TILEB);       // [H][NP] logits
  float* sg = sb + H * NP;                                              // gamma [DZ]
  const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3, r = lane >> 2;
  for (int k = tid; k < DZ; k += THREADS) sg[k] = LNW[k];
  // B fragments of Wb^T [k = d][n = h]: b0 = (Wb[n][16ks + 2q], +1), b1 = (+8, +9), n = lane / 4
  uint32_t bw[8][2];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
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
    uint32_t a[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) ldm_a(tile, ks, lane, a[ks]);
    // row statistics: this lane holds 32 values of row r (a0, a2) and of row r+8 (a1, a3); a quad completes a row
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const float2 x0 = unpack2(a[ks][0]), x2 = unpack2(a[ks][2]), x1 = unpack2(a[ks][1]), x3 = unpack2(a[ks][3]);
      s0 += x0.x + x0.y + x2.x + x2.y; s1 += x1.x + x1.y + x3.x + x3.y;
    }
    const float m0 = qsum(s0) * (1.f / DZ), m1 = qsum(s1) * (1.f / DZ);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const float2 x0 = unpack2(a[ks][0]), x2 = unpack2(a[ks][2]), x1 = unpack2(a[ks][1]), x3 = unpack2(a[ks][3]);
      v0 += (x0.x - m0) * (x0.x - m0) + (x0.y - m0) * (x0.y - m0) + (x2.x - m0) * (x2.x - m0) + (x2.y - m0) * (x2.y - m0);
      v1 += (x1.x - m1) * (x1.x - m1) + (x1.y - m1) * (x1.y - m1) + (x3.x - m1) * (x3.x - m1) + (x3.y - m1) * (x3.y - m1);
    }
    const float rs0 = 1.f / sqrtf(qsum(v0) * (1.f / DZ) + eps), rs1 = 1.f / sqrtf(qsum(v1) * (1.f / DZ) + eps);
    // zn = bf16((x - mean) * rstd * gamma) into A fragments, then the projection
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sg + col), g1 = *reinterpret_cast<const float2*>(sg + col + 8);
      const float2 x0 = unpack2(a[ks][0]), x1 = unpack2(a[ks][1]), x2 = unpack2(a[ks][2]), x3 = unpack2(a[ks][3]);
      uint32_t zn[4];
      zn[0] = pack2((x0.x - m0) * rs0 * g0.x, (x0.y - m0) * rs0 * g0.y);
      zn[1] = pack2((x1.x - m1) * rs1 * g0.x, (x1.y - m1) * rs1 * g0.y);
      zn[2] = pack2((x2.x - m0) * rs0 * g1.x, (x2.y - m0) * rs0 * g1.y);
      zn[3] = pack2((x3.x - m1) * rs1 * g1.x, (x3.y - m1) * rs1 * g1.y);
      mma_bf16(c, zn, bw[ks][0], bw[ks][1]);
    }
    // c0, c1: (row r, heads 2q, 2q+1); c2, c3: (row r+8)
    const int j0 = gi * GJ;
    const float mk0 = __bfloat162float(MASK[(long)i * N + j0 + r]), mk1 = __bfloat162float(MASK[(long)i * N + j0 + r + 8]);
    sb[(2 * q) * NP + j0 + r] = mk0 > 0.5f ? bf16r(c[0]) : -1e30f;
    sb[(2 * q + 1) * NP + j0 + r] = mk0 > 0.5f ? bf16r(c[1]) : -1e30f;
    sb[(2 * q) * NP + j0 + r + 8] = mk1 > 0.5f ? bf16r(c[2]) : -1e30f;
    sb[(2 * q + 1) * NP + j0 + r + 8] = mk1 > 0.5f ? bf16r(c[3]) : -1e30f;
    __syncwarp();                                                       // this stage may be refilled next iteration
  }
  cp_wait<0>();
  __syncthreads();
  {                                                                     // softmax over j for head `warp`
    const int h = warp;
    float mx = -1e30f;
    for (int j = lane; j < N; j += 32) mx = fmaxf(mx, sb[h * NP + j]);
    mx = wmax(mx);
    float den = 0.f;
    for (int j = lane; j < N; j += 32) den += __expf(sb[h * NP + j] - mx);
    den = wsum(den);
    const float inv = 1.f / den;
    for (int j = lane; j < N; j += 32) W[((long)h * N + i) * N + j] = __float2bfloat16(__expf(sb[h * NP + j] - mx) * inv);
  }
}

// ---------------------------------------------------------------- backward ----------------------------------------------------------------
// db = bf16(w (dw - sum_j w dw)); dzn = db . Wb (mma, K = 8 heads padded to 16); dz = LN_bwd(dzn) (registers);
// dWb^T[d][h] += zn^T . db (mma, A = zn tile read transposed with ldmatrix.trans); dgamma partials in registers.
// A fragments of the transposed 16x16 sub-tile (d rows 16*mt.., j cols) of a [16 j][128 d] tile
__device__ __forceinline__ void ldm_at(uint32_t tile, int mt, int lane, uint32_t* a) {
  const int m = lane >> 3, jrow = (lane & 7) + ((m >> 1) << 3), ch = 2 * mt + (m & 1);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(tile + tile_off(jrow, ch)));
}
__global__ void __launch_bounds__(THREADS, 2) pair_bwd_kernel(const __nv_bfloat16* __restrict__ Z, const __nv_bfloat16* __restrict__ W16, const float* __restrict__ DW,
                                                              const float* __restrict__ LNW, const __nv_bfloat16* __restrict__ WB,
                                                              __nv_bfloat16* __restrict__ DZO, float* __restrict__ PWB, float* __restrict__ PLN, int N, float eps,
                                                              const bool* __restrict__ KEY) {
  // Per row i.  db = bf16(w (dw - sum_j w dw)); dzn = db . Wb (mma, K = 8 heads padded to 16, computed twice: once for the
  // row sums, once for dz); dz = LN_bwd(dzn).  The parameter gradients come from two per-row reductions instead of
  // per-element accumulators:  M^T[d][h] = sum_j xhat[j][d] db[j][h] (mma, A = the xhat tile read transposed) and
  // dWb[h][d] = gamma[d] M^T[d][h],  dgamma[d] = sum_h Wb[h][d] M^T[d][h]  (dzn = db . Wb is linear in db, zn = xhat gamma).
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* tiles = smem;                                          // [NWARP][3][TILEB]: two x stages + one aux (xhat / dz) tile per warp
  float* sg = reinterpret_cast<float*>(smem + NWARP * 3 * TILEB);       // gamma [DZ]
  float* acc = sg + DZ;                                                 // [H][DZ] M^T (as [h][d])
  uint32_t* wbt2 = reinterpret_cast<uint32_t*>(acc + H * DZ);  // [DZ][4]: pack(Wb[2q][d], Wb[2q+1][d])
  __nv_bfloat16* sdb = reinterpret_cast<__nv_bfloat16*>(wbt2 + DZ * 4); // [N][H] db (bf16)
  float* sdw = reinterpret_cast<float*>(tiles);                         // staging of dw / w rows [H][N] fp32 each, aliased on the tile area
  float* sw = sdw + H * N;
  const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3, r = lane >> 2;
  for (int k = tid; k < DZ; k += THREADS) sg[k] = LNW[k];
  for (int k = tid; k < H * DZ; k += THREADS) acc[k] = 0.f;
  for (int k = tid; k < DZ * 4; k += THREADS) { const int d = k >> 2, qq = k & 3; wbt2[k] = pack2(__bfloat162float(WB[(2 * qq) * DZ + d]), __bfloat162float(WB[(2 * qq + 1) * DZ + d])); }
  for (int k0 = 0; k0 < H * N; k0 += 8 * THREADS) {                     // staged in chunks of 8 per thread: all 16 loads in flight before the stores
    float vd[8], vw[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const int k = k0 + m * THREADS + tid;
      if (k < H * N) { const int h = k / N, j = k - h * N; const long base = ((long)h * N + i) * N + j; vd[m] = DW[base]; vw[m] = __bfloat162float(W16[base]); }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) { const int k = k0 + m * THREADS + tid; if (k < H * N) { sdw[k] = vd[m]; sw[k] = vw[m]; } }
  }
  __syncthreads();
  {                                                                     // softmax backward, head `warp`: sdot, db (bf16)
    const int h = warp;
    float sdot = 0.f;
    for (int j = lane; j < N; j += 32) sdot += sw[h * N + j] * sdw[h * N + j];
    sdot = wsum(sdot);
    // masked_fill has zero derivative at every masked key, including the
    // all-masked case where finite-min logits produce uniform probabilities.
    for (int j = lane; j < N; j += 32) sdb[j * H + h] = __float2bfloat16(KEY && !KEY[j] ? 0.f : sw[h * N + j] * (sdw[h * N + j] - sdot));
  }
  __syncthreads();                                                      // the staging area becomes the tile ring
  float cm[8][4];                                                        // M^T accumulators: rows d = 16 mt + r / + 8, cols h = 2q, 2q+1
#pragma unroll
  for (int mt = 0; mt < 8; ++mt) { cm[mt][0] = cm[mt][1] = cm[mt][2] = cm[mt][3] = 0.f; }
  const uint32_t tbase = sa(tiles + warp * 3 * TILEB), taux = tbase + 2 * TILEB;
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
    for (int ks = 0; ks < 8; ++ks) {
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
    // pass A (per d-half): dzn = db . Wb; xhat tile (bf16) into the aux buffer; row sums of g = dzn gamma and g xhat
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      float dzn[8][4];
#pragma unroll
      for (int t = 0; t < 8; ++t) { const int nt = hf * 8 + t; dzn[t][0] = dzn[t][1] = dzn[t][2] = dzn[t][3] = 0.f; mma_bf16(dzn[t], adb, wbt2[(nt * 8 + r) * 4 + q], 0u); }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int ks = hf * 4 + kk, col = 16 * ks + 2 * q;
        uint32_t a[4]; ldm_a(tile, ks, lane, a);
        const float2 x0 = unpack2(a[0]), x1 = unpack2(a[1]), x2 = unpack2(a[2]), x3 = unpack2(a[3]);
        const float2 g0 = *reinterpret_cast<const float2*>(sg + col), g1 = *reinterpret_cast<const float2*>(sg + col + 8);
        const float xa = (x0.x - m0) * rs0, xb = (x0.y - m0) * rs0, xc = (x2.x - m0) * rs0, xd = (x2.y - m0) * rs0;
        const float xe = (x1.x - m1) * rs1, xf = (x1.y - m1) * rs1, xg = (x3.x - m1) * rs1, xhh = (x3.y - m1) * rs1;
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(taux + tile_off(r, 2 * ks) + 4 * q), "r"(pack2(xa, xb)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(taux + tile_off(r + 8, 2 * ks) + 4 * q), "r"(pack2(xe, xf)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(taux + tile_off(r, 2 * ks + 1) + 4 * q), "r"(pack2(xc, xd)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(taux + tile_off(r + 8, 2 * ks + 1) + 4 * q), "r"(pack2(xg, xhh)) : "memory");
        const float* d0 = dzn[2 * kk]; const float* d1 = dzn[2 * kk + 1];
        const float ga = d0[0] * g0.x, gb = d0[1] * g0.y, gc = d1[0] * g1.x, gd = d1[1] * g1.y;
        const float ge = d0[2] * g0.x, gf = d0[3] * g0.y, gg = d1[2] * g1.x, gh = d1[3] * g1.y;
        gs0 += ga + gb + gc + gd; gx0 += ga * xa + gb * xb + gc * xc + gd * xd;
        gs1 += ge + gf + gg + gh; gx1 += ge * xe + gf * xf + gg * xg + gh * xhh;
      }
    }
    __syncwarp();
    {                                                                   // M^T += xhat^T . db
      const __nv_bfloat16 e0 = sdb[(j0 + 2 * q) * H + r], e1 = sdb[(j0 + 2 * q + 1) * H + r], e2 = sdb[(j0 + 2 * q + 8) * H + r], e3 = sdb[(j0 + 2 * q + 9) * H + r];
      const uint32_t bdb0 = pack2(__bfloat162float(e0), __bfloat162float(e1)), bdb1 = pack2(__bfloat162float(e2), __bfloat162float(e3));
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) { uint32_t at[4]; ldm_at(taux, mt, lane, at); mma_bf16(cm[mt], at, bdb0, bdb1); }
    }
    gs0 = qsum(gs0) * (1.f / DZ); gx0 = qsum(gx0) * (1.f / DZ); gs1 = qsum(gs1) * (1.f / DZ); gx1 = qsum(gx1) * (1.f / DZ);
    // pass B (per d-half): dz into the x tile
#pragma unroll
    for (int pass = 1; pass < 2; ++pass) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
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
          const float* d0 = dzn[2 * kk]; const float* d1 = dzn[2 * kk + 1];
          const float ga = d0[0] * g0.x, gb = d0[1] * g0.y, gc = d1[0] * g1.x, gd = d1[1] * g1.y;
          const float ge = d0[2] * g0.x, gf = d0[3] * g0.y, gg = d1[2] * g1.x, gh = d1[3] * g1.y;
          if (pass == 0) {
            gs0 += ga + gb + gc + gd; gx0 += ga * xa + gb * xb + gc * xc + gd * xd;
            gs1 += ge + gf + gg + gh; gx1 += ge * xe + gf * xf + gg * xg + gh * xhh;
          } else {
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
    for (int qq = 0; qq < 8; ++qq) {                                    // dz tile -> global (coalesced 16-byte chunks)
      const int v = qq * 32 + lane, rr = v >> 4, ch = v & 15;
      uint4 u;
      asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(tile + tile_off(rr, ch)));
      *reinterpret_cast<uint4*>(DZO + ((long)i * N + j0 + rr) * DZ + ch * 8) = u;
    }
    __syncwarp();
  }
  cp_wait<0>();
  // ---- block reductions: M^T (as acc[h][d]) ----
#if MW_PWA_CUDA_AUX
  // The input ring is dead after every warp drains its cp.async pipeline.
  // Reuse it for fixed-order warp partials instead of unordered shared atomics.
  __syncthreads();
  float* warp_partials = reinterpret_cast<float*>(tiles);
#pragma unroll
  for (int mt = 0; mt < 8; ++mt) {
    float* p = warp_partials + warp * H * DZ;
    p[(2*q)*DZ+16*mt+r]=cm[mt][0]; p[(2*q+1)*DZ+16*mt+r]=cm[mt][1];
    p[(2*q)*DZ+16*mt+r+8]=cm[mt][2]; p[(2*q+1)*DZ+16*mt+r+8]=cm[mt][3];
  }
  __syncthreads();
  for(int k=tid;k<H*DZ;k+=THREADS) {
    float value=0.f;
#pragma unroll
    for(int w=0;w<NWARP;++w)value+=warp_partials[w*H*DZ+k];
    acc[k]=value;
  }
#else
#pragma unroll
  for (int mt = 0; mt < 8; ++mt) {
    atomicAdd(acc + (2 * q) * DZ + 16 * mt + r, cm[mt][0]); atomicAdd(acc + (2 * q + 1) * DZ + 16 * mt + r, cm[mt][1]);
    atomicAdd(acc + (2 * q) * DZ + 16 * mt + r + 8, cm[mt][2]); atomicAdd(acc + (2 * q + 1) * DZ + 16 * mt + r + 8, cm[mt][3]);
  }
#endif
  __syncthreads();
  // dWb[h][d] = gamma[d] M[h][d];  dgamma[d] = sum_h Wb[h][d] M[h][d]
  for (int k = tid; k < H * DZ; k += THREADS) { const int d = k % DZ; PWB[(long)i * H * DZ + k] = sg[d] * acc[k]; }
  for (int d = tid; d < DZ; d += THREADS) {
    float pg = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) pg += __bfloat162float(WB[h * DZ + d]) * acc[h * DZ + d];
    PLN[(long)i * DZ + d] = pg;
  }
}
}  // namespace

// Live graph-safe preparation; no cached casts of mutable master parameters.
namespace {
struct CopyDesc { const void* src; void* dst; long size; int src_type, dst_type; };
struct CopyBatch { CopyDesc v[12]; int count; };
__device__ float read_value(const void* p, int type, long i) {
  return type == 0 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]) : static_cast<const float*>(p)[i];
}
__global__ void pwa_cast_kernel(CopyBatch args) {
  for (int k = blockIdx.y; k < args.count; k += gridDim.y) {
    const auto a = args.v[k];
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < a.size; i += gridDim.x * blockDim.x) {
      const float value = read_value(a.src, a.src_type, i);
      if (a.dst_type == 0) static_cast<__nv_bfloat16*>(a.dst)[i] = __float2bfloat16(value);
      else static_cast<float*>(a.dst)[i] = value;
    }
  }
}
__global__ void pwa_mask_kernel(const bool* key, __nv_bfloat16* out, int n) {
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < (long)n*n; i += gridDim.x * blockDim.x)
    out[i] = __float2bfloat16(key == nullptr || key[i % n] ? 1.f : 0.f);
}
__global__ void pwa_transpose_kernel(const __nv_bfloat16* a, __nv_bfloat16* b, int rows, int cols) {
  __shared__ __nv_bfloat16 tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  const long offset = (long)blockIdx.z * rows * cols;
  for (int j=0;j<32;j+=8) if (x<cols && y+j<rows) tile[threadIdx.y+j][threadIdx.x]=a[offset+(long)(y+j)*cols+x];
  __syncthreads();
  x=blockIdx.y*32+threadIdx.x; y=blockIdx.x*32+threadIdx.y;
  for (int j=0;j<32;j+=8) if(x<rows && y+j<cols) b[offset+(long)(y+j)*rows+x]=tile[threadIdx.x][threadIdx.y+j];
}
// Two-stage, fixed-order column sums of the per-row partials: stage 1 sums `rpb` rows per block (grid.y chunks), stage 2 sums the chunks in
// order.  The one-stage form below used 4 blocks and walked all N rows per thread (latency-bound, 0.16 ms at N = 768).
constexpr int RCOLS = H * DZ + DZ;
__global__ void pwa_pair_reduce1(const float* pw, const float* pl, float* part, int rows, int rpb) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= RCOLS) return;
  const int r0 = blockIdx.y * rpb, r1 = min(rows, r0 + rpb);
  float v = 0.f;
  if (c < H * DZ) { for (int r = r0; r < r1; ++r) v += pw[(long)r * H * DZ + c]; }
  else { const int cc = c - H * DZ; for (int r = r0; r < r1; ++r) v += pl[(long)r * DZ + cc]; }
  part[(long)blockIdx.y * RCOLS + c] = v;
}
__global__ void pwa_pair_reduce2(const float* part, float* dw, float* dg, int chunks) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= RCOLS) return;
  float v = 0.f;
  for (int y = 0; y < chunks; ++y) v += part[(long)y * RCOLS + c];
  if (c < H * DZ) dw[c] = v; else dg[c - H * DZ] = v;
}
__global__ void pwa_pair_reduce(const float* pw, const float* pl, float* dw, float* dg, int rows) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c<H*DZ) { float v=0; for(int r=0;r<rows;++r)v+=pw[r*H*DZ+c]; dw[c]=v; }
  if(c<DZ) { float g=0;for(int r=0;r<rows;++r)g+=pl[r*DZ+c];dg[c]=g; }
}
__global__ void pwa_wgv_kernel(const __nv_bfloat16* g,const __nv_bfloat16* v,__nv_bfloat16* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<64*512){int row=i/512,col=i%512;out[i]=col<256?g[col*64+row]:v[(col-256)*64+row];}
}
}
std::vector<torch::Tensor> pwa_cast(std::vector<torch::Tensor> src, std::vector<int64_t> types, bool force) {
  TORCH_CHECK(src.size()<=12 && src.size()==types.size(), "at most 12 matching casts");
  CopyBatch args{}; std::vector<torch::Tensor> out;
  long largest=0;
  for(size_t k=0;k<src.size();++k) {
    const auto& x=src[k];TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&(x.scalar_type()==torch::kFloat||x.scalar_type()==torch::kBFloat16),"contiguous fp32/bf16 casts only");
    TORCH_CHECK(types[k]==0||types[k]==1,"dtype: 0 bf16, 1 fp32");
    const auto type=types[k]==0?torch::kBFloat16:torch::kFloat;
    if(!force&&x.scalar_type()==type){out.push_back(x);continue;}
    auto y=torch::empty_like(x,x.options().dtype(type));out.push_back(y);
    args.v[args.count++]={x.data_ptr(),y.data_ptr(),x.numel(),int(x.scalar_type()==torch::kFloat),int(types[k])};
    largest=std::max(largest,x.numel());
  }
  if(args.count) {
    cudaKernel_t kernel;
    C10_CUDA_CHECK(cudaGetKernel(&kernel, pwa_cast_kernel));
    void* launch_args[] = {&args};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
        dim3(std::min<long>((largest+255)/256,132),args.count), dim3(256),
        launch_args, 0, at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
torch::Tensor pwa_mask(torch::Tensor like, c10::optional<torch::Tensor> mask, int64_t n) {
  if(mask)TORCH_CHECK(mask->is_cuda()&&mask->is_contiguous()&&mask->scalar_type()==torch::kBool&&mask->numel()==n,"contiguous bool token mask");
  auto out=torch::empty({n,n},like.options().dtype(torch::kBFloat16));
  const bool* key = mask ? mask->data_ptr<bool>() : nullptr;
  auto* result = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  int width = static_cast<int>(n);
  cudaKernel_t kernel;
  C10_CUDA_CHECK(cudaGetKernel(&kernel, pwa_mask_kernel));
  void* launch_args[] = {&key, &result, &width};
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
      dim3(std::min<int64_t>((n*n+255)/256,132)), dim3(256),
      launch_args, 0, at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor pwa_transpose(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==torch::kBFloat16&&(x.dim()==2||x.dim()==3),"2D or 3D contiguous bf16 transpose");
  const int r=x.size(-2),c=x.size(-1),batch=x.dim()==3?x.size(0):1;
  auto sizes=x.sizes().vec();std::swap(sizes[sizes.size()-1],sizes[sizes.size()-2]);auto y=torch::empty(sizes,x.options());
  pwa_transpose_kernel<<<dim3((c+31)/32,(r+31)/32,batch),dim3(32,8),0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),r,c);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
torch::Tensor pwa_wgv(torch::Tensor g,torch::Tensor v) {
  for(const auto& x:{g,v})TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==torch::kBFloat16&&x.sizes()==torch::IntArrayRef({256,64}),"bf16 [256,64] weight");
  auto out=torch::empty({64,512},g.options());
  pwa_wgv_kernel<<<128,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()),reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor pair_fwd3(torch::Tensor z, torch::Tensor mask, torch::Tensor ln_w, double eps, torch::Tensor wb) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.is_contiguous() && z.dim() == 3 && z.size(2) == DZ && z.size(0) == z.size(1), "z: [N, N, 128] bf16");
  const int N = (int)z.size(0);
  TORCH_CHECK(N % GJ == 0 && N <= 1024, "N must be a multiple of 16 (and <= 1024)");
  TORCH_CHECK(mask.scalar_type() == torch::kBFloat16 && mask.is_contiguous() && mask.numel() == (long)N * N, "mask: [N, N] bf16");
  TORCH_CHECK(ln_w.scalar_type() == torch::kFloat && ln_w.is_contiguous() && ln_w.numel() == DZ, "LN scale fp32[128]");
  auto wb16 = wb.to(torch::kBFloat16).contiguous();
  TORCH_CHECK(wb16.dim() == 2 && wb16.size(0) == H && wb16.size(1) == DZ, "wb: [8, 128]");
  auto w = torch::empty({H, N, N}, z.options());
  const int NP = N + 4;
  const int smem = NWARP * 2 * TILEB + (H * NP + DZ) * 4;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NWARP * 2 * TILEB + (H * 1028 + DZ) * 4)); attr = true; }
  pair_fwd_kernel<<<N, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(mask.data_ptr<at::BFloat16>()),
      ln_w.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(wb16.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>()), N, NP, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w;
}

std::vector<torch::Tensor> pair_bwd3(torch::Tensor z, torch::Tensor w16, torch::Tensor dw, torch::Tensor ln_w, double eps, torch::Tensor wb, c10::optional<torch::Tensor> key) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.is_contiguous() && z.dim() == 3 && z.size(2) == DZ && z.size(0) == z.size(1), "z: [N, N, 128] bf16");
  const int N = (int)z.size(0);
  if (key.has_value()) TORCH_CHECK(key->is_cuda() && key->device() == z.device() && key->scalar_type() == at::kBool && key->is_contiguous() && key->numel() == N, "key: contiguous CUDA bool [N]");
  TORCH_CHECK(N % GJ == 0 && N <= 1024, "N must be a multiple of 16 (and <= 1024)");
  TORCH_CHECK(w16.scalar_type() == torch::kBFloat16 && w16.is_contiguous() && w16.sizes() == torch::IntArrayRef({H, N, N}), "w16: [8, N, N] bf16");
  auto dwc = dw.contiguous();
  TORCH_CHECK(dwc.scalar_type() == torch::kFloat && dwc.sizes() == torch::IntArrayRef({H, N, N}), "dw: [8, N, N] fp32");
  auto wb16 = wb.to(torch::kBFloat16).contiguous();
  auto dz = torch::empty_like(z);
  auto pwb = torch::empty({N, H, DZ}, z.options().dtype(torch::kFloat));
  auto pln = torch::empty({N, DZ}, z.options().dtype(torch::kFloat));
  TORCH_CHECK(2 * H * N * 4 <= NWARP * 3 * TILEB, "the w / dw staging must fit in the tile area");
  const int smem = NWARP * 3 * TILEB + (DZ + H * DZ) * 4 + DZ * 4 * 4 + N * H * 2;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NWARP * 3 * TILEB + (DZ + H * DZ) * 4 + DZ * 4 * 4 + 1024 * H * 2)); attr = true; }
  pair_bwd_kernel<<<N, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(w16.data_ptr<at::BFloat16>()), dwc.data_ptr<float>(),
      ln_w.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(wb16.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(dz.data_ptr<at::BFloat16>()), pwb.data_ptr<float>(), pln.data_ptr<float>(), N, (float)eps,
      key.has_value() ? key->data_ptr<bool>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#if MW_PWA_CUDA_AUX
  auto dweight=torch::empty({H,DZ},pwb.options()), dg=torch::empty({DZ},pln.options());
  {
    constexpr int RPB = 32;
    const int chunks = (int)((N + RPB - 1) / RPB);
    auto part = torch::empty({chunks, RCOLS}, pwb.options());
    pwa_pair_reduce1<<<dim3((RCOLS + 255) / 256, chunks), 256, 0, at::cuda::getCurrentCUDAStream()>>>(pwb.data_ptr<float>(), pln.data_ptr<float>(), part.data_ptr<float>(), (int)N, RPB);
    pwa_pair_reduce2<<<(RCOLS + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), dweight.data_ptr<float>(), dg.data_ptr<float>(), chunks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dz,dweight,dg};
#else
  return {dz, pwb.sum(0), pln.sum(0)};
#endif
}

// Accuracy-first FP32 beta residual: the pair softmax is invariant to a
// constant logit shift, so its tiny numerical gradient must follow the oracle's
// exact FP32 ordering. These copies preserve values; the GEMMs remain cuBLAS.
__global__ void pwa_shadow_pack_kernel(const float* x,float* y,long n,int S,int N) {
  long i=(long)blockIdx.x*256+threadIdx.x;
  if(i<n){
    int c=i%32,s=(i/32)%S,q=(i/(32L*S))%N,h=i/(32L*S*N);
    y[i]=x[((long)s*N+q)*256+h*32+c];
  }
}
torch::Tensor pwa_shadow_pack(torch::Tensor x,int64_t S,int64_t N) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kFloat&&x.numel()==S*N*256,"PWA shadow rows FP32 [S,N,256]");
  auto y=torch::empty({8,N,S*32},x.options());
  pwa_shadow_pack_kernel<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel(),S,N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
// Preserve every FP32 value while grouping adjacent MSA and token rows into
// one tile.  The scalar layout copy above jumps across the entire MSA tensor
// between warps; this producer reads complete 256-channel rows and reorders
// them in shared memory before writing the head-major contraction operand.
__global__ void pwa_shadow_pack_tiled_kernel(const float* x,float* y,int S,int N) {
  extern __shared__ float tile[];
  const int s0=blockIdx.x*8,n0=blockIdx.y*8;
  for(int i=threadIdx.x;i<8*8*256;i+=blockDim.x){
    int c=i%256,n=(i/256)%8,s=i/(8*256);
    tile[i]=x[((long)(s0+s)*N+n0+n)*256+c];
  }
  __syncthreads();
  for(int i=threadIdx.x;i<8*8*256;i+=blockDim.x){
    int c=i%32,s=(i/32)%8,n=(i/(32*8))%8,h=i/(32*8*8);
    y[(((long)h*N+n0+n)*S+s0+s)*32+c]=tile[(s*8+n)*256+h*32+c];
  }
}
torch::Tensor pwa_shadow_pack_tiled(torch::Tensor x,int64_t S,int64_t N) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kFloat&&x.numel()==S*N*256
    &&S>0&&N>0&&S%8==0&&N%8==0,"PWA tiled shadow rows FP32 [S,N,256], multiples of 8");
  auto y=torch::empty({8,N,S*32},x.options());
  constexpr int smem=8*8*256*sizeof(float);
  C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_shadow_pack_tiled_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  pwa_shadow_pack_tiled_kernel<<<dim3(S/8,N/8),256,smem,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),S,N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void pwa_shadow_dropout_kernel(const float* x,const __nv_bfloat16* keep,float* y,long n,int stride,float scale) {
  long i=(long)blockIdx.x*256+threadIdx.x;
  if(i<n)y[i]=__fmul_rn(__fmul_rn(x[i],scale),__bfloat162float(keep[i%stride]));
}
torch::Tensor pwa_shadow_dropout(torch::Tensor x,torch::Tensor keep,double probability) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kFloat&&keep.is_cuda()&&keep.device()==x.device()
    &&keep.is_contiguous()&&keep.scalar_type()==at::kBFloat16&&keep.numel()>0&&x.numel()%keep.numel()==0,"PWA shadow dropout layout");
  TORCH_CHECK(probability>=0.&&probability<1.,"PWA shadow dropout probability");
  auto y=torch::empty_like(x);
  // Probe 22580 verified the oracle's double-scalar reciprocal ordering:
  // convert to FP32 after division, not before (one ULP different at p=.15).
  const float scale=static_cast<float>(1.0/(1.0-probability));
  pwa_shadow_dropout_kernel<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),reinterpret_cast<const __nv_bfloat16*>(keep.data_ptr()),y.data_ptr<float>(),x.numel(),keep.numel(),scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("shadow_pack", &pwa_shadow_pack); mod.def("shadow_dropout", &pwa_shadow_dropout);
  mod.def("shadow_pack_tiled", &pwa_shadow_pack_tiled);
  mod.def("cast", &pwa_cast); mod.def("mask", &pwa_mask); mod.def("transpose", &pwa_transpose); mod.def("wgv", &pwa_wgv);
  mod.def("pair_fwd3", &pair_fwd3, "pair LN -> proj_z -> mask -> softmax (tensor cores)", py::arg("z"), py::arg("mask"), py::arg("ln_w"), py::arg("eps"), py::arg("wb"));
  mod.def("pair_bwd3", &pair_bwd3, "pair backward (tensor cores): (dz, dWb, dgamma)", py::arg("z"), py::arg("w16"), py::arg("dw"), py::arg("ln_w"), py::arg("eps"), py::arg("wb"), py::arg("key") = c10::nullopt);
}
