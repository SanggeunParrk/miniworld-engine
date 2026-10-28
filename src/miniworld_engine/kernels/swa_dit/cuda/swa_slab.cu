// ESMFold2 SWA atom block: every stage but the window attention as "slab" kernels (CUDA sm_90a, C = 128, H = 4, D = 32, SwiGLU hidden
// 256, bf16 operands, fp32 accumulate, mma.sync m16n8k16).  Math and rounding points are the Triton kernels':
//   qkvg fwd    x = rn(RMS(q) (1 + scale_a) + shift_a);  p = rn(x W^T) for W = Wq, Wk, Wv, Wg;  Q / K = rn(rope(rn(headRMS(p))));
//               V = p_v;  G = p_g   (Q / K / V head-major [N, H, S, D], G row-major; saves x, p_q, p_k)          _swa_qkvg_fwd_kernel
//   oproj fwd   att = rn(rn(sigmoid(g) O) Wo^T);  q1 = rn(q + rn(rn(ga) att))                                       _swa_oproj_ffn_fwd_kernel,
//   ffn fwd     y = rn(RMS(q1) (1 + scale_f) + shift_f);  h = rn(silu(y Wa^T) (y Wb^T));  ffn = rn(h Wd^T);        the two halves
//               out = rn(q1 + rn(rn(gf) ffn))
//   ffn bwd     dffn = rn(dq2 rn(gf));  dh = dffn Wd;  da = rn(dh b sa (1 + a (1 - sa))), db = rn(dh a sa), h = rn(a sa b) (a, b
//               recomputed from y);  dy = da Wa + db Wb;  dq1 = rn(dq2 + rstd (dyh - xh mean(dyh xh))), dyh = dy (1 + scale_f);
//               d scale_f = sum_aug dy xh, d shift_f = sum_aug dy, d gate_f = sum_aug dq2 ffn; writes dffn, h, [da | db]  _swa_ffn_bwd_kernel
//   oproj bwd   datt = rn(dq1 rn(ga));  gated = rn(sg O);  dgated = datt Wo;  dO = rn(dgated sg);  dG = rn(dgated O sg (1 - sg));
//               D_h = rowsum_h(dO O);  d gate_a = sum_aug dq1 att                                                   _swa_oproj_bwd_kernel
//   qkvg bwd    dp_q = rn(headRMS_bwd(pq, unrope(dQ))) (k alike);  dx = [dp_q | dp_k | dv | dG] [Wq; Wk; Wv; Wg];
//               d scale_a = sum_aug dx xh, d shift_a = sum_aug dx;  dq = rn(dq1 + rstd (dxh - xh mean(dxh xh)))   _swa_qkvg_bwd_kernel
//
// Why these next to the wgmma kernels of swa_{qkvg,ffn}_fwd.cu / swa_ffn_bwd.cu: the block mostly runs at small row counts (the
// input embedder: ONE sequence of ~4k atoms, M = 4096), where 64-row wgmma tiles give ~64 CTAs, each a long serial pipeline.  Here a
// slab of 16 rows is owned by a GROUP of 4 warps that split its GEMMs 4 ways, so a kernel takes about one short slab chain.  The
// wgmma kernels stay for large M (dispatch.py picks, `swa_dit_rows_impl`).
//
// Structure of every kernel: persistent CTAs of 2 groups (8 warps); the CTA's weights sit in shared memory for its whole life, copied
// with cp.async straight from the weight tensors (bank-swizzled per row, no preprocessing launch).  Per slab:
//   build   the 4 warps build the GEMM's A operand (x, gated, y, dffn, datt: RMSNorm / modulation / gating of the slab's rows) together,
//           4 rows each, into a natural-order bf16 smem tile [16][128] (and write it out when it is a saved tensor);
//   GEMM    each warp runs its quarter (an N quarter, or a K quarter with the partials exchanged through smem), A by ldmatrix from the
//           tile, B by ldmatrix from the weight, B fragments two steps ahead;
//   epilog  on the accumulators: lane (g, t) owns rows g and g + 8 of the slab.  The GEMM's output layout comes from the weight rows
//           ldmatrix is pointed at: "lane = head" (lane t holds head t's 32 channels: the q / k RMSNorm and RoPE are lane-local) or
//           "warp = head" (warp c holds head c, lane t its 8-channel slice: 64 B per quad, full-sector stores); where a GEMM's output
//           is in the MMA's natural layout the results are restaged through smem so they leave as 16 B stores.
// Slabs of the backward kernels are 16/SP atoms x SP augments of one batch element (row rho -> augment a0 + rho % SP, atom
// s0 + rho / SP): the modulation gradients are pre-summed over the slab's augments with shuffles, then one vector red.add per (atom,
// channels) -- or a plain store when a slab holds every augment (A == SP: one writer per element; each kernel owns its dmod columns).
// Buffers of the backward: the attention backward writes dq | dk | dv row-major into columns 0 | C | 2C of dP [M, 4C], the
// out-projection backward writes dG into 3C..4C, the qkvg backward turns dq, dk into dp_q, dp_k in place: dP is the operand of the
// weight gradients [dWq; dWk; dWv] = dP[:, :3C]^T x and dWg = dP[:, 3C:]^T x.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <algorithm>
#include "swa_ptx.cuh"
#include "pdl.cuh"

namespace {
constexpr int C = 128, H = 4, D = 32, NH = 256, MODW = 6 * C;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void group_sync(int grp) { asm volatile("bar.sync %0, 128;\n" :: "r"(1 + grp) : "memory"); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ void red4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void red2(float* p, float a, float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(p), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ float quad_sum(float v) {       // over lanes t = 0..3 of a quad
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
__device__ __forceinline__ float oct_sum(float v) {        // over the 8 lanes of a build row
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 4);
}

// 32 bf16 (one head) of one row as 16 packed pairs; 8 bf16 as 4
struct Row {
  uint32_t u[16];
};
struct Qtr {
  uint32_t u[4];
};
__device__ __forceinline__ void ld_row(Row& r, const bf16* p, bool ok) {
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 v = ok ? __ldg(reinterpret_cast<const uint4*>(p) + q) : make_uint4(0u, 0u, 0u, 0u);
    r.u[4 * q] = v.x; r.u[4 * q + 1] = v.y; r.u[4 * q + 2] = v.z; r.u[4 * q + 3] = v.w;
  }
}
__device__ __forceinline__ void st_row(bf16* p, const Row& r, bool ok) {
  if (!ok) return;
#pragma unroll
  for (int q = 0; q < 4; ++q) reinterpret_cast<uint4*>(p)[q] = make_uint4(r.u[4 * q], r.u[4 * q + 1], r.u[4 * q + 2], r.u[4 * q + 3]);
}
__device__ __forceinline__ void ld_q(Qtr& r, const bf16* p, bool ok) {
  const uint4 v = ok ? __ldg(reinterpret_cast<const uint4*>(p)) : make_uint4(0u, 0u, 0u, 0u);
  r.u[0] = v.x; r.u[1] = v.y; r.u[2] = v.z; r.u[3] = v.w;
}
__device__ __forceinline__ void st_q(bf16* p, const Qtr& r, bool ok) {
  if (ok) *reinterpret_cast<uint4*>(p) = make_uint4(r.u[0], r.u[1], r.u[2], r.u[3]);
}
__device__ __forceinline__ void ld_f8(float (&f)[8], const float* p) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
// the 16 bf16 / fp32 of a build lane (columns 16 bc .. 16 bc + 15)
struct B16 {
  uint32_t u[8];
};
__device__ __forceinline__ void ld_b16(B16& r, const bf16* p, bool ok) {
  const uint4 a = ok ? __ldg(reinterpret_cast<const uint4*>(p)) : make_uint4(0u, 0u, 0u, 0u);
  const uint4 b = ok ? __ldg(reinterpret_cast<const uint4*>(p) + 1) : make_uint4(0u, 0u, 0u, 0u);
  r.u[0] = a.x; r.u[1] = a.y; r.u[2] = a.z; r.u[3] = a.w; r.u[4] = b.x; r.u[5] = b.y; r.u[6] = b.z; r.u[7] = b.w;
}
__device__ __forceinline__ void st_b16(bf16* p, const B16& r, bool ok) {
  if (!ok) return;
  reinterpret_cast<uint4*>(p)[0] = make_uint4(r.u[0], r.u[1], r.u[2], r.u[3]);
  reinterpret_cast<uint4*>(p)[1] = make_uint4(r.u[4], r.u[5], r.u[6], r.u[7]);
}
__device__ __forceinline__ void ld_f16(float (&f)[16], const float* p) {
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(p) + q);
    f[4 * q] = v.x; f[4 * q + 1] = v.y; f[4 * q + 2] = v.z; f[4 * q + 3] = v.w;
  }
}
template <int N> __device__ __forceinline__ void zero(float (&acc)[N][4]) {
#pragma unroll
  for (int j = 0; j < N; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
}

// ---- smem tiles [16][W] bf16 in natural column order, RB bytes per row, 16 B chunk j at (j ^ (r & 7)) ---------------------------------
template <int RB> __device__ __forceinline__ uint32_t toff(const unsigned char* tile, int r, int j) { return sa(tile) + r * RB + ((j ^ (r & 7)) << 4); }
template <int RB> __device__ __forceinline__ unsigned char* tptr(unsigned char* tile, int r, int j) { return tile + r * RB + ((j ^ (r & 7)) << 4); }
template <int RB> __device__ __forceinline__ void tile_a(const unsigned char* tile, int s, int lane, uint32_t (&a)[4]) {
  ldsm4(toff<RB>(tile, (lane & 7) + ((lane >> 3) & 1) * 8, 2 * s + (lane >> 4)), a);
}
// a build lane's 16 values (row rho, columns 16 bc ..) into a [16][128] tile
__device__ __forceinline__ void tile_put(unsigned char* tile, int rho, int bc, const B16& v) {
  *reinterpret_cast<uint4*>(tptr<256>(tile, rho, 2 * bc)) = make_uint4(v.u[0], v.u[1], v.u[2], v.u[3]);
  *reinterpret_cast<uint4*>(tptr<256>(tile, rho, 2 * bc + 1)) = make_uint4(v.u[4], v.u[5], v.u[6], v.u[7]);
}

// ---- weights in smem: [rows][RB / 2] bf16 in natural column order; 16 B chunk j of row r at (j ^ key(r)).  The key separates the 8
// rows one ldmatrix reads: KSIG rows 32a + 2j + b, KTAU rows 32c + 8a + 2i + b, KNAT 8 consecutive rows.
enum Key { KSIG = 0, KTAU = 1, KNAT = 2 };
__device__ __forceinline__ int wkey(int key, int r) {
  return key == KSIG ? ((((r >> 5) & 3) << 1) | (r & 1)) : key == KTAU ? ((((r >> 3) & 3) << 1) | (r & 1)) : (r & 7);
}
template <int KEY, int RB> __device__ __forceinline__ uint32_t woff(const unsigned char* img, int r, int j) {
  return sa(img) + r * RB + ((j ^ wkey(KEY, r)) << 4);
}
// rows r0 .. r0 + rows of the smem weight from w [rows][RB / 2] (whole CTA, cp.async; the caller waits)
template <int KEY, int RB> __device__ __forceinline__ void fill_w(unsigned char* img, const bf16* w, int rows, int r0) {
  constexpr int CPR = RB / 16;
  for (int e = threadIdx.x; e < rows * CPR; e += blockDim.x) {
    const int r = e / CPR, j = e % CPR;
    cp16(woff<KEY, RB>(img, r0 + r, j), w + (long)e * 8);
  }
}

// ---- GEMMs: acc[2 NP][4] += A (16 x 16 KS) . B over a smem weight; B fragments two steps ahead of their MMAs; afn(s, a) gives A.
// gemm_t: weight rows = k (transposed ldmatrix; rows k0 + 16 s + kk), stored columns = n; n8 blocks 2 np0 .. 2 (np0 + NP)
template <int NP, int KS, int KEY, int RB, typename AF>
__device__ __forceinline__ void gemm_t(float (&acc)[2 * NP][4], AF afn, const unsigned char* img, int k0, int np0, int lane) {
  const int mi = lane >> 3, kk = (lane & 7) + 8 * (mi & 1), ch = mi >> 1;
  constexpr int NI = KS * NP;
  uint32_t b[3][4], a[4];
  auto load = [&](int i, uint32_t (&d)[4]) { ldsm4t(woff<KEY, RB>(img, k0 + kk + 16 * (i / NP), 2 * (np0 + i % NP) + ch), d); };
  load(0, b[0]);
  if (NI > 1) load(1, b[1]);
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    if (i + 2 < NI) load(i + 2, b[(i + 2) % 3]);
    const int np = i % NP;
    if (np == 0) afn(i / NP, a);
    mma(acc[2 * np], a, b[i % 3][0], b[i % 3][1]);
    mma(acc[2 * np + 1], a, b[i % 3][2], b[i % 3][3]);
  }
}
// gemm_n: weight rows = n (ldmatrix, not transposed), stored columns = k.  The row that n8 block blk's fragment column c reads:
//   RSIG 32 (c >> 1) + 2 blk + (c & 1)  (output lane = head);  RTAU 32 (blk >> 2) + 8 (c >> 1) + 2 (blk & 3) + (c & 1)
//   (output warp = head);  RNAT 8 blk + c  (natural)
enum RowPat { RSIG = 0, RTAU = 1, RNAT = 2 };
template <int RP> __device__ __forceinline__ int nrow(int blk, int c) {
  return RP == RSIG ? 32 * (c >> 1) + 2 * blk + (c & 1) : RP == RTAU ? 32 * (blk >> 2) + 8 * (c >> 1) + 2 * (blk & 3) + (c & 1) : 8 * blk + c;
}
template <int NP, int KS, int KEY, int RB, int RP, typename AF>
__device__ __forceinline__ void gemm_n(float (&acc)[2 * NP][4], AF afn, const unsigned char* img, int n0, int np0, int lane) {
  const int mi = lane >> 3, c = lane & 7;
  constexpr int NI = KS * NP;
  uint32_t b[3][4], a[4];
  auto load = [&](int i, uint32_t (&d)[4]) {
    const int blk = 2 * (np0 + i % NP) + (mi >> 1);
    ldsm4(woff<KEY, RB>(img, n0 + nrow<RP>(blk, c), 2 * (i / NP) + (mi & 1)), d);
  };
  load(0, b[0]);
  if (NI > 1) load(1, b[1]);
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    if (i + 2 < NI) load(i + 2, b[(i + 2) % 3]);
    const int np = i % NP;
    if (np == 0) afn(i / NP, a);
    mma(acc[2 * np], a, b[i % 3][0], b[i % 3][1]);
    mma(acc[2 * np + 1], a, b[i % 3][2], b[i % 3][3]);
  }
}

// ---- slab geometry --------------------------------------------------------------------------------------------------------------------
struct Geo {
  int S, A, B, nat, nag, nslab;
  int atomic;
};
template <int SP> struct Slab {
  static constexpr int AT = 16 / SP;
  int b, a0, s0;
  __device__ Slab(const Geo& g, int t) {
    const int at = t % g.nat, r = t / g.nat, ag = r % g.nag;
    b = r / g.nag; a0 = ag * SP; s0 = at * AT;
  }
  __device__ bool valid(const Geo& g, int rho) const { return a0 + rho % SP < g.A && s0 + rho / SP < g.S; }
  __device__ long row(const Geo& g, int rho) const { return valid(g, rho) ? ((long)(a0 + rho % SP) * g.B + b) * g.S + s0 + rho / SP : 0; }
  __device__ long mrow(const Geo& g, int rho) const { const int s = s0 + rho / SP; return (long)b * g.S + (s < g.S ? s : g.S - 1); }
};
// Pre-sum over the slab's augments of 8 values per row (v0: row g, v1: row g + 8; zero on invalid rows), then into dmod at columns
// col + off(i) for value i: SLICE off(i) = i (8 contiguous), PAIRS off(i) = 8 (i >> 1) + (i & 1) (4 pairs, 8 apart).
template <int SP, bool PAIRS>
__device__ __forceinline__ void presum(float* dmod, const Geo& geo, const Slab<SP>& sl, float (&v0)[8], float (&v1)[8], int col, int lane) {
  const int g = lane >> 2;
  constexpr int LS = SP < 8 ? SP : 8;
  if (SP == 16) {
#pragma unroll
    for (int i = 0; i < 8; ++i) v0[i] += v1[i];
  }
#pragma unroll
  for (int o = 4; o < 4 * LS; o <<= 1)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v0[i] += __shfl_xor_sync(0xffffffffu, v0[i], o);
      if (SP != 16) v1[i] += __shfl_xor_sync(0xffffffffu, v1[i], o);
    }
  if (g % LS != 0) return;
#pragma unroll
  for (int r = 0; r < (SP == 16 ? 1 : 2); ++r) {
    const int s = sl.s0 + (g + 8 * r) / SP;
    if (s >= geo.S) continue;
    float* p = dmod + ((long)sl.b * geo.S + s) * MODW + col;
    const float* v = r ? v1 : v0;
    if (PAIRS) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (geo.atomic) red2(p + 8 * i, v[2 * i], v[2 * i + 1]);
        else *reinterpret_cast<float2*>(p + 8 * i) = make_float2(v[2 * i], v[2 * i + 1]);
      }
    } else if (geo.atomic) { red4(p, v[0], v[1], v[2], v[3]); red4(p + 4, v[4], v[5], v[6], v[7]); }
    else { reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]); reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]); }
  }
}
// forward kernels: slab t = rows 16t .. 16t + 15 of the flattened sequence
struct FRow {
  long R, n, s, m;
  bool ok;
  __device__ FRow(long R_, long M, int S, int B) : R(R_), ok(R_ < M) {
    const long r = ok ? R : 0;
    n = r / S; s = r - n * S; m = (n % B) * S + s;
    if (!ok) R = 0;
  }
};

// ======================================================================================================================================
// qkvg forward: warp qc computes projection qc (q, k, v, g) of the slab, output lane = head (the head RMSNorm / RoPE lane-local).
struct QFArgs {
  const bf16 *q, *wqkv, *wg;
  const float *mod, *cs, *sn;
  bf16 *Q, *K, *V, *G, *X, *PQ, *PK;
  long M;
  int S, B, save, nslab;
  float eps, qk_eps;
};
constexpr int QF_W = 4 * C * 256, QF_TILE = 16 * 256;
constexpr int QF_SMEM = QF_W + 2 * 2 * QF_TILE;              // W | per group, double-buffered: x

__global__ void __launch_bounds__(256) qkvg_fwd_kernel(QFArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, grp = w >> 2, qc = w & 3, g = lane >> 2, t = lane & 3;
  const int br = lane >> 3, bc = lane & 7, rho = 4 * qc + br;
  fill_w<KSIG, 256>(smem, a.wqkv, 3 * C, 0);
  fill_w<KSIG, 256>(smem, a.wg, C, 3 * C);
  pdl_gate();
  const int ng = blockDim.x >> 7;
  bool first = true;
  int it = 0;
#pragma unroll 1
  for (int ts = blockIdx.x * ng + grp; ts < a.nslab; ts += gridDim.x * ng, ++it) {
    unsigned char* tile = smem + QF_W + (grp * 2 + (it & 1)) * QF_TILE;
    {                                                      // build: x of row rho, columns 16 bc ..
      const FRow rb(16L * ts + rho, a.M, a.S, a.B);
      B16 qv, xv;
      ld_b16(qv, a.q + rb.R * C + 16 * bc, rb.ok);
      float sh[16], sc[16];
      ld_f16(sh, a.mod + rb.m * MODW + 16 * bc);
      ld_f16(sc, a.mod + rb.m * MODW + C + 16 * bc);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float2 v = unpack(qv.u[i]); ss += v.x * v.x + v.y * v.y; }
      const float rs = rsqrtf(oct_sum(ss) * (1.f / C) + a.eps);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 v = unpack(qv.u[i]);
        xv.u[i] = pack(v.x * rs * (1.f + sc[2 * i]) + sh[2 * i], v.y * rs * (1.f + sc[2 * i + 1]) + sh[2 * i + 1]);
      }
      tile_put(tile, rho, bc, xv);
      if (a.save) st_b16(a.X + rb.R * C + 16 * bc, xv, rb.ok);
    }
    const FRow r0(16L * ts + g, a.M, a.S, a.B), r1(16L * ts + g + 8, a.M, a.S, a.B);
    float cs[2][16], sn[2][16];                            // RoPE tables of rows g, g + 8 (q / k warps), in flight during the GEMM
    if (qc < 2) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const long m = r ? r1.m : r0.m;
        ld_f16(cs[r], a.cs + m * (D / 2));
        ld_f16(sn[r], a.sn + m * (D / 2));
      }
    }
    if (first) { cp_wait_all(); __syncthreads(); first = false; }   // the weights have landed
    else group_sync(grp);
    float acc[16][4];
    zero(acc);
    gemm_n<8, 8, KSIG, 256, RSIG>(acc, [&](int s, uint32_t (&o)[4]) { tile_a<256>(tile, s, lane, o); }, smem, 128 * qc, 0, lane);
    // lane t holds p's head t of rows g, g + 8 (channel 32t + 2j + e in acc[j])
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const FRow& rr = r ? r1 : r0;
      Row p;
#pragma unroll
      for (int j = 0; j < 16; ++j) p.u[j] = pack(acc[j][2 * r], acc[j][2 * r + 1]);
      const long hm = ((rr.n * H + t) * a.S + rr.s) * D;          // head-major row of head t
      if (qc == 3) { st_row(a.G + rr.R * C + 32 * t, p, rr.ok); continue; }
      if (qc == 2) { st_row(a.V + hm, p, rr.ok); continue; }
      if (a.save) st_row((qc == 0 ? a.PQ : a.PK) + rr.R * C + 32 * t, p, rr.ok);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) { const float2 v = unpack(p.u[i]); ss += v.x * v.x + v.y * v.y; }
      const float rq = rsqrtf(ss * (1.f / D) + a.qk_eps);
      Row o;
#pragma unroll
      for (int i = 0; i < 8; ++i) {                        // pair i: d = 2i, 2i + 1; pair i + 8: d + 16
        const float2 v1 = unpack(p.u[i]), v2 = unpack(p.u[i + 8]);
        const float x10 = rnb(v1.x * rq), x11 = rnb(v1.y * rq), x20 = rnb(v2.x * rq), x21 = rnb(v2.y * rq);
        const float c0 = cs[r][2 * i], c1 = cs[r][2 * i + 1], s0 = sn[r][2 * i], s1 = sn[r][2 * i + 1];
        o.u[i] = pack(x10 * c0 - x20 * s0, x11 * c1 - x21 * s1);
        o.u[i + 8] = pack(x20 * c0 + x10 * s0, x21 * c1 + x11 * s1);
      }
      st_row((qc == 0 ? a.Q : a.K) + hm, o, rr.ok);
    }
  }
  if (first) { cp_wait_all(); __syncthreads(); }
}

// ======================================================================================================================================
// out-projection forward: warp qc computes att's head qc (warp = head).
struct OFArgs {
  const bf16 *q, *G, *O, *wo;
  const float* mod;
  bf16 *q1, *att;
  long M;
  int S, B, save, nslab;
};
constexpr int OF_W = C * 256, OF_TILE = 16 * 256;
constexpr int OF_SMEM = OF_W + 2 * 2 * OF_TILE;

__global__ void __launch_bounds__(256) oproj_fwd_kernel(OFArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, grp = w >> 2, qc = w & 3, g = lane >> 2, t = lane & 3;
  const int br = lane >> 3, bc = lane & 7, rho = 4 * qc + br;
  fill_w<KTAU, 256>(smem, a.wo, C, 0);
  pdl_gate();
  const int ng = blockDim.x >> 7, sc = 32 * qc + 8 * t;
  bool first = true;
  int it = 0;
#pragma unroll 1
  for (int ts = blockIdx.x * ng + grp; ts < a.nslab; ts += gridDim.x * ng, ++it) {
    unsigned char* tile = smem + OF_W + (grp * 2 + (it & 1)) * OF_TILE;
    {                                                      // build: gated = rn(sigmoid(g) o)
      const FRow rb(16L * ts + rho, a.M, a.S, a.B);
      B16 gv, ov, xv;
      ld_b16(gv, a.G + rb.R * C + 16 * bc, rb.ok);
      ld_b16(ov, a.O + rb.R * C + 16 * bc, rb.ok);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 u = unpack(gv.u[i]), v = unpack(ov.u[i]);
        xv.u[i] = pack(sigmoidf_(u.x) * v.x, sigmoidf_(u.y) * v.y);
      }
      tile_put(tile, rho, bc, xv);
    }
    const FRow r0(16L * ts + g, a.M, a.S, a.B), r1(16L * ts + g + 8, a.M, a.S, a.B);
    Qtr qs0, qs1;
    ld_q(qs0, a.q + r0.R * C + sc, r0.ok); ld_q(qs1, a.q + r1.R * C + sc, r1.ok);
    float ga0[8], ga1[8];
    ld_f8(ga0, a.mod + r0.m * MODW + 2 * C + sc); ld_f8(ga1, a.mod + r1.m * MODW + 2 * C + sc);
    if (first) { cp_wait_all(); __syncthreads(); first = false; }
    else group_sync(grp);
    float acc[4][4];
    zero(acc);
    gemm_n<2, 8, KTAU, 256, RTAU>(acc, [&](int s, uint32_t (&o)[4]) { tile_a<256>(tile, s, lane, o); }, smem, 0, 2 * qc, lane);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const FRow& rr = r ? r1 : r0;
      const Qtr& qq = r ? qs1 : qs0;
      const float* ga = r ? ga1 : ga0;
      Qtr at, q1;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        at.u[i] = pack(acc[i][2 * r], acc[i][2 * r + 1]);
        const float2 av = unpack(at.u[i]), qv = unpack(qq.u[i]);
        q1.u[i] = pack(qv.x + rnb(rnb(ga[2 * i]) * av.x), qv.y + rnb(rnb(ga[2 * i + 1]) * av.y));
      }
      st_q(a.q1 + rr.R * C + sc, q1, rr.ok);
      if (a.save) st_q(a.att + rr.R * C + sc, at, rr.ok);
    }
  }
  if (first) { cp_wait_all(); __syncthreads(); }
}

// ======================================================================================================================================
// FFN forward: y built together; warp qc computes hidden units 64qc .. 64qc + 63 (a | b, h); h goes through a smem tile; warp qc
// computes the output's head qc (warp = head) over all 256 hidden units.
struct FFArgs {
  const bf16 *q1, *wu, *wd;
  const float* mod;
  bf16 *out, *y, *ffn;
  long M;
  int S, B, save, nslab;
  float eps;
};
constexpr int FF_WU = 2 * NH * 256, FF_WD = C * 512, FF_YT = 16 * 256, FF_HT = 16 * 512;
constexpr int FF_SMEM = FF_WU + FF_WD + 2 * (FF_YT + FF_HT);   // Wu | Wd | per group: y, h

__global__ void __launch_bounds__(256) ffn_fwd_kernel(FFArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, grp = w >> 2, qc = w & 3, g = lane >> 2, t = lane & 3;
  const int br = lane >> 3, bc = lane & 7, rho = 4 * qc + br;
  unsigned char* wu = smem;
  unsigned char* wd = smem + FF_WU;
  unsigned char* ytile = smem + FF_WU + FF_WD + grp * (FF_YT + FF_HT);
  unsigned char* htile = ytile + FF_YT;
  fill_w<KNAT, 256>(wu, a.wu, 2 * NH, 0);
  fill_w<KTAU, 512>(wd, a.wd, C, 0);
  pdl_gate();
  const int ng = blockDim.x >> 7, sc = 32 * qc + 8 * t;
  bool first = true;
#pragma unroll 1
  for (int ts = blockIdx.x * ng + grp; ts < a.nslab; ts += gridDim.x * ng) {
    {                                                      // build: y = rn(RMS(q1) (1 + scale_f) + shift_f)
      const FRow rb(16L * ts + rho, a.M, a.S, a.B);
      B16 qv, yv;
      ld_b16(qv, a.q1 + rb.R * C + 16 * bc, rb.ok);
      float sh[16], scf[16];
      ld_f16(sh, a.mod + rb.m * MODW + 3 * C + 16 * bc);
      ld_f16(scf, a.mod + rb.m * MODW + 4 * C + 16 * bc);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float2 v = unpack(qv.u[i]); ss += v.x * v.x + v.y * v.y; }
      const float rs = rsqrtf(oct_sum(ss) * (1.f / C) + a.eps);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 v = unpack(qv.u[i]);
        yv.u[i] = pack(v.x * rs * (1.f + scf[2 * i]) + sh[2 * i], v.y * rs * (1.f + scf[2 * i + 1]) + sh[2 * i + 1]);
      }
      tile_put(ytile, rho, bc, yv);
      if (a.save) st_b16(a.y + rb.R * C + 16 * bc, yv, rb.ok);
    }
    const FRow r0(16L * ts + g, a.M, a.S, a.B), r1(16L * ts + g + 8, a.M, a.S, a.B);
    Qtr qs0, qs1;
    ld_q(qs0, a.q1 + r0.R * C + sc, r0.ok); ld_q(qs1, a.q1 + r1.R * C + sc, r1.ok);
    float gf0[8], gf1[8];
    ld_f8(gf0, a.mod + r0.m * MODW + 5 * C + sc); ld_f8(gf1, a.mod + r1.m * MODW + 5 * C + sc);
    if (first) { cp_wait_all(); __syncthreads(); first = false; }
    else group_sync(grp);
    {
      auto ya = [&](int s, uint32_t (&o)[4]) { tile_a<256>(ytile, s, lane, o); };
      float ha[8][4], hb[8][4];
      zero(ha); zero(hb);
      gemm_n<4, 8, KNAT, 256, RNAT>(ha, ya, wu, 64 * qc, 0, lane);
      gemm_n<4, 8, KNAT, 256, RNAT>(hb, ya, wu, NH + 64 * qc, 0, lane);
      // h = rn(silu(a) b), hidden 64qc + 8jb + 2t + e -> the h tile
#pragma unroll
      for (int jb = 0; jb < 8; ++jb)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float x0 = ha[jb][2 * r], x1 = ha[jb][2 * r + 1];
          *reinterpret_cast<uint32_t*>(tptr<512>(htile, g + 8 * r, 8 * qc + jb) + 4 * t) =
              pack(x0 * sigmoidf_(x0) * hb[jb][2 * r], x1 * sigmoidf_(x1) * hb[jb][2 * r + 1]);
        }
    }
    group_sync(grp);
    float fa[4][4];
    zero(fa);
    gemm_n<2, 16, KTAU, 512, RTAU>(fa, [&](int s, uint32_t (&o)[4]) { tile_a<512>(htile, s, lane, o); }, wd, 0, 2 * qc, lane);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const FRow& rr = r ? r1 : r0;
      const Qtr& qs = r ? qs1 : qs0;
      const float* gf = r ? gf1 : gf0;
      Qtr ff, o;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ff.u[i] = pack(fa[i][2 * r], fa[i][2 * r + 1]);
        const float2 fv = unpack(ff.u[i]), qv = unpack(qs.u[i]);
        o.u[i] = pack(qv.x + rnb(rnb(gf[2 * i]) * fv.x), qv.y + rnb(rnb(gf[2 * i + 1]) * fv.y));
      }
      st_q(a.out + rr.R * C + sc, o, rr.ok);
      if (a.save) st_q(a.ffn + rr.R * C + sc, ff, rr.ok);
    }
  }
  if (first) { cp_wait_all(); __syncthreads(); }
}

// ======================================================================================================================================
// FFN backward: dffn and y built together; warp qc computes dh, a, b, da, db, h for hidden units 64qc .. + 63; [da | db] goes through a
// smem tile (and out from there); warp qc computes dy's head qc (natural layout: lane t holds channels 32qc + 8i + 2t + e) over all 512;
// dq1 leaves through a staging tile.
struct FBArgs {
  const bf16 *dq2, *q1, *y, *ffn, *wu, *wd;
  const float* mod;
  float* dmod;
  bf16 *dq1, *dffn, *h, *dab;
  float eps;
  Geo geo;
};
constexpr int FB_R = 16 * 1024;                             // per group: dffn | y tiles (build), then [da | db] [16][512], staging at 8 KB
constexpr int FB_X = 512;                                   // per group: row dots [4][16], rstd [2][16] (by slab parity)
constexpr int FB_SMEM = FF_WU + FF_WD + 2 * (FB_R + FB_X);

template <int SP>
__global__ void __launch_bounds__(256) ffn_bwd_kernel(FBArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, grp = w >> 2, qc = w & 3, g = lane >> 2, t = lane & 3;
  const int br = lane >> 3, bc = lane & 7, rho = 4 * qc + br;
  const Geo geo = a.geo;
  unsigned char* wu = smem;
  unsigned char* wd = smem + FF_WU;
  unsigned char* reg = smem + FF_WU + FF_WD + grp * (FB_R + FB_X);
  unsigned char* dtile = reg;                              // dffn [16][128]
  unsigned char* ytile = reg + 16 * 256;                   // y [16][128]
  unsigned char* stg = reg + 8192 + qc * 1024;             // dq1 staging of warp qc: [16][32] bf16
  float* dots = reinterpret_cast<float*>(reg + FB_R);      // [4][16] per-warp row partials; rs [2][16] after
  fill_w<KNAT, 256>(wu, a.wu, 2 * NH, 0);
  fill_w<KNAT, 512>(wd, a.wd, C, 0);
  pdl_gate();
  const int ng = blockDim.x >> 7, sc = 32 * qc + 8 * t, nc = 32 * qc + 2 * t;
  bool first = true;
  int it = 0;
#pragma unroll 1
  for (int ts = blockIdx.x * ng + grp; ts < geo.nslab; ts += gridDim.x * ng, ++it) {
    const Slab<SP> sl(geo, ts);
    {                                                      // build: dffn = rn(dq2 rn(gf)), y (copied), rstd of q1
      const bool ok = sl.valid(geo, rho);
      const long R = sl.row(geo, rho), m = sl.mrow(geo, rho);
      B16 dv, yv, qv;
      ld_b16(dv, a.dq2 + R * C + 16 * bc, ok);
      ld_b16(yv, a.y + R * C + 16 * bc, ok);
      ld_b16(qv, a.q1 + R * C + 16 * bc, ok);
      float gf[16];
      ld_f16(gf, a.mod + m * MODW + 5 * C + 16 * bc);
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float2 v = unpack(dv.u[i]); dv.u[i] = pack(v.x * rnb(gf[2 * i]), v.y * rnb(gf[2 * i + 1])); }
      tile_put(dtile, rho, bc, dv);
      tile_put(ytile, rho, bc, yv);
      st_b16(a.dffn + R * C + 16 * bc, dv, ok);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float2 v = unpack(qv.u[i]); ss += v.x * v.x + v.y * v.y; }
      ss = oct_sum(ss);
      if (bc == 0) dots[64 + (it & 1) * 16 + rho] = rsqrtf(ss * (1.f / C) + a.eps);
    }
    const bool ok0 = sl.valid(geo, g), ok1 = sl.valid(geo, g + 8);
    const long R0 = sl.row(geo, g), R1 = sl.row(geo, g + 8), m0 = sl.mrow(geo, g), m1 = sl.mrow(geo, g + 8);
    {                                                      // d gate_f = dq2 * ffn on the warp = head slice
      Qtr d0, d1, f0, f1;
      ld_q(d0, a.dq2 + R0 * C + sc, ok0); ld_q(d1, a.dq2 + R1 * C + sc, ok1);
      ld_q(f0, a.ffn + R0 * C + sc, ok0); ld_q(f1, a.ffn + R1 * C + sc, ok1);
      float p0[8], p1[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 x0 = unpack(d0.u[i]), y0 = unpack(f0.u[i]), x1 = unpack(d1.u[i]), y1 = unpack(f1.u[i]);
        p0[2 * i] = x0.x * y0.x; p0[2 * i + 1] = x0.y * y0.y; p1[2 * i] = x1.x * y1.x; p1[2 * i + 1] = x1.y * y1.y;
      }
      presum<SP, false>(a.dmod, geo, sl, p0, p1, 5 * C + sc, lane);
    }
    if (first) { cp_wait_all(); __syncthreads(); first = false; }
    else group_sync(grp);
    float dh[8][4], ha[8][4], hb[8][4];
    zero(dh); zero(ha); zero(hb);
    gemm_t<4, 8, KNAT, 512>(dh, [&](int s, uint32_t (&o)[4]) { tile_a<256>(dtile, s, lane, o); }, wd, 0, 4 * qc, lane);   // hidden 64qc ..
    {
      auto ya = [&](int s, uint32_t (&o)[4]) { tile_a<256>(ytile, s, lane, o); };
      gemm_n<4, 8, KNAT, 256, RNAT>(ha, ya, wu, 64 * qc, 0, lane);
      gemm_n<4, 8, KNAT, 256, RNAT>(hb, ya, wu, NH + 64 * qc, 0, lane);
    }
    group_sync(grp);                                       // every warp is done with the dffn / y tiles: the region becomes [da | db]
#pragma unroll
    for (int jb = 0; jb < 8; ++jb)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float da[2], db[2], hv[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float x = ha[jb][2 * r + e], bb = hb[jb][2 * r + e], s_ = sigmoidf_(x), d = dh[jb][2 * r + e];
          da[e] = d * bb * s_ * (1.f + x * (1.f - s_));
          db[e] = d * x * s_;
          hv[e] = x * s_ * bb;
        }
        const int row = g + 8 * r;
        *reinterpret_cast<uint32_t*>(tptr<1024>(reg, row, 8 * qc + jb) + 4 * t) = pack(da[0], da[1]);
        *reinterpret_cast<uint32_t*>(tptr<1024>(reg, row, 32 + 8 * qc + jb) + 4 * t) = pack(db[0], db[1]);
        if (r ? ok1 : ok0) *reinterpret_cast<uint32_t*>(a.h + (r ? R1 : R0) * NH + 64 * qc + 8 * jb + 2 * t) = pack(hv[0], hv[1]);
      }
    group_sync(grp);
#pragma unroll
    for (int k = 0; k < 8; ++k) {                          // [da | db] rows 4qc .. 4qc + 3 -> global (1 KB each)
      const int e = lane + 32 * k, rr = 4 * qc + (e >> 6), ch = e & 63;
      if (sl.valid(geo, rr)) *reinterpret_cast<uint4*>(a.dab + sl.row(geo, rr) * (2 * NH) + 8 * ch) = *reinterpret_cast<const uint4*>(tptr<1024>(reg, rr, ch));
    }
    float dy[4][4];
    zero(dy);
    gemm_t<2, 32, KNAT, 256>(dy, [&](int s, uint32_t (&o)[4]) { tile_a<1024>(reg, s, lane, o); }, wu, 0, 2 * qc, lane);
    // epilogue: lane (g, t) holds dy at channels nc + 8i + e of rows g, g + 8
    const float* rsb = dots + 64 + (it & 1) * 16;
    const float rs[2] = {rsb[g], rsb[g + 8]};
    float xh[2][8], d2[2][8], scf[2][8];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long R = r ? R1 : R0;
      const bool ok = r ? ok1 : ok0;
      const long m = r ? m1 : m0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t qv = ok ? __ldg(reinterpret_cast<const uint32_t*>(a.q1 + R * C + nc + 8 * i)) : 0u;
        const uint32_t dv = ok ? __ldg(reinterpret_cast<const uint32_t*>(a.dq2 + R * C + nc + 8 * i)) : 0u;
        const float2 sv = __ldg(reinterpret_cast<const float2*>(a.mod + m * MODW + 4 * C + nc + 8 * i));
        const float2 qf = unpack(qv), df = unpack(dv);
        xh[r][2 * i] = qf.x * rs[r]; xh[r][2 * i + 1] = qf.y * rs[r];
        d2[r][2 * i] = df.x; d2[r][2 * i + 1] = df.y;
        scf[r][2 * i] = sv.x; scf[r][2 * i + 1] = sv.y;
      }
    }
    {
      float v0[8], v1[8];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) { v0[2 * i + e] = ok0 ? dy[i][e] : 0.f; v1[2 * i + e] = ok1 ? dy[i][2 + e] : 0.f; }
      presum<SP, true>(a.dmod, geo, sl, v0, v1, 3 * C + nc, lane);           // d shift_f
    }
    {
      float v0[8], v1[8];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) { v0[2 * i + e] = dy[i][e] * xh[0][2 * i + e]; v1[2 * i + e] = dy[i][2 + e] * xh[1][2 * i + e]; }
      presum<SP, true>(a.dmod, geo, sl, v0, v1, 4 * C + nc, lane);           // d scale_f
    }
    float part[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float s_ = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float hh = dy[i][2 * r + e] * (1.f + scf[r][2 * i + e]);
          dy[i][2 * r + e] = hh;
          s_ += hh * xh[r][2 * i + e];
        }
      part[r] = quad_sum(s_);
    }
    if (t == 0) { dots[qc * 16 + g] = part[0]; dots[qc * 16 + g + 8] = part[1]; }
    group_sync(grp);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = g + 8 * r;
      const float dot = (dots[row] + dots[16 + row] + dots[32 + row] + dots[48 + row]) * (1.f / C);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        *reinterpret_cast<uint32_t*>(stg + row * 64 + ((i ^ ((row >> 1) & 3)) << 4) + 4 * t) =
            pack(d2[r][2 * i] + rs[r] * (dy[i][2 * r] - xh[r][2 * i] * dot), d2[r][2 * i + 1] + rs[r] * (dy[i][2 * r + 1] - xh[r][2 * i + 1] * dot));
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 2; ++k) {                          // staging [16][32] -> dq1 (64 B per row)
      const int e = lane + 32 * k, row = e >> 2, ch = e & 3;
      if (sl.valid(geo, row))
        *reinterpret_cast<uint4*>(a.dq1 + sl.row(geo, row) * C + 32 * qc + 8 * ch) =
            *reinterpret_cast<const uint4*>(stg + row * 64 + ((ch ^ ((row >> 1) & 3)) << 4));
    }
  }
  if (first) { cp_wait_all(); __syncthreads(); }
}

// ======================================================================================================================================
// out-projection backward: datt built together (datt and gated written out from the build); warp qc computes head qc of dgated (natural
// layout: lane t, channels 32qc + 8i + 2t + e); dO and dG leave through staging tiles.
struct OArgs {
  const bf16 *dq1, *g, *o, *att, *wo;
  const float* mod;
  float *dmod, *Dv;
  bf16 *dO, *dG, *datt, *gated;
  int ldg;                                                  // row stride of dG (a column block of the dP buffer)
  Geo geo;
};
constexpr int OB_W = C * 256, OB_TILE = 16 * 256, OB_STG = 4 * 2 * 1024;
constexpr int OB_SMEM = OB_W + 2 * (2 * OB_TILE + OB_STG);   // Wo | per group: datt (double-buffered), dO / dG staging per warp

template <int SP>
__global__ void __launch_bounds__(256) oproj_bwd_kernel(OArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, grp = w >> 2, qc = w & 3, g = lane >> 2, t = lane & 3;
  const int br = lane >> 3, bc = lane & 7, rho = 4 * qc + br;
  const Geo geo = a.geo;
  unsigned char* base = smem + OB_W + grp * (2 * OB_TILE + OB_STG);
  unsigned char* sdo = base + 2 * OB_TILE + qc * 2048;     // [16][32] bf16 dO, then dG
  unsigned char* sdg = sdo + 1024;
  fill_w<KNAT, 256>(smem, a.wo, C, 0);
  pdl_gate();
  const int ng = blockDim.x >> 7, sc = 32 * qc + 8 * t, nc = 32 * qc + 2 * t;
  bool first = true;
  int it = 0;
#pragma unroll 1
  for (int ts = blockIdx.x * ng + grp; ts < geo.nslab; ts += gridDim.x * ng, ++it) {
    unsigned char* tile = base + (it & 1) * OB_TILE;
    const Slab<SP> sl(geo, ts);
    {                                                      // build: datt = rn(dq1 rn(ga)) (tile + out), gated = rn(sg o) (out)
      const bool ok = sl.valid(geo, rho);
      const long R = sl.row(geo, rho), m = sl.mrow(geo, rho);
      B16 dv, gv, ov, xv, gt;
      ld_b16(dv, a.dq1 + R * C + 16 * bc, ok);
      ld_b16(gv, a.g + R * C + 16 * bc, ok);
      ld_b16(ov, a.o + R * C + 16 * bc, ok);
      float ga[16];
      ld_f16(ga, a.mod + m * MODW + 2 * C + 16 * bc);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 d = unpack(dv.u[i]), u = unpack(gv.u[i]), v = unpack(ov.u[i]);
        xv.u[i] = pack(d.x * rnb(ga[2 * i]), d.y * rnb(ga[2 * i + 1]));
        gt.u[i] = pack(sigmoidf_(u.x) * v.x, sigmoidf_(u.y) * v.y);
      }
      tile_put(tile, rho, bc, xv);
      st_b16(a.datt + R * C + 16 * bc, xv, ok);
      st_b16(a.gated + R * C + 16 * bc, gt, ok);
    }
    const bool ok0 = sl.valid(geo, g), ok1 = sl.valid(geo, g + 8);
    const long R0 = sl.row(geo, g), R1 = sl.row(geo, g + 8);
    {                                                      // d gate_a = dq1 * att on the warp = head slice
      Qtr d0, d1, x0, x1;
      ld_q(d0, a.dq1 + R0 * C + sc, ok0); ld_q(d1, a.dq1 + R1 * C + sc, ok1);
      ld_q(x0, a.att + R0 * C + sc, ok0); ld_q(x1, a.att + R1 * C + sc, ok1);
      float p0[8], p1[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 u0 = unpack(d0.u[i]), v0 = unpack(x0.u[i]), u1 = unpack(d1.u[i]), v1 = unpack(x1.u[i]);
        p0[2 * i] = u0.x * v0.x; p0[2 * i + 1] = u0.y * v0.y; p1[2 * i] = u1.x * v1.x; p1[2 * i + 1] = u1.y * v1.y;
      }
      presum<SP, false>(a.dmod, geo, sl, p0, p1, 2 * C + sc, lane);
    }
    uint32_t gp[2][4], op[2][4];                           // g, o at the lane's output channels (in flight during the GEMM)
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const long R = r ? R1 : R0;
        const bool ok = r ? ok1 : ok0;
        gp[r][i] = ok ? __ldg(reinterpret_cast<const uint32_t*>(a.g + R * C + nc + 8 * i)) : 0u;
        op[r][i] = ok ? __ldg(reinterpret_cast<const uint32_t*>(a.o + R * C + nc + 8 * i)) : 0u;
      }
    if (first) { cp_wait_all(); __syncthreads(); first = false; }
    else group_sync(grp);
    float acc[4][4];
    zero(acc);
    gemm_t<2, 8, KNAT, 256>(acc, [&](int s, uint32_t (&o)[4]) { tile_a<256>(tile, s, lane, o); }, smem, 0, 2 * qc, lane);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = g + 8 * r;
      float dsum = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 gv = unpack(gp[r][i]), ov = unpack(op[r][i]);
        const float s0 = sigmoidf_(gv.x), s1 = sigmoidf_(gv.y), d0 = acc[i][2 * r], d1 = acc[i][2 * r + 1];
        const uint32_t dov = pack(d0 * s0, d1 * s1);
        const float2 dof = unpack(dov);
        dsum += dof.x * ov.x + dof.y * ov.y;
        const int off = row * 64 + ((i ^ ((row >> 1) & 3)) << 4) + 4 * t;
        *reinterpret_cast<uint32_t*>(sdo + off) = dov;
        *reinterpret_cast<uint32_t*>(sdg + off) = pack(d0 * ov.x * s0 * (1.f - s0), d1 * ov.y * s1 * (1.f - s1));
      }
      dsum = quad_sum(dsum);
      if ((r ? ok1 : ok0) && t == 0) {
        const int a_ = sl.a0 + row % SP, s = sl.s0 + row / SP;
        a.Dv[(((long)a_ * geo.B + sl.b) * H + qc) * geo.S + s] = dsum;
      }
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 2; ++k) {                          // staging -> dO, dG (64 B per row)
      const int e = lane + 32 * k, row = e >> 2, ch = e & 3;
      if (!sl.valid(geo, row)) continue;
      const long R = sl.row(geo, row);
      const int off = row * 64 + ((ch ^ ((row >> 1) & 3)) << 4);
      *reinterpret_cast<uint4*>(a.dO + R * C + 32 * qc + 8 * ch) = *reinterpret_cast<const uint4*>(sdo + off);
      *reinterpret_cast<uint4*>(a.dG + R * a.ldg + 32 * qc + 8 * ch) = *reinterpret_cast<const uint4*>(sdg + off);
    }
    __syncwarp();
  }
  if (first) { cp_wait_all(); __syncthreads(); }
}

// ======================================================================================================================================
// qkvg backward: warp qc takes projection qc -- its RoPE / RMSNorm backward (lane = head), dp_qc through the warp's own smem tile as the
// A operand of its 128 x 128 slice of dx (natural layout); the four partials are exchanged through smem so that warp qc finishes head qc
// (lane t: channels 32qc + 8i + 2t + e); dq leaves through the warp's tile.
struct QArgs {
  const bf16 *q, *pq, *pk, *dq1, *wqkv, *wg;
  const float *mod, *cs, *sn;
  bf16* dP;                                                 // [M, 4C]: dq | dk | dv | dG in, dp_q | dp_k | dv | dG out
  bf16* dq;
  float* dmod;
  float eps, qk_eps;
  Geo geo;
};
constexpr int QB_W = 4 * C * 256, QB_XCH = 4 * 4 * 4 * 32 * 16;
constexpr int QB_SMEM = QB_W + 2 * (4 * 16 * 256 + QB_XCH + 256);   // W | per group: 4 warp tiles, exchange, row dots

// dy (the gradient of the RoPE output of head t) and p (the saved pre-norm projection) of one row -> dp = rn(headRMS_bwd(p, unrope(dy)))
// in place of dy.  cs / sn: the row's 16 RoPE frequencies (pair d, d + 16).
__device__ __forceinline__ void head_bwd(Row& dy, const Row& p, const float (&cs)[16], const float (&sn)[16], float qk_eps) {
  auto unrope = [&](int i, float (&u)[4]) {                // dx1 = dy1 c + dy2 s ; dx2 = dy2 c - dy1 s   (pair i, pair i + 8)
    const float2 a = unpack(dy.u[i]), b = unpack(dy.u[i + 8]);
    u[0] = a.x * cs[2 * i] + b.x * sn[2 * i];         u[1] = a.y * cs[2 * i + 1] + b.y * sn[2 * i + 1];
    u[2] = b.x * cs[2 * i] - a.x * sn[2 * i];         u[3] = b.y * cs[2 * i + 1] - a.y * sn[2 * i + 1];
  };
  float ss = 0.f, dot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float u[4];
    unrope(i, u);
    const float2 a = unpack(p.u[i]), b = unpack(p.u[i + 8]);
    ss += a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
    dot += u[0] * a.x + u[1] * a.y + u[2] * b.x + u[3] * b.y;
  }
  const float rr = rsqrtf(ss * (1.f / D) + qk_eps);
  const float k = rr * rr * dot * (1.f / D);              // dp = rr (ux - p rr mean(ux p rr)) = rr (ux - p k)
  Row out;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float u[4];
    unrope(i, u);
    const float2 a = unpack(p.u[i]), b = unpack(p.u[i + 8]);
    out.u[i] = pack(rr * (u[0] - a.x * k), rr * (u[1] - a.y * k));
    out.u[i + 8] = pack(rr * (u[2] - b.x * k), rr * (u[3] - b.y * k));
  }
  dy = out;
}

template <int SP>
__global__ void __launch_bounds__(256) qkvg_bwd_kernel(QArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, grp = w >> 2, qc = w & 3, g = lane >> 2, t = lane & 3;
  const Geo geo = a.geo;
  unsigned char* gbase = smem + QB_W + grp * (4 * 16 * 256 + QB_XCH + 256);
  unsigned char* tile = gbase + qc * 16 * 256;             // this warp's [16][128]: dp_qc, then dq staging
  float4* xch = reinterpret_cast<float4*>(gbase + 4 * 16 * 256);
  float* dots = reinterpret_cast<float*>(gbase + 4 * 16 * 256 + QB_XCH);
  fill_w<KNAT, 256>(smem, a.wqkv, 3 * C, 0);
  fill_w<KNAT, 256>(smem, a.wg, C, 3 * C);
  pdl_gate();
  const int ng = blockDim.x >> 7, nc = 32 * qc + 2 * t;
  bool first = true;
#pragma unroll 1
  for (int ts = blockIdx.x * ng + grp; ts < geo.nslab; ts += gridDim.x * ng) {
    const Slab<SP> sl(geo, ts);
    const bool ok0 = sl.valid(geo, g), ok1 = sl.valid(geo, g + 8);
    const long R0 = sl.row(geo, g), R1 = sl.row(geo, g + 8), m0 = sl.mrow(geo, g), m1 = sl.mrow(geo, g + 8);
    const long p0 = R0 * (4 * C) + qc * C + 32 * t, p1 = R1 * (4 * C) + qc * C + 32 * t;    // the lane's head in block qc of dP
    Row x0, x1;
    ld_row(x0, a.dP + p0, ok0); ld_row(x1, a.dP + p1, ok1);
    // the epilogue's operands (in flight during the GEMM): q rows (RMSNorm statistics, lane = head), q / dq1 / scale_a at the lane's
    // output channels nc + 8i + e
    Row q0, q1;
    ld_row(q0, a.q + R0 * C + 32 * t, ok0); ld_row(q1, a.q + R1 * C + 32 * t, ok1);
    uint32_t qp[2][4], dp1[2][4];
    float2 scp[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const long R = r ? R1 : R0;
        const bool ok = r ? ok1 : ok0;
        qp[r][i] = ok ? __ldg(reinterpret_cast<const uint32_t*>(a.q + R * C + nc + 8 * i)) : 0u;
        dp1[r][i] = ok ? __ldg(reinterpret_cast<const uint32_t*>(a.dq1 + R * C + nc + 8 * i)) : 0u;
        scp[r][i] = __ldg(reinterpret_cast<const float2*>(a.mod + (r ? m1 : m0) * MODW + C + nc + 8 * i));
      }
    if (qc < 2) {                                          // RoPE and the per-head RMSNorm, backwards
      const bf16* pp = qc == 0 ? a.pq : a.pk;
      Row y0, y1;
      ld_row(y0, pp + R0 * C + 32 * t, ok0); ld_row(y1, pp + R1 * C + 32 * t, ok1);
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float cs[16], sn[16];
        const long m = r ? m1 : m0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float4 u = __ldg(reinterpret_cast<const float4*>(a.cs + m * (D / 2)) + q);
          const float4 v = __ldg(reinterpret_cast<const float4*>(a.sn + m * (D / 2)) + q);
          cs[4 * q] = u.x; cs[4 * q + 1] = u.y; cs[4 * q + 2] = u.z; cs[4 * q + 3] = u.w;
          sn[4 * q] = v.x; sn[4 * q + 1] = v.y; sn[4 * q + 2] = v.z; sn[4 * q + 3] = v.w;
        }
        head_bwd(r ? x1 : x0, r ? y1 : y0, cs, sn, a.qk_eps);
      }
      st_row(a.dP + p0, x0, ok0); st_row(a.dP + p1, x1, ok1);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {                          // dp_qc (lane = head rows) -> the warp's tile, natural order
      const Row& xx = r ? x1 : x0;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        *reinterpret_cast<uint4*>(tptr<256>(tile, g + 8 * r, 4 * t + q)) = make_uint4(xx.u[4 * q], xx.u[4 * q + 1], xx.u[4 * q + 2], xx.u[4 * q + 3]);
    }
    __syncwarp();
    if (first) { cp_wait_all(); __syncthreads(); first = false; }
    float acc[16][4];
    zero(acc);
    gemm_t<8, 8, KNAT, 256>(acc, [&](int s, uint32_t (&o)[4]) { tile_a<256>(tile, s, lane, o); }, smem, qc * C, 0, lane);
    // exchange: n8 block j of the partial (channels 8j + 2t + e) goes to warp j >> 2, slot j & 3, same lane
#pragma unroll
    for (int j = 0; j < 16; ++j)
      xch[((qc * 4 + (j >> 2)) * 4 + (j & 3)) * 32 + lane] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
    group_sync(grp);
    float dx[4][4] = {};
#pragma unroll
    for (int src = 0; src < 4; ++src)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 v = xch[((src * 4 + qc) * 4 + i) * 32 + lane];
        dx[i][0] += v.x; dx[i][1] += v.y; dx[i][2] += v.z; dx[i][3] += v.w;
      }
    float rs[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const Row& qq = r ? q1 : q0;
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) { const float2 v = unpack(qq.u[i]); ss += v.x * v.x + v.y * v.y; }
      rs[r] = rsqrtf(quad_sum(ss) * (1.f / C) + a.eps);
    }
    float xh[2][8];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) { const float2 v = unpack(qp[r][i]); xh[r][2 * i] = v.x * rs[r]; xh[r][2 * i + 1] = v.y * rs[r]; }
    {
      float v0[8], v1[8];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) { v0[2 * i + e] = ok0 ? dx[i][e] : 0.f; v1[2 * i + e] = ok1 ? dx[i][2 + e] : 0.f; }
      presum<SP, true>(a.dmod, geo, sl, v0, v1, nc, lane);                   // d shift_a
    }
    {
      float v0[8], v1[8];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) { v0[2 * i + e] = dx[i][e] * xh[0][2 * i + e]; v1[2 * i + e] = dx[i][2 + e] * xh[1][2 * i + e]; }
      presum<SP, true>(a.dmod, geo, sl, v0, v1, C + nc, lane);               // d scale_a
    }
    float part[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float s_ = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float sc2[2] = {scp[r][i].x, scp[r][i].y};
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float h = dx[i][2 * r + e] * (1.f + sc2[e]);
          dx[i][2 * r + e] = h;                            // dxh
          s_ += h * xh[r][2 * i + e];
        }
      }
      part[r] = quad_sum(s_);
    }
    if (t == 0) { dots[qc * 16 + g] = part[0]; dots[qc * 16 + g + 8] = part[1]; }
    group_sync(grp);
#pragma unroll
    for (int r = 0; r < 2; ++r) {                          // dq -> the warp's tile as [16][32] staging (the GEMM is done with it)
      const int row = g + 8 * r;
      const float dot = (dots[row] + dots[16 + row] + dots[32 + row] + dots[48 + row]) * (1.f / C);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 b = unpack(dp1[r][i]);
        *reinterpret_cast<uint32_t*>(tile + row * 64 + ((i ^ ((row >> 1) & 3)) << 4) + 4 * t) =
            pack(b.x + rs[r] * (dx[i][2 * r] - xh[r][2 * i] * dot), b.y + rs[r] * (dx[i][2 * r + 1] - xh[r][2 * i + 1] * dot));
      }
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int e = lane + 32 * k, row = e >> 2, ch = e & 3;
      if (sl.valid(geo, row))
        *reinterpret_cast<uint4*>(a.dq + sl.row(geo, row) * C + 32 * qc + 8 * ch) =
            *reinterpret_cast<const uint4*>(tile + row * 64 + ((ch ^ ((row >> 1) & 3)) << 4));
    }
    __syncwarp();
  }
  if (first) { cp_wait_all(); __syncthreads(); }
}

// ======================================================================================================================================
int num_sms(int dev) { static int s = 0; if (s == 0) cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }
int pick_sp(long A) { return A % 16 == 0 ? 16 : A % 8 == 0 ? 8 : A % 4 == 0 ? 4 : A % 2 == 0 ? 2 : 1; }
Geo make_geo(long A, long B, long S, int sp) {
  Geo g; g.S = (int)S; g.A = (int)A; g.B = (int)B;
  g.nat = (int)((S + 16 / sp - 1) / (16 / sp)); g.nag = (int)((A + sp - 1) / sp); g.nslab = g.nat * g.nag * (int)B;
  g.atomic = A != sp;
  return g;
}
void check_rows(const torch::Tensor& t, long M, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.numel() == M * C, name, ": [M, 128] bf16 contiguous");
}
void check_mod(const torch::Tensor& t, long rows, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat && t.is_contiguous() && t.numel() == rows * MODW, name, ": [B*S, 768] fp32");
}
void check_w(const torch::Tensor& t, long rows, long cols, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.size(0) == rows && t.size(1) == cols, name,
              ": weight shape / bf16 contiguous");
}
const bf16* bp(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr()); }
bf16* bpm(torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr()); }
template <typename F> void by_sp(int sp, F f) {
  switch (sp) {
    case 16: f(std::integral_constant<int, 16>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    default: f(std::integral_constant<int, 1>{}); break;
  }
}
// CTAs of 2 slab groups (8 warps): as many as fit at once (the kernels stride over the slabs), never more than the slabs need
template <typename K> int resident(K kernel, int smem, int sms) {
  int occ = 1;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, 256, smem));
  return sms * std::max(occ, 1);
}
int grid_of(int nslab, int res) { return std::max(1, std::min((nslab + 1) / 2, res)); }
}  // namespace

// ---- entry points ------------------------------------------------------------------------------------------------------------------
// q [M, 128] bf16; mod [B*S, 768] fp32; cos / sin [B*S, 16] fp32; wqkv [384, 128], wg [128, 128] bf16.
// Returns [Q, K, V (head-major [N, 4, S, 32]), G [M, 128], X, PQ, PK ([M, 128]; [1, 128] unless save)].
std::vector<torch::Tensor> qkvg_fwd(torch::Tensor q, torch::Tensor mod, torch::Tensor cs, torch::Tensor sn, torch::Tensor wqkv, torch::Tensor wg,
                                    int64_t N, int64_t B, int64_t S, double eps, double qk_eps, bool save) {
  const long M = N * S;
  check_rows(q, M, "q"); check_mod(mod, B * S, "mod"); check_w(wqkv, 3 * C, C, "wqkv"); check_w(wg, C, C, "wg");
  for (const auto* t : {&cs, &sn}) TORCH_CHECK(t->scalar_type() == torch::kFloat && t->is_contiguous() && t->numel() >= B * S * (D / 2), "cos / sin [B*S, 16] fp32");
  auto opt = q.options();
  auto Q = torch::empty({N, H, S, D}, opt), K = torch::empty_like(Q), V = torch::empty_like(Q), G = torch::empty({M, C}, opt);
  const long Ms = save ? M : 1;
  auto X = torch::empty({Ms, C}, opt), PQ = torch::empty({Ms, C}, opt), PK = torch::empty({Ms, C}, opt);
  if (M == 0) return {Q, K, V, G, X, PQ, PK};
  QFArgs a;
  a.q = bp(q); a.wqkv = bp(wqkv); a.wg = bp(wg); a.mod = mod.data_ptr<float>(); a.cs = cs.data_ptr<float>(); a.sn = sn.data_ptr<float>();
  a.Q = bpm(Q); a.K = bpm(K); a.V = bpm(V); a.G = bpm(G); a.X = bpm(X); a.PQ = bpm(PQ); a.PK = bpm(PK);
  a.M = M; a.S = (int)S; a.B = (int)B; a.save = save ? 1 : 0; a.nslab = (int)((M + 15) / 16); a.eps = (float)eps; a.qk_eps = (float)qk_eps;
  static int res = 0;
  if (res == 0) res = resident(qkvg_fwd_kernel, QF_SMEM, num_sms(q.device().index()));
  pdl_launch(qkvg_fwd_kernel, grid_of(a.nslab, res), 256, QF_SMEM, at::cuda::getCurrentCUDAStream(), false, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {Q, K, V, G, X, PQ, PK};
}

// q (block input), G, O [M, 128] bf16; mod [B*S, 768] fp32; wo [128, 128].  Returns [q1, att ([1, 128] unless save)].
std::vector<torch::Tensor> oproj_fwd(torch::Tensor q, torch::Tensor G, torch::Tensor O, torch::Tensor mod, torch::Tensor wo, int64_t N, int64_t B,
                                     int64_t S, bool save) {
  const long M = N * S;
  check_rows(q, M, "q"); check_rows(G, M, "G"); check_rows(O, M, "O"); check_mod(mod, B * S, "mod"); check_w(wo, C, C, "wo");
  auto q1 = torch::empty({M, C}, q.options()), att = torch::empty({save ? M : 1, C}, q.options());
  if (M == 0) return {q1, att};
  OFArgs a;
  a.q = bp(q); a.G = bp(G); a.O = bp(O); a.wo = bp(wo); a.mod = mod.data_ptr<float>(); a.q1 = bpm(q1); a.att = bpm(att);
  a.M = M; a.S = (int)S; a.B = (int)B; a.save = save ? 1 : 0; a.nslab = (int)((M + 15) / 16);
  static int res = 0;
  if (res == 0) res = resident(oproj_fwd_kernel, OF_SMEM, num_sms(q.device().index()));
  pdl_launch(oproj_fwd_kernel, grid_of(a.nslab, res), 256, OF_SMEM, at::cuda::getCurrentCUDAStream(), true, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q1, att};
}

// q1 [M, 128] bf16; mod [B*S, 768] fp32; wu [512, 128], wd [128, 256] bf16.  Returns [out, y, ffn] (y / ffn [1, 128] unless save).
std::vector<torch::Tensor> ffn_fwd(torch::Tensor q1, torch::Tensor mod, torch::Tensor wu, torch::Tensor wd, int64_t N, int64_t B, int64_t S, double eps,
                                   bool save) {
  const long M = N * S;
  check_rows(q1, M, "q1"); check_mod(mod, B * S, "mod"); check_w(wu, 2 * NH, C, "wu"); check_w(wd, C, NH, "wd");
  auto out = torch::empty({M, C}, q1.options()), y = torch::empty({save ? M : 1, C}, q1.options()), ffn = torch::empty({save ? M : 1, C}, q1.options());
  if (M == 0) return {out, y, ffn};
  FFArgs a;
  a.q1 = bp(q1); a.wu = bp(wu); a.wd = bp(wd); a.mod = mod.data_ptr<float>(); a.out = bpm(out); a.y = bpm(y); a.ffn = bpm(ffn);
  a.M = M; a.S = (int)S; a.B = (int)B; a.save = save ? 1 : 0; a.nslab = (int)((M + 15) / 16); a.eps = (float)eps;
  static int res = 0;
  if (res == 0) res = resident(ffn_fwd_kernel, FF_SMEM, num_sms(q1.device().index()));
  pdl_launch(ffn_fwd_kernel, grid_of(a.nslab, res), 256, FF_SMEM, at::cuda::getCurrentCUDAStream(), true, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, y, ffn};
}

// dq2, q1, y, ffn [M, 128] bf16; mod / dmod [B*S, 768] fp32 (dmod: shift_f, scale_f, gate_f columns written); wu [512, 128], wd [128, 256].
// Returns [dq1 [M, 128], dffn [M, 128], h [M, 256], dab [M, 512]].
std::vector<torch::Tensor> ffn_bwd(torch::Tensor dq2, torch::Tensor q1, torch::Tensor y, torch::Tensor ffn, torch::Tensor mod, torch::Tensor dmod,
                                   torch::Tensor wu, torch::Tensor wd, int64_t A, int64_t B, int64_t S, double eps) {
  const long M = A * B * S;
  check_rows(dq2, M, "dq2"); check_rows(q1, M, "q1"); check_rows(y, M, "y"); check_rows(ffn, M, "ffn");
  check_mod(mod, B * S, "mod"); check_mod(dmod, B * S, "dmod"); check_w(wu, 2 * NH, C, "wu"); check_w(wd, C, NH, "wd");
  auto opt = dq2.options();
  auto dq1 = torch::empty({M, C}, opt), dffn = torch::empty({M, C}, opt), h = torch::empty({M, NH}, opt), dab = torch::empty({M, 2 * NH}, opt);
  if (M == 0) return {dq1, dffn, h, dab};
  FBArgs a;
  a.dq2 = bp(dq2); a.q1 = bp(q1); a.y = bp(y); a.ffn = bp(ffn); a.wu = bp(wu); a.wd = bp(wd); a.mod = mod.data_ptr<float>();
  a.dmod = dmod.data_ptr<float>(); a.dq1 = bpm(dq1); a.dffn = bpm(dffn); a.h = bpm(h); a.dab = bpm(dab); a.eps = (float)eps;
  const int sp = pick_sp(A);
  a.geo = make_geo(A, B, S, sp);
  const int sms = num_sms(dq2.device().index());
  by_sp(sp, [&](auto spc) {
    constexpr int SP = decltype(spc)::value;
    static int res = 0;
    if (res == 0) res = resident(ffn_bwd_kernel<SP>, FB_SMEM, sms);
    pdl_launch(ffn_bwd_kernel<SP>, grid_of(a.geo.nslab, res), 256, FB_SMEM, at::cuda::getCurrentCUDAStream(), true, a);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq1, dffn, h, dab};
}

// dq1, g, o, att [M, 128] bf16; mod / dmod [B*S, 768] fp32 (dmod: gate_a columns written); wo [128, 128].
// dP [M, 512] bf16: dG goes to its columns 384..511.  Returns [dO, datt, gated ([M, 128] bf16), Dv [N, 4, S] fp32].
std::vector<torch::Tensor> oproj_bwd(torch::Tensor dq1, torch::Tensor g, torch::Tensor o, torch::Tensor att, torch::Tensor mod, torch::Tensor dmod,
                                     torch::Tensor wo, torch::Tensor dP, int64_t A, int64_t B, int64_t S) {
  const long M = A * B * S;
  check_rows(dq1, M, "dq1"); check_rows(g, M, "g"); check_rows(o, M, "o"); check_rows(att, M, "att");
  check_mod(mod, B * S, "mod"); check_mod(dmod, B * S, "dmod"); check_w(wo, C, C, "wo");
  TORCH_CHECK(dP.is_contiguous() && dP.scalar_type() == torch::kBFloat16 && dP.numel() == M * 4 * C, "dP [M, 512] bf16");
  auto opt = dq1.options();
  auto dO = torch::empty({M, C}, opt), datt = torch::empty({M, C}, opt), gated = torch::empty({M, C}, opt);
  auto Dv = torch::empty({A * B, H, S}, opt.dtype(torch::kFloat));
  if (M == 0) return {dO, datt, gated, Dv};
  OArgs a;
  a.dq1 = bp(dq1); a.g = bp(g); a.o = bp(o); a.att = bp(att); a.wo = bp(wo); a.mod = mod.data_ptr<float>(); a.dmod = dmod.data_ptr<float>();
  a.Dv = Dv.data_ptr<float>(); a.dO = bpm(dO); a.datt = bpm(datt); a.gated = bpm(gated);
  a.dG = bpm(dP) + 3 * C; a.ldg = 4 * C;
  const int sp = pick_sp(A);
  a.geo = make_geo(A, B, S, sp);
  const int sms = num_sms(dq1.device().index());
  by_sp(sp, [&](auto spc) {
    constexpr int SP = decltype(spc)::value;
    static int res = 0;
    if (res == 0) res = resident(oproj_bwd_kernel<SP>, OB_SMEM, sms);
    pdl_launch(oproj_bwd_kernel<SP>, grid_of(a.geo.nslab, res), 256, OB_SMEM, at::cuda::getCurrentCUDAStream(), true, a);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, datt, gated, Dv};
}

// q, pq, pk, dq1 [M, 128] bf16; mod / dmod [B*S, 768] fp32 (dmod: shift_a, scale_a columns written); cos / sin [B*S, 16] fp32;
// wqkv [384, 128], wg [128, 128]; dP [M, 512] bf16 holding dq | dk | dv | dG -> dp_q | dp_k | dv | dG.  Returns dq [M, 128].
torch::Tensor qkvg_bwd(torch::Tensor q, torch::Tensor pq, torch::Tensor pk, torch::Tensor dq1, torch::Tensor mod, torch::Tensor dmod, torch::Tensor cs,
                       torch::Tensor sn, torch::Tensor wqkv, torch::Tensor wg, torch::Tensor dP, int64_t A, int64_t B, int64_t S, double eps,
                       double qk_eps) {
  const long M = A * B * S;
  check_rows(q, M, "q"); check_rows(pq, M, "pq"); check_rows(pk, M, "pk"); check_rows(dq1, M, "dq1");
  check_mod(mod, B * S, "mod"); check_mod(dmod, B * S, "dmod"); check_w(wqkv, 3 * C, C, "wqkv"); check_w(wg, C, C, "wg");
  for (const auto* t : {&cs, &sn}) TORCH_CHECK(t->scalar_type() == torch::kFloat && t->is_contiguous() && t->numel() >= B * S * (D / 2), "cos / sin [B*S, 16] fp32");
  TORCH_CHECK(dP.is_contiguous() && dP.scalar_type() == torch::kBFloat16 && dP.numel() == M * 4 * C, "dP [M, 512] bf16");
  auto dq = torch::empty({M, C}, q.options());
  if (M == 0) return dq;
  QArgs a;
  a.q = bp(q); a.pq = bp(pq); a.pk = bp(pk); a.dq1 = bp(dq1); a.wqkv = bp(wqkv); a.wg = bp(wg);
  a.mod = mod.data_ptr<float>(); a.cs = cs.data_ptr<float>(); a.sn = sn.data_ptr<float>();
  a.dP = bpm(dP); a.dq = bpm(dq); a.dmod = dmod.data_ptr<float>(); a.eps = (float)eps; a.qk_eps = (float)qk_eps;
  const int sp = pick_sp(A);
  a.geo = make_geo(A, B, S, sp);
  const int sms = num_sms(q.device().index());
  by_sp(sp, [&](auto spc) {
    constexpr int SP = decltype(spc)::value;
    static int res = 0;
    if (res == 0) res = resident(qkvg_bwd_kernel<SP>, QB_SMEM, sms);
    pdl_launch(qkvg_bwd_kernel<SP>, grid_of(a.geo.nslab, res), 256, QB_SMEM, at::cuda::getCurrentCUDAStream(), true, a);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dq;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkvg_fwd", &qkvg_fwd, "SWA atom block: RMSNorm/adaLN + Q/K/V/G projections + qk-norm + RoPE (slab kernel)");
  m.def("oproj_fwd", &oproj_fwd, "SWA atom block: gated out-projection + residual (slab kernel)");
  m.def("ffn_fwd", &ffn_fwd, "SWA atom block: RMSNorm/adaLN + SwiGLU FFN + gated residual (slab kernel)");
  m.def("ffn_bwd", &ffn_bwd, "SWA atom block: FFN half backward (slab kernel)");
  m.def("oproj_bwd", &oproj_bwd, "SWA atom block: out-projection backward (slab kernel)");
  m.def("qkvg_bwd", &qkvg_bwd, "SWA atom block: qkvg backward (slab kernel)");
}
