// Token-pair initialisation of the input feature embedder on sm_90a (H100), forward and backward; d_pair = 128, fp32.
//
//   z[b,i,j,:] = left[b,i,:] + right[b,j,:] + Wrel[:, c1] + Wrel[:, 66 + c2] + Wrel[:, 132 + c3] + same_entity * Wrel[:, 138]
//                + Wbond[:, bond[b,i,j]]           (c1, c2, c3: AlphaFold 3's relative-position classes of the token pair)
//
// The reference builds the 139-wide one-hot per pair and runs a Linear over it; a one-hot times a matrix is a row gather, so
// these kernels never materialise it.  tbl [nbin][128] = [Wrel^T ; Wbond^T] (bins 0..138 relative position, 139 / 140 bond);
// ids [5][B L] int32 = asym, residue, token, entity, sym; bond [B L L] uint8.
//
//   forward   one pass writing z (the only large tensor): a CTA writes R rows (R <= 8) of one batch element; the class-row table arrives by one bulk
//             (TMA) copy that overlaps the packing of the rows bins, and right[j] is loaded once for the R rows (plain CUDA cores).
//   backward  BITWISE DETERMINISTIC (the SM100 kernel's global atomics are not): a CTA takes a quarter of the channels (32, one
//             per lane) of RB rows over every column; its four warps split the columns (j = warp mod 4).  Per position a warp
//             adds dz to its private shared-memory class-bin gradient (all lanes hit the same bin: no atomics, no conflicts),
//             to the row sums (registers) and to the column sum (written straight to a per-row-block partial).  At the end the
//             warps' sums are added in fixed order (dleft, dW partials) and a second kernel adds the partials in fixed order
//             (dright over the row blocks, dW over the CTAs).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int P = 128;
constexpr int NWARP = 8;
constexpr int RB = 8;                              // rows per backward CTA
constexpr int BW = 4;                              // warps per backward CTA

__device__ __forceinline__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ void operator+=(float4& a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

struct Row { int asym, res, tok, ent, sym; };

__device__ __forceinline__ Row load_row(const int* ids, int BL, int idx) {
  return Row{ids[idx], ids[BL + idx], ids[2 * BL + idx], ids[3 * BL + idx], ids[4 * BL + idx]};
}

__device__ __forceinline__ void pair_bins(const Row& a, const Row& b, int rmax, int smax, int& bin1, int& bin2, int& bin3, int& se) {
  const bool same_chain = a.asym == b.asym, same_res = a.res == b.res, same_ent = a.ent == b.ent;
  const int nr = 2 * rmax + 2, ns = 2 * smax + 2;
  int dres = min(max(a.res - b.res + rmax, 0), 2 * rmax);
  int dtok = min(max(a.tok - b.tok + rmax, 0), 2 * rmax);
  int dch = min(max(a.sym - b.sym + smax, 0), 2 * smax);
  bin1 = same_chain ? dres : nr - 1;
  bin2 = nr + ((same_chain && same_res) ? dtok : nr - 1);
  bin3 = 2 * nr + (same_ent ? dch : ns - 1);
  se = same_ent;
}

__device__ __forceinline__ uint32_t pack_pair(const Row& a, const Row& b, uint8_t bond, int rmax, int smax) {
  int b1, b2, b3, se;
  pair_bins(a, b, rmax, smax, b1, b2, b3, se);
  return (uint32_t)b1 | ((uint32_t)b2 << 8) | ((uint32_t)b3 << 16) | ((uint32_t)se << 24) | ((uint32_t)bond << 25);
}

__device__ __forceinline__ void pack_rows(uint32_t* pk, const int* ids, const uint8_t* bond, int BL, int L, int b, int i0, int nrows,
                                          int rmax, int smax) {
  for (int idx = threadIdx.x; idx < nrows * L; idx += blockDim.x) {
    const int r = idx / L, j = idx % L, ri = b * L + i0 + r;
    pk[idx] = pack_pair(load_row(ids, BL, ri), load_row(ids, BL, b * L + j), bond[(size_t)ri * L + j], rmax, smax);
  }
}


// ------------------------------------------------------------------------------------------------ forward v2
// A CTA writes R rows of z for one batch element.  The class-row table is fetched by one bulk copy (TMA) that overlaps the packing of the R rows' bins; each
// warp then owns a group of UNR columns at a time: it loads right[j] ONCE and writes the R rows' outputs for them (right is read 1/R times as often as
// when every row is a CTA of its own).  The sum order per element is that of tpi_fwd: left + right, then the class rows, the entity row, the bond row.
#ifndef TPI_UNR
#define TPI_UNR 4
#endif
constexpr int UNR = TPI_UNR;
constexpr int RMAX = 8;

__device__ __forceinline__ void tma_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst), b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(d), "l"(src), "r"(bytes), "r"(b)
               : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@p bra DONE_%=;\nbra WAIT_%=;\nDONE_%=:\n}" ::"r"(b), "r"(parity)
               : "memory");
}

__global__ void __launch_bounds__(256)
tpi_fwd(const float* __restrict__ left, const float* __restrict__ right, const float* __restrict__ tbl, const int* __restrict__ ids,
         const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax, int R) {
  extern __shared__ __align__(128) float sm[];
  const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1, nbin = nrel + 2;
  uint32_t* pk = (uint32_t*)(sm + nbin * P);                        // [R][L]
  __shared__ __align__(8) uint64_t bar;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
  const int tiles = (L + R - 1) / R;
  const int b = blockIdx.x / tiles, i0 = (blockIdx.x % tiles) * R;
  const int nrows = min(R, L - i0);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&bar)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) tma_load(sm, tbl, (uint32_t)(nbin * P * 4), &bar);
  pack_rows(pk, ids, bond, BL, L, b, i0, nrows, rmax, smax);
  float4 lf[RMAX];
#pragma unroll
  for (int r = 0; r < RMAX; r++) lf[r] = r < nrows ? ((const float4*)(left + (size_t)(b * L + i0 + r) * P))[lane] : make_float4(0, 0, 0, 0);
  __syncthreads();
  bar_wait(&bar, 0);
  const float4* T = (const float4*)sm;
  float4* obase = (float4*)(out + ((size_t)(b * L + i0) * L) * P);
  for (int j0 = warp * UNR; j0 < L; j0 += 8 * UNR) {
    float4 rt[UNR];
#pragma unroll
    for (int u = 0; u < UNR; u++) rt[u] = ((const float4*)(right + (size_t)(b * L + min(j0 + u, L - 1)) * P))[lane];
#pragma unroll
    for (int r = 0; r < RMAX; r++) if (r < nrows) {
      float4 o[UNR];
#pragma unroll
      for (int u = 0; u < UNR; u++) {
        const int j = min(j0 + u, L - 1);
        const uint32_t w = pk[r * L + j];
        o[u] = lf[r] + rt[u];
        o[u] += T[(w & 255) * 32 + lane];
        o[u] += T[((w >> 8) & 255) * 32 + lane];
        o[u] += T[((w >> 16) & 255) * 32 + lane];
        if ((w >> 24) & 1) o[u] += T[(nrel - 1) * 32 + lane];
        o[u] += T[(nrel + ((w >> 25) & 1)) * 32 + lane];
      }
#pragma unroll
      for (int u = 0; u < UNR; u++)
        if (j0 + u < L) __stcs(obase + ((size_t)r * L + j0 + u) * 32 + lane, o[u]);
    }
  }
}
// ------------------------------------------------------------------------------------------------ backward
__global__ void __launch_bounds__(BW * 32)
tpi_bwd(const float* __restrict__ g, const int* __restrict__ ids, const uint8_t* __restrict__ bond, float* __restrict__ dleft,
        float* __restrict__ dpart, float* __restrict__ wpart, int B, int L, int rmax, int smax) {
  extern __shared__ __align__(16) float sm[];
  const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1, nbin = nrel + 2;
  float* acc = sm;                                   // [BW][nbin][32]
  float* red = acc + BW * nbin * 32;                 // [BW][RB][32]
  uint32_t* pk = (uint32_t*)(red + BW * RB * 32);    // [RB][L]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
  const int nrb = (L + RB - 1) / RB;
  const int q = blockIdx.x & 3, rb = (blockIdx.x >> 2) % nrb, b = blockIdx.x / (4 * nrb);
  const int i0 = rb * RB, nrows = min(RB, L - i0);
  float* my = acc + warp * nbin * 32;
  for (int idx = lane; idx < nbin * 32; idx += 32) my[idx] = 0.f;
  pack_rows(pk, ids, bond, BL, L, b, i0, nrows, rmax, smax);
  __syncthreads();
  float rs[RB];
#pragma unroll
  for (int r = 0; r < RB; r++) rs[r] = 0.f;
  const float* gb = g + ((size_t)(b * L + i0) * L) * P + 32 * q + lane;
  constexpr int DEPTH = 4;                           // positions in flight per warp: DEPTH RB loads of 128 bytes
  float gq[DEPTH][RB];
#pragma unroll
  for (int d = 0; d < DEPTH; d++)
#pragma unroll
    for (int r = 0; r < RB; r++) gq[d][r] = (warp + d * BW < L && r < nrows) ? __ldg(gb + ((size_t)r * L + warp + d * BW) * P) : 0.f;
  for (int j = warp; j < L; j += BW) {
    float cur[RB];
#pragma unroll
    for (int r = 0; r < RB; r++) cur[r] = gq[0][r];
#pragma unroll
    for (int d = 0; d + 1 < DEPTH; d++)
#pragma unroll
      for (int r = 0; r < RB; r++) gq[d][r] = gq[d + 1][r];
    const int jn = j + DEPTH * BW;
#pragma unroll
    for (int r = 0; r < RB; r++) gq[DEPTH - 1][r] = (jn < L && r < nrows) ? __ldg(gb + ((size_t)r * L + jn) * P) : 0.f;
    float col = 0.f;
#pragma unroll
    for (int r = 0; r < RB; r++) {
      if (r >= nrows) continue;
      const float v = cur[r];
      rs[r] += v;
      col += v;
      const uint32_t w = pk[r * L + j];
#ifndef TPI_NODW
      my[(w & 255) * 32 + lane] += v;
      my[((w >> 8) & 255) * 32 + lane] += v;
      my[((w >> 16) & 255) * 32 + lane] += v;
      if ((w >> 24) & 1) my[(nrel - 1) * 32 + lane] += v;
      my[(nrel + ((w >> 25) & 1)) * 32 + lane] += v;
#endif
    }
    dpart[((size_t)(b * nrb + rb) * L + j) * P + 32 * q + lane] = col;
  }
#pragma unroll
  for (int r = 0; r < RB; r++) red[(warp * RB + r) * 32 + lane] = rs[r];
  __syncthreads();
  for (int idx = threadIdx.x; idx < RB * 32; idx += BW * 32) {
    const int r = idx / 32, l = idx % 32;
    if (r >= nrows) continue;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < BW; w++) s += red[(w * RB + r) * 32 + l];
    dleft[(size_t)(b * L + i0 + r) * P + 32 * q + l] = s;
  }
  for (int idx = threadIdx.x; idx < nbin * 32; idx += BW * 32) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < BW; w++) s += acc[w * nbin * 32 + idx];
    wpart[(size_t)blockIdx.x * nbin * 32 + idx] = s;
  }
}

// dright[b, j, c] = sum over the row blocks; dtbl[bin, c] = sum over the CTAs (b, rb) of the channel quarter c / 32: fixed order
__global__ void __launch_bounds__(128) tpi_reduce(const float* __restrict__ dpart, const float* __restrict__ wpart,
                                                   float* __restrict__ dright, float* __restrict__ dtbl, int B, int L, int nbin) {
  const int nrb = (L + RB - 1) / RB;
  const int nright = B * L;
  if ((int)blockIdx.x < nright) {                     // one (b, j) row of 128 channels per block
    const int b = blockIdx.x / L, j = blockIdx.x % L;
    float s = 0.f;
    for (int rb = 0; rb < nrb; rb++) s += dpart[((size_t)(b * nrb + rb) * L + j) * P + threadIdx.x];
    dright[(size_t)blockIdx.x * P + threadIdx.x] = s;
  } else {                                            // one bin per block
    const int bin = blockIdx.x - nright, q = threadIdx.x >> 5, l = threadIdx.x & 31;
    float s = 0.f;
    for (int b = 0; b < B; b++)
      for (int rb = 0; rb < nrb; rb++) s += wpart[((size_t)((b * nrb + rb) * 4 + q) * nbin + bin) * 32 + l];
    dtbl[(size_t)bin * P + threadIdx.x] = s;
  }
}

namespace {
inline int nrel_of(int rmax, int smax) { return 2 * (2 * rmax + 2) + (2 * smax + 2) + 1; }
}

void forward(torch::Tensor left, torch::Tensor right, torch::Tensor tbl, torch::Tensor ids, torch::Tensor bond, torch::Tensor out,
              int64_t rmax, int64_t smax, int64_t R) {
  const at::cuda::CUDAGuard guard(left.device());
  TORCH_CHECK(R >= 1 && R <= RMAX, "rows per CTA: 1..", RMAX);
  const int B = left.size(0), L = left.size(1), nbin = nrel_of(rmax, smax) + 2;
  const int tiles = (L + R - 1) / R;
  const size_t smem = (size_t)nbin * P * 4 + (size_t)R * L * 4;
  cudaFuncSetAttribute(tpi_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  tpi_fwd<<<B * tiles, 32 * NWARP, smem, at::cuda::getCurrentCUDAStream()>>>(
      left.data_ptr<float>(), right.data_ptr<float>(), tbl.data_ptr<float>(), ids.data_ptr<int>(), bond.data_ptr<uint8_t>(),
      out.data_ptr<float>(), B, L, (int)rmax, (int)smax, (int)R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// dleft [B L 128], dright [B L 128], dtbl [nbin 128]: all written (no zero-fill needed); dpart / wpart workspaces
void backward(torch::Tensor g, torch::Tensor ids, torch::Tensor bond, torch::Tensor dleft, torch::Tensor dright, torch::Tensor dtbl,
              torch::Tensor dpart, torch::Tensor wpart, int64_t rmax, int64_t smax) {
  const at::cuda::CUDAGuard guard(g.device());
  const int B = g.size(0), L = g.size(1), nbin = nrel_of(rmax, smax) + 2, nrb = (L + RB - 1) / RB;
  const size_t smem = ((size_t)BW * nbin * 32 + (size_t)BW * RB * 32 + (size_t)RB * L) * 4;
  cudaFuncSetAttribute(tpi_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  auto st = at::cuda::getCurrentCUDAStream();
  tpi_bwd<<<B * nrb * 4, BW * 32, smem, st>>>(g.data_ptr<float>(), ids.data_ptr<int>(), bond.data_ptr<uint8_t>(),
                                              dleft.data_ptr<float>(), dpart.data_ptr<float>(), wpart.data_ptr<float>(), B, L,
                                              (int)rmax, (int)smax);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tpi_reduce<<<B * L + nbin, 128, 0, st>>>(dpart.data_ptr<float>(), wpart.data_ptr<float>(), dright.data_ptr<float>(),
                                           dtbl.data_ptr<float>(), B, L, nbin);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
}
