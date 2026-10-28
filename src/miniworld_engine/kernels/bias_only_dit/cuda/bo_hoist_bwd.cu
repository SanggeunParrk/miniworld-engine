// Bias-only token DiT, the HOISTED pair bias's backward in ONE kernel (bf16, mma.sync m16n8k16): the three steps of
// ``integrations/bias_only_dit_hoist.py _hoist_bwd`` that stream the pair -- d LN0 = dbias_all^T W'_all (a cuBLAS GEMM with fp32 output),
// the LayerNorm backward rows (``ln0_bwd_rows``) and dW'_all = dbias_all Y (a cuBLAS GEMM over the hi / lo split of LN0) -- fused, so d LN0
// (fp32, 4 bytes per pair element) and Y (4 bytes) never reach HBM.
//
// Per 64-row item of the pair [R, 128] (persistent CTAs, items double-buffered by cp.async):
//   pair rows [64, 128] and dbias_all [N, 64 rows] (N = blocks x heads <= 192) land in shared memory;
//   LN0 of the rows with the forward kernel's arithmetic (mean, rstd, y); Y = [bf16(y) | bf16(y - bf16(y))] to shared memory;
//   d LN0 [64, 128] = dbias_all^T W'_all on mma (K = N), the LayerNorm backward on the accumulators (row sums over 128 channels: the
//   quad's two shuffles and one exchange between the two warps that share a row), d pair out as 16-byte rows;
//   dW' [N, 128] += dbias_all Y_hi + dbias_all Y_lo on mma, the accumulators staying in registers across items and leaving as one fp32 partial per CTA.
// Reads the pair and dbias_all once, writes d pair once: the HBM traffic of the three steps without the two fp32 / split temporaries.
// No atomics; the partials are summed by the caller (a fixed order): bitwise repeatable.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "bo_ptx.cuh"

namespace {
using bf = __nv_bfloat16;

using namespace bop;
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&u);
  return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

constexpr int RB = 64;          // pair rows per item
constexpr int ZP = 136;         // row pitch (elements) of the pair / weight / Y tiles
constexpr int DPITCH = RB + 8;  // row pitch of the dbias tile [head][row]

// NPT = N / 16 (heads x blocks in m16 tiles, 1..12)
template <int NPT>
__global__ void __launch_bounds__(256, 1) hoist_bwd_k(const bf* __restrict__ Z, const bf* __restrict__ WF, const bf* __restrict__ DB, bf* __restrict__ DZ,
                                                      float* __restrict__ DWP, long R, int N, float eps) {
  constexpr int NP = NPT * 16;
  extern __shared__ __align__(16) uint8_t sm_raw[];
  bf* Wfs = reinterpret_cast<bf*>(sm_raw);                // [NP][ZP]
  bf* Zs = Wfs + NP * ZP;                                 // 2 x [RB][ZP]
  bf* Ds = Zs + 2 * RB * ZP;                              // 2 x [NP][DPITCH]
  bf* Ys = Ds + 2 * NP * DPITCH;                          // [2][RB][ZP]: hi, lo
  float2* RS = reinterpret_cast<float2*>(Ys + 2 * RB * ZP);   // [RB] (mean, rstd)
  float2* XS = RS + RB;                                   // [2][RB] partial (sum d, sum d y) of the two channel halves
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const long nitems = R / RB;

  auto prefetch = [&](int s, long item) {
    const long r0 = item * RB;
    const uint32_t zs = smem_u32(Zs + s * RB * ZP), ds = smem_u32(Ds + s * NP * DPITCH);
    for (int i = tid; i < RB * 16; i += 256) {
      const int r = i >> 4, c = i & 15;
      cp_async16(zs + (r * ZP + c * 8) * 2, Z + (r0 + r) * 128 + c * 8);
    }
    for (int i = tid; i < N * 8; i += 256) {
      const int h = i >> 3, c = i & 7;
      cp_async16(ds + (h * DPITCH + c * 8) * 2, DB + (long)h * R + r0 + c * 8);
    }
  };
  if ((long)blockIdx.x < nitems) prefetch(0, blockIdx.x);
  cp_commit();
  for (int i = tid; i < N * 16; i += 256) {                // weights; rows N .. NP of every tile zero (padding heads weigh nothing)
    const int h = i >> 4, c = i & 15;
    *reinterpret_cast<uint4*>(Wfs + h * ZP + c * 8) = *reinterpret_cast<const uint4*>(WF + (long)h * 128 + c * 8);
  }
  for (int i = tid; i < (NP - N) * 16; i += 256) *reinterpret_cast<uint4*>(Wfs + (N + (i >> 4)) * ZP + (i & 15) * 8) = make_uint4(0, 0, 0, 0);
  for (int s = 0; s < 2; ++s)
    for (int i = tid; i < (NP - N) * 8; i += 256) *reinterpret_cast<uint4*>(Ds + s * NP * DPITCH + (N + (i >> 3)) * DPITCH + (i & 7) * 8) = make_uint4(0, 0, 0, 0);

  // dW' accumulators: warp (mg = warp / 2: m-tiles 3 mg .. 3 mg + 2, nh = warp % 2: channels 64 nh ..)
  float dw[3][8][4];
#pragma unroll
  for (int j = 0; j < 3; ++j)
#pragma unroll
    for (int n = 0; n < 8; ++n) dw[j][n][0] = dw[j][n][1] = dw[j][n][2] = dw[j][n][3] = 0.f;
  const int mg = warp >> 1, nh = warp & 1, rg = warp & 3, ch = warp >> 2;
  const int mi = lane >> 3;
  const uint32_t wfa = smem_u32(Wfs), ysa = smem_u32(Ys);

  int it = 0;
  for (long item = blockIdx.x; item < nitems; item += gridDim.x, ++it) {
    const int s = it & 1;
    cp_wait<0>();
    __syncthreads();                                       // item landed; the other stage's last reader (it - 1) is done
    if (item + gridDim.x < nitems) prefetch(s ^ 1, item + gridDim.x);
    cp_commit();
    bf* zt = Zs + s * RB * ZP;
    const uint32_t dsa = smem_u32(Ds + s * NP * DPITCH);

    // (C) LN0 of this warp's 8 rows with the forward kernel's arithmetic (the eight rows' reductions interleaved); Y = [hi | lo]
    {
      float y[8][4], sm[8], sq[8];
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const uint2 u = *reinterpret_cast<const uint2*>(zt + (warp * 8 + rr) * ZP + lane * 4);
        const float2 p0 = unpack2(u.x), p1 = unpack2(u.y);
        y[rr][0] = p0.x; y[rr][1] = p0.y; y[rr][2] = p1.x; y[rr][3] = p1.y;
        sm[rr] = y[rr][0] + y[rr][1] + y[rr][2] + y[rr][3];
      }
#pragma unroll
      for (int o = 16; o; o >>= 1)
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) sm[rr] += __shfl_xor_sync(0xffffffffu, sm[rr], o);
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        sm[rr] = sm[rr] / 128.f;                            // mean
#pragma unroll
        for (int e = 0; e < 4; ++e) y[rr][e] -= sm[rr];
        sq[rr] = y[rr][0] * y[rr][0] + y[rr][1] * y[rr][1] + y[rr][2] * y[rr][2] + y[rr][3] * y[rr][3];
      }
#pragma unroll
      for (int o = 16; o; o >>= 1)
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) sq[rr] += __shfl_xor_sync(0xffffffffu, sq[rr], o);
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int r = warp * 8 + rr;
        const float rstd = rsqrtf(sq[rr] / 128.f + eps);
#pragma unroll
        for (int e = 0; e < 4; ++e) y[rr][e] *= rstd;
        const bf h0 = __float2bfloat16_rn(y[rr][0]), h1 = __float2bfloat16_rn(y[rr][1]), h2 = __float2bfloat16_rn(y[rr][2]), h3 = __float2bfloat16_rn(y[rr][3]);
        uint2 hi, lo;
        hi.x = pack2(__bfloat162float(h0), __bfloat162float(h1)); hi.y = pack2(__bfloat162float(h2), __bfloat162float(h3));
        lo.x = pack2(y[rr][0] - __bfloat162float(h0), y[rr][1] - __bfloat162float(h1));
        lo.y = pack2(y[rr][2] - __bfloat162float(h2), y[rr][3] - __bfloat162float(h3));
        *reinterpret_cast<uint2*>(Ys + r * ZP + lane * 4) = hi;
        *reinterpret_cast<uint2*>(Ys + RB * ZP + r * ZP + lane * 4) = lo;
        if (lane == 0) RS[r] = make_float2(sm[rr], rstd);
      }
    }
    __syncthreads();

    // (E) d LN0 [16 rg .., 64 ch ..] = dbias^T W': A from the dbias tile (transposed), B from the weights
    float dl[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) dl[n][0] = dl[n][1] = dl[n][2] = dl[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < NPT; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldsm4t(dsa + ((ks * 16 + (mi >> 1) * 8 + (lane & 7)) * DPITCH + rg * 16 + (mi & 1) * 8) * 2, a0, a1, a2, a3);
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm4t(wfa + ((ks * 16 + (mi & 1) * 8 + (lane & 7)) * ZP + ch * 64 + np * 16 + (mi >> 1) * 8) * 2, b0, b1, b2, b3);
        mma16816(dl[2 * np], a0, a1, a2, a3, b0, b1);
        mma16816(dl[2 * np + 1], a0, a1, a2, a3, b2, b3);
      }
    }
    // (G) dW' += dbias Y_hi + dbias Y_lo: A = the dbias tile (rows = heads), B = Y (rows = pair rows)
#pragma unroll
    for (int yl = 0; yl < 2; ++yl) {
#pragma unroll
      for (int kk = 0; kk < RB / 16; ++kk) {
        uint32_t bb[4][4];
#pragma unroll
        for (int np = 0; np < 4; ++np)
          ldsm4t(ysa + (yl * RB * ZP + (kk * 16 + (mi & 1) * 8 + (lane & 7)) * ZP + nh * 64 + np * 16 + (mi >> 1) * 8) * 2, bb[np][0], bb[np][1], bb[np][2], bb[np][3]);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (mg * 3 + j < NPT) {
            uint32_t a0, a1, a2, a3;
            ldsm4(dsa + (((mg * 3 + j) * 16 + (mi & 1) * 8 + (lane & 7)) * DPITCH + kk * 16 + (mi >> 1) * 8) * 2, a0, a1, a2, a3);
#pragma unroll
            for (int np = 0; np < 4; ++np) {
              mma16816(dw[j][2 * np], a0, a1, a2, a3, bb[np][0], bb[np][1]);
              mma16816(dw[j][2 * np + 1], a0, a1, a2, a3, bb[np][2], bb[np][3]);
            }
          }
        }
      }
    }
    // row sums of d LN0 and d LN0 * y over this warp's 64 channels (rows g, g + 8)
    float sd[2] = {0.f, 0.f}, sy[2] = {0.f, 0.f};
    float yv[8][2][2];                                      // y at the accumulator positions, the forward's arithmetic
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float2 ms = RS[rg * 16 + g + half * 8];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float2 x = unpack2(*reinterpret_cast<const uint32_t*>(zt + (rg * 16 + g + half * 8) * ZP + ch * 64 + n * 8 + 2 * t));
        const float y0 = (x.x - ms.x) * ms.y, y1 = (x.y - ms.x) * ms.y;
        yv[n][half][0] = y0; yv[n][half][1] = y1;
        sd[half] += dl[n][2 * half] + dl[n][2 * half + 1];
        sy[half] = fmaf(dl[n][2 * half], y0, sy[half]);
        sy[half] = fmaf(dl[n][2 * half + 1], y1, sy[half]);
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      sd[half] += __shfl_xor_sync(0xffffffffu, sd[half], 1); sd[half] += __shfl_xor_sync(0xffffffffu, sd[half], 2);
      sy[half] += __shfl_xor_sync(0xffffffffu, sy[half], 1); sy[half] += __shfl_xor_sync(0xffffffffu, sy[half], 2);
      if (t == 0) XS[ch * RB + rg * 16 + g + half * 8] = make_float2(sd[half], sy[half]);
    }

    __syncthreads();                                        // XS complete

    // (F) d pair = rstd (d LN0 - mean(d LN0) - y mean(d LN0 y)), in place over the pair rows (each position read and written by one lane)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = rg * 16 + g + half * 8;
      const float2 a = XS[r], b = XS[RB + r], ms = RS[r];
      const float m1 = (a.x + b.x) / 128.f, m2 = (a.y + b.y) / 128.f;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float o0 = ms.y * (dl[n][2 * half] - m1 - yv[n][half][0] * m2), o1 = ms.y * (dl[n][2 * half + 1] - m1 - yv[n][half][1] * m2);
        *reinterpret_cast<uint32_t*>(zt + r * ZP + ch * 64 + n * 8 + 2 * t) = pack2(o0, o1);
      }
    }
    __syncthreads();
    bf* dz = DZ + item * RB * 128;
    for (int i = tid; i < RB * 16; i += 256) {
      const int r = i >> 4, c = i & 15;
      *reinterpret_cast<uint4*>(dz + (long)r * 128 + c * 8) = *reinterpret_cast<const uint4*>(zt + r * ZP + c * 8);
    }
  }
  // the CTA's dW' partial [NP, 128] fp32
  float* out = DWP + (long)blockIdx.x * NP * 128;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    if (mg * 3 + j < NPT) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = (mg * 3 + j) * 16 + g + half * 8, col = nh * 64 + n * 8 + 2 * t;
          *reinterpret_cast<float2*>(out + row * 128 + col) = make_float2(dw[j][n][2 * half], dw[j][n][2 * half + 1]);
        }
    }
  }
}

size_t smem_bytes(int npt) {
  const size_t np = npt * 16;
  return (np * ZP + 2 * RB * ZP + 2 * np * DPITCH + 2 * RB * ZP) * 2 + (RB + 2 * RB) * 8;
}
template <int NPT> void launch(const at::Tensor& z, const at::Tensor& wf, const at::Tensor& db, at::Tensor& dz, at::Tensor& dwp, int64_t R, int N, double eps, int grid) {
  auto k = hoist_bwd_k<NPT>;
  const size_t smem = smem_bytes(NPT);
  static bool attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess;
  TORCH_CHECK(attr && smem <= 232448, "hoist_bwd: shared memory");
  k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf*>(z.data_ptr()), reinterpret_cast<const bf*>(wf.data_ptr()),
                                                           reinterpret_cast<const bf*>(db.data_ptr()), reinterpret_cast<bf*>(dz.data_ptr()),
                                                           reinterpret_cast<float*>(dwp.data_ptr()), R, N, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// z [R, 128] bf16 (the pair rows), wf [N, 128] bf16, db [N, R] bf16 -> dz [R, 128] bf16 and dwp [grid, 16 ceil(N / 16), 128] fp32
// (the caller sums dwp over the grid). R a multiple of 64, N <= 192.
void hoist_bwd(at::Tensor z, at::Tensor wf, at::Tensor db, at::Tensor dz, at::Tensor dwp, double eps) {
  const int64_t R = z.size(0), N = wf.size(0);
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kBFloat16 && z.is_contiguous() && z.size(1) == 128, "hoist_bwd: z bf16 [R, 128]");
  TORCH_CHECK(wf.scalar_type() == at::kBFloat16 && wf.is_contiguous() && wf.size(1) == 128, "hoist_bwd: wf bf16 [N, 128]");
  TORCH_CHECK(db.scalar_type() == at::kBFloat16 && db.is_contiguous() && db.size(0) == N && db.size(1) == R, "hoist_bwd: db bf16 [N, R]");
  TORCH_CHECK(dz.scalar_type() == at::kBFloat16 && dz.is_contiguous() && dz.sizes() == z.sizes(), "hoist_bwd: dz like z");
  TORCH_CHECK(R % RB == 0 && N >= 1 && N <= 192, "hoist_bwd: R a multiple of 64, at most 192 rows of dbias");
  const int npt = (int)((N + 15) / 16);
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = (int)std::min<int64_t>(nsm, R / RB);
  TORCH_CHECK(dwp.scalar_type() == at::kFloat && dwp.is_contiguous() && dwp.numel() == (int64_t)grid * npt * 16 * 128, "hoist_bwd: dwp [", grid, ", ", npt * 16, ", 128] fp32");
  const at::cuda::CUDAGuard g(z.device());
  switch (npt) {
#define C_(K) case K: launch<K>(z, wf, db, dz, dwp, R, (int)N, eps, grid); break;
    C_(1) C_(2) C_(3) C_(4) C_(5) C_(6) C_(7) C_(8) C_(9) C_(10) C_(11) C_(12)
#undef C_
    default: TORCH_CHECK(false, "hoist_bwd: N");
  }
}
int64_t hoist_bwd_grid() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hoist_bwd", &hoist_bwd);
  m.def("hoist_bwd_grid", &hoist_bwd_grid);
}
