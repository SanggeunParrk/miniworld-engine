// Pair -> bias projection on the tensor cores for d_pair 256 (and 128), 8 heads: the forward of ``apb/apb_rows.cu`` ``pair_bias_fwd_k``
// with the pair width a template parameter (that file hard-wires 128). bias[h, i, j] = scale-folded Wf[h] . LN(pair[i, j]) in bf16,
// head-major, -big on masked keys; also writes the row statistics (mean, rstd) of every pair row for the backward.
//
// Persistent: a block walks tiles of 128 consecutive keys of one query row (128 x DP x 2 bytes, one bulk copy each, NSF stages deep);
// a warp takes 16 rows of the tile. Wf . LN(x) = rstd (Wf . x - mean sum_c Wf): the mma runs on the raw bf16 words and the row
// statistics come from the tensor cores too (a Gram step on the same registers: no per-element float work). See apb_rows.cu for the
// fragment mapping.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace {
using bf = __nv_bfloat16;

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ void gram_step(float (&g0)[4], float (&g1)[4], float (&cs)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
  constexpr uint32_t ONES = 0x3F803F80u;
  mma16816(g0, a0, a1, a2, a3, a0, a2);
  mma16816(g1, a0, a1, a2, a3, a1, a3);
  mma16816(cs, a0, a1, a2, a3, ONES, ONES);
}
template <int DP>
__device__ __forceinline__ void gram_stats(const float (&g0)[4], const float (&g1)[4], const float (&cs)[4], float eps, float& mu0, float& rs0,
                                           float& mu1, float& rs1) {
  const int lane = threadIdx.x & 31, g = lane >> 2, src = 4 * g + (g >> 1);
  const float s0 = __shfl_sync(0xffffffffu, (g & 1) ? g0[1] : g0[0], src), s1 = __shfl_sync(0xffffffffu, (g & 1) ? g1[3] : g1[2], src);
  mu0 = cs[0] * (1.f / DP); mu1 = cs[2] * (1.f / DP);
  rs0 = rsqrtf(fmaxf(s0 * (1.f / DP) - mu0 * mu0, 0.f) + eps);
  rs1 = rsqrtf(fmaxf(s1 * (1.f / DP) - mu1 * mu1, 0.f) + eps);
}
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(b)), "r"(n)); }
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra.uni W%=;\n }"
               ::"r"(su32(b)), "r"(parity) : "memory");
}

constexpr int FW = 8, FNT = 32 * FW, NH = 8, NHP = 8;
template <int DP> constexpr int stages() { return DP > 128 ? 3 : 2; }
template <int DP> constexpr int tile_bytes() { return 128 * DP * 2; }
template <int DP> constexpr int fwd_smem() { return stages<DP>() * tile_bytes<DP>() + 2 * NHP * (128 + 4) * 4 + stages<DP>() * 8; }

template <int DP>
__global__ void __launch_bounds__(FNT) pair_mma_fwd_k(const bf* __restrict__ Z, const bf* __restrict__ W, const bool* __restrict__ MASK,
                                                      bf* __restrict__ OUT, float2* __restrict__ ST, int L, float eps, float neg) {
  constexpr int NSF = stages<DP>(), FT = tile_bytes<DP>(), KS = DP / 16, UU = DP / 32;
  extern __shared__ __align__(128) uint8_t smem[];
  float (*out_s)[NHP][128 + 4] = reinterpret_cast<float (*)[NHP][128 + 4]>(smem + NSF * FT);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NSF * FT + 2 * NHP * (128 + 4) * 4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + 127) / 128;
  const long ntiles = (long)L * ntj;
  auto issue = [&](long n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0);
    uint64_t* fb = &full[n % NSF];
    mbar_expect_tx(fb, (uint32_t)rows * DP * 2);
    bulk_g2s(smem + (n % NSF) * FT, Z + ((long)i * L + j0) * DP, (uint32_t)rows * DP * 2, fb);
  };
  if (threadIdx.x == 0) {
    for (int k = 0; k < NSF; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int k = 0; k < NSF; ++k) issue(k);
  // Wf's B fragments (lane (g, q): head g) and sum_c Wf of heads 2q, 2q + 1 for this lane's C columns
  uint32_t b[KS][2];
  float sw0, sw1;
  {
    float sw = 0.f;
#pragma unroll
    for (int s = 0; s < KS; ++s) {
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + g * DP + 32 * (s >> 1) + 8 * q + 4 * (s & 1));
      b[s][0] = wp[0]; b[s][1] = wp[1];
      const float2 u = unpack2(b[s][0]), v = unpack2(b[s][1]);
      sw += u.x + u.y + v.x + v.y;
    }
    sw = quad_sum(sw);
    sw0 = __shfl_sync(0xffffffffu, sw, 8 * q); sw1 = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
  }
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0), st = (int)(n % NSF);
    mbar_wait(&full[st], (uint32_t)((n / NSF) & 1));
    const uint8_t* tile = smem + st * FT;
    float (*os)[128 + 4] = out_s[n & 1];
    const int rb = warp * 16;
    if (rb < rows) {
      float c[4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
#pragma unroll
      for (int u = 0; u < UU; ++u) {
        const uint4 r0 = *reinterpret_cast<const uint4*>(tile + (rb + g) * DP * 2 + 64 * u + 16 * q);
        const uint4 r1 = *reinterpret_cast<const uint4*>(tile + (rb + g + 8) * DP * 2 + 64 * u + 16 * q);
        const uint32_t u0[4] = {r0.x, r0.y, r0.z, r0.w}, u1[4] = {r1.x, r1.y, r1.z, r1.w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mma16816(c, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], b[2 * u + h][0], b[2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float m0, rs0, m1, rs1;
      gram_stats<DP>(g0, g1, cs, eps, m0, rs0, m1, rs1);
      const int jl = rb + g, h0 = 2 * q;
      os[h0][jl] = rs0 * (c[0] - m0 * sw0); os[h0 + 1][jl] = rs0 * (c[1] - m0 * sw1);
      os[h0][jl + 8] = rs1 * (c[2] - m1 * sw0); os[h0 + 1][jl + 8] = rs1 * (c[3] - m1 * sw1);
      if (q == 0) {
        ST[(long)i * L + j0 + jl] = make_float2(m0, rs0);
        ST[(long)i * L + j0 + jl + 8] = make_float2(m1, rs1);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) issue(n + NSF);
    for (int task = threadIdx.x; task < NH * 16; task += FNT) {
      const int h = task >> 4, ch = task & 15, j = j0 + ch * 8;
      if (ch * 8 >= rows) continue;
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float a = os[h][ch * 8 + 2 * e], bb = os[h][ch * 8 + 2 * e + 1];
        if (MASK) { if (!MASK[j + 2 * e]) a = neg; if (!MASK[j + 2 * e + 1]) bb = neg; }
        o[e] = pack2(a, bb);
      }
      *reinterpret_cast<uint4*>(OUT + ((long)h * L + i) * L + j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}

int nsm_of(const at::Tensor& t) {
  int n = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, t.get_device()));
  return n;
}
template <int DP>
void launch(const at::Tensor& z, const at::Tensor& w, const c10::optional<at::Tensor>& mask, at::Tensor& out, at::Tensor& st, int64_t L, double eps,
            double neg) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_mma_fwd_k<DP>, cudaFuncAttributeMaxDynamicSharedMemorySize, fwd_smem<DP>()));
    attr = true;
  }
  const long ntiles = L * ((L + 127) / 128);
  const unsigned blocks = (unsigned)std::min<long>((DP > 128 ? 1L : 2L) * nsm_of(z), ntiles);
  pair_mma_fwd_k<DP><<<blocks, FNT, fwd_smem<DP>(), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf*>(z.data_ptr()), reinterpret_cast<const bf*>(w.data_ptr()), mask ? mask->data_ptr<bool>() : nullptr,
      reinterpret_cast<bf*>(out.data_ptr()), reinterpret_cast<float2*>(st.data_ptr()), (int)L, (float)eps, (float)neg);
}
}  // namespace

// z [L L, DP] bf16, wf [8, DP] bf16 (LN weight, head weights and any scale folded), mask [L] bool or none -> (bias [8, L, L] bf16, stats [L L, 2] fp32)
std::vector<at::Tensor> pair_mma_fwd(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double neg) {
  const int64_t dp = wf.size(1);
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kBFloat16 && z.is_contiguous() && wf.scalar_type() == at::kBFloat16 && wf.is_contiguous(), "pair / wf: contiguous bf16");
  TORCH_CHECK(wf.size(0) == 8 && z.numel() == L * L * dp && L % 16 == 0, "pair_mma_fwd shapes: 8 heads, pair [L L, DP], L % 16");
  if (mask) TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "mask [L] bool");
  const at::cuda::CUDAGuard gd(z.device());
  auto out = at::empty({8, L, L}, z.options());
  auto st = at::empty({L * L, 2}, z.options().dtype(at::kFloat));
  if (dp == 256) launch<256>(z, wf, mask, out, st, L, eps, neg);
  else if (dp == 128) launch<128>(z, wf, mask, out, st, L, eps, neg);
  else TORCH_CHECK(false, "pair_mma_fwd: d_pair 128 or 256");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, st};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("pair_mma_fwd", &pair_mma_fwd); }
