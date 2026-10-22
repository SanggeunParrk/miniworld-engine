// triattn_dh_sm90.cu -- triangle-attention core for head widths 32 / 64 (Chai-1's 4 x 64 trunk heads; the wgmma core in triattn_sm90.cu
// takes 32 only), inference, sm_90a, plain mma.sync (m16n8k16 bf16, fp32 accumulation), FlashAttention-2 schedule:
//
//   u[b,n,s,h,:] = sigmoid(g[b,n,s,h,:] (+ gb[h,:])) * softmax_k( scale q[b,n,s,h,:] . k[b,n,k,h,:] + bias[b,h,s,k] ) @ v[b,n,k,h,:]
//
// q | k | v | g are column blocks of one projection row (b, n, s) of width 4 HD (HD = H D): q at h D, k at HD + h D, v at 2 HD + h D,
// g at 3 HD + h D. bias [B, H, S, S] bf16 in natural units (masked keys -inf), shared by the N pair rows. u [B N S, HD] bf16.
//
// CTA = 64 queries of one (b, n, h); 4 warps x 16 queries; key tiles of 64 through a two-stage cp.async ring (K, V and the bias tile).
// Per key tile a warp computes S = Q K^T (16 x 64), adds the bias, keeps a running maximum (log2 units) and sum, and accumulates
// O += P V with P rounded to bf16 in registers (the S accumulator re-read as the A operand).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
typedef __nv_bfloat16 bf16;
#define DEVI __device__ __forceinline__

constexpr int BM = 64, BN = 64, NW = 4, THREADS = 32 * NW;
constexpr float LOG2E = 1.4426950408889634f;

DEVI uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void cp16(uint32_t d, const void* s) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s) : "memory"); }
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEVI void ldsm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void ldsm4t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pk(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEVI float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// shared tiles: rows of D (+8 pad) bf16 for Q / K / V, 64 (+8 pad) for the bias -- conflict-free ldmatrix
template <int D> struct Smem {
  static constexpr int LD = D + 8, LB = BN + 8;
  bf16 q[BM * LD];
  bf16 k[2][BN * LD];
  bf16 v[2][BN * LD];
  bf16 b[2][BM * LB];
};

template <int D>
__global__ void __launch_bounds__(THREADS) triattn_dh_fwd(const bf16* __restrict__ P, const bf16* __restrict__ BIAS, bf16* __restrict__ U,
    const float* __restrict__ GB, int S, int H, float scale) {
  constexpr int LD = Smem<D>::LD, LB = Smem<D>::LB, KD = D / 16, ND = D / 8;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<D>& sm = *reinterpret_cast<Smem<D>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int q0 = blockIdx.x * BM, h = blockIdx.y;
  const long bn = blockIdx.z;                      // b * N + n (N == S)
  const int b = (int)(bn / S);
  const int HD = H * D, ld = 4 * HD;
  const bf16* rows = P + bn * S * ld;              // the S projection rows of pair row (b, n)
  const bf16* bias = BIAS + ((long)b * H + h) * S * S;

  // Q tile and the first K | V | bias stage
  constexpr int QC = BM * D / 8;                   // 16-byte chunks of a Q / K / V tile
  for (int c = tid; c < QC; c += THREADS) {
    const int r = c / (D / 8), col = (c % (D / 8)) * 8;
    cp16(su(&sm.q[r * LD + col]), rows + (long)(q0 + r) * ld + h * D + col);
  }
  auto stage = [&](int j, int s) {
    const int k0 = j * BN;
    for (int c = tid; c < QC; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      cp16(su(&sm.k[s][r * LD + col]), rows + (long)(k0 + r) * ld + HD + h * D + col);
      cp16(su(&sm.v[s][r * LD + col]), rows + (long)(k0 + r) * ld + 2 * HD + h * D + col);
    }
    for (int c = tid; c < BM * BN / 8; c += THREADS) {
      const int r = c / (BN / 8), col = (c % (BN / 8)) * 8;
      cp16(su(&sm.b[s][r * LB + col]), bias + (long)(q0 + r) * S + k0 + col);
    }
  };
  stage(0, 0);
  cp_commit();

  uint32_t qa[KD][4];
  float o[ND][4];
#pragma unroll
  for (int n = 0; n < ND; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sc = scale * LOG2E;
  const int ntiles = S / BN;
  for (int j = 0; j < ntiles; ++j) {
    const int s = j & 1;
    if (j + 1 < ntiles) stage(j + 1, s ^ 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) ldsm4(qa[kk], su(&sm.q[(warp * 16 + lane % 16) * LD + kk * 16 + (lane / 16) * 8]));
    }
    // S = Q K^T: 8 n-tiles of 8 keys
    float sacc[BN / 8][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
      for (int n = 0; n < BN / 8; n += 2) {
        uint32_t kb[4];
        ldsm4(kb, su(&sm.k[s][(n * 8 + (lane % 8) + (lane / 16) * 8) * LD + kk * 16 + ((lane / 8) % 2) * 8]));
        mma(sacc[n], qa[kk], kb[0], kb[1]);
        mma(sacc[n + 1], qa[kk], kb[2], kb[3]);
      }
    }
    // log2-unit logits with the bias; running maximum and sum per row (g, g + 8)
    const int r0 = warp * 16 + g;
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      const int col = n * 8 + 2 * t;
      const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&sm.b[s][r0 * LB + col]);
      const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&sm.b[s][(r0 + 8) * LB + col]);
      sacc[n][0] = sacc[n][0] * sc + __low2float(b0) * LOG2E;
      sacc[n][1] = sacc[n][1] * sc + __high2float(b0) * LOG2E;
      sacc[n][2] = sacc[n][2] * sc + __low2float(b1) * LOG2E;
      sacc[n][3] = sacc[n][3] * sc + __high2float(b1) * LOG2E;
      mx0 = fmaxf(mx0, fmaxf(sacc[n][0], sacc[n][1]));
      mx1 = fmaxf(mx1, fmaxf(sacc[n][2], sacc[n][3]));
    }
    mx0 = quad_max(mx0);
    mx1 = quad_max(mx1);
    // a row whose keys are all masked so far keeps a finite offset (its p are 0)
    const float off0 = mx0 == -INFINITY ? 0.f : mx0, off1 = mx1 == -INFINITY ? 0.f : mx1;
    const float a0 = ex2((m0 == -INFINITY ? off0 : m0) - off0), a1 = ex2((m1 == -INFINITY ? off1 : m1) - off1);
    m0 = mx0;
    m1 = mx1;
    l0 *= a0;
    l1 *= a1;
#pragma unroll
    for (int n = 0; n < ND; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
    uint32_t pa[BN / 16][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      const float p0 = ex2(sacc[n][0] - off0), p1 = ex2(sacc[n][1] - off0), p2 = ex2(sacc[n][2] - off1), p3 = ex2(sacc[n][3] - off1);
      l0 += p0 + p1;
      l1 += p2 + p3;
      pa[n / 2][(n % 2) * 2] = pk(p0, p1);
      pa[n / 2][(n % 2) * 2 + 1] = pk(p2, p3);
    }
    // O += P V: k = 64 keys (4 steps of 16), n = D dims
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
      for (int n = 0; n < ND; n += 2) {
        uint32_t vb[4];
        ldsm4t(vb, su(&sm.v[s][(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * LD + n * 8 + (lane / 16) * 8]));
        mma(o[n], pa[kk], vb[0], vb[1]);
        mma(o[n + 1], pa[kk], vb[2], vb[3]);
      }
    }
    __syncthreads();
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  // u = sigmoid(g (+ gb)) o, bf16 into U [B N S, HD]
  const long row0 = bn * S + q0 + warp * 16 + g;
#pragma unroll
  for (int n = 0; n < ND; ++n) {
    const int col = h * D + n * 8 + 2 * t;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const long row = row0 + half * 8;
      const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(P + row * ld + 3 * HD + col);
      float ga = __low2float(gv), gb = __high2float(gv);
      if (GB != nullptr) { ga += GB[col]; gb += GB[col + 1]; }
      const float inv = half ? i1 : i0;
      const float v0 = o[n][2 * half] * inv / (1.f + __expf(-ga)), v1 = o[n][2 * half + 1] * inv / (1.f + __expf(-gb));
      *reinterpret_cast<uint32_t*>(U + row * HD + col) = pk(v0, v1);
    }
  }
}

template <int D>
void launch(const at::Tensor& proj, const at::Tensor& bias, at::Tensor& u, const float* gb, int B, int S, int H, double scale) {
  auto f = triattn_dh_fwd<D>;
  const int smem = (int)sizeof(Smem<D>);
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(S / BM, H, (unsigned)B * S);
  f<<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(proj.data_ptr()),
      reinterpret_cast<const bf16*>(bias.data_ptr()), reinterpret_cast<bf16*>(u.data_ptr()), gb, S, H, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// proj [B S S, 4 H D] bf16 (q | k | v | g), bias [B, H, S, S] bf16 -> u [B S S, H D] bf16. gate_bias: fp32 [H D] or empty.
at::Tensor triattn_dh_forward(at::Tensor proj, at::Tensor bias, at::Tensor gate_bias, int64_t H, int64_t D, double scale) {
  TORCH_CHECK(D == 32 || D == 64, "head width 32 or 64");
  TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::kBFloat16 && proj.is_contiguous() && proj.dim() == 2 && proj.size(1) == 4 * H * D,
              "proj: contiguous bf16 [B S S, 4 H D]");
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.dim() == 4 && bias.size(1) == H && bias.size(2) == bias.size(3),
              "bias: contiguous bf16 [B, H, S, S]");
  const int B = (int)bias.size(0), S = (int)bias.size(2);
  TORCH_CHECK(S % 64 == 0 && proj.size(0) == (int64_t)B * S * S, "S a multiple of 64, proj rows B S S");
  TORCH_CHECK(gate_bias.numel() == 0 || (gate_bias.scalar_type() == at::kFloat && gate_bias.is_contiguous() && gate_bias.numel() == H * D),
              "gate_bias: fp32 [H D] or empty");
  const at::cuda::OptionalCUDAGuard guard(proj.device());
  auto u = at::empty({proj.size(0), H * D}, proj.options());
  const float* gb = gate_bias.numel() ? gate_bias.data_ptr<float>() : nullptr;
  if (D == 64) launch<64>(proj, bias, u, gb, B, S, (int)H, scale);
  else launch<32>(proj, bias, u, gb, B, S, (int)H, scale);
  return u;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("triattn_dh_forward", &triattn_dh_forward); }
