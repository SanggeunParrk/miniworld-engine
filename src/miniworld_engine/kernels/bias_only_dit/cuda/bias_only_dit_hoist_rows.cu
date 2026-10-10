// Bias-only token DiT: the row kernel of the HOISTED pair bias's backward (integrations/bias_only_dit_hoist.py).
//
// The hoist makes every block's attention logits from ONE LayerNorm of the pair rows without affine (LN0: the token DiT rows'
// layernorm128_rows) and ONE cuBLAS GEMM against all blocks' gamma-folded weights. Its backward is two cuBLAS GEMMs
// (d LN0 = dbias_all^T W'_all with fp32 output; dW'_all = dbias_all Y) and this kernel between them:
//
//   ln0_bwd_rows   dx = rstd (dy - mean(dy) - y mean(dy y)),   y = (x - mean) rstd          (128 channels, no affine)
//                  and Y, the dW' GEMM's operand made from the same fp32 y:
//                    bf16 path  Y = [hi | lo] [R, 256] bf16, hi = bf16(y), lo = bf16(y - hi)   (hi + lo = y to ~2^-16 relative)
//                    fp32 path  Y = y [R, 128] fp32 (the TF32 GEMM's operand)
//
// Why the split: dW'_b = dbias_b LN0 feeds d gamma_b = sum_h dW'_b o W_b. With LN0 rounded to bf16 for the GEMM, every product
// carries LN0's rounding (2^-9) -- an error the PyTorch block does not have in d gamma (its LayerNorm backward reads LN0 in fp32).
// The two-term operand makes the GEMM's sum of the bf16 dbias against y exact to fp32 accumulation; the GEMM reads K = 256 columns
// instead of 128. LN0 is no longer kept from the forward: this pass recomputes it.
//
// x the pair rows as the forward read them (bf16 or fp32), dy = d LN0 (fp32), dx in its own dtype (bf16 or fp32). The row
// statistics are recomputed from x with the forward kernel's arithmetic (layernorm128_rows_kernel: the same sums in the same
// order), so y is the forward's fp32 value.
//
// SYNC PROTOCOL. None beyond the warp: one warp per RPW consecutive rows (all of their loads issued before any math); a lane owns
// channels 4 lane .. 4 lane + 3 of each row (and the matching columns of both halves of Y); the four row sums (mean, variance,
// mean(dy), mean(dy y)) are xor-shuffle reductions over the full warp, executed by all 32 lanes on warp-uniform control flow (a row
// past R is computed on zeros and not stored). No shared memory, no block barrier, no atomics, no communication between warps or
// blocks; every output element is written exactly once, by one lane. The results are a fixed function of the inputs (a fixed
// reduction tree): bitwise repeatable for any grid, allocator state or concurrent work.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <string>
#include <tuple>
#include <vector>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

constexpr int CW = 128;         // the pair width: a float4 per lane
constexpr int WPB = 8;          // warps per block
constexpr int RPW = 2;          // rows per warp

// Y's row: y itself (fp32, 128 columns) or [bf16(y) | bf16(y - bf16(y))] (bf16, 256 columns)
template <typename YT> struct YOp;
template <> struct YOp<float> {
  static constexpr int COLS = CW;
  static __device__ __forceinline__ void store(float* row, int lane, float4 y) { V4<float>::store(row + lane * 4, y); }
};
template <> struct YOp<bf> {
  static constexpr int COLS = 2 * CW;
  static __device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
  static __device__ __forceinline__ void store(bf* row, int lane, float4 y) {
    const float4 hi = make_float4(rnd(y.x), rnd(y.y), rnd(y.z), rnd(y.w));     // exact in bf16: the store below keeps it
    V4<bf>::store(row + lane * 4, hi);
    V4<bf>::store(row + CW + lane * 4, make_float4(y.x - hi.x, y.y - hi.y, y.z - hi.z, y.w - hi.w));   // y - hi exact in fp32
  }
};

template <typename XT, typename OT, typename YT>
__global__ void __launch_bounds__(WPB * 32) ln0_bwd_rows_kernel(const XT* __restrict__ X, const float* __restrict__ DY,
    OT* __restrict__ DX, YT* __restrict__ Y, long R, float eps) {
  const long row0 = ((long)blockIdx.x * WPB + threadIdx.x / 32) * RPW;
  const int lane = threadIdx.x % 32;
  float4 x[RPW], d[RPW];
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    const long r = row0 + k;
    if (r < R) {
      x[k] = V4<XT>::load(X + r * CW + lane * 4);
      d[k] = V4<float>::load(DY + r * CW + lane * 4);
    } else {
      x[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      d[k] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    // the forward's statistics (layernorm128_rows_kernel), then y
    float4 y = x[k];
    const float mean = warp_sum(y.x + y.y + y.z + y.w) / 128.f;
    y.x -= mean; y.y -= mean; y.z -= mean; y.w -= mean;
    const float rstd = rsqrtf(warp_sum(y.x * y.x + y.y * y.y + y.z * y.z + y.w * y.w) / 128.f + eps);
    y.x *= rstd; y.y *= rstd; y.z *= rstd; y.w *= rstd;
    const float4 g = d[k];
    const float m1 = warp_sum(g.x + g.y + g.z + g.w) / 128.f;                        // mean(dy)
    const float m2 = warp_sum(g.x * y.x + g.y * y.y + g.z * y.z + g.w * y.w) / 128.f;  // mean(dy y)
    const float4 o = make_float4(rstd * (g.x - m1 - y.x * m2), rstd * (g.y - m1 - y.y * m2), rstd * (g.z - m1 - y.z * m2),
                                 rstd * (g.w - m1 - y.w * m2));
    const long r = row0 + k;
    if (r < R) {
      V4<OT>::store(DX + r * CW + lane * 4, o);
      YOp<YT>::store(Y + r * YOp<YT>::COLS, lane, y);
    }
  }
}

#define HR_DISPATCH(T, NAME, ...)                                                                    \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    TORCH_CHECK((T) == at::kBFloat16, "bf16 or fp32");                                                 \
    using NAME = bf; return __VA_ARGS__();                                                           \
  }()

void rows(const at::Tensor& t, const char* name, int64_t R, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(0) == R && t.size(1) == cols && t.is_contiguous(), name, ": contiguous [R, ",
              cols, "]");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, name, ": bf16 or fp32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

// dx = LN0 backward of dy at x (no affine) and Y = the dW' GEMM operand of y = LN0(x): x, dx bf16 or fp32 [R, 128], dy fp32 [R, 128],
// y bf16 [R, 256] (hi | lo) or fp32 [R, 128]; all contiguous
void ln0_bwd_rows(at::Tensor x, at::Tensor dy, at::Tensor dx, at::Tensor y, double eps) {
  const int64_t R = x.size(0);
  rows(x, "x", R, CW); rows(dy, "dy", R, CW); rows(dx, "dx", R, CW);
  rows(y, "y", R, y.scalar_type() == at::kBFloat16 ? 2 * CW : CW);
  TORCH_CHECK(dy.scalar_type() == at::kFloat, "ln0_bwd_rows: dy fp32");
  TORCH_CHECK(x.device() == dy.device() && x.device() == dx.device() && x.device() == y.device(), "ln0_bwd_rows: one device");
  if (R == 0) return;
  const at::cuda::CUDAGuard g(x.device());
  const int64_t per = (int64_t)WPB * RPW;
  HR_DISPATCH(x.scalar_type(), XT, [&] {
    HR_DISPATCH(dx.scalar_type(), OT, [&] {
      HR_DISPATCH(y.scalar_type(), YT, [&] {
        ln0_bwd_rows_kernel<XT, OT, YT><<<(unsigned)((R + per - 1) / per), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const XT*>(x.data_ptr()), reinterpret_cast<const float*>(dy.data_ptr()),
            reinterpret_cast<OT*>(dx.data_ptr()), reinterpret_cast<YT*>(y.data_ptr()), (long)R, (float)eps);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// (name, registers, local-memory bytes) of every kernel of this extension (the tests hold them to no local memory)
std::vector<std::tuple<std::string, int64_t, int64_t>> func_attrs() {
  std::vector<std::tuple<std::string, int64_t, int64_t>> out;
  auto add = [&](const char* n, const void* f) {
    cudaFuncAttributes a{};
    TORCH_CHECK(cudaFuncGetAttributes(&a, f) == cudaSuccess, "cudaFuncGetAttributes ", n);
    out.emplace_back(n, (int64_t)a.numRegs, (int64_t)a.localSizeBytes);
  };
#define FA(XT, OT, YT) add("ln0_bwd_rows_kernel<" #XT ", " #OT ", " #YT ">", (const void*)ln0_bwd_rows_kernel<XT, OT, YT>)
  FA(bf, bf, bf); FA(bf, float, bf); FA(float, bf, bf); FA(float, float, bf);
  FA(bf, bf, float); FA(bf, float, float); FA(float, bf, float); FA(float, float, float);
#undef FA
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln0_bwd_rows_cuda", &ln0_bwd_rows);
  m.def("func_attrs", &func_attrs);
}
