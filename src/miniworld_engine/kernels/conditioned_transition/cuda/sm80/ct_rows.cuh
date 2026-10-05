// ct_rows.cuh -- the row passes of the ConditionedTransition tail on A100 (sm_80): the SwiGLU and the output gate with its residual, forward and backward.  The GEMMs between
// them are cuBLAS; the AdaLN in front of the tail is ``kernels/adaln/cuda/sm80``.
//
//   h = silu(a) b,  [a | b] = xa [Wa; Wb]^T                  swiglu_fwd / swiglu_bwd (the backward recomputes h for the weight gradient of the squeeze)
//   y = x + sigmoid(g) z,  g = cond Wsc^T + bsc, z = h Ws^T   gate_res_fwd (x optional: the update alone) / gate_res_bwd
//
// Elementwise passes are flat (a thread owns 4 consecutive columns); the column sums of the gate backward leave a block as one row of a [blocks, d] fp32 partial buffer.
#pragma once
#include "adaln_common.cuh"

namespace adl {

// H [M, N] = silu(a) b; AB [M, 2N] = [a | b] (row stride sab)
template <typename T>
__global__ void __launch_bounds__(256) swiglu_fwd_kernel(const T* __restrict__ AB, long sab, T* __restrict__ H, long M, int N4) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < M * N4; i += (long)gridDim.x * 256) {
    const long row = i / N4;
    const int c = (int)(i - row * N4) * 4;
    const float4 a = V4<T>::load(AB + row * sab + c), b = V4<T>::load(AB + row * sab + 4 * N4 + c);
    V4<T>::store(H + row * 4 * N4 + c, mul4(mul4(a, sig4(a)), b));
  }
}

// h = silu(a) b: da = dh b s (1 + a (1 - s)), db = dh a s with s = sigmoid(a); DAB [M, 2N] = [da | db]; H [M, N] = h is written for the squeeze's weight gradient
template <typename T>
__global__ void __launch_bounds__(256) swiglu_bwd_kernel(const T* __restrict__ DH, const T* __restrict__ AB, long sab, T* __restrict__ DAB, T* __restrict__ H, long M,
                                                         int N4) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < M * N4; i += (long)gridDim.x * 256) {
    const long row = i / N4;
    const int c = (int)(i - row * N4) * 4, N = 4 * N4;
    const float4 dh = V4<T>::load(DH + row * N + c), a = V4<T>::load(AB + row * sab + c), b = V4<T>::load(AB + row * sab + N + c);
    const float4 s = sig4(a), as = mul4(a, s);
    const float4 da = make_float4(dh.x * b.x * s.x * (1.f + a.x * (1.f - s.x)), dh.y * b.y * s.y * (1.f + a.y * (1.f - s.y)),
                                  dh.z * b.z * s.z * (1.f + a.z * (1.f - s.z)), dh.w * b.w * s.w * (1.f + a.w * (1.f - s.w)));
    V4<T>::store(DAB + row * 2 * N + c, da);
    V4<T>::store(DAB + row * 2 * N + N + c, mul4(dh, as));
    V4<T>::store(H + row * N + c, mul4(as, b));
  }
}

// Y [M, d] = (X) + sigmoid(G[r % P]) Z; X null: the update alone
template <typename T>
__global__ void __launch_bounds__(256) gate_res_fwd_kernel(const T* __restrict__ X, const T* __restrict__ Z, const T* __restrict__ G, T* __restrict__ Y, long M, int D4, long P) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < M * D4; i += (long)gridDim.x * 256) {
    const long row = i / D4;
    const int c = (int)(i - row * D4) * 4, D = 4 * D4;
    const float4 gz = mul4(sig4(V4<T>::load(G + (row % P) * D + c)), V4<T>::load(Z + row * D + c));
    V4<T>::store(Y + row * D + c, X != nullptr ? add4(V4<T>::load(X + row * D + c), gz) : gz);
  }
}

// dz = sigmoid(g) dy, dg = dy z s (1 - s) (both stored in T: the operands of the GEMMs), the column sums of dg (fp32, unrounded) -> PB [blocks, d]
template <int NT, int R, typename T>
__global__ void __launch_bounds__(NT * R) gate_res_bwd_kernel(const T* __restrict__ DY, const T* __restrict__ Z, const T* __restrict__ G, T* __restrict__ DZ, T* __restrict__ DG,
                                                              float* __restrict__ PB, long M) {
  constexpr int D = NT * 4;
  __shared__ float4 csum[R][NT];
  const int ty = threadIdx.y, tx = threadIdx.x, col = tx * 4;
  float4 acc = zero4();
  for (long chunk = blockIdx.x; chunk * R < M; chunk += gridDim.x) {
    const long row = chunk * R + ty;
    if (row < M) {
      const float4 dy = V4<T>::load(DY + row * D + col), z = V4<T>::load(Z + row * D + col), s = sig4(V4<T>::load(G + row * D + col));
      const float4 dg = make_float4(dy.x * z.x * s.x * (1.f - s.x), dy.y * z.y * s.y * (1.f - s.y), dy.z * z.z * s.z * (1.f - s.z), dy.w * z.w * s.w * (1.f - s.w));
      V4<T>::store(DZ + row * D + col, mul4(dy, s));
      V4<T>::store(DG + row * D + col, dg);
      acc = add4(acc, dg);
    }
  }
  csum[ty][tx] = acc;
  __syncthreads();
  if (ty == 0) {
    float4 t = csum[0][tx];
#pragma unroll
    for (int r = 1; r < R; ++r) t = add4(t, csum[r][tx]);
    V4<float>::store(PB + (long)blockIdx.x * D + col, t);
  }
}

}  // namespace adl
