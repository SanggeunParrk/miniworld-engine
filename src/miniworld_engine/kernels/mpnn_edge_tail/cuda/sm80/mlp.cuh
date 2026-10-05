// mlp.cuh -- the ProteinMPNN edge-message MLP on the A100 (sm_80): update = rn(rn(gelu(rn(gelu(x) Wh^T + bh))) Wo^T + bo), both products in one kernel (the tail's layers 2 and 3 without
// the gathers, dropout, residual and LayerNorm), and its backward: G2 = rn(rn(go Wo) gelu'(hid)), grad_x = rn(rn(G2 Wh) gelu'(x)) (the Triton path's and the bf16 autograd chain's rounding of the product before the derivative), the weight gradients from dw.cuh (jobs: (go, hid) -> dWo and
// (G2, x) -> dWh).  ``hid`` is the "compute" policy's saved projection; with RECOMP the backward kernel recomputes it from x (the "memory" policy: nothing but the input is kept).
//
// The structure is tail_fwd.cuh / tail_bwd.cuh's: resident weight images (forward: Wh | Wo; backward: Wo^T | Wh^T | Wh), a warp = 16 rows, 16-byte vectors straight into the fragments,
// two halves of 64 output channels per product.
#pragma once
#include "tail_fwd.cuh"

namespace me80 {

struct MlpFwdParams {
  const __nv_bfloat16* x;        // [rows][128]
  const uint8_t* img;            // layer 0 = Wh, 1 = Wo (pack.cuh forward image)
  const float* tab;              // bh | bo
  __nv_bfloat16* out;            // [rows][128]
  __nv_bfloat16* hid;            // [rows][128] or nullptr
  int rows;
};

constexpr int MLP_FWD_SMEM = 2 * LAYER_BYTES + TAB_FLOATS * 4;

template <int NW, bool SAVE>
__global__ void __launch_bounds__(NW * 32, 1) mlp_fwd_kernel(const MlpFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* tab = reinterpret_cast<const float*>(smem_raw + 2 * LAYER_BYTES);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 2 * LAYER_BYTES / 16; i += NW * 32) cp_async16(sb + i * 16, p.img + (size_t)i * 16);
  cp_async_commit();
  {
    float* t = reinterpret_cast<float*>(smem_raw + 2 * LAYER_BYTES);
    for (int i = tid; i < 2 * 128; i += NW * 32) t[i] = p.tab[i];
  }
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wq[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wq[kb] = woff(g8, 4 * kb + q4);

  const int ntile = (p.rows + 15) >> 4;
  for (int tile = blockIdx.x * NW + warp; tile < ntile; tile += gridDim.x * NW) {
    int row[2], rr[2];
    bool ok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      row[hh] = tile * 16 + g8 + 8 * hh;
      ok[hh] = row[hh] < p.rows;
      rr[hh] = ok[hh] ? row[hh] : p.rows - 1;
    }
    uint4 xv[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) xv[hh][kb] = ldg128_stream(p.x + (size_t)rr[hh] * D + 8 * q4 + 32 * kb);

    // act1 = rn(gelu(x)): the A fragments of product 1
    uint32_t a[8][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t xp = comp(xv[hh][kb], s);
          A_SLOT(a, kb, s, hh) = pack_bf16(gelu(bf16lo(xp)), gelu(bf16hi(xp)));
        }

    // product 1: hid = rn(acc + bh); act2 = rn(gelu(hid))
    uint32_t a2[8][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float acc[8][4];
      gemm_half(acc, a, sb + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          const float4 b0 = *reinterpret_cast<const float4*>(tab + 32 * aa + 8 * q4), b1 = *reinterpret_cast<const float4*>(tab + 32 * aa + 8 * q4 + 4);
          const float bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
          uint32_t hp[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) hp[s] = pack_bf16(acc[4 * j + s][2 * hh] + bias[2 * s], acc[4 * j + s][2 * hh + 1] + bias[2 * s + 1]);
          if (SAVE && ok[hh]) stg128_stream(p.hid + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(hp[0], hp[1], hp[2], hp[3]));
#pragma unroll
          for (int s = 0; s < 4; ++s) A_SLOT(a2, aa, s, hh) = pack_bf16(gelu(bf16lo(hp[s])), gelu(bf16hi(hp[s])));
        }
    }

    // product 2: out = rn(acc + bo)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float acc[8][4];
      gemm_half(acc, a2, sb + LAYER_BYTES + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          const float4 b0 = *reinterpret_cast<const float4*>(tab + 128 + 32 * aa + 8 * q4), b1 = *reinterpret_cast<const float4*>(tab + 128 + 32 * aa + 8 * q4 + 4);
          const float bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
          uint32_t op[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) op[s] = pack_bf16(acc[4 * j + s][2 * hh] + bias[2 * s], acc[4 * j + s][2 * hh + 1] + bias[2 * s + 1]);
          if (ok[hh]) stg128(p.out + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(op[0], op[1], op[2], op[3]));
        }
    }
  }
}

struct MlpBwdParams {
  const __nv_bfloat16* go;       // [rows][128] gradient of the update
  const __nv_bfloat16* x;        // [rows][128]
  const __nv_bfloat16* hid;      // [rows][128] saved projection (!RECOMP)
  const uint8_t* img;            // layer 0 = Wo^T, 1 = Wh^T, 2 = Wh (RECOMP only)
  const float* tab;              // bh | bo
  __nv_bfloat16* g2;             // [rows][128] out
  __nv_bfloat16* gx;             // [rows][128] out
  __nv_bfloat16* hid_out;        // [rows][128] out (RECOMP: the recomputed projection for the weight-gradient kernel)
  int rows;
};

template <int NW, bool RECOMP>
__global__ void __launch_bounds__(NW * 32, 1) mlp_bwd_kernel(const MlpBwdParams p) {
  constexpr int NL = RECOMP ? 3 : 2;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* tab = reinterpret_cast<const float*>(smem_raw + NL * LAYER_BYTES);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < NL * LAYER_BYTES / 16; i += NW * 32) cp_async16(sb + i * 16, p.img + (size_t)i * 16);
  cp_async_commit();
  {
    float* t = reinterpret_cast<float*>(smem_raw + NL * LAYER_BYTES);
    for (int i = tid; i < 2 * 128; i += NW * 32) t[i] = p.tab[i];
  }
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wq[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wq[kb] = woff(g8, 4 * kb + q4);

  const int ntile = (p.rows + 15) >> 4;
  for (int tile = blockIdx.x * NW + warp; tile < ntile; tile += gridDim.x * NW) {
    int row[2], rr[2];
    bool ok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      row[hh] = tile * 16 + g8 + 8 * hh;
      ok[hh] = row[hh] < p.rows;
      rr[hh] = ok[hh] ? row[hh] : p.rows - 1;
    }
    // hid: saved, or recomputed from x
    uint32_t hq[2][4][4];
    if (RECOMP) {
      uint4 xv[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) xv[hh][kb] = ldg128_stream(p.x + (size_t)rr[hh] * D + 8 * q4 + 32 * kb);
      uint32_t a[8][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t xp = comp(xv[hh][kb], s);
            A_SLOT(a, kb, s, hh) = pack_bf16(gelu(bf16lo(xp)), gelu(bf16hi(xp)));
          }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float acc[8][4];
        gemm_half(acc, a, sb + 2 * LAYER_BYTES + h * 16384, wq);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int aa = 2 * h + j;
            const float4 b0 = *reinterpret_cast<const float4*>(tab + 32 * aa + 8 * q4), b1 = *reinterpret_cast<const float4*>(tab + 32 * aa + 8 * q4 + 4);
            const float bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int s = 0; s < 4; ++s) hq[hh][aa][s] = pack_bf16(acc[4 * j + s][2 * hh] + bias[2 * s], acc[4 * j + s][2 * hh + 1] + bias[2 * s + 1]);
            if (ok[hh]) stg128(p.hid_out + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(hq[hh][aa][0], hq[hh][aa][1], hq[hh][aa][2], hq[hh][aa][3]));
          }
      }
    } else {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int aa = 0; aa < 4; ++aa) {
          const uint4 v = ldg128_stream(p.hid + (size_t)rr[hh] * D + 8 * q4 + 32 * aa);
          hq[hh][aa][0] = v.x; hq[hh][aa][1] = v.y; hq[hh][aa][2] = v.z; hq[hh][aa][3] = v.w;
        }
    }

    // the update's gradient: the A fragments of dX3
    uint32_t a[8][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const uint4 v = ldg128_stream(p.go + (size_t)rr[hh] * D + 8 * q4 + 32 * kb);
        a[2 * kb][hh] = v.x; a[2 * kb][2 + hh] = v.y; a[2 * kb + 1][hh] = v.z; a[2 * kb + 1][2 + hh] = v.w;
      }

    // dX3: G2 = rn((go Wo) gelu'(hid))
    uint32_t a2[8][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float acc[8][4];
      gemm_half(acc, a, sb + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          uint32_t g2p[4];
#pragma unroll
          for (int s = 0; s < 4; ++s)
            g2p[s] = pack_bf16(round_bf16f(acc[4 * j + s][2 * hh]) * gelu_grad(bf16lo(hq[hh][aa][s])), round_bf16f(acc[4 * j + s][2 * hh + 1]) * gelu_grad(bf16hi(hq[hh][aa][s])));
          if (ok[hh]) stg128(p.g2 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(g2p[0], g2p[1], g2p[2], g2p[3]));
#pragma unroll
          for (int s = 0; s < 4; ++s) A_SLOT(a2, aa, s, hh) = g2p[s];
        }
    }

    // dX2: grad_x = rn((G2 Wh) gelu'(x))
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 xv[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) xv[hh][j] = ldg128_stream(p.x + (size_t)rr[hh] * D + 8 * q4 + 32 * (2 * h + j));
      float acc[8][4];
      gemm_half(acc, a2, sb + LAYER_BYTES + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          uint32_t gxp[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t xp = comp(xv[hh][j], s);
            gxp[s] = pack_bf16(round_bf16f(acc[4 * j + s][2 * hh]) * gelu_grad(bf16lo(xp)), round_bf16f(acc[4 * j + s][2 * hh + 1]) * gelu_grad(bf16hi(xp)));
          }
          if (ok[hh]) stg128(p.gx + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(gxp[0], gxp[1], gxp[2], gxp[3]));
        }
    }
  }
}

}  // namespace me80
