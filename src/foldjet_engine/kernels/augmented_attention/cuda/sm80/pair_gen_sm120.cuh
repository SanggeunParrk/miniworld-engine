// pair_gen_sm120.cuh -- the generic pair bias (``pair_bias_gen_kernel``, aux_sm80.cuh) with the pair staged through shared memory, for the RTX 5090 build
// (MWE_SM120) only. Same arithmetic, same order of every sum (channels in order; per row the H dot products, the sum and the sum of squares as there), so the
// result is bitwise the one of pair_bias_gen_kernel. What changes is how the pair reaches the FMAs. There a thread loads its own 8-channel chunk (32 B) straight
// from global memory: a warp's load touches 32 rows C channels apart, and each 128-B line of the pair is fetched in four pieces at four different times -- on the
// 5090 that ran at ~570 GB/s once a row is 1 KB or more (fp32 d_pair 256, bf16 512). Here a block (8 rows i x 32 columns j, 128 threads, two rows a thread) stages
// 128 B of every element (32 fp32 / 64 bf16 channels: whole lines, eight threads to a line) through a two-stage cp.async ring.
#pragma once

namespace aa80 {

constexpr int PG_NT = 128, PG_TJ = 32, PG_ROWS = 8;                  // threads, columns and rows of a block
constexpr int PG_LINE = 128, PG_PITCH = PG_LINE;                     // bytes of an element per stage, unpadded: its 16-B chunk c is stored at c ^ (e & 7)
                                                                     // (element e), so the 16-B reads of eight consecutive elements hit distinct banks
constexpr int PG_ELEMS = PG_ROWS * PG_TJ;
constexpr int PG_STAGE = PG_ELEMS * PG_PITCH;
constexpr int PG_NS = 2;

template <int H> constexpr int pg_smem(int C) { return ((C * H + H) * 4 + 15) / 16 * 16 + PG_NS * PG_STAGE; }

__device__ __forceinline__ void pg_read4(const unsigned char* p, float (&x)[4], const float*) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
}
__device__ __forceinline__ void pg_read4(const unsigned char* p, float (&x)[4], const __nv_bfloat16*) {
  const uint2 a = *reinterpret_cast<const uint2*>(p);
  x[0] = __uint_as_float(a.x << 16); x[1] = __uint_as_float(a.x & 0xffff0000u); x[2] = __uint_as_float(a.y << 16); x[3] = __uint_as_float(a.y & 0xffff0000u);
}

// grid (N / 32, N / 8); C a multiple of the channels of a line (32 fp32 / 64 bf16)
template <int H, bool PAD, typename TI, typename TO>
__global__ void __launch_bounds__(PG_NT) pair_bias_gen_staged_kernel(const TI* __restrict__ z, const float* __restrict__ wcm, TO* __restrict__ bo, int N, int NZ, int C,
                                                                     const unsigned char* __restrict__ kv, float eps, float oscale, float fill) {
  PbGenCfg<H>();
  constexpr int R = 2, CK = PG_LINE / (int)sizeof(TI), GPE = PG_LINE / 16;   // channels of a stage; 16-B granules of an element's line
  constexpr int ES = 4 * (int)sizeof(TI);                                  // staged bytes of 4 channels
  extern __shared__ __align__(16) unsigned char pgsm[];
  float* gsm = reinterpret_cast<float*>(pgsm);                             // W' channel-major [C][H], then sum_c W' [H]
  float* ws = gsm + (size_t)C * H;
  unsigned char* ring = pgsm + ((size_t)(C * H + H) * 4 + 15) / 16 * 16;
  const uint32_t ring_s = (uint32_t)__cvta_generic_to_shared(ring);
  const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
  const int i0 = blockIdx.y * PG_ROWS, j0 = blockIdx.x * PG_TJ, j = j0 + tx;
  const int NCH = C / CK;

  auto issue = [&](int ch) {                                               // stage ch % 2 <- channels [ch CK, (ch + 1) CK) of the block's elements
    if (ch < NCH) {
      const uint32_t st = ring_s + (uint32_t)(ch % PG_NS) * PG_STAGE;
#pragma unroll
      for (int k = 0; k < PG_ELEMS * GPE / PG_NT; ++k) {
        const int q = tid + PG_NT * k, e = q / GPE, part = q - e * GPE, ri = e / PG_TJ, jj = j0 + (e % PG_TJ), ii = i0 + ri;
        const bool in = !PAD || (ii < NZ && jj < NZ);
        const unsigned char* src = reinterpret_cast<const unsigned char*>(z + (in ? ((size_t)ii * NZ + jj) * C + (size_t)ch * CK : (size_t)0)) + part * 16;
        cp_async16(st + (uint32_t)(e * PG_PITCH + ((part ^ (e & 7)) << 4)), src, in ? 16u : 0u);
      }
    }
    cp_async_commit();
  };
  issue(0);
  for (int e = tid; e < C * H; e += PG_NT) gsm[e] = wcm[e];
  __syncthreads();
  if (tid < H) { float s = 0.f; for (int c = 0; c < C; ++c) s += gsm[c * H + tid]; ws[tid] = s; }

  const bool jin = !PAD || j < NZ;
  const bool valid = jin && (kv == nullptr || kv[j]);
  float d[R][H], s[R], s2[R];
  bool live[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int i = i0 + ty + 4 * r;
    live[r] = !PAD || (i < NZ && jin);
    s[r] = s2[r] = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) d[r][h] = 0.f;
  }
#pragma unroll 1
  for (int ch = 0; ch < NCH; ++ch) {
    cp_async_wait<0>();
    __syncthreads();                                                       // stage ch landed; every thread is done with ch - 1, whose stage ch + 1 now takes
    issue(ch + 1);
    const unsigned char* st = ring + (size_t)(ch % PG_NS) * PG_STAGE;
#pragma unroll 2
    for (int c4 = 0; c4 < CK / 4; ++c4) {
      float x[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int e = (ty + 4 * r) * PG_TJ + tx, b = c4 * ES;
        if (live[r]) pg_read4(st + e * PG_PITCH + ((((b >> 4) ^ (e & 7))) << 4) + (b & 15), x[r], z);
        else x[r][0] = x[r][1] = x[r][2] = x[r][3] = 0.f;
      }
      const int c0 = ch * CK + 4 * c4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float4* wr = reinterpret_cast<const float4*>(gsm + (size_t)(c0 + e) * H);
#pragma unroll
        for (int hq = 0; hq < H / 4; ++hq) {
          const float4 w = wr[hq];
#pragma unroll
          for (int r = 0; r < R; ++r) {
            d[r][4 * hq] = fmaf(x[r][e], w.x, d[r][4 * hq]); d[r][4 * hq + 1] = fmaf(x[r][e], w.y, d[r][4 * hq + 1]);
            d[r][4 * hq + 2] = fmaf(x[r][e], w.z, d[r][4 * hq + 2]); d[r][4 * hq + 3] = fmaf(x[r][e], w.w, d[r][4 * hq + 3]);
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) { s[r] += x[r][e]; s2[r] = fmaf(x[r][e], x[r][e], s2[r]); }
      }
    }
  }
  const float invc = 1.f / (float)C;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int i = i0 + ty + 4 * r;
    const float mean = s[r] * invc, rstd = rsqrtf(fmaxf(s2[r] * invc - mean * mean, 0.f) + eps);
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const float y = live[r] ? rstd * (d[r][h] - mean * ws[h]) : 0.f;     // a padded query row or key: 0
      pb_store(bo + ((size_t)h * N + i) * N + j, valid ? y * oscale : fill);
    }
  }
}

}  // namespace aa80
