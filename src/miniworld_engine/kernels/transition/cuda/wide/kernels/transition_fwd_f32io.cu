// transition_fwd_f32io.cu -- the fp32 inference Transition forward at D = 64 / 128 (hidden n D, n = 2 / 4) as ONE fused sm_90a kernel:
//
//   y = x + Ws (silu(Wa LN(x)) * (Wb LN(x)))        x, y fp32 [M][D]; LayerNorm statistics and normalisation in fp32
//
// The structure of transition_fwd_d64.cu (LNFRAG): a persistent CTA per 64-row tile per consumer warpgroup, the LayerNorm in the wgmma
// A-fragment layout (xn never touches shared memory), a two-slot TMA weight ring over the hidden chunks, SwiGLU on the fp32 accumulators
// straight into the A fragments of the squeeze GEMM, the residual added in fp32 over the x tile, one TMA store per tile half.
//
// Numerics: the products run on fp16 tensor cores with fp32 accumulation. fp16 has TF32's 10-bit mantissa (both round to nearest here),
// so every operand is rounded exactly as the TF32 path rounds it (xn, Wa, Wb, h, Ws); fp16's narrower exponent range is covered by four
// fp32 device scalars ``sc`` the host makes once per parameter version (fused_wide_sm90a._f32_forms; 1 for any ordinary magnitude):
//   sc[0] = 1 / s_x            xn16 = xn / s_x
//   sc[1] = s_x s_w            a = acc_a sc[1], b = acc_b sc[1]   (Wa16 = Wa / s_w, Wb16 = Wb / s_w)
//   sc[2] = 1 / s_h            h16 = h / s_h
//   sc[3] = s_h s_ws           y = x + acc sc[3]                  (Ws16 = Ws / s_ws)
// The traffic is x in and y out (512 B / row at D = 64): the three-launch TF32 path moved xn and h through HBM as well.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

TMN_DEVI uint32_t pack_f16(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
TMN_DEVI float f32io_sigmoid(float a) {                // the kit's sigmoid (as transition_fwd_d64.cu): rcp.approx(1 + ex2.approx(-a log2 e))
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));
}
TMN_DEVI void f32io_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

// D[64 x N fp32] (+)= A[64 x 16 fp16, registers] B[16 x N fp16, smem descriptor]; TB = 0: B K-major, 1: B MN-major
template <int TB> TMN_DEVI void f16_rs_n64(float (&d)[32], const uint32_t (&a)[4], uint64_t desc, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, %38;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(accumulate), "n"(TB));
}
template <int TB> TMN_DEVI void f16_rs_n128(float (&d)[64], const uint32_t (&a)[4], uint64_t desc, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, %70;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(accumulate), "n"(TB));
}

// The build's layout. HS: hidden units per chunk (64: the [Wa_j; Wb_j] product is m64n128, 32: m64n64); NWG: consumer warpgroups per CTA
// (64 rows each); NBUF: x tile buffers; CPS: CTAs per SM (the launch bound and the persistent grid).
template <int D, int N, int HS, int NWG, int NBUF, int CPS>
struct F32Cfg {
  static constexpr int H = N * D, NCH = H / HS, ROWS = 64 * NWG;
  static constexpr int KB = D / 64;                    // 64-wide K blocks of the expand GEMM (one 128-byte swizzle atom of fp16 each)
  static constexpr int NA = D / 64;                    // 64-wide N atoms of the squeeze GEMM
  static constexpr int CB = D / 32;                    // 32-wide column blocks of the fp32 x tile (one 128-byte swizzle atom each)
  static constexpr int W1 = 2 * HS * 128;              // one K block of the packed [Wa_j; Wb_j] tile: 2 HS rows of 128 B
  static constexpr int F_WST = KB * W1;                // then Ws^T_j: NA atoms of [HS hidden][64 d]
  static constexpr int F_SLOT = F_WST + NA * HS * 128;
  static constexpr int F_X = 2 * F_SLOT;               // x tile: [NWG halves][CB blocks][64 rows][128 B], NBUF buffers
  static constexpr int F_XB = NWG * CB * 8192;
  static constexpr int F_BAR = F_X + NBUF * F_XB;
  static constexpr int SMEM = F_BAR + 256;
  static_assert(D % 64 == 0 && H % HS == 0 && (HS == 32 || HS == 64), "tile shape");
  static_assert((SMEM + 1024) * CPS <= 231424, "shared memory budget (each block also reserves 1 KB)");
};

template <int D, int N, int HS, int NWG, int NBUF, int CPS>
__global__ void __launch_bounds__(128 * NWG, CPS)
transition_fwd_f32io(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
                     const __grid_constant__ CUtensorMap mwst, const __grid_constant__ CUtensorMap mout, const float* __restrict__ gamma,
                     const float* __restrict__ beta, const float* __restrict__ sc, int tiles, float eps) {
  using C = F32Cfg<D, N, HS, NWG, NBUF, CPS>;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, ncta = gridDim.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + 2; uint64_t* x_full = bars + 4; uint64_t* x_free = bars + 4 + NBUF;
  const int n_local = (tiles > cta) ? (tiles - cta + ncta - 1) / ncta : 0;
  const uint32_t wmax = (uint32_t)n_local * C::NCH;
  auto issue_w = [&](uint32_t seq) {                    // chunk seq % NCH into slot seq & 1
    const int s = seq & 1, j = (int)(seq % C::NCH); uint8_t* b = sm + s * C::F_SLOT;
    mbar_arrive_expect_tx(w_full + s, C::F_SLOT);
#pragma unroll
    for (int kb = 0; kb < C::KB; ++kb) {
      tma_load_2d(b + kb * C::W1, &mwa, w_full + s, kb * 64, j * HS);
      tma_load_2d(b + kb * C::W1 + HS * 128, &mwb, w_full + s, kb * 64, j * HS);
    }
#pragma unroll
    for (int na = 0; na < C::NA; ++na) tma_load_2d(b + C::F_WST + na * HS * 128, &mwst, w_full + s, na * 64, j * HS);
  };
  auto issue_x = [&](int i) {
    const int buf = i % NBUF, row = (cta + i * ncta) * C::ROWS; uint8_t* b = sm + C::F_X + buf * C::F_XB;
    mbar_arrive_expect_tx(x_full + buf, C::F_XB);
#pragma unroll
    for (int h = 0; h < NWG; ++h)
#pragma unroll
      for (int cb = 0; cb < C::CB; ++cb) tma_load_2d(b + (h * C::CB + cb) * 8192, &mx, x_full + buf, cb * 32, row + h * 64);
  };
  if (tid == 0) {
    tma_prefetch_desc(&mx); tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb); tma_prefetch_desc(&mwst); tma_prefetch_desc(&mout);
    mbar_init(w_full, 1); mbar_init(w_full + 1, 1); mbar_init(w_free, NWG); mbar_init(w_free + 1, NWG);
#pragma unroll
    for (int b = 0; b < NBUF; ++b) { mbar_init(x_full + b, 1); mbar_init(x_free + b, NWG); }
    fence_barrier_init();
  }
  __syncthreads();
  uint32_t wseq = 0;
  if (tid == 0) { if (wmax > 0) issue_w(0); if (wmax > 1) issue_w(1); for (int i = 0; i < NBUF && i < n_local; ++i) issue_x(i); }
  const float s_inx = __ldg(sc), s_ab = __ldg(sc + 1), s_inh = __ldg(sc + 2), s_out = __ldg(sc + 3);
  const int q = lane & 3, r0 = wg * 64 + 16 * warp + (lane >> 2);   // fragment rows r0 and r0 + 8 of the tile, columns 2 q + {0, 1} (+ 8)
  // byte address of (row r of the tile, fp32 column c) in an x buffer
  auto xaddr = [&](uint32_t xu, int r, int c) -> uint32_t {
    return xu + (uint32_t)(((r >> 6) * C::CB + (c >> 5)) * 8192) + swz128((uint32_t)(r & 63), (uint32_t)((c & 31) * 4));
  };

  for (int i = 0; i < n_local; ++i) {
    const int buf = i % NBUF, trow = (cta + i * ncta) * C::ROWS;
    const uint32_t xu = su + C::F_X + buf * C::F_XB;
    mbar_wait(x_full + buf, (uint32_t)((i / NBUF) & 1));
    // ---------------------------------------------------------------- LayerNorm (fp32) in the wgmma A-fragment layout -> fp16 xn
    // register reg of k-step ks: row r0 + 8 (reg & 1), columns 16 ks + 8 (reg >> 1) + 2 q + {0, 1}; a quad of lanes holds one row
    uint32_t af[D / 16][4];
    {
      float xv[D / 16][4][2];
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
          const float2 v = lds64f(xaddr(xu, r0 + 8 * (reg & 1), 16 * ks + 8 * (reg >> 1) + 2 * q));
          xv[ks][reg][0] = v.x; xv[ks][reg][1] = v.y;
        }
      float mean[2], rs[2];
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        float s_ = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) s_ += (xv[ks][R][0] + xv[ks][R][1]) + (xv[ks][R + 2][0] + xv[ks][R + 2][1]);
        mean[R] = s_;
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
        for (int R = 0; R < 2; ++R) mean[R] += __shfl_xor_sync(0xffffffffu, mean[R], o);
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        mean[R] *= (1.f / D);
        float v_ = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
          for (int hc = 0; hc < 2; ++hc)
#pragma unroll
            for (int e = 0; e < 2; ++e) { const float d_ = xv[ks][R + 2 * hc][e] - mean[R]; v_ += d_ * d_; }
        rs[R] = v_;
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
        for (int R = 0; R < 2; ++R) rs[R] += __shfl_xor_sync(0xffffffffu, rs[R], o);
#pragma unroll
      for (int R = 0; R < 2; ++R) rs[R] = rsqrtf(rs[R] * (1.f / D) + eps);
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
          const int c = 16 * ks + 8 * (reg >> 1) + 2 * q, R = reg & 1;
          const float2 g = __ldg(reinterpret_cast<const float2*>(gamma + c)), b = __ldg(reinterpret_cast<const float2*>(beta + c));
          af[ks][reg] = pack_f16(((xv[ks][reg][0] - mean[R]) * rs[R] * g.x + b.x) * s_inx, ((xv[ks][reg][1] - mean[R]) * rs[R] * g.y + b.y) * s_inx);
        }
    }
    // ---------------------------------------------------------------- the hidden chunks
    float acc[C::NA][32];
#pragma unroll
    for (int na = 0; na < C::NA; ++na)
#pragma unroll
      for (int e = 0; e < 32; ++e) acc[na][e] = 0.f;
    for (int j = 0; j < C::NCH; ++j, ++wseq) {
      const int s = (int)(wseq & 1);
      const uint32_t slot = su + s * C::F_SLOT;
      mbar_wait(w_full + s, (uint32_t)((wseq >> 1) & 1));
      float AB[HS];                                     // m64 n(2 HS): a in groups 0 .. HS / 8 - 1, b in the rest
#pragma unroll
      for (int e = 0; e < HS; ++e) AB[e] = 0.f;
      fence_regs(AB);
#pragma unroll
      for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        const uint64_t desc = smem_desc(slot + (uint32_t)((ks >> 2) * C::W1 + (ks & 3) * 32), 16, 1024, 1);
        if constexpr (HS == 64) f16_rs_n128<0>(AB, af[ks], desc, ks > 0);
        else f16_rs_n64<0>(AB, af[ks], desc, ks > 0);
      }
      wgmma_commit();
      wgmma_wait<1>();                                  // also retires the previous chunk's squeeze update
#pragma unroll
      for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
      if (j > 0 || i > 0) {
        const int ps = (int)((wseq - 1) & 1);
        if (wtid == 0) mbar_arrive(w_free + ps);
        if (tid == 0) { mbar_wait(w_free + ps, (uint32_t)(((wseq - 1) >> 1) & 1)); if (wseq + 1 < wmax) issue_w(wseq + 1); }
      }
      wgmma_wait<0>(); fence_regs(AB);
      uint32_t fh[HS / 16][4];                          // h16 = fp16(silu(a) b / s_h) in the A-fragment layout of the squeeze GEMM
#pragma unroll
      for (int g = 0; g < HS / 8; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r] * s_ab, a1 = AB[e0 + 2 * r + 1] * s_ab;
          const float b0 = AB[e0 + HS / 2 + 2 * r] * s_ab, b1 = AB[e0 + HS / 2 + 2 * r + 1] * s_ab;
          fh[ks][o + r] = pack_f16(a0 * f32io_sigmoid(a0) * b0 * s_inh, a1 * f32io_sigmoid(a1) * b1 * s_inh);
        }
      }
#pragma unroll
      for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < HS / 16; ++ks)
#pragma unroll
        for (int na = 0; na < C::NA; ++na)
          f16_rs_n64<1>(acc[na], fh[ks], smem_desc(slot + (uint32_t)(C::F_WST + na * HS * 128 + ks * 2048), 8192, 1024, 1), 1);
      wgmma_commit();
    }
    wgmma_wait<0>();
#pragma unroll
    for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
    // ---------------------------------------------------------------- y = x + acc s_out (fp32), in place over the x tile, one TMA store per half
#pragma unroll
    for (int na = 0; na < C::NA; ++na)
#pragma unroll
      for (int g = 0; g < 8; ++g)
#pragma unroll
        for (int R = 0; R < 2; ++R) {
          const uint32_t ad = xaddr(xu, r0 + 8 * R, 64 * na + 8 * g + 2 * q);
          const float2 v = lds64f(ad);
          sts64f(ad, v.x + acc[na][4 * g + 2 * R] * s_out, v.y + acc[na][4 * g + 2 * R + 1] * s_out);
        }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (wtid == 0) {
#pragma unroll
      for (int cb = 0; cb < C::CB; ++cb) f32io_store_2d(&mout, sm + C::F_X + buf * C::F_XB + (wg * C::CB + cb) * 8192, cb * 32, trow + wg * 64);
      tma_store_commit();
      tma_store_wait_read<0>();                         // the store has read the tile: the buffer may be refilled
      mbar_arrive(x_free + buf);
    }
    if (tid == 0 && i + NBUF < n_local) { mbar_wait(x_free + buf, (uint32_t)((i / NBUF) & 1)); issue_x(i + NBUF); }
  }
  if (wtid == 0) tma_store_wait_all();
}
