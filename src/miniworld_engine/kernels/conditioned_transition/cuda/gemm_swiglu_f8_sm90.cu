// gemm_swiglu_f8_sm90.cu -- EXPERIMENTAL (engine 3.0.0 line, opt-in, default off): the token DiT transition's expand + SwiGLU
// (gemm_swiglu_sm90.cu, same tiles, ring and epilogue) on e4m3 operands: wgmma m64n{256,128}k32 e4m3 x e4m3 -> fp32.
//
//   h = silu(a) * b,  a = sx[m] swa[n] (x8 Wa8^T)[m, n],  b = sx[m] swb[n] (x8 Wb8^T)[m, n]
//
// x8 [M, K] e4m3 with a per-row scale sx (token_dit_fp8_rows.cu writes both), Wab8 [2H, K] e4m3 packed per 2 HT rows exactly as the
// bf16 kernel's wab (HT of Wa, the same HT of Wb), sw [2H] fp32 per packed row (per output channel). A 128-B slab row is 128 e4m3 of
// K; a k-step of the wgmma is 32 B as in the bf16 kernel, so the descriptors and the TMA boxes are the bf16 kernel's byte for byte.
// h is bf16 ([M, H], 128-B swizzled stmatrix staging and TMA stores, as the bf16 kernel). The ext also holds the fp8 row passes.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

TMN_DEVI void mma_f8_n256(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, 1, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(a), "l"(b));
}

TMN_DEVI void mma_f8_n128(float (&d)[64], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(a), "l"(b));
}

constexpr int TBM = 128;
template <int HTW> struct Cfg {
  static constexpr int TBK = 128;                                         // K elements (bytes) a 128-B slab row
  static constexpr int NST = 4;
  static constexpr int HT = HTW, TBN = 2 * HTW;
  static constexpr int F_SA = TBM * 128, F_SB = TBN * 128, F_ST = F_SA + F_SB;
  static constexpr int STGW = 64 * HT * 2;
  static constexpr int STG = NST * F_ST, BAR = STG + 2 * STGW, SMEM = BAR + 256;
  static_assert(SMEM + 1024 <= 232448, "shared memory budget");
};

TMN_DEVI float sigmoid_(float a) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f); }
TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

template <int HTW>
__global__ void __launch_bounds__(384, 1)
gemm_swiglu_f8_sm90(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB,
                    const __grid_constant__ CUtensorMap mH, const float* __restrict__ sx, const float* __restrict__ sw, int M, int K, int H) {
  using C = Cfg<HTW>;
  constexpr int NSTAGE = C::NST, TBK = C::TBK, HT = C::HT, TBN = C::TBN, F_SA = C::F_SA, F_ST = C::F_ST;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::BAR);
  uint64_t* empty = full + NSTAGE;
  const int MT = M / TBM, NT = H / HT, KB = K / TBK, ntiles = MT * NT;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mA); tma_prefetch_desc(&mB);
    uint32_t it = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      const int m0 = (t % MT) * TBM, n0 = (t / MT) * TBN;
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        mbar_arrive_expect_tx(full + s, F_ST);
        tma_load_2d(sm + s * F_ST, &mA, full + s, kb * TBK, m0);
        tma_load_2d(sm + s * F_ST + F_SA, &mB, full + s, kb * TBK, n0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  uint8_t* stg = sm + C::STG + cw * C::STGW;
  uint32_t it = 0;
  bool first = true;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const int m0 = (t % MT) * TBM, nt = t / MT;
    float acc[TBN / 2];
#pragma unroll
    for (int e = 0; e < TBN / 2; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_ST + cw * 8192, b0 = su + s * F_ST + F_SA;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if constexpr (HTW == 128) mma_f8_n256(acc, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
        else mma_f8_n128(acc, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
      }
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
    if (!first && wtid == 0) tma_store_wait_read<0>();
    first = false;
    named_bar_sync(1 + cw, 128);
    // accumulator rows: 16 warp + lane / 4 (+ 8 for rb = 1) of this warpgroup's 64; columns 8 g + 2 (lane & 3) (+ 1)
    const float s0 = sx[m0 + 64 * cw + 16 * warp + (lane >> 2)], s1 = sx[m0 + 64 * cw + 16 * warp + (lane >> 2) + 8];
    const float* swt = sw + (long)nt * TBN;
    const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
    for (int gp = 0; gp < HT / 16; ++gp) {
      uint32_t hp[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + HT / 8) + 2 * rb;
        const int c = 8 * g + 2 * (lane & 3);
        const float sr = rb ? s1 : s0;
        const float2 wa = *reinterpret_cast<const float2*>(swt + c), wb = *reinterpret_cast<const float2*>(swt + HT + c);
        const float x0 = acc[ia] * (sr * wa.x), x1 = acc[ia + 1] * (sr * wa.y);
        const float h0 = x0 * sigmoid_(x0) * (acc[ib] * (sr * wb.x)), h1 = x1 * sigmoid_(x1) * (acc[ib + 1] * (sr * wb.y));
        hp[q] = pack_bf16(h0, h1);
      }
      const int col = 8 * (2 * (gp & 3) + (mi >> 1));
      stsm_x4(smem_u32(stg + (gp >> 2) * 8192) + swz128((uint32_t)mrow, (uint32_t)(col * 2)), hp[0], hp[1], hp[2], hp[3]);
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
#pragma unroll
      for (int b = 0; b < HT / 64; ++b) tma_store_2d(&mH, stg + b * 8192, nt * HT + b * 64, m0 + 64 * cw);
      tma_store_commit();
    }
  }
  if (wtid == 0) tma_store_wait_all();
}

// ------------------------------------------------------------------------------------------------ host
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "no TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
CUtensorMap map2d(void* p, int esize, uint64_t rows, uint64_t cols, uint64_t rs, uint32_t bc, uint32_t br) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {cols, rows}, strides[1] = {rs * esize};
  const cuuint32_t box[2] = {bc, br}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, esize == 1 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, dims, strides, box,
                        elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "tensor map encode failed");
  return map;
}
template <int HTW>
void run(torch::Tensor& x, torch::Tensor& wab, torch::Tensor& h, torch::Tensor& sx, torch::Tensor& sw, int64_t sms, bool pdl) {
  using C = Cfg<HTW>;
  const int64_t M = x.size(0), K = x.size(1), H = h.size(1);
  TORCH_CHECK(M % TBM == 0 && K % C::TBK == 0 && H % C::HT == 0, "shapes: M % 128, K % 128, H % HT");
  const auto mA = map2d(x.data_ptr(), 1, M, K, x.stride(0), C::TBK, TBM), mB = map2d(wab.data_ptr(), 1, 2 * H, K, K, C::TBK, C::TBN),
             mH = map2d(h.data_ptr(), 2, M, H, H, 64, 64);
  const int tiles = (int)((M / TBM) * (H / C::HT));
  cudaFuncSetAttribute(gemm_swiglu_f8_sm90<HTW>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM + 1024);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((unsigned)std::min<int64_t>(tiles, sms));
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = C::SMEM + 1024;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, gemm_swiglu_f8_sm90<HTW>, mA, mB, mH, (const float*)sx.data_ptr(), (const float*)sw.data_ptr(),
                                             (int)M, (int)K, (int)H);
  TORCH_CHECK(err == cudaSuccess, "gemm_swiglu_f8 launch: ", cudaGetErrorString(err));
}
}  // namespace

// x8 [M, K] e4m3 (row stride may exceed K), sx [M] fp32, wab8 [2H, K] e4m3 packed per 2 ht rows, sw [2H] fp32 in the same order,
// h [M, H] bf16. M % 128, K % 128, H % ht.
void gemm_swiglu_f8(torch::Tensor x, torch::Tensor wab, torch::Tensor h, torch::Tensor sx, torch::Tensor sw, int64_t sms, int64_t ht, bool pdl) {
  TORCH_CHECK(x.scalar_type() == torch::kFloat8_e4m3fn && wab.scalar_type() == torch::kFloat8_e4m3fn && h.scalar_type() == torch::kBFloat16,
              "e4m3 x / wab, bf16 h");
  TORCH_CHECK(sx.scalar_type() == torch::kFloat && sw.scalar_type() == torch::kFloat && sx.is_contiguous() && sw.is_contiguous()
              && sx.numel() == x.size(0) && sw.numel() == wab.size(0), "fp32 scales sx [M], sw [2H]");
  TORCH_CHECK(x.stride(1) == 1 && wab.is_contiguous() && h.is_contiguous() && wab.size(0) == 2 * h.size(1) && wab.size(1) == x.size(1)
              && h.size(0) == x.size(0), "shapes");
  const at::cuda::CUDAGuard guard(x.device());
  if (ht == 64) run<64>(x, wab, h, sx, sw, sms, pdl);
  else { TORCH_CHECK(ht == 128, "ht 64 or 128"); run<128>(x, wab, h, sx, sw, sms, pdl); }
}

void adaln_in_rows_cf_q8(at::Tensor xin, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor c, at::Tensor stats,
                         at::Tensor out8, at::Tensor srow, double eps, double ceps, bool pdl);
void adaln_in_rows_cf_par_q8(at::Tensor xin, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor c, at::Tensor stats,
                             at::Tensor out, at::Tensor out8, at::Tensor srow, double eps, double ceps, bool pdl, bool sig);
void resgate_adaln_cf_q8(at::Tensor xin, at::Tensor ya, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor stats,
                         at::Tensor out8, at::Tensor srow, double eps, bool pdl);
void quant_rows(at::Tensor x, at::Tensor out8, at::Tensor srow, bool pdl);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_swiglu_f8", &gemm_swiglu_f8);
  m.def("adaln_in_rows_cf_q8", &adaln_in_rows_cf_q8);
  m.def("adaln_in_rows_cf_par_q8", &adaln_in_rows_cf_par_q8);
  m.def("resgate_adaln_cf_q8", &resgate_adaln_cf_q8);
  m.def("quant_rows", &quant_rows);
}
