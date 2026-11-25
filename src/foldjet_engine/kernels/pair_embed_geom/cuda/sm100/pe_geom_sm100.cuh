// pe_geom_sm100.cuh -- B200 (sm_100a) forward of the encoder backbone pair (PairEmbedGeom, 2026-11-17) and the host entry points of the round-3
// kernels, included by pair_embed_geom_sm100.cu after pe_ln_sm100.cuh.
//
// The RTX 5090 one-launch kernels this card ran (pg120_fwd / pg120_fwd_blk: a warp a pair, every lane unpacking 8 bf16 channels of up to ten
// table rows: ~13 instructions per 8 channels and row) were instruction bound at L384 (54 us, S1) and latency bound at L128 (15.8 us).
// Here: the decoder pair's design (pe_ca_sm100.cuh: persistent warps over chunks of 8 / 16 / 32 consecutive pairs, lane l bins pair l of the
// chunk with pg_fwd's arithmetic from inputs that were requested a chunk earlier, a group of G lanes a pair) on the bf16 table through the
// L1 (pe100ln::pair8: the rows are added in pg_fwd's order: the pair's bits; an fp32 widening of the table -- half the instructions a row -- was 2x slower: 325 KB do not stay in the L1), the LayerNorm by
// pe100ln::ln_hat. The 16-byte codes (training) are pg_fwd's.
#pragma once

namespace pe100g {

constexpr int WARPS = 4;
using pe100ln::f2;

template <typename IdxT>
struct Raw {
  V3 Ni, CAi, CBi, Nj, CAj, Cj, CBj;
  IdxT ii, ij, hi, hj;
  uint8_t mi, mj, ai, aj, bi, bj;
};

__device__ __forceinline__ V3 atom_at(const float* __restrict__ bb, unsigned row, int a) {
  const float* p = bb + ((size_t)row * 4 + a) * 3;
  return {__ldg(p), __ldg(p + 1), __ldg(p + 2)};
}

template <typename IdxT>
__device__ __forceinline__ Raw<IdxT> load_raw(const float* __restrict__ bb, const uint8_t* __restrict__ ca, const uint8_t* __restrict__ cb,
                                              const uint8_t* __restrict__ mask, const IdxT* __restrict__ idx, const IdxT* __restrict__ chains, unsigned n,
                                              unsigned p, unsigned P) {
  Raw<IdxT> r;
  r.mi = r.mj = r.ai = r.aj = r.bi = r.bj = 0;
  r.ii = r.ij = r.hi = r.hj = 0;
  r.Ni = r.CAi = r.CBi = r.Nj = r.CAj = r.Cj = r.CBj = {0.f, 0.f, 0.f};
  if (p < P) {
    const unsigned row = p / n;                          // b * n + i
    const unsigned rj = (row / n) * n + (p - row * n);   // b * n + j
    r.mi = __ldg(mask + row); r.mj = __ldg(mask + rj);
    r.ai = __ldg(ca + row); r.aj = __ldg(ca + rj);
    r.bi = __ldg(cb + row); r.bj = __ldg(cb + rj);
    r.ii = __ldg(idx + row); r.ij = __ldg(idx + rj);
    r.hi = __ldg(chains + row); r.hj = __ldg(chains + rj);
    r.Ni = atom_at(bb, row, 0); r.CAi = atom_at(bb, row, 1); r.CBi = atom_at(bb, row, 3);
    r.Nj = atom_at(bb, rj, 0); r.CAj = atom_at(bb, rj, 1); r.Cj = atom_at(bb, rj, 2); r.CBj = atom_at(bb, rj, 3);
  }
  return r;
}

// pg_fwd's binning: the stored code (word 0 with its live bit, the angle bins, the chain product's bits)
template <typename IdxT>
__device__ __forceinline__ uint4 code_of(const Raw<IdxT>& r, const float* dlim_s, const float* alim_s) {
  uint32_t w0 = MASKED, w1 = 0;
  float cv = 0.f;
  if (r.mi && r.mj) {
    const int s = (int)min(max((long long)r.ii - (long long)r.ij + 63, 0LL), 126LL);
    w0 = (uint32_t)s;
    cv = (float)(r.hi * r.hj);
    if (r.ai && r.aj) {
      const V3 Ni = r.Ni, CAi = r.CAi, CBi = r.CBi, Nj = r.Nj, CAj = r.CAj, Cj = r.Cj, CBj = r.CBj;
      const float d0 = nrm(sub(CAi, Nj)), d1 = nrm(sub(CAi, CAj)), d2 = nrm(sub(CAi, Cj)), d3 = r.bj ? nrm(sub(CAi, CBj)) : 0.f;
      w0 |= F_DIST | ((uint32_t)bucket(d0, dlim_s, 20) << 8) | ((uint32_t)bucket(d1, dlim_s, 20) << 13) | ((uint32_t)bucket(d2, dlim_s, 20) << 18) |
            ((uint32_t)bucket(d3, dlim_s, 20) << 23);
      if (r.bi) {
        const float a0 = dihedral(Ni, CAi, CBi, CBj), a1 = dihedral(Nj, CAj, CBj, CBi), a2 = bond(CAi, CBi, CBj), a3 = bond(CAj, CBj, CBi),
                    a4 = dihedral(CAi, CBi, CBj, CAj);
        w0 |= F_ORI;
        w1 = (uint32_t)bucket(a0, alim_s, 20) | ((uint32_t)bucket(a1, alim_s, 20) << 5) | ((uint32_t)bucket(a2, alim_s, 20) << 10) |
             ((uint32_t)bucket(a3, alim_s, 20) << 15) | ((uint32_t)bucket(a4, alim_s, 20) << 20);
      }
    }
  }
  return make_uint4(w0 ^ MASKED, w1, __float_as_uint(cv), 0u);
}

// out [P][256] bf16 (its LayerNorm when LN), codes [P] (16 bytes) when CODES; tw: the bf16 table [317][256]; G lanes a pair, ch pairs a chunk
// MB: CTAs per SM the registers must allow (1: no cap; 6: at most 80 registers a thread, 24 warps per SM)
template <typename IdxT, int G, bool LN, bool CODES, int MB>
__global__ void __launch_bounds__(WARPS * 32, MB)
fwd(const __nv_bfloat16* __restrict__ tw, const float* __restrict__ bb, const uint8_t* __restrict__ ca, const uint8_t* __restrict__ cb, const uint8_t* __restrict__ mask,
    const IdxT* __restrict__ idx, const IdxT* __restrict__ chains, const float* __restrict__ dlim, const float* __restrict__ alim,
    __nv_bfloat16* __restrict__ out, uint4* __restrict__ codes, unsigned n, unsigned P, float eps, int ch) {
  constexpr int NVL = 32 / G, PPS = 32 / G;
  __shared__ float dlim_s[32], alim_s[32];
  if (threadIdx.x < 32) { dlim_s[threadIdx.x] = threadIdx.x < 20 ? dlim[threadIdx.x] : 0.f; alim_s[threadIdx.x] = threadIdx.x < 20 ? alim[threadIdx.x] : 0.f; }
  __syncthreads();
  const int lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  const unsigned nchunk = (P + ch - 1) / ch, wstride = gridDim.x * WARPS;
  unsigned q = blockIdx.x * WARPS + (threadIdx.x >> 5);
  Raw<IdxT> cur = load_raw(bb, ca, cb, mask, idx, chains, n, q < nchunk && lane < ch ? q * ch + lane : P, P);
  bool waited = false;
  while (q < nchunk) {
    const unsigned qn = q + wstride;
    const Raw<IdxT> nxt = load_raw(bb, ca, cb, mask, idx, chains, n, qn < nchunk && lane < ch ? qn * ch + lane : P, P);   // in flight meanwhile
    const uint4 code = code_of(cur, dlim_s, alim_s);
    if (!waited) { pdl_wait(); waited = true; }               // the table (pg120_table, a training step) from here on
    const unsigned pb = q * ch;
    if constexpr (CODES) {
      if (lane < ch && pb + lane < P) codes[pb + lane] = code;
    }
#pragma unroll 1
    for (int t = 0; t < ch / PPS; t++) {
      if (pb + t * PPS >= P) break;                            // warp-uniform
      const unsigned p = pb + t * PPS + grp;
      const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, code.x, t * PPS + grp), w1 = __shfl_sync(0xFFFFFFFFu, code.y, t * PPS + grp);
      const float pc = __uint_as_float(__shfl_sync(0xFFFFFFFFu, code.z, t * PPS + grp));
      f2 x[NVL][4];
#pragma unroll
      for (int k = 0; k < NVL; k++) pe100ln::pair8<0, __nv_bfloat16>(tw, w0, w1, pc, (k * G + g) * 8, x[k]);
      if constexpr (LN) pe100ln::ln_hat<G>(x, eps);
      if (p < P) {
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          const uint4 u = make_uint4(pe100ln::pack2(x[k][0]), pe100ln::pack2(x[k][1]), pe100ln::pack2(x[k][2]), pe100ln::pack2(x[k][3]));
          __stcs(reinterpret_cast<uint4*>(out + (size_t)p * C + (k * G + g) * 8), u);
        }
      }
    }
    cur = nxt;
    q = qn;
  }
}

// fwd with the whole bf16 table in shared memory (317 x 512 B = 162 KB: one CTA of NW warps per SM): the eleven row gathers a pair are
// ld.shared instead of L1 / L2 traffic (the L1 path re-reads ~5.6 KB a pair from the L2 once the table does not stay in the L1)
constexpr int SMT_BYTES = NT * C * 2;
template <typename IdxT, int G, bool LN, bool CODES, int NW>
__global__ void __launch_bounds__(NW * 32, 1)
fwd_smt(const __nv_bfloat16* __restrict__ tw, const float* __restrict__ bb, const uint8_t* __restrict__ ca, const uint8_t* __restrict__ cb,
        const uint8_t* __restrict__ mask, const IdxT* __restrict__ idx, const IdxT* __restrict__ chains, const float* __restrict__ dlim,
        const float* __restrict__ alim, __nv_bfloat16* __restrict__ out, uint4* __restrict__ codes, unsigned n, unsigned P, float eps, int ch) {
  constexpr int NVL = 32 / G, PPS = 32 / G;
  extern __shared__ __align__(16) uint4 tsm_v[];
  __shared__ float dlim_s[32], alim_s[32];
  const int lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  const unsigned nchunk = (P + ch - 1) / ch, wstride = gridDim.x * NW;
  unsigned q = blockIdx.x * NW + (threadIdx.x >> 5);
  Raw<IdxT> cur = load_raw(bb, ca, cb, mask, idx, chains, n, q < nchunk && lane < ch ? q * ch + lane : P, P);
  __shared__ __align__(8) unsigned long long mbar;
  const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
  if (threadIdx.x < 32) { dlim_s[threadIdx.x] = threadIdx.x < 20 ? dlim[threadIdx.x] : 0.f; alim_s[threadIdx.x] = threadIdx.x < 20 ? alim[threadIdx.x] : 0.f; }
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {                                      // the table by four bulk copies (TMA, 1-D) while the first chunk is binned
    pdl_wait();                                                // the table (pg120_table, a training step) from here on
    const unsigned dst = (unsigned)__cvta_generic_to_shared(tsm_v);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"((unsigned)SMT_BYTES) : "memory");
    constexpr unsigned PIECE = SMT_BYTES / 4;
#pragma unroll
    for (int k = 0; k < 4; k++)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst + k * PIECE),
                   "l"(reinterpret_cast<const char*>(tw) + k * PIECE), "r"(PIECE), "r"(mb)
                   : "memory");
  }
  uint4 code = code_of(cur, dlim_s, alim_s);
  {
    uint32_t done = 0;
    while (!done)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(mb) : "memory");
  }
  const __nv_bfloat16* tsm = reinterpret_cast<const __nv_bfloat16*>(tsm_v);
  while (q < nchunk) {
    const unsigned qn = q + wstride;
    const Raw<IdxT> nxt = load_raw(bb, ca, cb, mask, idx, chains, n, qn < nchunk && lane < ch ? qn * ch + lane : P, P);
    const unsigned pb = q * ch;
    if constexpr (CODES) {
      if (lane < ch && pb + lane < P) codes[pb + lane] = code;
    }
#pragma unroll 1
    for (int t = 0; t < ch / PPS; t++) {
      if (pb + t * PPS >= P) break;
      const unsigned p = pb + t * PPS + grp;
      const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, code.x, t * PPS + grp), w1 = __shfl_sync(0xFFFFFFFFu, code.y, t * PPS + grp);
      const float pc = __uint_as_float(__shfl_sync(0xFFFFFFFFu, code.z, t * PPS + grp));
      f2 x[NVL][4];
#pragma unroll
      for (int k = 0; k < NVL; k++) pe100ln::pair8<0, __nv_bfloat16, true>(tsm, w0, w1, pc, (k * G + g) * 8, x[k]);
      if constexpr (LN) pe100ln::ln_hat<G>(x, eps);
      if (p < P) {
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          const uint4 u = make_uint4(pe100ln::pack2(x[k][0]), pe100ln::pack2(x[k][1]), pe100ln::pack2(x[k][2]), pe100ln::pack2(x[k][3]));
          __stcs(reinterpret_cast<uint4*>(out + (size_t)p * C + (k * G + g) * 8), u);
        }
      }
    }
    if (qn < nchunk) code = code_of(nxt, dlim_s, alim_s);
    q = qn;
  }
}

}  // namespace pe100g

namespace {
// CTAs (of 128 threads, 4 warps) of a persistent launch over `chunks` warp chunks: at most cps per SM
inline dim3 pe100_grid(long long chunks, int64_t cps) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return dim3((unsigned)std::max<long long>(1, std::min<long long>((long long)sms * std::max<int64_t>(1, cps), (chunks + 3) / 4)));
}
inline void pe100_check_ch(int64_t ch) { TORCH_CHECK(ch == 8 || ch == 16 || ch == 32, "pair_embed sm100: chunks of 8, 16 or 32 pairs"); }
}  // namespace

// The encoder pair [B, n, n, 256] bf16 (its LayerNorm when ln) from the bf16 table [317, 256], bb [B, n, 4, 3] fp32, the masks as
// bytes, idx / chains int32 or int64; codes [B, n, n, 4] int32, or empty (inference). g lanes a pair (8 / 16 / 32), ch pairs a warp chunk, at most
// cps CTAs per SM (from 6 on: the build with the registers capped for 6 CTAs per SM).
// A training step passes the module's [256, 317] weight: the table is made here (pg120_table) and returned (else tw itself is returned).
torch::Tensor forward_b200(torch::Tensor tw, torch::Tensor bb, torch::Tensor ca, torch::Tensor cb, torch::Tensor mask, torch::Tensor idx, torch::Tensor chains,
                  torch::Tensor dlim, torch::Tensor alim, torch::Tensor out, torch::Tensor codes, int64_t ln, double eps, int64_t g, int64_t cps, int64_t ch) {
  const at::cuda::CUDAGuard guard(tw.device());
  pe100_check_ch(ch);
  TORCH_CHECK(dlim.numel() == 20 && alim.numel() == 20, "20 bin limits");
  auto st = at::cuda::getCurrentCUDAStream();
  bool pdl = false;
  if (idx.numel() > 0) tw = table_of(tw, NT, st, pdl);   // a [256, 317] weight: its table by pg120_table, this launch behind it by PDL
  TORCH_CHECK(tw.scalar_type() == at::kBFloat16 && tw.size(0) == NT && tw.size(1) == C && tw.is_contiguous() && bb.scalar_type() == at::kFloat &&
                  bb.is_contiguous() && ca.element_size() == 1 && cb.element_size() == 1 && mask.element_size() == 1 && ca.is_contiguous() &&
                  cb.is_contiguous() && mask.is_contiguous() && idx.is_contiguous() && chains.is_contiguous() && idx.scalar_type() == chains.scalar_type(),
              "forward_b200: bf16 table [317, 256], contiguous fp32 bb, byte masks, idx / chains of one integer type");
  const long long B = idx.size(0), n = idx.size(1), PL = B * n * n;
  TORCH_CHECK(PL < (1ll << 31), "at most 2^31 pairs");
  if (PL == 0) return tw;
  const bool wc = codes.numel() > 0;
  TORCH_CHECK(!wc || (codes.numel() == PL * 4 && codes.scalar_type() == at::kInt && codes.is_contiguous()), "codes: contiguous int32 [B, n, n, 4]");
  uint4* cp = wc ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr;
  const dim3 grid = pe100_grid((PL + ch - 1) / ch, cps), block(pe100g::WARPS * 32);
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto* m1 = static_cast<const uint8_t*>(ca.data_ptr());
  const auto* m2 = static_cast<const uint8_t*>(cb.data_ptr());
  const auto* m3 = static_cast<const uint8_t*>(mask.data_ptr());
  auto go = [&](auto ip, auto hp) {
    using I = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
#define PE_G(G, LN, CO)                                                                                                                         \
  launch_pdl(cps >= 6 ? pe100g::fwd<I, G, LN, CO, 6> : pe100g::fwd<I, G, LN, CO, 1>, grid, block, st, pdl, reinterpret_cast<const __nv_bfloat16*>(tw.data_ptr<at::BFloat16>()), bb.data_ptr<float>(), m1, m2, m3, ip, hp, dlim.data_ptr<float>(),      \
                                                    alim.data_ptr<float>(), op, cp, (unsigned)n, (unsigned)PL, (float)eps, (int)ch)
#define PE_GG(G)                                                                                                                                \
  do {                                                                                                                                          \
    if (ln && wc) PE_G(G, true, true);                                                                                                          \
    else if (ln) PE_G(G, true, false);                                                                                                          \
    else if (wc) PE_G(G, false, true);                                                                                                          \
    else PE_G(G, false, false);                                                                                                                 \
  } while (0)
    if (g == 8) PE_GG(8);
    else if (g == 16) PE_GG(16);
    else PE_GG(32);
#undef PE_GG
#undef PE_G
  };
  if (idx.scalar_type() == at::kLong) go(static_cast<const long long*>(idx.data_ptr()), static_cast<const long long*>(chains.data_ptr()));
  else {
    TORCH_CHECK(idx.scalar_type() == at::kInt, "idx: int32 or int64");
    go(static_cast<const int*>(idx.data_ptr()), static_cast<const int*>(chains.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tw;
}

// forward_b200 with the table in shared memory (pe100g::fwd_smt): one CTA of nw warps (8 / 16 / 24 / 32) per SM
torch::Tensor forward_b200s(torch::Tensor tw, torch::Tensor bb, torch::Tensor ca, torch::Tensor cb, torch::Tensor mask, torch::Tensor idx, torch::Tensor chains,
                            torch::Tensor dlim, torch::Tensor alim, torch::Tensor out, torch::Tensor codes, int64_t ln, double eps, int64_t g, int64_t nw, int64_t ch) {
  const at::cuda::CUDAGuard guard(tw.device());
  pe100_check_ch(ch);
  TORCH_CHECK(dlim.numel() == 20 && alim.numel() == 20, "20 bin limits");
  auto st = at::cuda::getCurrentCUDAStream();
  bool pdl = false;
  if (idx.numel() > 0) tw = table_of(tw, NT, st, pdl);
  TORCH_CHECK(tw.scalar_type() == at::kBFloat16 && tw.size(0) == NT && tw.size(1) == C && tw.is_contiguous() && bb.scalar_type() == at::kFloat &&
                  bb.is_contiguous() && ca.element_size() == 1 && cb.element_size() == 1 && mask.element_size() == 1 && ca.is_contiguous() &&
                  cb.is_contiguous() && mask.is_contiguous() && idx.is_contiguous() && chains.is_contiguous() && idx.scalar_type() == chains.scalar_type(),
              "forward_b200s: bf16 table [317, 256], contiguous fp32 bb, byte masks, idx / chains of one integer type");
  const long long B = idx.size(0), n = idx.size(1), PL = B * n * n;
  TORCH_CHECK(PL < (1ll << 31), "at most 2^31 pairs");
  if (PL == 0) return tw;
  const bool wc = codes.numel() > 0;
  TORCH_CHECK(!wc || (codes.numel() == PL * 4 && codes.scalar_type() == at::kInt && codes.is_contiguous()), "codes: contiguous int32 [B, n, n, 4]");
  uint4* cp = wc ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long long chunks = (PL + ch - 1) / ch;
  const dim3 grid((unsigned)std::max<long long>(1, std::min<long long>(sms, (chunks + nw - 1) / nw))), block((unsigned)nw * 32);
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto* m1 = static_cast<const uint8_t*>(ca.data_ptr());
  const auto* m2 = static_cast<const uint8_t*>(cb.data_ptr());
  const auto* m3 = static_cast<const uint8_t*>(mask.data_ptr());
  const auto* tp = reinterpret_cast<const __nv_bfloat16*>(tw.data_ptr<at::BFloat16>());
  auto launch = [&](auto kernel, auto ip, auto hp) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pe100g::SMT_BYTES));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = pe100g::SMT_BYTES;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, tp, (const float*)bb.data_ptr<float>(), m1, m2, m3, ip, hp, (const float*)dlim.data_ptr<float>(),
                                      (const float*)alim.data_ptr<float>(), op, cp, (unsigned)n, (unsigned)PL, (float)eps, (int)ch));
  };
  auto go = [&](auto ip, auto hp) {
    using I = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
#define PE_S(G, LN, CO)                                                                                                        \
  do {                                                                                                                         \
    if (nw == 8) launch(pe100g::fwd_smt<I, G, LN, CO, 8>, ip, hp);                                                             \
    else if (nw == 16) launch(pe100g::fwd_smt<I, G, LN, CO, 16>, ip, hp);                                                      \
    else if (nw == 24) launch(pe100g::fwd_smt<I, G, LN, CO, 24>, ip, hp);                                                      \
    else launch(pe100g::fwd_smt<I, G, LN, CO, 32>, ip, hp);                                                                    \
  } while (0)
#define PE_SG(G)                                                                                                               \
  do {                                                                                                                         \
    if (ln && wc) PE_S(G, true, true);                                                                                         \
    else if (ln) PE_S(G, true, false);                                                                                         \
    else if (wc) PE_S(G, false, true);                                                                                         \
    else PE_S(G, false, false);                                                                                                \
  } while (0)
    if (g == 16) PE_SG(16);
    else PE_SG(32);
#undef PE_SG
#undef PE_S
  };
  if (idx.scalar_type() == at::kLong) go(static_cast<const long long*>(idx.data_ptr()), static_cast<const long long*>(chains.data_ptr()));
  else {
    TORCH_CHECK(idx.scalar_type() == at::kInt, "idx: int32 or int64");
    go(static_cast<const int*>(idx.data_ptr()), static_cast<const int*>(chains.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tw;
}

// d pair [P][256] bf16 from d LayerNorm(pair) = dh (the pair recomputed from the 16-byte codes and the bf16 table: mode 0 the encoder table
// [317, 256], mode 1 the bf16 decoder table [157, 256]); g lanes a pair (8 / 16 / 32), ch pairs a warp chunk, at most cps CTAs per SM
void ln_backward_b200(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor dx, int64_t mode, double eps, int64_t g, int64_t cps,
                      int64_t ch) {
  const at::cuda::CUDAGuard guard(tbl.device());
  pe100_check_ch(ch);
  const long long PL = dh.numel() / C;
  const int ntab = mode == 0 ? NT : NT_CA;
  TORCH_CHECK(PL < (1ll << 31) && codes.numel() == PL * 4 && dx.numel() == dh.numel() && tbl.size(0) == ntab && tbl.size(1) == C &&
                  tbl.scalar_type() == at::kBFloat16 && dh.scalar_type() == at::kBFloat16 &&
                  dx.scalar_type() == at::kBFloat16 && tbl.is_contiguous() && dh.is_contiguous() && dx.is_contiguous() && codes.is_contiguous(),
              "ln_backward_b200: tbl [NT, 256] bf16, dh / dx [.., 256] contiguous bf16, codes [.., 4] int32");
  if (PL == 0) return;
  const unsigned P = (unsigned)PL;
  auto* hp = reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>());
  auto* xp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid = pe100_grid((PL + ch - 1) / ch, cps), block(pe100ln::WARPS * 32);
  const float e = (float)eps;
  if (mode == 0) {
    auto* tp = reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>());
    if (g == 8) pe100ln::ln_bwd<0, __nv_bfloat16, 8><<<grid, block, 0, st>>>(tp, cp, hp, xp, P, e, (int)ch);
    else if (g == 16) pe100ln::ln_bwd<0, __nv_bfloat16, 16><<<grid, block, 0, st>>>(tp, cp, hp, xp, P, e, (int)ch);
    else pe100ln::ln_bwd<0, __nv_bfloat16, 32><<<grid, block, 0, st>>>(tp, cp, hp, xp, P, e, (int)ch);
  } else {
    auto* tp = reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>());
    if (g == 8) pe100ln::ln_bwd<1, __nv_bfloat16, 8><<<grid, block, 0, st>>>(tp, cp, hp, xp, P, e, (int)ch);
    else if (g == 16) pe100ln::ln_bwd<1, __nv_bfloat16, 16><<<grid, block, 0, st>>>(tp, cp, hp, xp, P, e, (int)ch);
    else pe100ln::ln_bwd<1, __nv_bfloat16, 32><<<grid, block, 0, st>>>(tp, cp, hp, xp, P, e, (int)ch);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The weight gradient dw [256, NTAB] fp32 (the weight's layout; every entry the bf16 rounding of its fp32 sum, as the bf16 table gradient) from
// d pair = g [B, n, n, 256] bf16 and the codes: the tcgen05 table gradient over nblk pair slices (tpart [nblk][NTAB][256] fp32 scratch) and
// its reduce. mode 0: encoder (317 rows), 1: decoder (157).
void backward_w_b200(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dw, int64_t mode, int64_t nblk, int64_t rv,
                     torch::Tensor bar) {
  const at::cuda::CUDAGuard guard(g.device());
  const long long P = g.numel() / C;
  TORCH_CHECK(P < (1ll << 31), "pair_embed backward: at most 2^31 pairs");
  const int ntab = mode == 0 ? NT : NT_CA;
  TORCH_CHECK(nblk >= 1 && tpart.numel() >= (long long)nblk * ntab * C && dw.numel() == (long long)ntab * C && dw.scalar_type() == at::kFloat &&
                  g.scalar_type() == at::kBFloat16 && g.is_contiguous() && codes.numel() == P * 4,
              "pair_embed backward: g contiguous bf16, codes [.., 4], tpart [nblk][NTAB][256], dw [256, NTAB] fp32");
  auto st = at::cuda::getCurrentCUDAStream();
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* tp = tpart.data_ptr<float>();
  // rv 2: the reduce inside the table-gradient launch (pe100::launch_fr; at most one CTA per SM), else a reduce launch: 0 reduce_parts_t, 1 _t2
  if (rv == 2 && 2 * nblk <= at::cuda::getCurrentDeviceProperties()->multiProcessorCount) {
    TORCH_CHECK(bar.numel() >= 2 && bar.scalar_type() == at::kInt, "pair_embed backward: bar, 2 zeroed int32 words");
    auto* bp = reinterpret_cast<unsigned*>(bar.data_ptr<int>());
    if (mode == 0) pe100::launch_fr(gp, P, cp, P, GeomRows<0>{}, ntab, 2, 160, tp, dw.data_ptr<float>(), bp, st, (int)nblk);
    else pe100::launch_fr(gp, P, cp, P, GeomRows<1>{}, ntab, 1, 160, tp, dw.data_ptr<float>(), bp, st, (int)nblk);
    return;
  }
  if (mode == 0) pe100::launch(gp, P, cp, P, 0, GeomRows<0>{}, ntab, 2, 160, tp, st, (int)nblk);
  else pe100::launch(gp, P, cp, P, 0, GeomRows<1>{}, ntab, 1, 160, tp, st, (int)nblk);
  if (rv == 1) pe100::reduce_parts_t2<true><<<(ntab * C / 2 + 255) / 256, 256, 0, st>>>(tp, dw.data_ptr<float>(), (int)nblk, ntab);
  else pe100::reduce_parts_t<true><<<(ntab * C / 4 + 255) / 256, 256, 0, st>>>(tp, dw.data_ptr<float>(), (int)nblk, ntab);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
