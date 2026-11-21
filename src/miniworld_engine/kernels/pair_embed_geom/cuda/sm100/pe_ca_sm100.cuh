// pe_ca_sm100.cuh -- B200 (sm_100a) inference forward of the decoder CA pair (PairEmbedCA, 2026-11-16), included by pair_embed_geom_sm100.cu
// after the A100 source (it uses that source's constants, bucket(), table_of() / pg120_table and launch_pdl / pdl_wait).
//
// The op writes B n^2 x 256 bf16: pair p = M_p * (T[sep_p] + T[127 + dbin_p]) (optionally its LayerNorm over the 256 channels, no affine).
// The RTX 5090's one-launch kernel (pg120_fwd_ca_blk: a non-persistent CTA of 32 pairs, every half-warp binning its pair from three dependent
// global loads before its first store) keeps few bytes in flight per SM on the B200, and its LayerNorm costs ~10 shuffles and ~10 scalar
// instructions per channel and pair, which bind the B200 before its HBM does. Here:
//  * a warp owns chunks of 32 consecutive pairs (flat (b, i, j) order, across rows); lane l bins pair 32 q + l (coalesced loads of the j side,
//    broadcast of the i side) and the codes go to the lanes by shuffles;
//  * a group of G lanes owns a pair (32 / G pairs per warp step, U steps interleaved): lane g of the group holds the 16-byte vectors of channels
//    k G 8 + 8 g (k < 256 / (8 G)), so every store instruction writes 8 G contiguous bytes per pair;
//  * persistent warps (grid-stride over the chunks) load the inputs of their next chunk before writing the current one, so the input latency
//    hides behind the stores; the table rows come through the L1 (80 KB, L1 / L2 resident);
//  * packed arithmetic: bf16(T[r0] + T[r1]) as one bf16x2 add per two channels (the correctly rounded sum: what rounding the fp32 sum gives),
//    the LayerNorm in fp32x2 (FADD2 / FFMA2 / FMUL2, sm_100), log2(G) shuffles per reduction.
// LayerNorm: mean of the bf16 sum, then the centred second moment, rsqrtf(var + eps), (x - mean) * rstd; fixed lane order and shuffle tree:
// bitwise reproducible run to run.
#pragma once

namespace pe100ca {

constexpr int WARPS = 4;   // warps per CTA (128 threads)

template <typename IdxT>
struct Raw {
  float xi, yi, zi, xj, yj, zj;
  IdxT ii, ij;
  uint8_t mi, mj;
};

template <typename IdxT>
__device__ __forceinline__ Raw<IdxT> load_raw(const float* __restrict__ ca, const uint8_t* __restrict__ mask, const IdxT* __restrict__ idx, int n, long long p,
                                              long long P) {
  Raw<IdxT> r;
  if (p < P) {
    const long long row = p / n;                   // b * n + i
    const long long b = row / n;
    const long long rj = b * n + (p - row * n);    // b * n + j
    r.mi = __ldg(mask + row);
    r.mj = __ldg(mask + rj);
    r.ii = __ldg(idx + row);
    r.ij = __ldg(idx + rj);
    r.xi = __ldg(ca + row * 3); r.yi = __ldg(ca + row * 3 + 1); r.zi = __ldg(ca + row * 3 + 2);
    r.xj = __ldg(ca + rj * 3); r.yj = __ldg(ca + rj * 3 + 1); r.zj = __ldg(ca + rj * 3 + 2);
  } else {
    r.mi = r.mj = 0;
    r.ii = r.ij = 0;
    r.xi = r.yi = r.zi = r.xj = r.yj = r.zj = 0.f;
  }
  return r;
}

// the pair's code: sep bin | CA-distance bin << 8 | LIVE (0: masked / out of range)
template <typename IdxT>
__device__ __forceinline__ uint32_t code_of(const Raw<IdxT>& r, const float* dlim_s, int nd) {
  if (!(r.mi && r.mj)) return 0u;
  const float dx = r.xi - r.xj, dy = r.yi - r.yj, dz = r.zi - r.zj;
  const float d = sqrtf(dx * dx + dy * dy + dz * dz);
  const long long s = (long long)r.ii - (long long)r.ij + 63;
  return (uint32_t)min(max(s, 0LL), 126LL) | ((uint32_t)bucket(d, dlim_s, nd) << 8) | LIVE;
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
  __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 s = __hadd2(x, y);
  return *reinterpret_cast<uint32_t*>(&s);
}
__device__ __forceinline__ float2 unpack2(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u)); }
__device__ __forceinline__ uint32_t pack2(float2 v) {
  __nv_bfloat162 h = __float22bfloat162_rn(v);
  return *reinterpret_cast<uint32_t*>(&h);
}

// G lanes per pair, U warp steps (of 32 / G pairs) interleaved
template <typename IdxT, int G, int U, bool LN, bool CS, bool CODES = false>
__global__ void __launch_bounds__(WARPS * 32)
fwd_ca(const __nv_bfloat16* __restrict__ tbl, const float* __restrict__ ca, const uint8_t* __restrict__ mask, const IdxT* __restrict__ idx,
       const float* __restrict__ dlim, __nv_bfloat16* __restrict__ out, int n, long long P, int nd, float eps, uint4* __restrict__ codes) {
  constexpr int PPS = 32 / G;          // pairs per warp step
  constexpr int NV = C / (8 * G);      // 16-byte vectors per lane and pair
  constexpr int STEPS = 32 / PPS;      // steps per chunk of 32 pairs (= G)
  static_assert(STEPS % U == 0, "U divides the steps of a chunk");
  __shared__ float dlim_s[32];
  if (threadIdx.x < 32) dlim_s[threadIdx.x] = threadIdx.x < nd ? dlim[threadIdx.x] : 0.f;
  __syncthreads();
  const int lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  const long long nchunk = (P + 31) >> 5, wstride = (long long)gridDim.x * WARPS;
  long long q = (long long)blockIdx.x * WARPS + (threadIdx.x >> 5);
  Raw<IdxT> cur = load_raw(ca, mask, idx, n, q * 32 + lane, P);
  bool waited = false;
  while (q < nchunk) {
    const long long qn = q + wstride;
    const Raw<IdxT> nxt = load_raw(ca, mask, idx, n, qn * 32 + lane, qn < nchunk ? P : 0);   // the next chunk's inputs, in flight meanwhile
    const uint32_t code = code_of(cur, dlim_s, nd);
    if (!waited) { pdl_wait(); waited = true; }               // the table (pg120_table) from here on
    const long long pb = q * 32;
    if constexpr (CODES) {                                     // the training forward: the 16-byte code the backward replays (word 0)
      if (pb + lane < P) codes[pb + lane] = make_uint4(code, 0u, 0u, 0u);
    }
#pragma unroll 1
    for (int t = 0; t < STEPS; t += U) {
      if (pb + t * PPS >= P) break;                            // warp-uniform
      uint32_t x[U][NV][4];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const uint32_t w = __shfl_sync(0xFFFFFFFFu, code, (t + u) * PPS + grp);
#pragma unroll
        for (int k = 0; k < NV; k++) {
          if (w & LIVE) {
            const int c = k * G * 8 + g * 8;
            const uint4 v = trow<true>(tbl, w & 127, c), h = trow<true>(tbl, R_CAD + ((w >> 8) & 31), c);
            x[u][k][0] = hadd2u(v.x, h.x); x[u][k][1] = hadd2u(v.y, h.y); x[u][k][2] = hadd2u(v.z, h.z); x[u][k][3] = hadd2u(v.w, h.w);
          } else {
            x[u][k][0] = x[u][k][1] = x[u][k][2] = x[u][k][3] = 0u;
          }
        }
      }
      if constexpr (LN) {
        float2 d[U][NV][4];
        float s[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
          float2 a = make_float2(0.f, 0.f);
#pragma unroll
          for (int k = 0; k < NV; k++)
#pragma unroll
            for (int e = 0; e < 4; e++) { d[u][k][e] = unpack2(x[u][k][e]); a = __fadd2_rn(a, d[u][k][e]); }
          s[u] = a.x + a.y;
        }
#pragma unroll
        for (int o = G / 2; o; o >>= 1)
#pragma unroll
          for (int u = 0; u < U; u++) s[u] += __shfl_xor_sync(0xFFFFFFFFu, s[u], o);
#pragma unroll
        for (int u = 0; u < U; u++) {
          const float m = s[u] * (1.f / C);
          const float2 nm = make_float2(-m, -m);
          float2 a = make_float2(0.f, 0.f);
#pragma unroll
          for (int k = 0; k < NV; k++)
#pragma unroll
            for (int e = 0; e < 4; e++) { d[u][k][e] = __fadd2_rn(d[u][k][e], nm); a = __ffma2_rn(d[u][k][e], d[u][k][e], a); }
          s[u] = a.x + a.y;
        }
#pragma unroll
        for (int o = G / 2; o; o >>= 1)
#pragma unroll
          for (int u = 0; u < U; u++) s[u] += __shfl_xor_sync(0xFFFFFFFFu, s[u], o);
#pragma unroll
        for (int u = 0; u < U; u++) {
          const float r = rsqrtf(s[u] * (1.f / C) + eps);
          const float2 r2 = make_float2(r, r);
#pragma unroll
          for (int k = 0; k < NV; k++)
#pragma unroll
            for (int e = 0; e < 4; e++) x[u][k][e] = pack2(__fmul2_rn(d[u][k][e], r2));
        }
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        const long long p = pb + (t + u) * PPS + grp;
        if (p < P) {
#pragma unroll
          for (int k = 0; k < NV; k++) {
            const uint4 w = make_uint4(x[u][k][0], x[u][k][1], x[u][k][2], x[u][k][3]);
            uint4* dst = reinterpret_cast<uint4*>(out + (size_t)p * C + k * G * 8 + g * 8);
            if constexpr (CS) __stcs(dst, w);
            else *dst = w;
          }
        }
      }
    }
    cur = nxt;
    q = qn;
  }
  if (!waited) pdl_wait();
}

}  // namespace pe100ca

// B200 inference: the decoder pair [B, n, n, 256] bf16 (its LayerNorm when ln) from the module's weight [256, 157] (fp32 / bf16) or a bf16
// table [157, 256], CA [B, n, 3] fp32, mask [B, n] bytes, idx [B, n] int32 / int64, the bin limits (<= 31); cps: CTAs (of 128 threads) per SM
// at most. The launch: 8 lanes a pair, one warp step at a time, streaming stores (measured 2026-11-16, measurer timing, cps 8 / 16 / 32 x
// lanes 8 / 16 / 32 x U 1 / 2 / 4 x cs 0 / 1 over S1 / S5 x L128 / 256 / 384 with and without the LayerNorm: the best or within 2 us
// everywhere; 32 lanes a pair 1.1-1.5x slower with the LayerNorm). Up to CA_SMALL pairs four warp steps interleaved (S1 L128: 12.1 vs 14.2 us;
// from S5 L128 / S1 L256 on one step is as fast or 2 us faster with the LayerNorm).
constexpr int CA_LANES = 8;
constexpr long long CA_SMALL = 128 * 128;
torch::Tensor forward_ca_b200(torch::Tensor tbl, torch::Tensor ca, torch::Tensor mask, torch::Tensor idx, torch::Tensor dlim, torch::Tensor out, torch::Tensor codes, int64_t ln,
                     double eps, int64_t cps) {
  const at::cuda::CUDAGuard guard(ca.device());
  TORCH_CHECK(dlim.numel() <= 31, "at most 31 bin limits");
  TORCH_CHECK(mask.element_size() == 1 && mask.is_contiguous() && idx.is_contiguous() && ca.is_contiguous() && ca.scalar_type() == at::kFloat,
              "mask: contiguous bool / uint8; idx: contiguous; ca: contiguous fp32");
  const int B = idx.size(0), n = idx.size(1);
  const long long P = (long long)B * n * n;
  if (P == 0) return tbl;
  auto st = at::cuda::getCurrentCUDAStream();
  bool pdl = false;
  const torch::Tensor t = table_of(tbl, NT_CA, st, pdl);
  const auto* tp = reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
  const auto* mp = static_cast<const uint8_t*>(mask.data_ptr());
  const float* xp = ca.data_ptr<float>();
  const float* lp = dlim.data_ptr<float>();
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long long chunks = (P + 31) / 32;
  const long long ctas = std::max<long long>(1, std::min<long long>((long long)sms * std::max<int64_t>(1, cps), (chunks + pe100ca::WARPS - 1) / pe100ca::WARPS));
  const dim3 grid((unsigned)ctas), block(pe100ca::WARPS * 32);
  const int nd = (int)dlim.numel();
  const float e = (float)eps;
  const bool wc = codes.numel() > 0;                           // training: the codes [B, n, n, 4] int32 too
  TORCH_CHECK(!wc || (codes.numel() == P * 4 && codes.scalar_type() == at::kInt && codes.is_contiguous()), "codes: contiguous int32 [B, n, n, 4]");
  uint4* cp = wc ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr;
  auto go = [&](auto ip) {
    using I = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    const bool small = P <= CA_SMALL;
#define PE_CA(U, LN, CO) launch_pdl(pe100ca::fwd_ca<I, CA_LANES, U, LN, true, CO>, grid, block, st, pdl, tp, xp, mp, ip, lp, op, n, P, nd, e, cp)
    if (wc) {
      if (ln && small) PE_CA(4, true, true);
      else if (ln) PE_CA(1, true, true);
      else if (small) PE_CA(4, false, true);
      else PE_CA(1, false, true);
    } else {
      if (ln && small) PE_CA(4, true, false);
      else if (ln) PE_CA(1, true, false);
      else if (small) PE_CA(4, false, false);
      else PE_CA(1, false, false);
    }
#undef PE_CA
  };
  if (idx.scalar_type() == at::kLong) go(reinterpret_cast<const long long*>(idx.data_ptr<int64_t>()));
  else {
    TORCH_CHECK(idx.scalar_type() == at::kInt, "idx: int32 or int64");
    go(static_cast<const int*>(idx.data_ptr<int>()));
  }
  return t;
}
