// apb_pair_fwd_rows_sm100.cuh -- B200 only (included by apb_rows_sm100.cu after the shared apb_rows.cu; no other build reads it).
// pair_bias_fwd2_k (2026-11-17, v21): the shared d_pair 128 inference pair -> bias pass (apb_rows.cu pair_bias_fwd_k: the same tiles,
// mma.sync fragments, LN fold, row statistics and stores, so the same bits) with the tile loads spread over warp 0's lanes. The shared
// kernel issues ONE 32 KB bulk copy per tile from thread 0; on the B200 one issuing thread starts about one bulk copy per ~600 clocks
// whatever its size, and a warp waits for the whole tile. Here lane w of warp 0 copies warp w's 16 rows (4 KB) into the stage onto its
// own mbarrier (PERW), so each warp starts on its rows as soon as they land; NSF2 stages; blocks per SM from the occupancy. Measured (standalone, L2 flushed, 2026-11-17): 2 stages x 2 blocks per SM
// 0-2 us under the shared pass at L 384-1024 (24 heads 2-5 us), the same bits; 4 / 6 stages in one block per SM 10-40 % slower; the
// loads alone (no mma, no stores) take 28.2 us for 105 MB at 2 x 2 x 32 KB in flight and 28.7 us at 6 x 32 KB: the pass sits at the
// card's read rate for this access pattern (~3.7 TB/s), not at an issue or in-flight limit.
#pragma once
namespace {
// HINT (v22): 0 plain bulk copy, 1 L2::cache_hint evict_first (createpolicy fractional 1.0). Measured (standalone, 16 x 128 L 640 / 1024,
// after the measurer's 256 MB zero fill): 30.8 / 61.5 -> 28.7 / 57.4 us; evict_last / evict_normal level with plain; after a read flush (clean L2)
// all forms 28.7 / 54-55 us: the dirty lines the flush leaves cost the plain pass ~2-4 us there, and evict_first gets most of it back
// (the read-once lines become the first victims). The loads alone (a probe build, no mma / stores): 28.7 -> 24.6 us (clean L2: 22.5).
template <int HINT>
__device__ __forceinline__ void bulk_g2s_h(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  if constexpr (HINT == 0) {
    bulk_g2s(dst, src, bytes, b);
  } else {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(b)), "l"(pol) : "memory");
  }
}
template <int NH, int NSF2>
constexpr int fwd2_smem() { return NSF2 * fwd_tile<DP>() + 2 * nhp<NH>() * (128 + 4) * 4 + NSF2 * FW * 8; }
template <int NH, int NSF2, int HINT = 0>
__global__ void __launch_bounds__(FNT) pair_bias_fwd2_k(const bf* __restrict__ Z, const bf* __restrict__ W, const bool* __restrict__ MASK,
                                                       bf* __restrict__ OUT, int L, float eps, float neg) {
  constexpr int NT8 = nhp<NH>() / 8, NHP = nhp<NH>(), FT = fwd_tile<DP>(), UU = DP / 32, TR = 128, TC = TR / 8;
  extern __shared__ __align__(128) uint8_t smem[];
  float (*out_s)[NHP][128 + 4] = reinterpret_cast<float (*)[NHP][128 + 4]>(smem + NSF2 * FT);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NSF2 * FT + 2 * NHP * (128 + 4) * 4);   // [stage][warp]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + TR - 1) / TR;
  const long ntiles = (long)L * ntj;
  __shared__ int klen_s;
  const bool skip_dead = MASK != nullptr;
  auto dead_tile = [&](long n, int j0) { return skip_dead && n >= NSF2 && j0 >= klen_s; };
  auto issue = [&](long n) {                                                // warp 0, lane w < FW: warp w's rows of the n-th tile
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles || lane >= FW) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TR, rows = min(TR, L - j0), rb = lane * 16;
    uint64_t* fb = &full[(n % NSF2) * FW + lane];
    if (dead_tile(n, j0) || rb >= rows) { mbar_expect_tx(fb, 0); return; }
    mbar_expect_tx(fb, 16u * DP * 2);
    bulk_g2s_h<HINT>(smem + (n % NSF2) * FT + rb * DP * 2, Z + ((long)i * L + j0 + rb) * DP, 16u * DP * 2, fb);
  };
  if (threadIdx.x < NSF2 * FW) mbar_init(&full[threadIdx.x], 1);
  if (threadIdx.x == 0) mbar_fence_init();
  __syncthreads();
  if (warp == 0)
    for (int k = 0; k < NSF2; ++k) issue(k);
  if (skip_dead && warp == FW - 1) {
    const int kl = warp_key_len(reinterpret_cast<const uint8_t*>(MASK), L);
    if (lane == 0) klen_s = kl;
  }
  uint32_t b[NT8][DP / 16][2];
  float sw0[NT8], sw1[NT8];
  wf_frags<NT8, NH, DP>(W, b, sw0, sw1);
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TR, rows = min(TR, L - j0), st = (int)(n % NSF2);
    const bool* mrow = MASK;
    const bool dead = dead_tile(n, j0);
    const uint8_t* tile = smem + st * FT;
    float (*os)[128 + 4] = out_s[n & 1];
    const int rb = warp * 16;
    if (!dead && rb < rows) {
      mbar_wait(&full[st * FW + warp], (uint32_t)((n / NSF2) & 1));
      float c[NT8][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
      uint4 r0[UU], r1[UU];
#pragma unroll
      for (int u = 0; u < UU; ++u) {
        r0[u] = *reinterpret_cast<const uint4*>(tile + (rb + g) * DP * 2 + 64 * u + 16 * q);
        r1[u] = *reinterpret_cast<const uint4*>(tile + (rb + g + 8) * DP * 2 + 64 * u + 16 * q);
      }
#pragma unroll
      for (int u = 0; u < UU; ++u) {
        const uint32_t u0[4] = {r0[u].x, r0[u].y, r0[u].z, r0[u].w}, u1[4] = {r1[u].x, r1[u].y, r1[u].z, r1[u].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt)
            mma16816(c[nt], u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], b[nt][2 * u + h][0], b[nt][2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float m0, rs0, m1, rs1;
      gram_stats<DP>(g0, g1, cs, eps, m0, rs0, m1, rs1);
      const int jl = rb + g;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const int h0 = 2 * q + 8 * nt;
        os[h0][jl] = rs0 * (c[nt][0] - m0 * sw0[nt]); os[h0 + 1][jl] = rs0 * (c[nt][1] - m0 * sw1[nt]);
        os[h0][jl + 8] = rs1 * (c[nt][2] - m1 * sw0[nt]); os[h0 + 1][jl + 8] = rs1 * (c[nt][3] - m1 * sw1[nt]);
      }
    } else {
      mbar_wait(&full[st * FW + warp], (uint32_t)((n / NSF2) & 1));         // the phase (an empty one) completes before the stage is reused
    }
    __syncthreads();                                                        // the stage is read; out_s[n & 1] written
    if (warp == 0) issue(n + NSF2);
    for (int task = threadIdx.x; task < NH * TC; task += FNT) {             // (head, 8 keys) per task: 16-B stores
      const int h = task / TC, ch = task % TC, j = j0 + ch * 8;
      if (ch * 8 >= rows) continue;
      if (dead) {
        const uint32_t nn = pack2(neg, neg);
        *reinterpret_cast<uint4*>(OUT + ((long)h * L + i) * L + j) = make_uint4(nn, nn, nn, nn);
        continue;
      }
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float a = os[h][ch * 8 + 2 * e], c1 = os[h][ch * 8 + 2 * e + 1];
        if (mrow) { if (!mrow[j + 2 * e]) a = neg; if (!mrow[j + 2 * e + 1]) c1 = neg; }
        o[e] = pack2(a, c1);
      }
      *reinterpret_cast<uint4*>(OUT + ((long)h * L + i) * L + j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}
template <int NH, int NSF2, int HINT = 0>
void pair_bias_fwd2_t(const at::Tensor& z, const at::Tensor& w, const c10::optional<at::Tensor>& mask, at::Tensor& out, int64_t L, double eps,
                      double neg, int64_t bps) {
  static int occ = 0;
  if (!occ) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_fwd2_k<NH, NSF2, HINT>, cudaFuncAttributeMaxDynamicSharedMemorySize, fwd2_smem<NH, NSF2>()));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, pair_bias_fwd2_k<NH, NSF2, HINT>, FNT, fwd2_smem<NH, NSF2>()));
  }
  const long per_sm = bps > 0 ? std::min<long>(bps, occ) : occ;
  const long ntiles = L * ((L + 127) / 128);
  pair_bias_fwd2_k<NH, NSF2, HINT><<<(unsigned)std::min<long>(per_sm * nsm_of(z), ntiles), FNT, fwd2_smem<NH, NSF2>(), S()>>>(
      CP<bf>(z), CP<bf>(w), mask ? CP<bool>(*mask) : nullptr, P<bf>(out), (int)L, (float)eps, (float)neg);
}
// pair_bias_fwd's d_pair 128 one-pair inference form (no stats / unmasked bias); ``stages`` 2 or 3, ``bps`` blocks per SM (0: the occupancy)
void pair_bias_fwd2(at::Tensor z, at::Tensor w, c10::optional<at::Tensor> mask, at::Tensor out, int64_t L, double eps, double neg,
                    int64_t stages, int64_t bps) {
  chk(z, at::kBFloat16, "pair"); chk(w, at::kBFloat16, "wf"); chk(out, at::kBFloat16, "bias");
  const int64_t nh = w.size(0);
  TORCH_CHECK(w.size(1) == DP && L % 16 == 0 && z.numel() == L * L * DP && out.numel() == nh * L * L, "pair_bias_fwd2 shapes");
  TORCH_CHECK(nh == 8 || nh == 12 || nh == 16 || nh == 24, "pair_bias_fwd2: 8, 12, 16 or 24 heads");
  TORCH_CHECK(stages == 2 || stages == 4 || stages == 12, "pair_bias_fwd2: 2 or 4 stages (12: 2 stages, the pair loads L2::evict_first)");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  const at::cuda::CUDAGuard gd(z.device());
#define PF2(NH) (stages == 2 ? pair_bias_fwd2_t<NH, 2>(z, w, mask, out, L, eps, neg, bps) : stages == 12 ? pair_bias_fwd2_t<NH, 2, 1>(z, w, mask, out, L, eps, neg, bps) : pair_bias_fwd2_t<NH, 4>(z, w, mask, out, L, eps, neg, bps))
  if (nh == 8) PF2(8); else if (nh == 12) PF2(12); else if (nh == 16) PF2(16); else PF2(24);
#undef PF2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
