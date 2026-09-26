// Triangle-attention forward for B200 (sm_100a): the cuEquivariance / opt_core `triangle_attention` op, D = 32, bf16 operands,
// fp32 softmax and accumulation, bf16 out -- the op the H100 kernel (opt_core kernels/triattn/cuda_sm90a) serves, on tcgen05 / TMEM.
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]        (no mask)
//
// A CTA task = one (b, h, 128-query tile) x R pair rows (the bias tile is shared by every pair row: R rows reuse each staged bias
// tile, as the H100 kernel's 3-row CTAs do).  Per 128-key tile j and row r (a "sub-step"):
//   MMA warp:      S = Q_r . K_{r,j}^T                (M = 128 q, N = 128 k, K = 32) into a TMEM buffer
//   softmax warps: x = S * scale*log2e + bias * log2e; p = 2^(x - m_r); l_r += p; P (bf16) written over the S buffer in TMEM
//   MMA warp:      O_r += P . V_{r,j}                  (A = P from TMEM, B = V MN-major, N = 32)
// The softmax offset m_r is the row maximum of the first key tile and is never moved afterwards (no running maximum, no O rescale):
// a later logit would have to exceed it by ~100 log2 units to overflow, which a guard counts (FLAGS) -- the H100 kernel's max-free
// softmax, with its offset fixed at the first tile.
// Warps: 0 = TMA producer (Q of the task's rows, K|V per sub-step, the bias tile per key tile), 1 = MMA issue, 2-5 = softmax + epilogue.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

namespace {
constexpr int D = 32, BM = 128, BN = 128, R = 4;
constexpr int QT = BM * D;                                 // Q tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][32], 64B swizzle (8 KiB)
constexpr int BT = BM * BN;                                // bias tile [2 key halves][128 q][64], 128B swizzle, bf16 (32 KiB)
constexpr int NKV = 5, NB = 2;
constexpr int THREADS = 192;
constexpr int SMEM = 1024 + (2 * R * QT + NKV * 2 * KVT + NB * BT) * 2 + 512;
constexpr int COL_S = 0, COL_O = 256;                      // S/P [2][128] | O [2 tasks][R rows][32]
constexpr uint32_t ID_S = idesc_bf16(128, BN, 0, 0);       // A = Q K-major, B = K K-major
constexpr uint32_t ID_O = idesc_bf16(128, D, 0, 1);        // A = P from TMEM, B = V MN-major
constexpr float L2E = 1.4426950408889634f;
static_assert(SMEM <= 232448, "one CTA per SM");

__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__global__ void __launch_bounds__(THREADS, 1) triattn_fwd_sm100(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*H*S][32] bf16, box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*H*S][32]
    int* __restrict__ FLAGS) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 tasks][R][QT]
  __nv_bfloat16* sKV = sQ + 2 * R * QT;                          // [NKV][K | V]
  __nv_bfloat16* sB = sKV + NKV * 2 * KVT;                       // [NB][BT]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sB + NB * BT);
  uint64_t* qf = bars;               // [2]   the task's Q tiles landed
  uint64_t* qe = qf + 2;             // [2]   the task's last QK^T retired
  uint64_t* kvf = qe + 2;            // [NKV]
  uint64_t* kve = kvf + NKV;         // [NKV] the PV GEMM of the stage retired
  uint64_t* bf = kve + NKV;          // [NB]
  uint64_t* be = bf + NB;            // [NB]  count 4: the softmax is done with the bias tile (all R rows)
  uint64_t* sf = be + NB;            // [2]   S of the sub-step ready
  uint64_t* pf = sf + 2;             // [2]   count 4: P written (over S)
  uint64_t* of = pf + 2;             // [2]   the task's O complete
  uint64_t* oe = of + 2;             // [2]   count 4: drained
  uint32_t* tslot = reinterpret_cast<uint32_t*>(oe + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < 2; ++k) { bar_init(qf + k, 1); bar_init(qe + k, 1); bar_init(sf + k, 1); bar_init(pf + k, 4); bar_init(of + k, 1); bar_init(oe + k, 4); }
    for (int k = 0; k < NKV; ++k) { bar_init(kvf + k, 1); bar_init(kve + k, 1); }
    for (int k = 0; k < NB; ++k) { bar_init(bf + k, 1); bar_init(be + k, 4); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  // task t: q tile fastest (the CTAs sharing a row group's K / V run together), then the row group, then (b, h)
  auto decode = [&](int t, int& bh, int& qt, int& g) { qt = t % QTILES; const int r = t / QTILES; g = r % NG; bh = r / NG; };
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H, h = bh % H; return ((b * N + n) * H + h) * S; };   // first (b,n,h) row

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        const int nr = min(R, N - g * R), qb = lt & 1;
        if (lt >= 2) wait(qe + qb, ((lt >> 1) - 1) & 1);
        expect_tx(qf + qb, nr * QT * 2);
        for (int r = 0; r < nr; ++r) load_2d(&qmap, sQ + (qb * R + r) * QT, qf + qb, 0, row0(bh, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1);
          expect_tx(bf + bs, BT * 2);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sB + bs * BT + hh * BM * 64, bf + bs, j * BN + hh * 64, bh * S + qt * BM);
          for (int r = 0; r < nr; ++r, ++x) {
            const int st = x % NKV;
            if (x >= NKV) wait(kve + st, ((x / NKV) - 1) & 1);
            expect_tx(kvf + st, 2 * KVT * 2);
            const int rw = row0(bh, g * R + r) + j * BN;
            load_2d(&kmap, sKV + st * 2 * KVT, kvf + st, 0, rw);
            load_2d(&vmap, sKV + st * 2 * KVT + KVT, kvf + st, 0, rw);
          }
        }
      }
    }
  } else if (warp == 1) {
    // flattened sub-steps: QK^T of sub-step x+1 is issued before the PV of sub-step x, so the softmax of x overlaps it
    int lt = 0, x = 0;
    int pend = -1, pend_st = 0, pend_r = 0, pend_j = 0, pend_ob = 0, pend_last = 0, pend_lt = 0;
    auto pv = [&]() {                            // O_r += P . V for the pending sub-step
      const int b = pend & 1;
      wait(pf + b, (pend >> 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* v = sKV + pend_st * 2 * KVT + KVT;
        const uint32_t o = tmem + COL_O + (pend_ob * R + pend_r) * D;
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks)
          mma_ts(o, tmem + COL_S + b * BN + ks * 8, sdesc(sa(v + ks * 16 * D), KVT * 2, 512, 4), ID_O, (pend_j | ks) ? 1u : 0u);
        mma_commit(kve + pend_st);
        if (pend_last) mma_commit(of + pend_ob);
      }
      __syncwarp();
      pend = -1;
    };
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      const int nr = min(R, N - g * R), qb = lt & 1, ob = lt & 1;
      wait(qf + qb, (lt >> 1) & 1);
      if (lt >= 2) wait(oe + ob, ((lt >> 1) - 1) & 1);   // the task two back has been drained from O buffer ob
      tc_fence_after();
      for (int j = 0; j < NK; ++j) {
        for (int r = 0; r < nr; ++r, ++x) {
          const int st = x % NKV, b = x & 1;
          wait(kvf + st, (x / NKV) & 1);
          tc_fence_after();
          if (elect_one()) {
            const __nv_bfloat16* q = sQ + (qb * R + r) * QT;
            const __nv_bfloat16* k = sKV + st * 2 * KVT;
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks)
              mma_ss(tmem + COL_S + b * BN, desc_k64(q + ks * 16), desc_k64(k + ks * 16), ID_S, ks ? 1u : 0u);
            mma_commit(sf + b);
            if (j == NK - 1 && r == nr - 1) mma_commit(qe + qb);
          }
          __syncwarp();
          if (pend >= 0) pv();
          pend = x; pend_st = st; pend_r = r; pend_j = j; pend_ob = ob; pend_last = (j == NK - 1 && r == nr - 1); pend_lt = lt;
        }
      }
    }
    if (pend >= 0) pv();
  } else {
    const int q = warp & 3, row = q * 32 + lane;   // TMEM lane = query row of the tile
    int lt = 0, x = 0, jb = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      const int nr = min(R, N - g * R), ob = lt & 1;
      float m[R], l[R];
#pragma unroll
      for (int r = 0; r < R; ++r) { m[r] = 0.f; l[r] = 0.f; }
      for (int j = 0; j < NK; ++j, ++jb) {
        const int bs = jb % NB;
        wait(bf + bs, (jb / NB) & 1);
        const __nv_bfloat16* brow = sB + bs * BT + row * 64;
        for (int r = 0; r < nr; ++r, ++x) {
          const int b = x & 1;
          wait(sf + b, (x >> 1) & 1);
          tc_fence_after();
          const uint32_t sb = tmem_at(tmem + COL_S + b * BN, q * 32, 0);
          float mr = m[r], lr = 0.f;
          if (j == 0) {                          // the row's offset: the first key tile's maximum
            float mx = -INFINITY;
#pragma unroll 1
            for (int c = 0; c < 4; ++c) {
              float s[32];
              tmem_ld32(sb + c * 32, s);
              tmem_wait_ld();
              const __nv_bfloat16* bp = brow + (c >> 1) * BM * 64;
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const uint4 u = *reinterpret_cast<const uint4*>(bp + ((((c & 1) * 4 + e) ^ (row & 7)) << 3));
                const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) {
                  const float2 bb = bf2f(w[k2]);
                  const int i = e * 8 + k2 * 2;
                  mx = fmaxf(mx, fmaxf(fmaf(bb.x, L2E, s[i] * scl), fmaf(bb.y, L2E, s[i + 1] * scl)));
                }
              }
            }
            mr = mx;
          }
#pragma unroll 1
          for (int c = 0; c < 4; ++c) {
            float s[32];
            tmem_ld32(sb + c * 32, s);
            tmem_wait_ld();
            const __nv_bfloat16* bp = brow + (c >> 1) * BM * 64;
            uint32_t pk[16];
            float2 acc = make_float2(0.f, 0.f);
            const float2 sc2 = make_float2(scl, scl), l2 = make_float2(L2E, L2E), nm = make_float2(-mr, -mr);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint4 u = *reinterpret_cast<const uint4*>(bp + ((((c & 1) * 4 + e) ^ (row & 7)) << 3));
              const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
              for (int k2 = 0; k2 < 4; ++k2) {
                const int i = e * 8 + k2 * 2;
                const float2 xx = add2(fma2(bf2f(w[k2]), l2, mul2(make_float2(s[i], s[i + 1]), sc2)), nm);
                const float2 p = make_float2(ex2f(xx.x), ex2f(xx.y));
                acc = add2(acc, p);
                pk[e * 4 + k2] = pack2(p.x, p.y);
              }
            }
            lr += acc.x + acc.y;
            tmem_st8(sb + c * 16, pk);           // P over S: chunk c's 16 columns cover S columns already read
            tmem_st8(sb + c * 16 + 8, pk + 8);
          }
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(pf + b);
          m[r] = mr; l[r] += lr;
        }
        __syncwarp();
        if (lane == 0) arrive(be + bs);
      }
      // ---- epilogue: O_r / l_r -> bf16 -> out ----
      wait(of + ob, (lt >> 1) & 1);
      tc_fence_after();
      int bad = 0;
      for (int r = 0; r < nr; ++r) {
        float o[32];
        tmem_ld32(tmem_at(tmem + COL_O + (ob * R + r) * D, q * 32, 0), o);
        tmem_wait_ld();
        const float inv = 1.f / l[r];
        bad |= !(l[r] > 0.f) || !(l[r] < 3.0e38f);
        uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * D);
#pragma unroll
        for (int c8 = 0; c8 < 4; ++c8) {
          uint4 w;
          w.x = pack2(o[c8 * 8 + 0] * inv, o[c8 * 8 + 1] * inv); w.y = pack2(o[c8 * 8 + 2] * inv, o[c8 * 8 + 3] * inv);
          w.z = pack2(o[c8 * 8 + 4] * inv, o[c8 * 8 + 5] * inv); w.w = pack2(o[c8 * 8 + 6] * inv, o[c8 * 8 + 7] * inv);
          dst[c8] = w;
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(oe + ob);
      if (bad) atomicAdd(FLAGS, 1);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

// q, k, v [B, N, H, S, 32] bf16 contiguous; bias [B, 1, H, S, S] (any float dtype; staged to bf16); S % 128 == 0; no mask.
// -> (out [B, N, H, S, 32] bf16, flags int32 [1]: rows whose softmax offset overflowed -- 0 on model data)
std::vector<torch::Tensor> triattn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, double scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, H, S, 32] bf16");
  for (const auto* t : {&q, &k, &v}) TORCH_CHECK(t->is_contiguous() && t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16, "q, k, v: contiguous, same shape");
  const long B = q.size(0), N = q.size(1), H = q.size(2), S = q.size(3);
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.numel() == B * H * S * S, "bias: [B, 1, H, S, S]");
  auto b16 = bias.to(torch::kBFloat16).contiguous();
  auto out = torch::empty_like(q);
  auto flags = torch::zeros({1}, q.options().dtype(torch::kInt32));
  const uint64_t rows = (uint64_t)(B * N * H * S);
  CUtensorMap qm = make_map<2>(q.data_ptr(), {(uint64_t)D, rows}, {(uint64_t)D}, {D, BM}, CU_TENSOR_MAP_SWIZZLE_64B, "q");
  CUtensorMap km = make_map<2>(k.data_ptr(), {(uint64_t)D, rows}, {(uint64_t)D}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_64B, "k");
  CUtensorMap vm = make_map<2>(v.data_ptr(), {(uint64_t)D, rows}, {(uint64_t)D}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_64B, "v");
  CUtensorMap bm = make_map<2>(b16.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  const int ntasks = (int)(B * H * (S / BM) * ((N + R - 1) / R));
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_fwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  const float scl = (float)scale * L2E;
  triattn_fwd_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)H, (int)S, ntasks, scl, qm, km, vm, bm,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), flags.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, flags};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd", &triattn_fwd, "sm100 triangle-attention forward (D = 32, bf16, no mask)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"));
}
