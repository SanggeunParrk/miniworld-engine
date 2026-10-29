// Triangle-attention forward for B200 (sm_100a) with 64-channel heads: Chai-1's trunk and confidence pair attention (4 x 64 at d_pair 256),
// RoseTTAFold3's template stack (4 x 64 at d_pair 64), IntelliFold-2's trunk (8 x 64 at d_pair 512).  bf16 operands, fp32 softmax and
// accumulation, bf16 out; inference (no LSE):
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]        (masked keys: bias = bf16 min)
//
// The head-dim-32 core's schedule (triattn_sm100.cu) on the operands of a 64-channel head, as its own kernel (the d_pair-128 core stays
// as it is): a CTA task = (b, h, 128-query tile) x R = 2 pair rows; per 128-key tile j and row r:
//   MMA warp:      S = Q_r . K_{r,j}^T     (M = 128, N = 128, K = 64: four k-steps on 128-byte-swizzled [128][64] tiles)
//   softmax warps: x = S * scale*log2e + bias * log2e; p = 2^(x - m_r) (m_r the running maximum, raised lazily past TA_TAU); P over S
//   MMA warp:      O_r += P . V_{r,j}       (A = P from TMEM, B = V MN-major [128 k][64], N = 64)
// TMEM: three 128-column S / P buffers + O of the two rows (2 x 64) = 512 columns.  Shared memory: Q of two tasks (2 x 2 x 16 KiB), three
// K | V stages (3 x 32 KiB), two bias tiles (2 x 32 KiB) and the row sums: 226.5 KiB, so the dynamic base carries no alignment slack (the
// kernel traps if it is not 1024-aligned, as tri_wfront does).
// Warps: 0 = TMA, 1 = MMA, 2-5 / 6-9 = the two softmax groups (row 0 / row 1 of the task), 10-13 = epilogue (O / l -> bf16 out).
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

#ifndef TA_TAU
#define TA_TAU 8.f       // a chunk raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
namespace {
constexpr int D = 64, BM = 128, BN = 128, R = 2;
constexpr int QT = BM * D;                                 // Q tile [128 q][64], 128B swizzle (16 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][64], 128B swizzle (16 KiB)
constexpr int BT = BM * BN;                                // bias tile [2 key halves][128 q][64], 128B swizzle, bf16 (32 KiB)
constexpr int NKV = 3, NB = 2, NS = 3;                     // K | V stages, bias stages, S / P TMEM buffers (NKV >= NS: QK^T runs NS - 1 ahead)
constexpr int THREADS = 448;
constexpr int SMEM = (2 * R * QT + NKV * 2 * KVT + NB * BT) * 2 + 2 * R * BM * 4 + 512;
constexpr int COL_S = 0, COL_O = NS * BN;                  // S/P [NS][128] | O [R rows][64]
static_assert(COL_O + R * D <= 512, "TMEM");
static_assert(NKV >= NS, "the MMA warp's QK^T look-ahead needs a K | V stage each");
constexpr uint32_t ID_S = idesc_bf16(128, BN, 0, 0);       // A = Q K-major, B = K K-major
constexpr uint32_t ID_O = idesc_bf16(128, D, 0, 1);        // A = P from TMEM, B = V MN-major
constexpr float L2E = 1.4426950408889634f;
static_assert(SMEM <= 232448, "one CTA per SM");

__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__global__ void __launch_bounds__(THREADS, 1) triattn_fwd64_sm100(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][H*64] bf16, box (64, 128) at column h*64, 128B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*64]
    int* __restrict__ FLAGS) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();                         // no alignment slack (see the header)
  unsigned char* smb = raw;
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 tasks][R][QT]
  __nv_bfloat16* sKV = sQ + 2 * R * QT;                          // [NKV][K | V]
  __nv_bfloat16* sB = sKV + NKV * 2 * KVT;                       // [NB][BT]
  float* sL = reinterpret_cast<float*>(sB + NB * BT);            // [2 tasks][R][128] row sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sL + 2 * R * BM);
  uint64_t* qf = bars;               // [2]   the task's Q tiles landed
  uint64_t* qe = qf + 2;             // [2]   the task's last QK^T retired
  uint64_t* kvf = qe + 2;            // [NKV]
  uint64_t* kve = kvf + NKV;         // [NKV] the PV GEMM of the stage retired
  uint64_t* bf = kve + NKV;          // [NB]
  uint64_t* be = bf + NB;            // [NB]  count 8: both softmax groups are done with the bias tile
  uint64_t* sf = be + NB;            // [NS]  S of the sub-step ready
  uint64_t* pf = sf + NS;            // [NS]  count 4: P written (over S)
  uint64_t* of = pf + NS;            // [1]   the task's O complete
  uint64_t* oe = of + 1;             // [1]   count 4: the epilogue drained O
  uint64_t* lf = oe + 1;             // [2]   count 8: both groups wrote the task's row sums
  uint32_t* tslot = reinterpret_cast<uint32_t*>(lf + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < 2; ++k) { bar_init(qf + k, 1); bar_init(qe + k, 1); }
    for (int k = 0; k < NS; ++k) { bar_init(sf + k, 1); bar_init(pf + k, 4); }
    bar_init(of, 1); bar_init(oe, 4); bar_init(lf, 8); bar_init(lf + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(kvf + k, 1); bar_init(kve + k, 1); }
    for (int k = 0; k < NB; ++k) { bar_init(bf + k, 1); bar_init(be + k, 8); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  // task t: q tile fastest (the CTAs sharing a row group's K / V run together), then the row group, then (b, h)
  auto decode = [&](int t, int& bh, int& qt, int& g) { qt = t % QTILES; const int r = t / QTILES; g = r % NG; bh = r / NG; };
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H; n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row
  auto hcol = [&](int bh) -> int { return (bh % H) * D; };

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        const int qb = lt & 1;
        if (lt >= 2) wait(qe + qb, ((lt >> 1) - 1) & 1);
        expect_tx(qf + qb, R * QT * 2);
        for (int r = 0; r < R; ++r) load_2d(&qmap, sQ + (qb * R + r) * QT, qf + qb, hcol(bh), row0(bh, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1);
          expect_tx(bf + bs, BT * 2);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sB + bs * BT + hh * BM * 64, bf + bs, j * BN + hh * 64, bh * S + qt * BM);
          for (int r = 0; r < R; ++r, ++x) {
            const int st = x % NKV;
            if (x >= NKV) wait(kve + st, ((x / NKV) - 1) & 1);
            expect_tx(kvf + st, 2 * KVT * 2);
            const int rw = row0(bh, g * R + r) + j * BN;
            load_2d(&kmap, sKV + st * 2 * KVT, kvf + st, hcol(bh), rw);
            load_2d(&vmap, sKV + st * 2 * KVT + KVT, kvf + st, hcol(bh), rw);
          }
        }
      }
    }
  } else if (warp == 1) {
    // flattened sub-steps y (task k = y / SPT, key tile j, row r): QK^T runs up to NS - 1 sub-steps ahead of the PV GEMMs
    const int SPT = NK * R;
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    auto qk = [&](int y) {
      const int k = y / SPT, w = y % SPT, r = w % R, st = y % NKV, b = y % NS, qb = k & 1;
      if (w == 0) wait(qf + qb, (k >> 1) & 1);
      wait(kvf + st, (y / NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* q = sQ + (qb * R + r) * QT;
        const __nv_bfloat16* kk = sKV + st * 2 * KVT;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + COL_S + b * BN, desc_k128(q + ks * 16), desc_k128(kk + ks * 16), ID_S, ks ? 1u : 0u);
        mma_commit(sf + b);
        if (w == SPT - 1) mma_commit(qe + qb);
      }
      __syncwarp();
    };
    auto pv = [&](int y) {                       // O_r += P . V
      const int k = y / SPT, w = y % SPT, j = w / R, r = w % R, st = y % NKV, b = y % NS;
      wait(pf + b, (y / NS) & 1);
      if (w == 0 && k >= 1) wait(oe, (k - 1) & 1);   // the previous task's O was drained
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* v = sKV + st * 2 * KVT + KVT;
        const uint32_t o = tmem + COL_O + r * D;
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks)
          mma_ts(o, tmem + COL_S + b * BN + ks * 8, desc_mn128(v + ks * 16 * D, KVT * 2), ID_O, (j | ks) ? 1u : 0u);
        mma_commit(kve + st);
        if (w == SPT - 1) mma_commit(of);
      }
      __syncwarp();
    };
    int y = 0;
    for (int p = 0; p < total; ++p) {
      while (y < total && y < p + NS) qk(y++);   // S buffer y % NS is free once PV(y - NS) is issued (the tensor pipe runs in order)
      pv(p);
    }
  } else if (warp < 10) {
    const int q = warp & 3, row = q * 32 + lane;   // TMEM lane = query row of the tile
    const int r = (warp - 2) >> 2;                 // this group's row of the task
    int lt = 0, x = 0, jb = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      float m = -INFINITY, l = 0.f;                // no finite logit seen yet
      for (int j = 0; j < NK; ++j, ++jb) {
        const int bs = jb % NB;
        wait(bf + bs, (jb / NB) & 1);
        const __nv_bfloat16* brow = sB + bs * BT + row * 64;
        const int xs = x + r, b = xs % NS;
        wait(sf + b, (xs / NS) & 1);
        tc_fence_after();
        const uint32_t sb = tmem_at(tmem + COL_S + b * BN, q * 32, 0);
        float mr = m, lr = 0.f;
        const float2 sc2 = make_float2(scl, scl), l2 = make_float2(L2E, L2E);
        auto logits = [&](int c, float* s32, float2 nm0) {   // x = S * scale*log2e + bias * log2e + nm0
          const __nv_bfloat16* bp = brow + (c >> 1) * BM * 64;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint4 u = *reinterpret_cast<const uint4*>(bp + ((((c & 1) * 4 + e) ^ (row & 7)) << 3));
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
              const int i2 = e * 8 + k2 * 2;
              const float2 xx = fma2(bf2f(w[k2]), l2, fma2(make_float2(s32[i2], s32[i2 + 1]), sc2, nm0));
              s32[i2] = xx.x; s32[i2 + 1] = xx.y;
            }
          }
        };
        {
          float2 acc = make_float2(0.f, 0.f);
          float s32[2][32];
          float lfall = 0.f, lfc[4];
          tmem_ld32(sb, s32[0]);
          tmem_wait_ld();
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if (c + 1 < 4) tmem_ld32(sb + (c + 1) * 32, s32[(c + 1) & 1]);
            float* sc = s32[c & 1];
            const bool have = mr > -INFINITY;                     // -inf until a finite logit was seen
            const float nm = have ? -mr : 0.f;
            logits(c, sc, make_float2(nm, nm));
            float c0 = sc[0], c1 = sc[1];
#pragma unroll
            for (int i2 = 2; i2 < 32; i2 += 2) { c0 = fmaxf(c0, sc[i2]); c1 = fmaxf(c1, sc[i2 + 1]); }
            const float cm = fmaxf(c0, c1);                        // the chunk's maximum, relative to the offset (absolute if !have)
            const bool up = have ? cm > TA_TAU : cm > -1e30f;
            if (up) {                                              // raise the offset: this chunk and the running sum here; the
              if (have) {                                          // P chunks already in TMEM and O_r after the loop (lfall)
                const float f = ex2f(-cm);
                acc.x *= f; acc.y *= f;
                lfall -= cm;
              }
              mr = have ? mr + cm : cm;
#pragma unroll
              for (int i2 = 0; i2 < 32; ++i2) sc[i2] -= cm;
            }
            lfc[c] = lfall;
            uint32_t pk[16];
#pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) {
              const float2 pv = make_float2(ex2f(sc[k2 * 2]), ex2f(sc[k2 * 2 + 1]));
              acc = add2(acc, pv);
              pk[k2] = pack2(pv.x, pv.y);
            }
            tmem_wait_ld();                    // chunk c+1 landed (and every S column chunk c's P covers was read)
            tmem_st8(sb + c * 16, pk);         // P over S: chunk c's 16 columns cover S columns already read
            tmem_st8(sb + c * 16 + 8, pk + 8);
          }
          lr = acc.x + acc.y;
          if (__any_sync(0xffffffffu, lfall != 0.f)) {           // rare, warp-uniform (tcgen05.ld / st are warp-collective)
            // the tile's P chunks written before a raise, and O_r (previous tiles) and l_r, to the final offset. PV(previous tile,
            // row r) = sub-step xs - R: waited on through its K/V stage barrier (R <= NS)
            tmem_wait_st();
#pragma unroll
            for (int c2 = 0; c2 < 3; ++c2) {
              const float f = ex2f(lfall - lfc[c2]);
              float pw[16];
              tmem_ld16(sb + c2 * 16, pw);
              tmem_wait_ld();
              uint32_t pq[16];
#pragma unroll
              for (int k2 = 0; k2 < 16; ++k2) {
                const float2 pp = bf2f(__float_as_uint(pw[k2]));
                pq[k2] = pack2(pp.x * f, pp.y * f);
              }
              tmem_st8(sb + c2 * 16, pq);
              tmem_st8(sb + c2 * 16 + 8, pq + 8);
            }
            const float fo = ex2f(lfall);
            if (j > 0) {
              const int yp = xs - R;
              wait(kve + yp % NKV, (yp / NKV) & 1);
              tc_fence_after();
#pragma unroll
              for (int hh = 0; hh < D / 32; ++hh) {
                const uint32_t oa = tmem_at(tmem + COL_O + r * D + hh * 32, q * 32, 0);
                float ov[32];
                tmem_ld32(oa, ov);
                tmem_wait_ld();
                uint32_t oq[32];
#pragma unroll
                for (int k2 = 0; k2 < 32; ++k2) oq[k2] = __float_as_uint(ov[k2] * fo);
#pragma unroll
                for (int k2 = 0; k2 < 32; k2 += 8) tmem_st8(oa + k2, oq + k2);
              }
            }
            l *= fo;
          }
        }
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(pf + b);
        m = mr; l += lr;
        x += R;
        __syncwarp();
        if (lane == 0) arrive(be + bs);
      }
      sL[((lt & 1) * R + r) * BM + row] = l;      // hand the row sum to the epilogue warps
      __syncwarp();
      if (lane == 0) arrive(lf + (lt & 1));
    }
  } else if (warp < 14) {
    // ---- epilogue warps: O_r / l_r -> bf16 -> out, overlapping the next task's softmax ----
    const int q = warp & 3, row = q * 32 + lane;
    int lt = 0, bad = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      wait(of, lt & 1);
      wait(lf + (lt & 1), (lt >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float lr = sL[((lt & 1) * R + r) * BM + row];
        const float inv = 1.f / lr;
        bad |= !(lr > 0.f) || !(lr < 3.0e38f);
        const bool store = g * R + r < N;          // else a padding row
        uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (H * D) + hcol(bh));
#pragma unroll
        for (int hh = 0; hh < D / 32; ++hh) {
          float o[32];
          tmem_ld32(tmem_at(tmem + COL_O + r * D + hh * 32, q * 32, 0), o);
          tmem_wait_ld();
          if (r == R - 1 && hh == D / 32 - 1) {    // O is free for the next task's PV GEMMs
            tc_fence_before();
            __syncwarp();
            if (lane == 0) arrive(oe);
          }
          if (store) {
#pragma unroll
            for (int c8 = 0; c8 < 4; ++c8) {
              uint4 w;
              w.x = pack2(o[c8 * 8 + 0] * inv, o[c8 * 8 + 1] * inv); w.y = pack2(o[c8 * 8 + 2] * inv, o[c8 * 8 + 3] * inv);
              w.z = pack2(o[c8 * 8 + 4] * inv, o[c8 * 8 + 5] * inv); w.w = pack2(o[c8 * 8 + 6] * inv, o[c8 * 8 + 7] * inv);
              dst[hh * 4 + c8] = w;
            }
          }
        }
      }
    }
    if (bad) atomicAdd(FLAGS, 1);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

// q, k, v [B, N, S, H, 64] bf16 (rows (b, n, s) of all heads at one common stride, e.g. column slices of one projection output, or
// contiguous); bias [B, H, S, S] bf16 (masked keys already filled); S % 128 == 0.
// -> (out [B, N, S, H, 64] bf16 contiguous, flags int32 [1]: rows whose sum left (0, 3e38), a per-device counter never re-zeroed)
// out_: a contiguous bf16 tensor like q to write into (a group of samples' part of a batch's out), or empty (fresh)
std::vector<torch::Tensor> triattn_fwd64(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, double scale,
                                         torch::Tensor out_) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 64] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld,
                "q, k, v: same shape, [.., H, 64] dense per row, rows at one common stride");
  TORCH_CHECK(ld >= H * D && ld % 8 == 0, "q / k / v row stride: >= H*64 and a multiple of 8 elements");
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous() && bias.numel() == B * H * S * S, "bias: [B, H, S, S] bf16");
  const bool og = out_.defined() && out_.numel() > 0;
  if (og) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == q.numel(), "out: contiguous bf16 like q");
  auto out = og ? out_.view(q.sizes()) : torch::empty(q.sizes(), q.options());
  static torch::Tensor flags_dev[64];
  torch::Tensor& flags = flags_dev[q.device().index()];
  if (!flags.defined()) flags = torch::zeros({1}, q.options().dtype(torch::kInt32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  CUtensorMap qm = make_map<2>(q.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "q");
  CUtensorMap km = make_map<2>(k.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "k");
  CUtensorMap vm = make_map<2>(v.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "v");
  CUtensorMap bm = make_map<2>(bias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  const long nt = B * H * (S / BM) * ((N + R - 1) / R);
  TORCH_CHECK(nt < (1L << 31), "too many tasks for one launch");
  const int ntasks = (int)nt;
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_fwd64_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  triattn_fwd64_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)H, (int)S, ntasks, (float)scale * L2E, qm, km, vm,
      bm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), flags.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, flags};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd64", &triattn_fwd64, "sm100 triangle-attention forward, 64-channel heads (bf16, inference): q/k/v/out [B, N, S, H, 64]",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"), py::arg("out"));
}
