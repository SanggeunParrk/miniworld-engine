// triattn_d64_sm90.cu -- triangle-attention core for heads of 64 channels (RoseTTAFold3's 4 x 64 heads, Chai-1's d_pair 256 trunk /
// confidence heads), inference, sm_90a, wgmma + TMA (raw PTX, sm90.cuh), fp32 logits / softmax / accumulation, P in bf16:
//
//   u[b,n,s,h,:] = sigmoid(g[b,n,s,h,:] (+ gb[h,:])) * softmax_k( scale q[b,n,s,h,:] . k[b,n,k,h,:] + bias[b,h,s,k] ) @ v[b,n,k,h,:]
//
// Operands as the mma.sync core of h100_dh (triattn_dh_sm90.cu): q | k | v | g are column blocks of one projection row (b, n, s) of width
// 4 HD (HD = 64 H) -- q at h 64, k at HD + h 64, v at 2 HD + h 64, g at 3 HD + h 64 --; bias [B, H, S, S] bf16 in natural units (masked
// keys -inf), shared by the N = S pair rows; u [B N S, HD] bf16.
//
// The schedule of triattn_sm90.cu (the D = 32 bf16 core) at head width 64: a CTA task = one (b, h, 128-query tile) x R = 2 pair rows;
// 384 threads: warpgroup 0 = TMA producer (one thread), warpgroups 1 / 2 = consumers of queries 0-63 / 64-127 of both rows. Per 64-key
// tile j the producer stages the bias tile once (shared by the two rows: the mma.sync core reads it once per pair row) and K | V of each
// row. A consumer alternates the rows: the G phase of (j, r) initialises the score accumulator with the bias fragment (ldmatrix: no
// per-logit bias add) and issues S += Q_r K_rj^T (RS wgmma, Q_r in registers for the whole task, pre-multiplied by `scale`: a power of
// two at head width 64, so exact); the PV of the previous sub-step (the other row) is issued; the X phase computes the exponentials of
// (j, r) while that PV runs: p = 2^(S log2e - m_r) with the row's running maximum m_r raised lazily (only when a tile's maximum exceeds
// it by more than TD_TAU log2 units; o_r and l_r rescaled then -- they are idle, the PV in flight belongs to the other row), so p <=
// 2^TD_TAU for any spread of the logits. P rounded to bf16 = the A fragment of [O_r | l_r] += P [V | 1] (m64n72 RS wgmma per 16 keys:
// the row sums come from the tensor core, from the same bf16 P; the ones block sits in each K | V stage after V).
// Key tiles of 64 (the 64 x 64 fp32 score tile and the 64 x 72 accumulators of two rows fit the consumers' registers); KT[b] (inference
// key_tiles) bounds sample b's key tiles: the tiles past its last attended key are -inf in every row (p = 0, m and l unchanged), so
// skipping them gives the same result. A row without an attended key gives u = 0 (as the mma.sync core).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "sm90.cuh"
using namespace sm90;

#ifndef TD_NKV
#define TD_NKV 4          // K | V ring stages (one per (key tile, row))
#endif
#ifndef TD_NB
#define TD_NB 3           // bias ring stages (one per key tile)
#endif
#ifndef TD_TAU
#define TD_TAU 8.f        // a tile raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
#ifndef TD_ORDER1_MINS
#define TD_ORDER1_MINS 2048   // from this S on the CTAs running together work on one head (its bias stays L2-resident across the row groups)
#endif
#ifndef TD_DYN
#define TD_DYN 1          // dynamic task schedule (a global counter)
#endif
namespace {
typedef __nv_bfloat16 bf16;
constexpr int D = 64, BM = 128, BN = 64, R = 2, NKV = TD_NKV, NB = TD_NB;
constexpr int QT = BM * D;                                 // Q tile [128 q][64], 128B swizzle (16 KiB)
constexpr int KVT = BN * D;                                // K or V tile [64 k][64], 128B swizzle (8 KiB)
constexpr int STG = 3 * KVT;                               // a K | V stage: K, V, ones (24 KiB)
constexpr int BT = BM * BN;                                // bias tile [128 q][64 k] bf16, 128B swizzle (16 KiB)
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + (NB * BT + R * QT + NKV * STG) * 2 + 256;
static_assert(SMEM <= 232448, "one CTA per SM");
constexpr float LOG2E = 1.4426950408889634f;

// [O | l] += P [V | 1]: m64n72k16, A = P from registers, B = V (MN-major) with the ones block as columns 64-71
__device__ __forceinline__ void mma_rs_n72(float (&d)[36], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
               "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
               "{%36, %37, %38, %39}, %40, p, 1, 1, 1;\n}\n"
               : SM90_D32(0), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
__device__ __forceinline__ uint32_t mul2(uint32_t x, __nv_bfloat162 s) {
  __nv_bfloat162 v = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&x), s);
  return *reinterpret_cast<uint32_t*>(&v);
}
}  // namespace

template <bool ORD1>      // ORD1: task order (query tile, row group, head); else (query tile, head, row group)
__global__ void __launch_bounds__(THREADS, 1) triattn_d64_sm90(
    int N, int H, int S, int ntasks, float scale,
    const __grid_constant__ CUtensorMap qmap,            // proj [B N S][4 H 64] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap kvmap,           // the same tensor, box (64, 64)
    const __grid_constant__ CUtensorMap bmap,            // bias [B H S][S] bf16, box (64, 128), 128B swizzle
    const bf16* __restrict__ P, long ld,                 // proj (the gate's columns), its row stride
    const float* __restrict__ GB,                        // gate bias fp32 [H 64], or nullptr
    bf16* __restrict__ U,                                // [B N S][H 64]
    const int* __restrict__ KT,                          // [B] key tiles of 64 per sample, or nullptr (S / 64)
    int* __restrict__ SCHED) {                           // zeroed task counter (dynamic schedule), or nullptr: static
  const int QTILES = S / BM, NKALL = S / BN, NG = (N + R - 1) / R, HD = H * D;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  // shared layout: bias ring [NB][BT] | Q [R][QT] | K | V ring [NKV][K | V | ones] | barriers
  const uint32_t sB = sa(smb), sQ = sB + NB * BT * 2, sKV = sQ + R * QT * 2, bar0 = sKV + NKV * STG * 2;
  const uint32_t qf = bar0, qe = bar0 + 8;
  auto kvf = [&](int s) { return bar0 + 16 + 8 * s; };
  auto kve = [&](int s) { return bar0 + 16 + 8 * (NKV + s); };
  auto bfu = [&](int s) { return bar0 + 16 + 8 * (2 * NKV + s); };
  auto bem = [&](int s) { return bar0 + 16 + 8 * (2 * NKV + NB + s); };
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  for (int i = tid; i < NKV * KVT / 2; i += THREADS) {     // the ones blocks: bf16 pairs (1, 1)
    const int s = i / (KVT / 2), e = i % (KVT / 2);
    reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (s * STG + 2 * KVT) * 2)[e] = 0x3F803F80u;
  }
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sB));
    bar_init(bars, 1); bar_init(bars + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(bars + 2 + k, 1); bar_init(bars + 2 + NKV + k, 8); }
    for (int k = 0; k < NB; ++k) { bar_init(bars + 2 + 2 * NKV + k, 1); bar_init(bars + 2 + 2 * NKV + NB + k, 8); }
    bar_init_fence();
  }
  fence_proxy_async();                                      // the ones blocks are read by the tensor core (async proxy)
  __syncthreads();
  auto decode = [&](int t, int& b, int& h, int& qt, int& g) {
    qt = t % QTILES; int r = t / QTILES;
    if (ORD1) { g = r % NG; r /= NG; h = r % H; b = r / H; }
    else      { h = r % H; r /= H; g = r % NG; b = r / NG; }
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };
  volatile int* slot = reinterpret_cast<volatile int*>(smb + (bar0 - sB) + 240);

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&qmap); prefetch_map(&kvmap); prefetch_map(&bmap);
      int lt = 0, st = 0, kph = 0, bs = 0, bph = 0;
      for (int t = blockIdx.x; t < ntasks; t = SCHED ? (int)gridDim.x + atomicAdd(SCHED, 1) : t + (int)gridDim.x, ++lt) {
        int b, h, qt, g; decode(t, b, h, qt, g);
        const int nk = KT != nullptr ? KT[b] : NKALL;
        wait(qe, (lt & 1) ^ 1);
        if (SCHED) *slot = t;
        expect_tx(qf, R * QT * 2);
        for (int r = 0; r < R; ++r) load_2d(&qmap, sQ + r * QT * 2, qf, h * D, row0(b, g * R + r) + qt * BM);
        for (int j = 0; j < nk; ++j) {
          wait(bem(bs), bph ^ 1);
          expect_tx(bfu(bs), BT * 2);
          load_2d(&bmap, sB + bs * BT * 2, bfu(bs), j * BN, (b * H + h) * S + qt * BM);
          if (++bs == NB) { bs = 0; bph ^= 1; }
          for (int r = 0; r < R; ++r) {
            wait(kve(st), kph ^ 1);
            expect_tx(kvf(st), 2 * KVT * 2);
            const int rw = row0(b, g * R + r) + j * BN;
            load_2d(&kvmap, sKV + st * STG * 2, kvf(st), HD + h * D, rw);
            load_2d(&kvmap, sKV + (st * STG + KVT) * 2, kvf(st), 2 * HD + h * D, rw);
            if (++st == NKV) { st = 0; kph ^= 1; }
          }
        }
      }
      if (SCHED) {
        wait(qe, (lt & 1) ^ 1);
        *slot = -1;
        arrive(qf);
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int qrow = 64 * w + 16 * wi;               // this warp's first query row in the tile
  const int lrow = qrow + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4, x7 = lane & 7;
  const uint32_t qlane = sQ + lrow * 128, blane = sB + lrow * 128;
  const uint64_t kd0 = desc_k128(sKV), vd0 = desc_mn128(sKV + KVT * 2, KVT * 2);
  const __nv_bfloat162 sc2 = __floats2bfloat162_rn(scale, scale);
  int gst = 0, gph = 0, bs = 0, bph = 0, xst = 0;
  uint32_t qa[R][4][4];
  float o[R][36], m[R][2];
  float s[32];                                      // the score accumulator, then the sub-step's probabilities (fp32)
  uint32_t p[16];                                   // the previous sub-step's P (bf16 pairs): the A fragment of its PV GEMM
  auto G = [&](int r) {
    if (r == 0) wait(bfu(bs), bph);
    const uint32_t btile = blane + bs * BT * 2;
#pragma unroll
    for (int i2 = 0; i2 < 4; ++i2) {
      uint32_t r4[4];
      ldsm_x4(btile + (((2 * i2 + lcol8) ^ x7) << 4), r4);
      const float2 a0 = bf2f(r4[0]), a1 = bf2f(r4[1]), a2 = bf2f(r4[2]), a3 = bf2f(r4[3]);
      s[8 * i2 + 0] = a0.x; s[8 * i2 + 1] = a0.y; s[8 * i2 + 2] = a1.x; s[8 * i2 + 3] = a1.y;
      s[8 * i2 + 4] = a2.x; s[8 * i2 + 5] = a2.y; s[8 * i2 + 6] = a3.x; s[8 * i2 + 7] = a3.y;
    }
    if (r == R - 1) {
      fence_regs(s);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(bem(bs));
      if (++bs == NB) { bs = 0; bph ^= 1; }
    }
    wait(kvf(gst), gph);
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_rs_n64<0>(s, qa[r][ks], desc_off(kd0, gst * STG * 2 + 32 * ks), 1u);
    wg_commit();
    if (++gst == NKV) { gst = 0; gph ^= 1; }
  };
  auto PV = [&](int r) {
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const uint32_t a[4] = {p[4 * kk], p[4 * kk + 1], p[4 * kk + 2], p[4 * kk + 3]};
      mma_rs_n72(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 2048));
    }
    wg_commit();
  };
  auto release = [&]() {
    __syncwarp();
    if (lane == 0) arrive(kve(xst));
    if (++xst == NKV) xst = 0;
  };
  auto X = [&](int r) {
    float m0 = fmaxf(s[0], s[1]), m1 = fmaxf(s[2], s[3]);
#pragma unroll
    for (int i = 1; i < 8; ++i) { m0 = fmaxf(m0, fmaxf(s[4 * i], s[4 * i + 1])); m1 = fmaxf(m1, fmaxf(s[4 * i + 2], s[4 * i + 3])); }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m0 *= LOG2E; m1 *= LOG2E;                       // -inf when the tile's keys are all masked
    const bool u0 = m0 > m[r][0] + TD_TAU, u1 = m1 > m[r][1] + TD_TAU;
    if (__any_sync(0xffffffffu, u0 | u1)) {
      const float n0 = u0 ? m0 : m[r][0], n1 = u1 ? m1 : m[r][1];
      const float f0 = u0 ? ex2f(m[r][0] - n0) : 1.f, f1 = u1 ? ex2f(m[r][1] - n1) : 1.f;     // 2^(-inf) = 0 scales an empty sum
#pragma unroll
      for (int i = 0; i < 9; ++i) { o[r][4 * i] *= f0; o[r][4 * i + 1] *= f0; o[r][4 * i + 2] *= f1; o[r][4 * i + 3] *= f1; }
      m[r][0] = n0; m[r][1] = n1;
    }
    const float nm0 = m[r][0] > -INFINITY ? -m[r][0] : 0.f, nm1 = m[r][1] > -INFINITY ? -m[r][1] : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int e = 0; e < 4; ++e) s[4 * i + e] = ex2f(fmaf(s[4 * i + e], LOG2E, e < 2 ? nm0 : nm1));
    }
  };
  auto pack = [&]() {
#pragma unroll
    for (int i = 0; i < 8; ++i) { p[2 * i] = PACKP(s[4 * i], s[4 * i + 1]); p[2 * i + 1] = PACKP(s[4 * i + 2], s[4 * i + 3]); }
  };
  auto step = [&](int r) {
    G(r);
    PV(1 - r);
    wg_wait<1>();
    fence_regs(s);
    X(r);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    release();
    pack();
  };
  int lt = 0;
  for (int t = blockIdx.x;; t += SCHED ? 0 : (int)gridDim.x, ++lt) {
    if (SCHED == nullptr && t >= ntasks) break;
    wait(qf, lt & 1);
    if (SCHED) {
      t = *slot;
      if (t < 0) break;
    }
    int b, h, qt, g; decode(t, b, h, qt, g);
    const int nk = KT != nullptr ? KT[b] : NKALL;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) ldsm_x4(qlane + r * QT * 2 + (((2 * ks + lcol8) ^ x7) << 4), qa[r][ks]);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) fence_regs(qa[r][ks]);
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) arrive(qe);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int e = 0; e < 4; ++e) qa[r][ks][e] = mul2(qa[r][ks][e], sc2);   // scale q (a power of two at D = 64: exact)
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int i = 0; i < 36; ++i) o[r][i] = 0.f;
      m[r][0] = m[r][1] = -INFINITY;
    }
    G(0);                                          // sub-step (0, 0): no previous PV
    wg_wait<0>();
    fence_regs(s);
    X(0);
    pack();
    for (int j = 0; j < nk; ++j) {
      step(1);
      if (j + 1 < nk) step(0);
    }
    PV(1);                                         // the last sub-step's PV
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    release();
    // ---- epilogue: u = sigmoid(g (+ gb)) O / l in fp32, bf16; a quad transposes its words so each lane stores 16 contiguous bytes
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = g * R + r;
      if (n >= N) continue;                      // a padding row: computed on row N - 1, not stored
      const float l0 = o[r][32], l1 = o[r][34];  // accumulator columns 64..71 = P 1: the row sums
      const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float inv = half ? inv1 : inv0;
        const long row = (long)row0(b, n) + qt * BM + qrow + g8 + 8 * half;
        const bf16* gp = P + row * ld + 3 * HD + h * D;
#pragma unroll
        for (int grp = 0; grp < 2; ++grp) {
          uint32_t in[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int blk = 4 * grp + i, col = 8 * blk + 2 * t4;
            const float2 gv = bf2f(*reinterpret_cast<const uint32_t*>(gp + col));
            float ga = gv.x, gb = gv.y;
            if (GB != nullptr) { ga += GB[h * D + col]; gb += GB[h * D + col + 1]; }
            in[i] = pack2(o[r][4 * blk + 2 * half] * inv / (1.f + __expf(-ga)), o[r][4 * blk + 2 * half + 1] * inv / (1.f + __expf(-gb)));
          }
          *reinterpret_cast<uint4*>(U + row * HD + h * D + 32 * grp + 8 * t4) = quad_transpose(in, lane);
        }
      }
    }
  }
}

// proj [B S S, 4 H 64] bf16 contiguous (q | k | v | g), bias [B, H, S, S] bf16 contiguous (natural units, masked keys -inf), S % 128 == 0;
// gate_bias fp32 [H 64] or empty; key_tiles int32 [B] (64-key tiles, h100_sm90.key_tiles(mask, S, 64)) or None -> u [B S S, H 64] bf16
at::Tensor triattn_d64_forward(at::Tensor proj, at::Tensor bias, at::Tensor gate_bias, int64_t H, double scale,
                               std::optional<at::Tensor> key_tiles) {
  TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::kBFloat16 && proj.is_contiguous() && proj.dim() == 2 && proj.size(1) == 4 * H * D,
              "proj: contiguous bf16 [B S S, 4 H 64]");
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.dim() == 4 && bias.size(1) == H && bias.size(2) == bias.size(3),
              "bias: contiguous bf16 [B, H, S, S]");
  const long B = bias.size(0), S = bias.size(2), N = S;
  TORCH_CHECK(S % BM == 0 && S > 0 && proj.size(0) == B * N * S, "S a multiple of 128, proj rows B S S");
  TORCH_CHECK(gate_bias.numel() == 0 || (gate_bias.scalar_type() == at::kFloat && gate_bias.is_contiguous() && gate_bias.numel() == H * D),
              "gate_bias: fp32 [H 64] or empty");
  const int* kt = nullptr;
  if (key_tiles.has_value()) {
    TORCH_CHECK(key_tiles->scalar_type() == at::kInt && key_tiles->is_contiguous() && key_tiles->numel() == B && key_tiles->is_cuda(),
                "key_tiles: int32 [B] on the device");
    kt = key_tiles->data_ptr<int>();
  }
  const c10::cuda::OptionalCUDAGuard guard(proj.device());
  const long ld = 4 * H * D;
  const uint64_t rows = (uint64_t)(B * N * S);
  TORCH_CHECK(rows < (1ull << 31) && (uint64_t)(B * H * S) < (1ull << 31), "row coordinates beyond int32");
  CUtensorMap qm = make_map<2>(proj.data_ptr(), {(uint64_t)ld, rows}, {(uint64_t)ld}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "q");
  CUtensorMap kvm = make_map<2>(proj.data_ptr(), {(uint64_t)ld, rows}, {(uint64_t)ld}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "k | v");
  CUtensorMap bm = make_map<2>(bias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  auto u = at::empty({proj.size(0), H * D}, proj.options());
  static bool attr = false;
  if (!attr) {
    for (auto k : {triattn_d64_sm90<false>, triattn_d64_sm90<true>})
      C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    attr = true;
  }
  const long ntasks = B * H * ((N + R - 1) / R) * (S / BM);
  TORCH_CHECK(ntasks < (1L << 31), "too many tasks");
  const int grid = (int)std::min<long>(ntasks, num_sms(proj.device().index()));
  const bool dyn = TD_DYN && ntasks >= 16L * grid;
  auto sched = at::zeros({dyn ? 1 : 0}, proj.options().dtype(at::kInt));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto kernel = S >= TD_ORDER1_MINS ? triattn_d64_sm90<true> : triattn_d64_sm90<false>;
  kernel<<<grid, THREADS, SMEM, stream>>>((int)N, (int)H, (int)S, (int)ntasks, (float)scale, qm, kvm, bm,
                                          reinterpret_cast<const bf16*>(proj.data_ptr()), ld,
                                          gate_bias.numel() ? gate_bias.data_ptr<float>() : nullptr, reinterpret_cast<bf16*>(u.data_ptr()), kt,
                                          dyn ? sched.data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return u;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_d64_forward", &triattn_d64_forward, "sm90 triangle-attention inference core for heads of 64 channels (wgmma, fp32 softmax): "
        "proj [B S S, 4 H 64] (q | k | v | g), bias [B, H, S, S] natural units -> u = sigmoid(g (+ gate_bias)) o [B S S, H 64]",
        py::arg("proj"), py::arg("bias"), py::arg("gate_bias"), py::arg("H"), py::arg("scale"), py::arg("key_tiles") = py::none());
}
