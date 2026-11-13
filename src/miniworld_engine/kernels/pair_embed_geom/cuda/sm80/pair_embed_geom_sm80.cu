// pair_embed_geom_sm80.cu -- the A100 (sm_80) build of sm90/pair_embed_geom.cu: the same kernels and host entry points, the source compiled
// unchanged (the H100 build's file stays byte-identical); pe_sm80.cuh supplies the sm_80 forms of its five sm_90-only helpers (the table
// gradient's warpgroup product as mma.sync on the same swizzled shared-memory operands).
//
// One addition, the table gradient's impl 3 (this card's choice for the decoder table, sm80/__init__.py): the H100 grid of the mma.sync
// kernel (pg_bwd_mma, 4 x NBLK = 132 CTAs of 5 warps: sized for the H100's 132 SMs) leaves the A100's 108 SMs a few warps each, so the
// pairs are split in KS contiguous chunks whose grids run concurrently on two streams (fork / join by events: capturable), each writing
// its own NBLK partials; the KS x NBLK partials are then summed in a fixed order (bitwise deterministic, as the H100 kernels).
// `backward` below is the module's entry point; the H100 host function (renamed backward_h100 here) serves every other impl.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "pe_sm80.cuh"

void backward(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl);
#define backward(...) backward_h100(__VA_ARGS__)   // the H100 definition only: the pybind entry `&backward` names the one below
#ifdef MWE_SM120
// RTX 5090 (sm_120a, 99 KB of shared memory a block): the LayerNorm backward of the encoder pair (mode 0) cannot keep its 317-row table
// (162 KB) in shared memory; `ln_backward` below reads it through the L1 there (mode 1, 80 KB, keeps the H100 kernel).
void ln_backward(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor dx, int64_t mode, double eps);
#define ln_backward(...) ln_backward_h100(__VA_ARGS__)
// the two forwards: `forward` / `forward_ca` below (gt 2: the RTX 5090 inference launches, sm120 section at the end); the H100 host functions
// serve gt 0 / 1
void forward_ca(torch::Tensor tbl, torch::Tensor ca, torch::Tensor mask, torch::Tensor idx, torch::Tensor dlim, torch::Tensor out, torch::Tensor codes,
                int64_t ln, double eps, int64_t jp, int64_t gt);
#define forward_ca(...) forward_ca_h100(__VA_ARGS__)
void forward(torch::Tensor tbl, torch::Tensor bb, torch::Tensor ca, torch::Tensor cb, torch::Tensor mask, torch::Tensor idx, torch::Tensor chains,
             torch::Tensor dlim, torch::Tensor alim, torch::Tensor out, torch::Tensor codes, int64_t ln, double eps, int64_t jp, int64_t gt);
#define forward(...) forward_h100(__VA_ARGS__)
#endif
#include "../sm90/pair_embed_geom.cu"
#undef backward
#ifdef MWE_SM120
#undef ln_backward
#undef forward_ca
#undef forward
#endif

namespace {
constexpr int KS = 2;   // concurrent pair chunks of impl 3

__global__ void pg80_reduce_tbl(const float* __restrict__ tpart, __nv_bfloat16* __restrict__ dtbl, int NTAB, int nparts) {
  const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (size_t)NTAB * C) return;
  float s = 0.f;
  for (int r = 0; r < nparts; r++) s += tpart[(size_t)r * NTAB * C + k];
  dtbl[k] = __float2bfloat16_rn(s);
}

struct Fork { cudaEvent_t start, done; };
Fork& fork_events() {
  static thread_local Fork f = [] {
    Fork e;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&e.start, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&e.done, cudaEventDisableTiming));
    return e;
  }();
  return f;
}
}  // namespace

#ifdef MWE_SM120
void backward_sm120(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl);
#endif

// impl 3: tpart fp32 scratch [KS][NBLK][NTAB][256]; impl 4 (RTX 5090 only): [NB][NTAB][256], NB = tpart's partial count; else the H100 entry
// point (tpart [NBLK][NTAB][256])
void backward(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl) {
#ifdef MWE_SM120
  if (impl >= 4) { backward_sm120(g, codes, tpart, dtbl, mode, impl); return; }
#endif
  if (impl != 3) { backward_h100(g, codes, tpart, dtbl, mode, impl); return; }
  const at::cuda::CUDAGuard guard(g.device());
  const int B = g.size(0), n = g.size(1), ntab = mode == 0 ? NT : NT_CA;
  const long long P = (long long)B * n * n, per = (P + KS - 1) / KS;
  TORCH_CHECK(tpart.numel() == (int64_t)KS * NBLK * ntab * C, "backward impl 3: tpart [", KS, ", NBLK, NTAB, 256] fp32");
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* tp = tpart.data_ptr<float>();
  auto main = at::cuda::getCurrentCUDAStream();
  auto side = at::cuda::getStreamFromPool(false, g.get_device());
  Fork& ev = fork_events();
  C10_CUDA_CHECK(cudaEventRecord(ev.start, main));
  C10_CUDA_CHECK(cudaStreamWaitEvent(side, ev.start, 0));
  for (int k = 0; k < KS; k++) {
    const long long p0 = k * per, np = std::max(0LL, std::min(P, p0 + per) - p0);
    cudaStream_t s = k == 0 ? main.stream() : side.stream();
    float* part = tp + (size_t)k * NBLK * ntab * C;
    if (mode == 0) pg_bwd_mma<0><<<dim3(4, NBLK), 13 * 32, 0, s>>>(gp + (size_t)p0 * C, cp + p0, np, part);
    else pg_bwd_mma<1><<<dim3(4, NBLK), 5 * 32, 0, s>>>(gp + (size_t)p0 * C, cp + p0, np, part);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  C10_CUDA_CHECK(cudaEventRecord(ev.done, side));
  C10_CUDA_CHECK(cudaStreamWaitEvent(main, ev.done, 0));
  pg80_reduce_tbl<<<(ntab * C + 255) / 256, 256, 0, main>>>(tp, reinterpret_cast<__nv_bfloat16*>(dtbl.data_ptr<at::BFloat16>()), ntab, KS * NBLK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef MWE_SM120
namespace {
// pg_ln_bwd with the table read from global memory (through the L1) instead of a shared-memory copy: the same warp-per-pair schedule, the
// same two-deep load pipeline and the same arithmetic (ln_bwd_group), so the result is bitwise the shared-memory kernel's.
template <int MODE, int UL>
__global__ void __launch_bounds__(16 * 32, 1)
pg120_ln_bwd_gt(const __nv_bfloat16* __restrict__ tbl, const uint4* codes, const __nv_bfloat16* dh, __nv_bfloat16* dx, long long P, float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  const long long stride = (long long)gridDim.x * 16;
  const long long p0 = (long long)blockIdx.x * 16 + warp;
  auto ldc = [&](long long p) { return p < P ? codes[p] : make_uint4(0u, 0u, 0u, 0u); };
  auto ldg = [&](long long p, const uint4& cd) { return (p < P && (cd.x & LIVE)) ? *reinterpret_cast<const uint4*>(dh + (size_t)p * C + c0) : make_uint4(0u, 0u, 0u, 0u); };
  uint4 cA[UL], gA[UL], cB[UL], gB[UL], cC[UL];
#pragma unroll
  for (int u = 0; u < UL; u++) { cA[u] = ldc(p0 + u * stride); cB[u] = ldc(p0 + (UL + u) * stride); }
#pragma unroll
  for (int u = 0; u < UL; u++) gA[u] = ldg(p0 + u * stride, cA[u]);
  for (long long p = p0; p < P; p += UL * stride) {
#pragma unroll
    for (int u = 0; u < UL; u++) gB[u] = ldg(p + (UL + u) * stride, cB[u]);
#pragma unroll
    for (int u = 0; u < UL; u++) cC[u] = ldc(p + (2 * UL + u) * stride);
    {
      uint4 o[UL];
      bool any = false;
#pragma unroll
      for (int u = 0; u < UL; u++) any |= (cA[u].x & LIVE) != 0;
      if (any) ln_bwd_group<MODE, UL>(tbl, cA, gA, c0, eps, o);
      else {
#pragma unroll
        for (int u = 0; u < UL; u++) o[u] = make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int u = 0; u < UL; u++) {
        const long long pu = p + u * stride;
        if (pu < P) *reinterpret_cast<uint4*>(dx + (size_t)pu * C + c0) = o[u];
      }
    }
#pragma unroll
    for (int u = 0; u < UL; u++) { cA[u] = cB[u]; gA[u] = gB[u]; cB[u] = cC[u]; }
  }
}
}  // namespace

void ln_table_grad_sm120(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor scratch, int64_t mode, double eps, int64_t impl);

// mode 0 / 1: d pair (encoder / decoder); mode >= 2: the fused LayerNorm + table gradient (ln_table_grad_sm120, dx = its fp32 scratch):
// 2 = the encoder, 3 = the decoder at 128 channels a CTA, 5 = the decoder at 64
void ln_backward(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor dx, int64_t mode, double eps) {
  static_assert((size_t)NT_CA * C * sizeof(__nv_bfloat16) <= 101376, "the decoder table must fit the 5090's 99 KB of shared memory");
  if (mode >= 2) { ln_table_grad_sm120(tbl, codes, dh, dx, (mode - 2) & 1, eps, (mode - 2) >> 1 ? 4 : 5); return; }
  if (mode != 0) { ln_backward_h100(tbl, codes, dh, dx, mode, eps); return; }
  const at::cuda::CUDAGuard guard(tbl.device());
  const long long P = (long long)dh.size(0) * dh.size(1) * dh.size(2);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  pg120_ln_bwd_gt<0, UL_G><<<sms, 16 * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>()), reinterpret_cast<const uint4*>(codes.data_ptr<int>()),
      reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>()), P, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- RTX 5090 table gradient (impl 4: 64 channels a CTA, impl 5: 128) ------------------------------------------------------------------
// The product dT = F^T G of pg_bwd_mma (the one-hot A fragments built in registers from the codes, B = a cp.async ring of 64-pair gradient
// tiles read by ldmatrix.trans, mma.sync m16n8k16, fp32 accumulators), with the table rows packed: 16-row m-tile k is table rows 16k .. 16k + 15
// (encoder: 20 m-tiles = 320 rows for the 317, decoder: 10 = 160 for the 157; pg_bwd_mma pads every feature group to 32 rows: 416 / 160), and
// the pairs cut in NB contiguous slices (NB = the partial count of the caller's scratch, sized for the card's 170 SMs) instead of the H100's 33.
// A table row r of the pair's code: hit iff (c.x & req) == req and ((word >> sh) & mk) == loc (word c.y for the angles), the chain row the chain
// product (bf16). Partials [NB][NTAB][256] fp32 summed in slice order (pg80_reduce_tbl): bitwise deterministic, no atomics.
namespace {
constexpr int TP5 = 64, ST5 = 4;
struct RowD { uint32_t req, mk, loc; int sh, sel, chain; };
template <int MODE>
__device__ __forceinline__ RowD row_desc(int r) {
  RowD d{LIVE, 127u, (uint32_t)r, 0, 0, 0};
  if (r < 127) return d;
  d.mk = 31u;
  if (MODE == 1) {
    d.sh = 8;
    d.loc = r < NT_CA ? (uint32_t)(r - R_CAD) : 0xFFFFFFFFu;
  } else if (r < R_ANG) {
    const int k = (r - R_DIST) / 21;
    d.req = LIVE | F_DIST; d.sh = 8 + 5 * k; d.loc = (uint32_t)(r - R_DIST - 21 * k);
  } else if (r < R_CHAIN) {
    const int k = (r - R_ANG) / 21;
    d.req = LIVE | F_ORI; d.sel = 1; d.sh = 5 * k; d.loc = (uint32_t)(r - R_ANG - 21 * k);
  } else if (r == R_CHAIN) {
    d.chain = 1;
  } else {
    d.loc = 0xFFFFFFFFu;
  }
  return d;
}
__device__ __forceinline__ uint32_t onehot_el(const RowD& d, const uint4& c, uint32_t cvb) {
  const uint32_t w = d.sel ? c.y : c.x;
  const uint32_t hit = ((c.x & d.req) == d.req && ((w >> d.sh) & d.mk) == d.loc) ? 0x3F80u : 0u;
  return d.chain ? ((c.x & LIVE) ? cvb : 0u) : hit;
}
template <int MODE> struct G5 {
  static constexpr int NTAB = MODE == 0 ? NT : NT_CA, MT = (NTAB + 15) / 16, WR = MT / 2;
};
// shared memory of pg120_bwd: ST stages of [64 pairs x 64 WC channels] bf16 + codes (+ LNF: the per-pair LayerNorm statistics, and the table's
// slice [NTAB][64 WC] bf16)
template <int MODE, int WC, int ST, bool LNF> constexpr size_t g5_smem() {
  return (size_t)ST * (TP5 * 128 * WC + TP5 * 16 + (LNF ? TP5 * 16 : 0)) + (LNF ? (size_t)G5<MODE>::NTAB * 128 * WC : 0);
}

// LNF: the pair's 8 channels [c0, c0 + 8) of the CTA's table slice Ts (row pitch 64 WC), summed as gather_rows sums them (the forward's order)
template <int MODE, int WC>
__device__ __forceinline__ void slice_rows(const __nv_bfloat16* Ts, const uint4& cd, int c0, float (&x)[8]) {
#pragma unroll
  for (int e = 0; e < 8; e++) x[e] = 0.f;
  const uint32_t w0 = cd.x ^ MASKED, w1 = cd.y;
  auto add = [&](int r, float scale) {
    const uint4 u = *reinterpret_cast<const uint4*>(Ts + (size_t)r * 64 * WC + c0);
    x[0] += scale * bf_lo(u.x); x[1] += scale * bf_hi(u.x); x[2] += scale * bf_lo(u.y); x[3] += scale * bf_hi(u.y);
    x[4] += scale * bf_lo(u.z); x[5] += scale * bf_hi(u.z); x[6] += scale * bf_lo(u.w); x[7] += scale * bf_hi(u.w);
  };
  add(w0 & 127, 1.f);
  if (MODE == 0) {
    if (w0 & F_DIST) {
      add(R_DIST + ((w0 >> 8) & 31), 1.f);
      add(R_DIST + 21 + ((w0 >> 13) & 31), 1.f);
      add(R_DIST + 42 + ((w0 >> 18) & 31), 1.f);
      add(R_DIST + 63 + ((w0 >> 23) & 31), 1.f);
    }
    if (w0 & F_ORI) {
#pragma unroll
      for (int k = 0; k < 5; k++) add(R_ANG + 21 * k + ((w1 >> (5 * k)) & 31), 1.f);
    }
    const float pc = __uint_as_float(cd.z);
    if (pc != 0.f) add(R_CHAIN, pc);
  } else {
    add(R_CAD + ((w0 >> 8) & 31), 1.f);
  }
}

// LNF: d pair = rstd (dh - mean(dh) - hat mean(dh hat)) of one tile, in place (dh -> d pair, bf16; zeros for padding), hat = (bf16(pair) - mean) rstd
// recomputed from the codes and the table slice; the statistics (mean, rstd, mean(dh), mean(dh hat)) per pair from pg120_ln_stats
template <int MODE, int WC, int NTH>
__device__ __forceinline__ void ln_convert(unsigned char* Bst, const uint4* Cst, const float4* Sst, const __nv_bfloat16* Ts, int tid) {
  for (int c = tid; c < TP5 * 8 * WC; c += NTH) {
    const int kp = c / (8 * WC), ch = c % (8 * WC);
    uint4* slot = reinterpret_cast<uint4*>(Bst + kp * 128 * WC + ((ch ^ (kp & 7)) << 4));
    const uint4 cd = Cst[kp];
    if (!(cd.x & LIVE)) { *slot = make_uint4(0u, 0u, 0u, 0u); continue; }
    const float4 s = Sst[kp];
    float x[8];
    slice_rows<MODE, WC>(Ts, cd, ch * 8, x);
    const uint4 gv = *slot;
    const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gv);
    uint4 r;
    __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const float2 gf = __bfloat1622float2(gh[k]);
      const float h0 = (__bfloat162float(__float2bfloat16_rn(x[2 * k])) - s.x) * s.y;
      const float h1 = (__bfloat162float(__float2bfloat16_rn(x[2 * k + 1])) - s.x) * s.y;
      oh[k] = __floats2bfloat162_rn(s.y * (gf.x - s.z - h0 * s.w), s.y * (gf.y - s.z - h1 * s.w));
    }
    *slot = r;
  }
}

// LNF false: the table gradient of g (= d pair); LNF true: of the LayerNorm backward of g (= d LayerNorm(pair)), the d pair never stored (each
// tile converted in shared memory one tile ahead of its products, ln_convert; tbl / stats / ln unused otherwise)
template <int MODE, int WC, int ST, bool LNF>
__global__ void __launch_bounds__(G5<MODE>::WR * WC * 32, 1)
pg120_bwd(const __nv_bfloat16* g, const uint4* codes, long long P, long long per, float* tpart, const __nv_bfloat16* tbl, const float4* stats) {
  constexpr int NTAB = G5<MODE>::NTAB, WR = G5<MODE>::WR, NTH = WR * WC * 32;
  constexpr int RB = 128 * WC, BST = TP5 * RB, LA = LNF ? 1 : 0;   // LA: tiles converted ahead
  static_assert(NTH >= TP5, "one thread per pair code of a tile");
  static_assert(ST >= 2 + LA, "stages");
  extern __shared__ __align__(128) unsigned char sm5[];
  unsigned char* Bs = sm5;                                   // [ST][TP5 pairs][64 WC channels] bf16, 16-byte chunks XOR-swizzled by (pair & 7)
  uint4* Cs = reinterpret_cast<uint4*>(sm5 + ST * BST);      // [ST][TP5] codes
  float4* Ss = reinterpret_cast<float4*>(sm5 + ST * (BST + TP5 * 16));                        // LNF: [ST][TP5] statistics
  __nv_bfloat16* Ts = reinterpret_cast<__nv_bfloat16*>(sm5 + ST * (BST + TP5 * 32));          // LNF: [NTAB][64 WC] table slice
  const int q = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int wr = warp % WR, wc = warp / WR;
  const long long p0 = (long long)t * per, p1 = min(P, p0 + per);
  const int ntiles = p1 > p0 ? (int)((p1 - p0 + TP5 - 1) / TP5) : 0;
  if (LNF)
    for (int k = tid; k < NTAB * 8 * WC; k += NTH)
      reinterpret_cast<uint4*>(Ts)[k] = *reinterpret_cast<const uint4*>(tbl + (size_t)(k / (8 * WC)) * C + q * 64 * WC + (k % (8 * WC)) * 8);
  RowD rd[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int hi = 0; hi < 2; hi++) rd[mt][hi] = row_desc<MODE>(32 * wr + 16 * mt + gid + 8 * hi);
  float acc[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

  auto load_tile = [&](int tile, int stage) {
    const long long pb = p0 + (long long)tile * TP5;
    for (int c = tid; c < TP5 * 8 * WC; c += NTH) {
      const int kp = c / (8 * WC), ch = c % (8 * WC);
      const long long pair = pb + kp;
      const bool pred = pair < p1;
      cp16(Bs + stage * BST + kp * RB + ((ch ^ (kp & 7)) << 4), g + (size_t)(pred ? pair : 0) * C + q * 64 * WC + ch * 8, pred);
    }
    if (tid < TP5) {
      const long long pair = pb + tid;
      const bool pred = pair < p1;
      cp16(&Cs[stage * TP5 + tid], codes + (pred ? pair : 0), pred);
      if (LNF) cp16(&Ss[stage * TP5 + tid], stats + (pred ? pair : 0), pred);
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < ntiles) load_tile(s, s);
    cp_commit();
  }
  if (LNF) {                                                  // tile 0 converted before the loop
    cp_wait<ST - 2>();
    __syncthreads();
    if (ntiles > 0) ln_convert<MODE, WC, NTH>(Bs, Cs, Ss, Ts, tid);
  }
  const unsigned bs_base = (unsigned)__cvta_generic_to_shared(Bs);
  for (int kt = 0; kt < ntiles; kt++) {
    cp_wait<ST - 2 - LA>();                                   // tile kt + LA has landed
    __syncthreads();                                          // ... visible to all; tile kt converted; tile kt - 1's stage free
    if (kt + ST - 1 < ntiles) load_tile(kt + ST - 1, (kt + ST - 1) % ST);
    cp_commit();
    if (LNF && kt + 1 < ntiles) {
      const int s1 = (kt + 1) % ST;
      ln_convert<MODE, WC, NTH>(Bs + s1 * BST, Cs + s1 * TP5, Ss + s1 * TP5, Ts, tid);
    }
    const int stage = kt % ST;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int kb = stage * TP5 + ks * 16 + 2 * tig;
      const uint4 c0 = Cs[kb], c1 = Cs[kb + 1], c2 = Cs[kb + 8], c3 = Cs[kb + 9];
      uint32_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
      if (MODE == 0) {
        v0 = bf16_bits(__uint_as_float(c0.z)); v1 = bf16_bits(__uint_as_float(c1.z));
        v2 = bf16_bits(__uint_as_float(c2.z)); v3 = bf16_bits(__uint_as_float(c3.z));
      }
      unsigned bq[8][2];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const unsigned addr = bs_base + stage * BST + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * RB +
                              ((wc * 8 + ((2 * j + (lane >> 4)) ^ (lane & 7))) << 4);
        ldsm4t(bq[2 * j][0], bq[2 * j][1], bq[2 * j + 1][0], bq[2 * j + 1][1], addr);
      }
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        const unsigned a0 = onehot_el(rd[mt][0], c0, v0) | (onehot_el(rd[mt][0], c1, v1) << 16);
        const unsigned a1 = onehot_el(rd[mt][1], c0, v0) | (onehot_el(rd[mt][1], c1, v1) << 16);
        const unsigned a2 = onehot_el(rd[mt][0], c2, v2) | (onehot_el(rd[mt][0], c3, v3) << 16);
        const unsigned a3 = onehot_el(rd[mt][1], c2, v2) | (onehot_el(rd[mt][1], c3, v3) << 16);
#pragma unroll
        for (int nt = 0; nt < 8; nt++) mma16816(acc[mt][nt], a0, a1, a2, a3, bq[nt][0], bq[nt][1]);
      }
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
#pragma unroll
    for (int hi = 0; hi < 2; hi++) {
      const int r = 32 * wr + 16 * mt + gid + 8 * hi;
      if (r >= NTAB) continue;
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        const float2 v = make_float2(acc[mt][nt][2 * hi], acc[mt][nt][2 * hi + 1]);
        *reinterpret_cast<float2*>(tpart + ((size_t)t * NTAB + r) * C + q * 64 * WC + wc * 64 + nt * 8 + 2 * tig) = v;
      }
    }
  }
}

template <int MODE, int WC, int ST, bool LNF>
void launch_bwd120(const __nv_bfloat16* gp, const uint4* cp, long long P, float* tp, int nb, cudaStream_t st, const __nv_bfloat16* tbl = nullptr,
                   const float4* stats = nullptr) {
  constexpr size_t smem = g5_smem<MODE, WC, ST, LNF>();
  static_assert(smem <= 101376, "the RTX 5090's 99 KB of shared memory a block");
  const long long per = ((P + nb - 1) / nb + TP5 - 1) / TP5 * TP5;
  auto k = pg120_bwd<MODE, WC, ST, LNF>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  k<<<dim3(4 / WC, nb), G5<MODE>::WR * WC * 32, smem, st>>>(gp, cp, P, per, tp, tbl, stats);
}

// The LayerNorm statistics of every pair for the fused backward: (mean, rstd) of bf16(pair) and (mean(dh), mean(dh hat)), as ln_bwd_group computes
// them (a warp per pair, the table through the L1, UL pairs in flight); zeros for padding. Memory-bound: reads dh once.
template <int MODE, int UL>
__global__ void __launch_bounds__(256, 3)
pg120_ln_stats(const __nv_bfloat16* __restrict__ tbl, const uint4* codes, const __nv_bfloat16* dh, float4* stats, long long P, float eps) {
  const int lane = threadIdx.x & 31, c0 = lane * 8;
  const long long nw = (long long)gridDim.x * (blockDim.x >> 5);
  for (long long p = ((long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * UL; p < P; p += nw * UL) {
    uint4 cd[UL], gv[UL];
#pragma unroll
    for (int u = 0; u < UL; u++) cd[u] = p + u < P ? codes[p + u] : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < UL; u++) gv[u] = (cd[u].x & LIVE) ? *reinterpret_cast<const uint4*>(dh + (size_t)(p + u) * C + c0) : make_uint4(0u, 0u, 0u, 0u);
    float x[UL][8], gf[UL][8], mean[UL], rstd[UL], t[UL], a[UL], bs[UL];
#pragma unroll
    for (int u = 0; u < UL; u++) {
      gather_rows<MODE>(tbl, cd[u], c0, x[u]);
      const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gv[u]);
#pragma unroll
      for (int k = 0; k < 4; k++) { const float2 f = __bfloat1622float2(gh[k]); gf[u][2 * k] = f.x; gf[u][2 * k + 1] = f.y; }
      t[u] = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) { x[u][e] = __bfloat162float(__float2bfloat16_rn(x[u][e])); t[u] += x[u][e]; }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1)
#pragma unroll
      for (int u = 0; u < UL; u++) t[u] += __shfl_xor_sync(0xffffffffu, t[u], off);
#pragma unroll
    for (int u = 0; u < UL; u++) {
      mean[u] = t[u] * (1.f / C);
      t[u] = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) { const float d = x[u][e] - mean[u]; t[u] += d * d; }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1)
#pragma unroll
      for (int u = 0; u < UL; u++) t[u] += __shfl_xor_sync(0xffffffffu, t[u], off);
#pragma unroll
    for (int u = 0; u < UL; u++) {
      rstd[u] = rsqrtf(t[u] * (1.f / C) + eps);
      a[u] = 0.f; bs[u] = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) { const float h = (x[u][e] - mean[u]) * rstd[u]; a[u] += gf[u][e]; bs[u] += gf[u][e] * h; }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1)
#pragma unroll
      for (int u = 0; u < UL; u++) { a[u] += __shfl_xor_sync(0xffffffffu, a[u], off); bs[u] += __shfl_xor_sync(0xffffffffu, bs[u], off); }
    if (lane < UL) {
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int u = 0; u < UL; u++)
        if (u == lane && (cd[u].x & LIVE)) s = make_float4(mean[u], rstd[u], a[u] * (1.f / C), bs[u] * (1.f / C));
      if (p + lane < P) stats[p + lane] = s;
    }
  }
}
}  // namespace

void backward_sm120(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl) {
  const at::cuda::CUDAGuard guard(g.device());
  const long long P = (long long)g.size(0) * g.size(1) * g.size(2);
  const int ntab = mode == 0 ? NT : NT_CA;
  const int nb = (int)(tpart.numel() / ((int64_t)ntab * C));
  TORCH_CHECK(nb >= 1 && tpart.numel() == (int64_t)nb * ntab * C, "backward impl 4/5: tpart [NB, NTAB, 256] fp32");
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* tp = tpart.data_ptr<float>();
  auto st = at::cuda::getCurrentCUDAStream();
  if (mode == 0) { if (impl == 5) launch_bwd120<0, 2, ST5, false>(gp, cp, P, tp, nb, st); else launch_bwd120<0, 1, ST5, false>(gp, cp, P, tp, nb, st); }
  else { if (impl == 5) launch_bwd120<1, 2, ST5, false>(gp, cp, P, tp, nb, st); else launch_bwd120<1, 1, ST5, false>(gp, cp, P, tp, nb, st); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pg80_reduce_tbl<<<(ntab * C + 255) / 256, 256, 0, st>>>(tp, reinterpret_cast<__nv_bfloat16*>(dtbl.data_ptr<at::BFloat16>()), ntab, nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The fused LayerNorm + table gradient (ln_backward's mode 2 / 3 = encoder / decoder): the table gradient straight from d LayerNorm(pair) = dh.
// scratch fp32: [NB][NTAB][256] partials | [P] float4 statistics | the table gradient [NTAB][256] bf16 (NTAB * 128 floats), NB from its size.
// impl (the decoder): 5 = 128 channels a CTA (3 stages), 4 = 64 (4 stages); the encoder: 64 channels, 4 stages.
void ln_table_grad_sm120(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor scratch, int64_t mode, double eps, int64_t impl) {
  const at::cuda::CUDAGuard guard(dh.device());
  const long long P = (long long)dh.size(0) * dh.size(1) * dh.size(2);
  const int ntab = mode == 0 ? NT : NT_CA;
  const long long rest = scratch.numel() - 4 * P - (long long)ntab * (C / 2);
  const int nb = (int)(rest / ((long long)ntab * C));
  TORCH_CHECK(nb >= 1 && rest == (long long)nb * ntab * C, "ln_table_grad: scratch [NB * NTAB * 256 + 4 P + NTAB * 128] fp32");
  auto* tp = reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>());
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* part = scratch.data_ptr<float>();
  float4* stats = reinterpret_cast<float4*>(part + (size_t)nb * ntab * C);
  auto* dtbl = reinterpret_cast<__nv_bfloat16*>(part + (size_t)nb * ntab * C + 4 * P);
  auto st = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (mode == 0) pg120_ln_stats<0, 2><<<3 * sms, 256, 0, st>>>(tp, cp, gp, stats, P, (float)eps);
  else pg120_ln_stats<1, 4><<<3 * sms, 256, 0, st>>>(tp, cp, gp, stats, P, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (mode == 0) launch_bwd120<0, 1, 4, true>(gp, cp, P, part, nb, st, tp, stats);
  else if (impl == 5) launch_bwd120<1, 2, 3, true>(gp, cp, P, part, nb, st, tp, stats);
  else launch_bwd120<1, 1, 4, true>(gp, cp, P, part, nb, st, tp, stats);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pg80_reduce_tbl<<<(ntab * C + 255) / 256, 256, 0, st>>>(part, dtbl, ntab, nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- RTX 5090 inference forwards (gt 2) -------------------------------------------------------------------------------------------------
// The forward-only launches of the two modules without their per-call preparation kernels (modules/pair_embed.py `_prepared`: the transposed
// bf16 table, the uint8 masks, the int32 indices): the table is the module's weight [256, NT] (fp32 or bf16), transposed and rounded to bf16
// here by one kernel (pg120_table: the values of `tbl.copy_(weight.t())`), the masks are read as bytes (bool or uint8), the indices as int32 or
// int64 (IdxT), the bin limits are the module's (cached per device by the caller). The kernels: pg_fwd's (encoder) and the decoder's below.
namespace {
__device__ __forceinline__ float as_f32(float v) { return v; }
__device__ __forceinline__ float as_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
// programmatic dependent launch: the forwards are launched with programmatic stream serialization behind pg120_table and wait for it only
// before their first table read (their binning overlaps the transpose); griddepcontrol.wait is a no-op for a launch without a primary
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <typename TI>
__global__ void pg120_table(const TI* __restrict__ w, __nv_bfloat16* __restrict__ t, int nt) {   // w [256][nt] -> t [nt][256] bf16
  pdl_trigger();
  __shared__ float tile[32][33];
  const int r0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
  for (int k = threadIdx.y; k < 32; k += 8) {
    const int r = r0 + threadIdx.x;
    tile[k][threadIdx.x] = r < nt ? as_f32(w[(size_t)(c0 + k) * nt + r]) : 0.f;
  }
  __syncthreads();
  for (int k = threadIdx.y; k < 32; k += 8) {
    const int r = r0 + k;
    if (r < nt) t[(size_t)r * C + c0 + threadIdx.x] = __float2bfloat16_rn(tile[threadIdx.x][k]);
  }
}

// `kernel<<<grid, block, 0, st>>>(args...)` with programmatic stream serialization (pdl): it may start while the previous kernel runs and
// waits for it at its griddepcontrol.wait
template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, bool pdl, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

// the bf16 table [nt][256]: `tbl` itself, or the transpose of a [256][nt] weight (fp32 / bf16)
torch::Tensor table_of(torch::Tensor tbl, int nt, cudaStream_t st, bool& transposed) {
  transposed = false;
  if (tbl.size(0) == nt && tbl.size(1) == C && tbl.scalar_type() == at::kBFloat16) return tbl;
  transposed = true;
  TORCH_CHECK(tbl.dim() == 2 && tbl.size(0) == C && tbl.size(1) == nt && tbl.is_contiguous(), "table: [NT, 256] bf16 or a contiguous [256, NT] weight");
  auto t = torch::empty({nt, C}, tbl.options().dtype(at::kBFloat16));
  const dim3 grid((nt + 31) / 32, C / 32), block(32, 8);
  auto* tp = reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
  if (tbl.scalar_type() == at::kFloat) pg120_table<float><<<grid, block, 0, st>>>(tbl.data_ptr<float>(), tp, nt);
  else {
    TORCH_CHECK(tbl.scalar_type() == at::kBFloat16, "table: fp32 or bf16 weight");
    pg120_table<__nv_bfloat16><<<grid, block, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>()), tp, nt);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return t;
}

// pg_fwd<true, true> (the H100 launch's choice) with IdxT indices and chains
template <typename IdxT>
__global__ void __launch_bounds__(GT_THREADS, GT_CPS)
pg120_fwd(const __nv_bfloat16* tbl, const float* bb, const uint8_t* ca, const uint8_t* cb, const uint8_t* mask, const IdxT* idx, const IdxT* chains,
          const float* dlim, const float* alim, __nv_bfloat16* out, uint4* codes, int B, int n, int ln, float eps, int jp) {
  __shared__ float dlim_s[32], alim_s[32];
  const int nwb = blockDim.x >> 5;
  if (threadIdx.x < 32) { dlim_s[threadIdx.x] = threadIdx.x < 20 ? dlim[threadIdx.x] : 0.f; alim_s[threadIdx.x] = threadIdx.x < 20 ? alim[threadIdx.x] : 0.f; }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  const int nch = (n + jp - 1) / jp;
  for (long long item = (long long)blockIdx.x * nwb + warp; item < (long long)B * n * nch; item += (long long)gridDim.x * nwb) {
    const int row = (int)(item / nch), j0 = (int)(item - (long long)row * nch) * jp;
    const int b = row / n, i = row % n;
    const bool mi = mask[b * n + i], cai = ca[b * n + i], cbi = cb[b * n + i];
    const V3 Ni = atom(bb, b, n, i, 0), CAi = atom(bb, b, n, i, 1), CBi = atom(bb, b, n, i, 3);
    __nv_bfloat16* orow = out + (size_t)row * n * C;
    const int j = j0 + lane;
    const int jend = min(n, j0 + jp);
    uint32_t w0 = MASKED, w1 = 0;
    float cv = 0.f;
    if (j < jend && mi && mask[b * n + j]) {
      const int s = (int)min(max((long long)idx[b * n + i] - (long long)idx[b * n + j] + 63, 0LL), 126LL);
      w0 = (uint32_t)s;
      cv = (float)(chains[b * n + i] * chains[b * n + j]);
      if (cai && ca[b * n + j]) {
        const V3 Nj = atom(bb, b, n, j, 0), CAj = atom(bb, b, n, j, 1), Cj = atom(bb, b, n, j, 2), CBj = atom(bb, b, n, j, 3);
        const float d0 = nrm(sub(CAi, Nj)), d1 = nrm(sub(CAi, CAj)), d2 = nrm(sub(CAi, Cj)), d3 = cb[b * n + j] ? nrm(sub(CAi, CBj)) : 0.f;
        w0 |= F_DIST | ((uint32_t)bucket(d0, dlim_s, 20) << 8) | ((uint32_t)bucket(d1, dlim_s, 20) << 13) | ((uint32_t)bucket(d2, dlim_s, 20) << 18) |
              ((uint32_t)bucket(d3, dlim_s, 20) << 23);
        if (cbi) {
          const float a0 = dihedral(Ni, CAi, CBi, CBj), a1 = dihedral(Nj, CAj, CBj, CBi), a2 = bond(CAi, CBi, CBj), a3 = bond(CAj, CBj, CBi),
                      a4 = dihedral(CAi, CBi, CBj, CAj);
          w0 |= F_ORI;
          w1 = (uint32_t)bucket(a0, alim_s, 20) | ((uint32_t)bucket(a1, alim_s, 20) << 5) | ((uint32_t)bucket(a2, alim_s, 20) << 10) |
               ((uint32_t)bucket(a3, alim_s, 20) << 15) | ((uint32_t)bucket(a4, alim_s, 20) << 20);
        }
      }
    }
    if (codes != nullptr && j < jend) codes[(size_t)row * n + j] = make_uint4(w0 ^ MASKED, w1, __float_as_uint(cv), 0u);
    const int cnt = jend - j0;
    for (int t0 = 0; t0 < cnt; t0 += PU) {
      float x[PU][8];
#pragma unroll
      for (int u = 0; u < PU; u++) {
        const int t = t0 + u;
        const uint32_t p0 = __shfl_sync(0xffffffffu, w0, t), p1 = __shfl_sync(0xffffffffu, w1, t);
        const float pc = __shfl_sync(0xffffffffu, cv, t);
#pragma unroll
        for (int e = 0; e < 8; e++) x[u][e] = 0.f;
        if (!(p0 & MASKED)) {
          auto add = [&](int r, float scale) {
            const uint4 v = trow<true>(tbl, r, c0);
            x[u][0] += scale * bf_lo(v.x); x[u][1] += scale * bf_hi(v.x); x[u][2] += scale * bf_lo(v.y); x[u][3] += scale * bf_hi(v.y);
            x[u][4] += scale * bf_lo(v.z); x[u][5] += scale * bf_hi(v.z); x[u][6] += scale * bf_lo(v.w); x[u][7] += scale * bf_hi(v.w);
          };
          add(p0 & 127, 1.f);
          if (p0 & F_DIST) {
            add(R_DIST + ((p0 >> 8) & 31), 1.f);
            add(R_DIST + 21 + ((p0 >> 13) & 31), 1.f);
            add(R_DIST + 42 + ((p0 >> 18) & 31), 1.f);
            add(R_DIST + 63 + ((p0 >> 23) & 31), 1.f);
          }
          if (p0 & F_ORI) {
#pragma unroll
            for (int k = 0; k < 5; k++) add(R_ANG + 21 * k + ((p1 >> (5 * k)) & 31), 1.f);
          }
          if (pc != 0.f) add(R_CHAIN, pc);
        }
      }
      store_pairs<true>(orow + (size_t)j0 * C + c0, t0, cnt, x, ln != 0, eps);
    }
  }
}

// The encoder, one non-persistent CTA per PB consecutive pairs (in (b, i, j) order, across rows): thread t bins pair t (pg_fwd's arithmetic),
// then warp w gathers and stores pairs 32 w .. 32 w + 31 PU at a time (pg_fwd's warp-per-pair layout and LayerNorm, store_pairs).
template <typename IdxT, int PB>
__global__ void __launch_bounds__(PB)
pg120_fwd_blk(const __nv_bfloat16* tbl, const float* bb, const uint8_t* ca, const uint8_t* cb, const uint8_t* mask, const IdxT* idx, const IdxT* chains,
              const float* dlim, const float* alim, __nv_bfloat16* out, uint4* codes, int n, long long P, int ln, float eps) {
  __shared__ float dlim_s[32], alim_s[32];
  __shared__ uint32_t w0s[PB], w1s[PB];
  __shared__ float cvs[PB];
  if (threadIdx.x < 32) { dlim_s[threadIdx.x] = threadIdx.x < 20 ? dlim[threadIdx.x] : 0.f; alim_s[threadIdx.x] = threadIdx.x < 20 ? alim[threadIdx.x] : 0.f; }
  __syncthreads();
  const long long pb = (long long)blockIdx.x * PB;
  {
    const long long p = pb + threadIdx.x;
    uint32_t w0 = MASKED, w1 = 0;
    float cv = 0.f;
    if (p < P) {
      const long long row = p / n;
      const int j = (int)(p - row * n), i = (int)(row % n);
      const int b = (int)(row / n);
      if (mask[b * n + i] && mask[b * n + j]) {
        const int s = (int)min(max((long long)idx[b * n + i] - (long long)idx[b * n + j] + 63, 0LL), 126LL);
        w0 = (uint32_t)s;
        cv = (float)(chains[b * n + i] * chains[b * n + j]);
        if (ca[b * n + i] && ca[b * n + j]) {
          const V3 Ni = atom(bb, b, n, i, 0), CAi = atom(bb, b, n, i, 1), CBi = atom(bb, b, n, i, 3);
          const V3 Nj = atom(bb, b, n, j, 0), CAj = atom(bb, b, n, j, 1), Cj = atom(bb, b, n, j, 2), CBj = atom(bb, b, n, j, 3);
          const float d0 = nrm(sub(CAi, Nj)), d1 = nrm(sub(CAi, CAj)), d2 = nrm(sub(CAi, Cj)), d3 = cb[b * n + j] ? nrm(sub(CAi, CBj)) : 0.f;
          w0 |= F_DIST | ((uint32_t)bucket(d0, dlim_s, 20) << 8) | ((uint32_t)bucket(d1, dlim_s, 20) << 13) | ((uint32_t)bucket(d2, dlim_s, 20) << 18) |
                ((uint32_t)bucket(d3, dlim_s, 20) << 23);
          if (cb[b * n + i]) {
            const float a0 = dihedral(Ni, CAi, CBi, CBj), a1 = dihedral(Nj, CAj, CBj, CBi), a2 = bond(CAi, CBi, CBj), a3 = bond(CAj, CBj, CBi),
                        a4 = dihedral(CAi, CBi, CBj, CAj);
            w0 |= F_ORI;
            w1 = (uint32_t)bucket(a0, alim_s, 20) | ((uint32_t)bucket(a1, alim_s, 20) << 5) | ((uint32_t)bucket(a2, alim_s, 20) << 10) |
                 ((uint32_t)bucket(a3, alim_s, 20) << 15) | ((uint32_t)bucket(a4, alim_s, 20) << 20);
          }
        }
      }
      if (codes != nullptr) codes[p] = make_uint4(w0 ^ MASKED, w1, __float_as_uint(cv), 0u);
    }
    w0s[threadIdx.x] = w0; w1s[threadIdx.x] = w1; cvs[threadIdx.x] = cv;
  }
  __syncthreads();
  pdl_wait();                                                 // the table (pg120_table) from here on
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  const long long p0 = pb + warp * 32;
  if (p0 >= P) return;
  const int cnt = (int)min(32LL, P - p0);
  for (int t0 = 0; t0 < cnt; t0 += PU) {
    float x[PU][8];
#pragma unroll
    for (int u = 0; u < PU; u++) {
      const int t = warp * 32 + t0 + u;
      const uint32_t q0 = w0s[t], q1 = w1s[t];
      const float pc = cvs[t];
#pragma unroll
      for (int e = 0; e < 8; e++) x[u][e] = 0.f;
      if (!(q0 & MASKED)) {
        auto add = [&](int r, float scale) {
          const uint4 v = trow<true>(tbl, r, c0);
          x[u][0] += scale * bf_lo(v.x); x[u][1] += scale * bf_hi(v.x); x[u][2] += scale * bf_lo(v.y); x[u][3] += scale * bf_hi(v.y);
          x[u][4] += scale * bf_lo(v.z); x[u][5] += scale * bf_hi(v.z); x[u][6] += scale * bf_lo(v.w); x[u][7] += scale * bf_hi(v.w);
        };
        add(q0 & 127, 1.f);
        if (q0 & F_DIST) {
          add(R_DIST + ((q0 >> 8) & 31), 1.f);
          add(R_DIST + 21 + ((q0 >> 13) & 31), 1.f);
          add(R_DIST + 42 + ((q0 >> 18) & 31), 1.f);
          add(R_DIST + 63 + ((q0 >> 23) & 31), 1.f);
        }
        if (q0 & F_ORI) {
#pragma unroll
          for (int k = 0; k < 5; k++) add(R_ANG + 21 * k + ((q1 >> (5 * k)) & 31), 1.f);
        }
        if (pc != 0.f) add(R_CHAIN, pc);
      }
    }
    store_pairs<true>(out + (size_t)p0 * C + c0, t0, cnt, x, ln != 0, eps);
  }
}

// The decoder, one non-persistent CTA per PB consecutive pairs (in (b, i, j) order, across rows): a half-warp per pair (16 lanes x 16 channels:
// channels 8 l .. 8 l + 7 and 128 + 8 l .. of lane l, so that a half-warp's store writes 256 contiguous bytes), its 16 lanes binning the pair
// redundantly (broadcast loads), no item loop: the launch is a stream of short CTAs, as a fill is.
template <typename IdxT, int PB>
__global__ void __launch_bounds__(PB * 16)
pg120_fwd_ca_blk(const __nv_bfloat16* __restrict__ tbl, const float* __restrict__ ca, const uint8_t* __restrict__ mask, const IdxT* __restrict__ idx,
                 const float* __restrict__ dlim, __nv_bfloat16* __restrict__ out, uint4* codes, int n, long long P, int nd, int ln, float eps) {
  __shared__ float dlim_s[32];
  if (threadIdx.x < 32) dlim_s[threadIdx.x] = threadIdx.x < nd ? dlim[threadIdx.x] : 0.f;
  __syncthreads();
  const int hl = threadIdx.x & 15, cc = hl * 8;
  const long long p = (long long)blockIdx.x * PB + (threadIdx.x >> 4);
  if (p >= P) return;                                         // whole half-warps leave together (the shuffles below are per half)
  const long long row = p / n;
  const int j = (int)(p - row * n), i = (int)(row % n);
  const long long b = row / n;
  uint32_t w0 = MASKED;
  if (mask[b * n + i] && mask[b * n + j]) {
    const float* pi = ca + (size_t)row * 3;
    const float* pj = ca + ((size_t)b * n + j) * 3;
    const float dx = pi[0] - pj[0], dy = pi[1] - pj[1], dz = pi[2] - pj[2];
    const float d = sqrtf(dx * dx + dy * dy + dz * dz);
    const long long s = (long long)idx[b * n + i] - (long long)idx[b * n + j] + 63;
    w0 = (uint32_t)min(max(s, 0LL), 126LL) | ((uint32_t)bucket(d, dlim_s, nd) << 8);
  }
  if (codes != nullptr && hl == 0) codes[p] = make_uint4(w0 ^ MASKED, 0u, 0u, 0u);
  float x[16];
#pragma unroll
  for (int e = 0; e < 16; e++) x[e] = 0.f;
  pdl_wait();                                                 // the table (pg120_table) from here on
  if (!(w0 & MASKED)) {
    const int r0 = w0 & 127, r1 = R_CAD + ((w0 >> 8) & 31);
#pragma unroll
    for (int k = 0; k < 2; k++) {
      const uint4 v = trow<true>(tbl, r0, cc + 128 * k), u = trow<true>(tbl, r1, cc + 128 * k);
      float* y = x + 8 * k;
      y[0] += bf_lo(v.x); y[1] += bf_hi(v.x); y[2] += bf_lo(v.y); y[3] += bf_hi(v.y);
      y[4] += bf_lo(v.z); y[5] += bf_hi(v.z); y[6] += bf_lo(v.w); y[7] += bf_hi(v.w);
      y[0] += bf_lo(u.x); y[1] += bf_hi(u.x); y[2] += bf_lo(u.y); y[3] += bf_hi(u.y);
      y[4] += bf_lo(u.z); y[5] += bf_hi(u.z); y[6] += bf_lo(u.w); y[7] += bf_hi(u.w);
    }
  }
  const unsigned hm = 0xFFFFu << (threadIdx.x & 16);          // this half-warp's lanes
  float s = 0.f, rs = 1.f;
  if (ln) {
#pragma unroll
    for (int e = 0; e < 16; e++) { x[e] = __bfloat162float(__float2bfloat16_rn(x[e])); s += x[e]; }
#pragma unroll
    for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(hm, s, o);
    s *= 1.f / C;
    float q = 0.f;
#pragma unroll
    for (int e = 0; e < 16; e++) { const float d = x[e] - s; q += d * d; }
#pragma unroll
    for (int o = 8; o; o >>= 1) q += __shfl_xor_sync(hm, q, o);
    rs = rsqrtf(q * (1.f / C) + eps);
  }
#pragma unroll
  for (int k = 0; k < 2; k++) {
    uint4 w;
    __nv_bfloat162* hw = reinterpret_cast<__nv_bfloat162*>(&w);
#pragma unroll
    for (int e = 0; e < 4; e++)
      hw[e] = ln ? __floats2bfloat162_rn((x[8 * k + 2 * e] - s) * rs, (x[8 * k + 2 * e + 1] - s) * rs) : __floats2bfloat162_rn(x[8 * k + 2 * e], x[8 * k + 2 * e + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)p * C + cc + 128 * k) = w;
  }
}

// The decoder: pg_fwd_ca's half-warp schedule (two pairs a warp, 16 lanes x 16 channels a pair, for the pair and its LayerNorm alike) with each
// lane's channels laid out so that a store instruction writes 256 contiguous bytes per half-warp (channels 8 l .. 8 l + 7 and 128 + 8 l .. 128 +
// 8 l + 7 of lane l of the half; the H100 kernel's lane holds 16 adjacent channels, each store half-filling 32-byte sectors), the bin inputs of
// the next item loaded ahead. The LayerNorm sums run over a different lane order than pg_fwd_ca's (the same fp32 statistics, other roundings).
template <bool CS, typename IdxT>
__global__ void __launch_bounds__(GT_THREADS, GT_CPS)
pg120_fwd_ca(const __nv_bfloat16* tbl, const float* ca, const uint8_t* mask, const IdxT* idx, const float* dlim, __nv_bfloat16* out, uint4* codes,
             int B, int n, int nd, int ln, float eps, int jp) {
  __shared__ float dlim_s[32];
  const int nwb = blockDim.x >> 5;
  if (threadIdx.x < 32) dlim_s[threadIdx.x] = threadIdx.x < nd ? dlim[threadIdx.x] : 0.f;
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, hh = lane >> 4, cc = (lane & 15) * 8;
  const int nch = (n + jp - 1) / jp;
  const long long total = (long long)B * n * nch, stride = (long long)gridDim.x * nwb;
  // the bin inputs of an item (this lane's pair), loaded one item ahead: their global-load latency overlaps the previous item's stores
  struct In { int row, j0, jend, s; float dx, dy, dz; bool live; };
  auto load_in = [&](long long item, In& r) {
    r.live = false;
    if (item >= total) return;
    r.row = (int)(item / nch);
    r.j0 = (int)(item - (long long)r.row * nch) * jp;
    r.jend = min(n, r.j0 + jp);
    const int b = r.row / n, i = r.row % n, j = r.j0 + lane;
    if (j < r.jend) {
      const float* pi = ca + (size_t)r.row * 3;
      const float* pj = ca + ((size_t)b * n + j) * 3;
      r.dx = pi[0] - pj[0]; r.dy = pi[1] - pj[1]; r.dz = pi[2] - pj[2];
      r.s = (int)max(min((long long)idx[b * n + i] - (long long)idx[b * n + j], 1LL << 20), -(1LL << 20));   // clamped to [0, 126] below
      r.live = mask[b * n + i] && mask[b * n + j];
    }
  };
  In cur, nxt;
  load_in((long long)blockIdx.x * nwb + warp, cur);
  for (long long item = (long long)blockIdx.x * nwb + warp; item < total; item += stride) {
    const int row = cur.row, j0 = cur.j0, jend = cur.jend, j = j0 + lane;
    uint32_t w0 = MASKED;
    if (cur.live) {
      const float d = sqrtf(cur.dx * cur.dx + cur.dy * cur.dy + cur.dz * cur.dz);
      w0 = (uint32_t)min(max(cur.s + 63, 0), 126) | ((uint32_t)bucket(d, dlim_s, nd) << 8);
    }
    load_in(item + stride, nxt);
    __nv_bfloat16* orow = out + (size_t)row * n * C;
    if (codes != nullptr && j < jend) codes[(size_t)row * n + j] = make_uint4(w0 ^ MASKED, 0u, 0u, 0u);
    const int cnt = jend - j0;
    for (int t0 = 0; t0 < cnt; t0 += 2 * HP) {
      float x[HP][16];
#pragma unroll
      for (int u = 0; u < HP; u++) {
        const uint32_t p0 = __shfl_sync(0xffffffffu, w0, t0 + 2 * u + hh);   // < 32; a lane past the chunk holds MASKED
#pragma unroll
        for (int e = 0; e < 16; e++) x[u][e] = 0.f;
        if (!(p0 & MASKED)) {
          auto add = [&](int r) {
#pragma unroll
            for (int k = 0; k < 2; k++) {
              const uint4 v = trow<true>(tbl, r, cc + 128 * k);
              float* y = x[u] + 8 * k;
              y[0] += bf_lo(v.x); y[1] += bf_hi(v.x); y[2] += bf_lo(v.y); y[3] += bf_hi(v.y);
              y[4] += bf_lo(v.z); y[5] += bf_hi(v.z); y[6] += bf_lo(v.w); y[7] += bf_hi(v.w);
            }
          };
          add(p0 & 127);
          add(R_CAD + ((p0 >> 8) & 31));
        }
      }
      float s[HP], rs[HP];
      if (ln) {
        float q[HP];
#pragma unroll
        for (int u = 0; u < HP; u++) {
          s[u] = 0.f;
#pragma unroll
          for (int e = 0; e < 16; e++) { x[u][e] = __bfloat162float(__float2bfloat16_rn(x[u][e])); s[u] += x[u][e]; }
        }
#pragma unroll
        for (int o = 8; o; o >>= 1)
#pragma unroll
          for (int u = 0; u < HP; u++) s[u] += __shfl_xor_sync(0xffffffffu, s[u], o);
#pragma unroll
        for (int u = 0; u < HP; u++) {
          s[u] *= 1.f / C;
          q[u] = 0.f;
#pragma unroll
          for (int e = 0; e < 16; e++) { const float d = x[u][e] - s[u]; q[u] += d * d; }
        }
#pragma unroll
        for (int o = 8; o; o >>= 1)
#pragma unroll
          for (int u = 0; u < HP; u++) q[u] += __shfl_xor_sync(0xffffffffu, q[u], o);
#pragma unroll
        for (int u = 0; u < HP; u++) rs[u] = rsqrtf(q[u] * (1.f / C) + eps);
      } else {
#pragma unroll
        for (int u = 0; u < HP; u++) { s[u] = 0.f; rs[u] = 1.f; }
      }
#pragma unroll
      for (int u = 0; u < HP; u++) {
        const int tt = t0 + 2 * u + hh;
        if (tt >= cnt) continue;
#pragma unroll
        for (int k = 0; k < 2; k++) {
          uint4 w;
          __nv_bfloat162* hw = reinterpret_cast<__nv_bfloat162*>(&w);
#pragma unroll
          for (int e = 0; e < 4; e++)
            hw[e] = ln ? __floats2bfloat162_rn((x[u][8 * k + 2 * e] - s[u]) * rs[u], (x[u][8 * k + 2 * e + 1] - s[u]) * rs[u])
                       : __floats2bfloat162_rn(x[u][8 * k + 2 * e], x[u][8 * k + 2 * e + 1]);
          uint4* d = reinterpret_cast<uint4*>(orow + (size_t)(j0 + tt) * C + cc + 128 * k);
          if constexpr (CS) __stcs(d, w);
          else *d = w;
        }
      }
    }
    cur = nxt;
  }
}
}  // namespace

void forward_ca(torch::Tensor tbl, torch::Tensor ca, torch::Tensor mask, torch::Tensor idx, torch::Tensor dlim, torch::Tensor out, torch::Tensor codes,
                int64_t ln, double eps, int64_t jp, int64_t gt) {
  if (gt < 2) { forward_ca_h100(tbl, ca, mask, idx, dlim, out, codes, ln, eps, jp, gt); return; }
  const at::cuda::CUDAGuard guard(ca.device());
  TORCH_CHECK(dlim.numel() <= 31, "at most 31 bin limits");
  TORCH_CHECK(mask.element_size() == 1 && mask.is_contiguous() && idx.is_contiguous(), "mask: contiguous bool / uint8; idx: contiguous");
  const int B = idx.size(0), n = idx.size(1);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto st = at::cuda::getCurrentCUDAStream();
  bool pdl = false;
  const torch::Tensor t = table_of(tbl, NT_CA, st, pdl);
  uint4* cp = codes.defined() && codes.numel() > 0 ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr;
  const auto* tp = reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
  const auto* mp = static_cast<const uint8_t*>(mask.data_ptr());
  const float* xp = ca.data_ptr<float>();
  const float* lp = dlim.data_ptr<float>();
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const long long P = (long long)B * n * n;
  const int nd = (int)dlim.numel(), l = (int)ln;
  const float e = (float)eps;
  if (gt == 2) {                                              // the launch: CTAs of 32 pairs
    const dim3 grid((unsigned)((P + 31) / 32)), block(512);
    if (idx.scalar_type() == at::kLong)
      launch_pdl(pg120_fwd_ca_blk<long long, 32>, grid, block, st, pdl, tp, xp, mp, reinterpret_cast<const long long*>(idx.data_ptr<int64_t>()), lp, op, cp, n,
                 P, nd, l, e);
    else
      launch_pdl(pg120_fwd_ca_blk<int, 32>, grid, block, st, pdl, tp, xp, mp, (const int*)idx.data_ptr<int>(), lp, op, cp, n, P, nd, l, e);
    return;
  }
  // gt 3: the persistent item kernel (A/B)
  const Launch L = fwd_launch(B, n, sms, 2 * GT_CPS, GT_THREADS / 32, (int)jp, 16);
  if (idx.scalar_type() == at::kLong)
    pg120_fwd_ca<false, long long><<<L.grid, L.nw * 32, 0, st>>>(tp, xp, mp, reinterpret_cast<const long long*>(idx.data_ptr<int64_t>()), lp, op, cp, B, n,
                                                               nd, l, e, L.jp);
  else
    pg120_fwd_ca<false, int><<<L.grid, L.nw * 32, 0, st>>>(tp, xp, mp, idx.data_ptr<int>(), lp, op, cp, B, n, nd, l, e, L.jp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void forward(torch::Tensor tbl, torch::Tensor bb, torch::Tensor ca, torch::Tensor cb, torch::Tensor mask, torch::Tensor idx, torch::Tensor chains,
             torch::Tensor dlim, torch::Tensor alim, torch::Tensor out, torch::Tensor codes, int64_t ln, double eps, int64_t jp, int64_t gt) {
  if (gt < 2) { forward_h100(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, out, codes, ln, eps, jp, gt); return; }
  const at::cuda::CUDAGuard guard(bb.device());
  TORCH_CHECK(dlim.numel() == 20 && alim.numel() == 20, "20 bin limits each");
  TORCH_CHECK(ca.element_size() == 1 && cb.element_size() == 1 && mask.element_size() == 1 && ca.is_contiguous() && cb.is_contiguous() &&
                  mask.is_contiguous(), "masks: contiguous bool / uint8");
  TORCH_CHECK(idx.scalar_type() == chains.scalar_type() && idx.is_contiguous() && chains.is_contiguous(), "idx / chains: one integer dtype, contiguous");
  const int B = idx.size(0), n = idx.size(1);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto st = at::cuda::getCurrentCUDAStream();
  bool pdl = false;
  const torch::Tensor t = table_of(tbl, NT, st, pdl);
  uint4* cp = codes.defined() && codes.numel() > 0 ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr;
  const Launch L = fwd_launch(B, n, sms, 2 * GT_CPS, GT_THREADS / 32, (int)jp);
  const auto* tp = reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
  const auto* m1 = static_cast<const uint8_t*>(ca.data_ptr());
  const auto* m2 = static_cast<const uint8_t*>(cb.data_ptr());
  const auto* m3 = static_cast<const uint8_t*>(mask.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  // gt 2: from 4 x 128 pairs per SM on, CTAs of 32 pairs (pg120_fwd_blk); below, the persistent item kernel (pg120_fwd), the faster there
  // (measured, the lean inputs, L2 flushed: L 128 20.5 vs 28.7 us, L 256 34.8 vs 36.6; L 384 61.4 vs 65.5, its LayerNorm 67.6 vs 71.7)
  const bool blk = gt == 2 ? (long long)B * n * n >= 4LL * sms * 128 : gt >= 4;
  if (blk) {
    // CTAs of PB pairs: 32 (gt 2 / 5); gt 4 / 6: 64 / 128 (A/B)
    const long long P = (long long)B * n * n;
    const int pb = gt == 4 ? 64 : gt == 6 ? 128 : 32;
    const dim3 grid((unsigned)((P + pb - 1) / pb)), block(pb);
    const float* bp = bb.data_ptr<float>();
    const float *dl = dlim.data_ptr<float>(), *al = alim.data_ptr<float>();
    const int l = (int)ln;
    const float e = (float)eps;
    auto go = [&](auto ip, auto hp) {
      using I = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
      if (pb == 128) launch_pdl(pg120_fwd_blk<I, 128>, grid, block, st, pdl, tp, bp, m1, m2, m3, ip, hp, dl, al, op, cp, n, P, l, e);
      else if (pb == 64) launch_pdl(pg120_fwd_blk<I, 64>, grid, block, st, pdl, tp, bp, m1, m2, m3, ip, hp, dl, al, op, cp, n, P, l, e);
      else launch_pdl(pg120_fwd_blk<I, 32>, grid, block, st, pdl, tp, bp, m1, m2, m3, ip, hp, dl, al, op, cp, n, P, l, e);
    };
    if (idx.scalar_type() == at::kLong)
      go(reinterpret_cast<const long long*>(idx.data_ptr<int64_t>()), reinterpret_cast<const long long*>(chains.data_ptr<int64_t>()));
    else
      go(static_cast<const int*>(idx.data_ptr<int>()), static_cast<const int*>(chains.data_ptr<int>()));
    return;
  }
  // the persistent item kernel (gt 2 below the threshold, gt 3)
  if (idx.scalar_type() == at::kLong)
    pg120_fwd<long long><<<L.grid, L.nw * 32, 0, st>>>(tp, bb.data_ptr<float>(), m1, m2, m3, reinterpret_cast<const long long*>(idx.data_ptr<int64_t>()),
                                                       reinterpret_cast<const long long*>(chains.data_ptr<int64_t>()), dlim.data_ptr<float>(),
                                                       alim.data_ptr<float>(), op, cp, B, n, (int)ln, (float)eps, L.jp);
  else
    pg120_fwd<int><<<L.grid, L.nw * 32, 0, st>>>(tp, bb.data_ptr<float>(), m1, m2, m3, idx.data_ptr<int>(), chains.data_ptr<int>(), dlim.data_ptr<float>(),
                                                 alim.data_ptr<float>(), op, cp, B, n, (int)ln, (float)eps, L.jp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
