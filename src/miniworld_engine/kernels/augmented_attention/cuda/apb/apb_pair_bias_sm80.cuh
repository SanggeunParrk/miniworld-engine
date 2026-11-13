// apb_pair_bias_sm80.cuh -- the pair -> bias passes of AttentionPairBias on sm_80 (A100). Included by apb_rows.cu under -DAPB_SM80, inside its
// anonymous namespace, in place of the sm_90+ kernels (bulk copies, TMA): the same arithmetic, the tiles move through a cp.async ring.
//
//   pair_bias_fwd   bias[h, i, j] = Wf[h] . LN(pair[i, j])  (d_pair 128; the LN weight is folded into Wf, its bias adds a per-head constant that the
//                   softmax cancels), head-major bf16 [NH][LP][LP], LP = L rounded up to a multiple of 128 (the attention core's tile): a key j >= L, a
//                   masked key and every entry of a padded query row i >= L carry `neg`. One read of the pair, nothing else of its size.
//
// Per tile (query row i, 128 consecutive keys j0..: 32 KB of the pair, contiguous) a warp takes 16 rows. LN folded into the projection,
//   Wf . LN(x) = rstd (Wf . x - mean sum_c Wf),
// so the mma (m16n8k16, bf16 -> fp32) runs on the raw bf16 words, and the row statistics come from the tensor cores as well (gram_step): the lane
// (g = lane / 4, q = lane % 4) holds rows g and g + 8 at the columns 32 u + 8 q + e, the dot product runs over K in any order, so Wf's B fragments use the
// same column map (apb_rows.cu's pair_bias_fwd_k explains it; this kernel only changes how the tile gets to shared memory).
//
// The tile ring: 256 threads copy 16-B granules with cp.async (a granule is 8 channels of one row); a row's 16 granules are stored at chunk c ^ 4 (row & 1),
// which makes the fragment reads (eight lanes = two rows x four granules) conflict-free. One barrier per tile: it publishes the landed tile and the
// previous tile's staged output, and frees the stage the next tile is copied into; the staged output (fp32, head-major) is written to global memory one
// tile later, under the next tile's loads.

constexpr int NS80 = 2;                                                   // ring stages
constexpr int T80 = 128 * DP * 2;                                         // bytes of a tile
template <int NH> constexpr int nhp80() { return (NH + 7) / 8 * 8; }      // heads rounded up to the mma's 8
template <int NH> constexpr int fwd80_smem() { return NS80 * T80 + 2 * nhp80<NH>() * (128 + 4) * 4; }

__device__ __forceinline__ uint32_t su32_80(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16_80(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit80() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait80() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
// byte offset of granule c (8 channels) of row r in a tile
__device__ __forceinline__ uint32_t swz80(int r, int c) { return (uint32_t)(r * (DP * 2) + ((c ^ ((r & 1) << 2)) << 4)); }

// Where Wf comes from: a packed bf16 Wf [NH, 128] (``WfBf``), or -- ``WfRaw``, the training step's -- the parameters as they are, Wf = bf16(Wb[h, c] ln_pair.weight[c] ws) formed in the
// registers (no pack kernel to wait for: the pair pass starts with the step, on a stream of its own). ``word(h, c)``: the bf16 pair (c, c + 1) of head h.
struct WfBf {
  const bf* W;
  __device__ __forceinline__ uint32_t word(int h, int c) const { return *reinterpret_cast<const uint32_t*>(W + h * DP + c); }
};
template <class WT>
struct WfRaw {
  const WT* WB;
  const float* LPW;
  float ws;
  __device__ __forceinline__ uint32_t word(int h, int c) const { return pack2(ldf(WB + h * DP + c) * LPW[c] * ws, ldf(WB + h * DP + c + 1) * LPW[c + 1] * ws); }
};

// Wf's B fragments (lane (g, q): head g + 8 nt; zero past NH) and sum_c Wf of heads 2q, 2q + 1 (+ 8 nt) for this lane's C columns
template <int NT8, int NH, class WSRC>
__device__ __forceinline__ void wf_frags80(const WSRC& W, uint32_t (&b)[NT8][8][2], float (&sw0)[NT8], float (&sw1)[NT8]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    float sw = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const bool live = g + 8 * nt < NH;
      const int hh = live ? g + 8 * nt : 0, cc = 32 * (s >> 1) + 8 * q + 4 * (s & 1);
      b[nt][s][0] = live ? W.word(hh, cc) : 0u; b[nt][s][1] = live ? W.word(hh, cc + 2) : 0u;
      const float2 u = unpack2(b[nt][s][0]), v = unpack2(b[nt][s][1]);
      sw += u.x + u.y + v.x + v.y;
    }
    sw = quad_sum(sw);                                                      // sum_c Wf[g + 8 nt, c]
    sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
  }
}

// Persistent, two blocks per SM, 8 warps: block b walks the tiles b, b + grid, ... of the LP x (LP / 128) tile grid (query row i, key tile j0).
template <int NH, class WSRC>
__global__ void __launch_bounds__(256, 2) pair_bias_fwd80_k(const bf* __restrict__ Z, const WSRC W, const bool* __restrict__ MASK,
    bf* __restrict__ OUT, int L, int LP, float eps, float neg) {
  constexpr int NT8 = nhp80<NH>() / 8, NHP = nhp80<NH>();
  extern __shared__ __align__(128) uint8_t smem[];
  float (*out_s)[NHP][128 + 4] = reinterpret_cast<float (*)[NHP][128 + 4]>(smem + NS80 * T80);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = LP / 128;
  const long ntiles = (long)LP * ntj;
  const uint32_t sbase = su32_80(smem);
  // the tiles past the last attended key (a padded tail) hold masked keys only: warp 0 takes the bound from the mask (key_len.cuh; 0: no key
  // attended, every tile masked), the stores write those tiles as neg without their pair rows (the same bits)
  __shared__ int klen_s;
  if (warp == 0) {
    const int kl = MASK != nullptr ? warp_key_len(reinterpret_cast<const uint8_t*>(MASK), L) : LP;
    if (lane == 0) klen_s = kl;
  }
  __syncthreads();
  // tile n of this block: its query row, first key, number of real keys in it (0 for a padded query row)
  auto tile_of = [&](long n, int& i, int& j0, int& rows) -> bool {
    const long t = blockIdx.x + n * (long)gridDim.x;
    if (t >= ntiles) return false;
    i = (int)(t / ntj); j0 = (int)(t % ntj) * 128;
    rows = i < L ? min(128, L - j0) : 0;
    if (j0 >= klen_s) rows = 0;                                            // every key of the tile masked: stored as neg below, not loaded or computed
    return true;
  };
  auto issue = [&](long n) {                                                // every thread: its granules of tile n into stage n % NS80 (always one group)
    int i, j0, rows;
    if (tile_of(n, i, j0, rows)) {
      const bf* src = Z + ((long)i * L + j0) * DP;
      const uint32_t dst = sbase + (uint32_t)(n % NS80) * T80;
      for (int gr = threadIdx.x; gr < rows * 16; gr += 256) {
        const int r = gr >> 4, c = gr & 15;
        cp_async16_80(dst + swz80(r, c), src + (long)r * DP + c * 8);
      }
    }
    cp_commit80();
  };
  auto store_tile = [&](long m) {                                           // the staged output of tile m: (head, 8 keys) per task, 16-B stores
    int i, j0, rows;
    tile_of(m, i, j0, rows);
    const float (*os)[128 + 4] = out_s[m & 1];
    for (int task = threadIdx.x; task < NH * 16; task += 256) {
      const int h = task >> 4, ch = task & 15, j = j0 + ch * 8;
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 v = *reinterpret_cast<const float2*>(&os[h][ch * 8 + 2 * e]);
        const int ja = j + 2 * e;
        const bool ka = i < L && ja < L && (!MASK || MASK[ja]), kb = i < L && ja + 1 < L && (!MASK || MASK[ja + 1]);
        o[e] = pack2(ka ? v.x : neg, kb ? v.y : neg);
      }
      *reinterpret_cast<uint4*>(OUT + ((long)h * LP + i) * LP + j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  };

  uint32_t b[NT8][8][2];
  float sw0[NT8], sw1[NT8];
  issue(0);                                                                 // the first tile is on its way while Wf's fragments load
  wf_frags80<NT8, NH>(W, b, sw0, sw1);
  for (long n = 0;; ++n) {
    int i = 0, j0 = 0, rows = 0;
    const bool live = tile_of(n, i, j0, rows);
    cp_wait80<0>();                                                         // tile n has landed
    __syncthreads();                                                        // ... for every thread; the staged output of tile n - 1 is complete; stage (n + 1) % 2 is free
    if (live) issue(n + 1);
    const int rb = warp * 16;
    if (live && rb < rows) {
      const uint8_t* tile = smem + (n % NS80) * T80;
      uint4 r0[4], r1[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        r0[u] = *reinterpret_cast<const uint4*>(tile + swz80(rb + g, 4 * u + q));
        r1[u] = *reinterpret_cast<const uint4*>(tile + swz80(rb + g + 8, 4 * u + q));
      }
      float c[NT8][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint32_t u0[4] = {r0[u].x, r0[u].y, r0[u].z, r0[u].w}, u1[4] = {r1[u].x, r1[u].y, r1[u].z, r1[u].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {                                       // k step s = 2 u + h: words 2h, 2h + 1
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt)
            mma16816(c[nt], u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], b[nt][2 * u + h][0], b[nt][2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float m0, rs0, m1, rs1;
      gram_stats(g0, g1, cs, eps, m0, rs0, m1, rs1);
      float (*os)[128 + 4] = out_s[n & 1];
      const int jl = rb + g;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const int h0 = 2 * q + 8 * nt;
        os[h0][jl] = rs0 * (c[nt][0] - m0 * sw0[nt]); os[h0 + 1][jl] = rs0 * (c[nt][1] - m0 * sw1[nt]);
        os[h0][jl + 8] = rs1 * (c[nt][2] - m1 * sw0[nt]); os[h0 + 1][jl + 8] = rs1 * (c[nt][3] - m1 * sw1[nt]);
      }
    }
    if (n > 0) store_tile(n - 1);
    if (!live) break;
  }
}

template <int NH, class WSRC>
void pair_bias_fwd80_t(const at::Tensor& z, const WSRC& w, const c10::optional<at::Tensor>& mask, at::Tensor& out, int64_t L, int64_t LP,
                       double eps, double neg) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_fwd80_k<NH, WSRC>, cudaFuncAttributeMaxDynamicSharedMemorySize, fwd80_smem<NH>()));
    attr = true;
  }
  const long ntiles = LP * (LP / 128);
  pair_bias_fwd80_k<NH, WSRC><<<(unsigned)std::min<long>(2L * nsm_of(z), ntiles), 256, fwd80_smem<NH>(), S()>>>(
      CP<bf>(z), w, mask ? CP<bool>(*mask) : nullptr, P<bf>(out), (int)L, (int)LP, (float)eps, (float)neg);
}

// z [L L, 128] bf16 (the real pair), w = Wf [NH, 128] bf16, mask [L] bool or none, out [NH, LP, LP] bf16 (LP: the padded length, a multiple of 128 >= L)
void pair_bias_fwd(at::Tensor z, at::Tensor w, c10::optional<at::Tensor> mask, at::Tensor out, int64_t L, double eps, double neg) {
  chk(z, at::kBFloat16, "pair"); chk(w, at::kBFloat16, "wf"); chk(out, at::kBFloat16, "bias");
  const int64_t nh = w.size(0);
  TORCH_CHECK(nh == 8 || nh == 12 || nh == 16 || nh == 24, "pair_bias (sm_80): 8, 12, 16 or 24 heads");
  const int64_t LP = (int64_t)std::llround(std::sqrt((double)out.numel() / (double)nh));
  TORCH_CHECK(L >= 1 && LP >= L && LP % 128 == 0 && LP - L < 128 && out.numel() == nh * LP * LP && z.numel() == L * L * DP && w.numel() == nh * DP,
              "pair_bias_fwd shapes: pair [L L, 128], wf [H, 128], bias [H, LP, LP] with LP = L rounded up to 128");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  const at::cuda::CUDAGuard gd(z.device());
  const WfBf wsrc{CP<bf>(w)};
  if (nh == 8) pair_bias_fwd80_t<8>(z, wsrc, mask, out, L, LP, eps, neg);
  else if (nh == 12) pair_bias_fwd80_t<12>(z, wsrc, mask, out, L, LP, eps, neg);
  else if (nh == 16) pair_bias_fwd80_t<16>(z, wsrc, mask, out, L, LP, eps, neg);
  else pair_bias_fwd80_t<24>(z, wsrc, mask, out, L, LP, eps, neg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// pair_bias_fwd with Wf formed from the parameters: wb [NH, 128] (bf16 or fp32), lpw ln_pair.weight [128] fp32, ws the scale (Wf = bf16(wb lpw ws): what ``prep_train`` packs)
void pair_bias_fwd_raw(at::Tensor z, at::Tensor wb, at::Tensor lpw, double ws, c10::optional<at::Tensor> mask, at::Tensor out, int64_t L, double eps, double neg) {
  chk(z, at::kBFloat16, "pair"); chk(out, at::kBFloat16, "bias"); chk(lpw, at::kFloat, "ln_pair.weight");
  const bool f32 = wb.scalar_type() == at::kFloat;
  TORCH_CHECK(f32 || wb.scalar_type() == at::kBFloat16, "pair_bias_fwd_raw: wb fp32 or bf16");
  const int64_t nh = wb.size(0);
  TORCH_CHECK(nh == 8 || nh == 12 || nh == 16 || nh == 24, "pair_bias (sm_80): 8, 12, 16 or 24 heads");
  const int64_t LP = (int64_t)std::llround(std::sqrt((double)out.numel() / (double)nh));
  TORCH_CHECK(L >= 1 && LP >= L && LP % 128 == 0 && LP - L < 128 && out.numel() == nh * LP * LP && z.numel() == L * L * DP && wb.numel() == nh * DP && lpw.numel() == DP,
              "pair_bias_fwd_raw shapes: pair [L L, 128], wb [H, 128], bias [H, LP, LP] with LP = L rounded up to 128");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  const at::cuda::CUDAGuard gd(z.device());
#define PB_RAW(NH_, WT_) pair_bias_fwd80_t<NH_>(z, WfRaw<WT_>{CP<WT_>(wb), CP<float>(lpw), (float)ws}, mask, out, L, LP, eps, neg)
  if (f32) { if (nh == 8) PB_RAW(8, float); else if (nh == 12) PB_RAW(12, float); else if (nh == 16) PB_RAW(16, float); else PB_RAW(24, float); }
  else { if (nh == 8) PB_RAW(8, bf); else if (nh == 12) PB_RAW(12, bf); else if (nh == 16) PB_RAW(16, bf); else PB_RAW(24, bf); }
#undef PB_RAW
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------------------- backward
// pair_bias_bwd   dpair = LN_bwd(Wf^T dbias) and dWf = dbias^T LN(pair) (+ the per-head dbias sums), over the forward's tiles (query row i, 128 keys): one more
//                 read of the pair, one write of dpair. The arithmetic is apb_rows.cu's pair_bias_bwd_k (its header comment derives it); what changes here is how
//                 the tile arrives: cp.async into a ring of (pair as two 128-B-swizzled halves of 64 channels | dbias [NH][128] fp32) stages, zero rows past the
//                 real keys (so a padded key adds exactly 0 to dWf), one barrier per tile. dbias is [NH][LP][LP] (the attention backward's, padded keys 0).
//                 8 heads: two blocks per SM; 12 / 16 heads (64 dWf accumulator registers): one block per SM, 3 stages.
#ifndef BWD80_STAGES
#define BWD80_STAGES 2
#endif
constexpr int BTP80 = 128 * DP * 2;
template <int NH> constexpr int bwd80_bps() { return NH == 8 ? 2 : 1; }
template <int NH> constexpr int bwd80_stages() { return NH == 8 ? BWD80_STAGES : 3; }
// DBT: the dtype of dbias, fp32 or bf16 (the attention backward's partial, already the bias gradient when one sample: no reduction pass between)
template <int NH, class DBT = float> constexpr int bwd80_tile() { return BTP80 + NH * 128 * (int)sizeof(DBT); }   // pair, then dbias
template <int NH, class DBT = float> constexpr int bwd80_smem() {
  constexpr int red = 8 * (NH * DP + NH) * 4, ring = bwd80_stages<NH>() * bwd80_tile<NH, DBT>();    // the end reduce reuses the ring
  return (ring > red ? ring : red) + 128 + 32 * 16 * 4 * (nhp80<NH>() / 8);
}
__device__ __forceinline__ uint32_t swzb80(int r, int c) {                  // byte offset of (row, channel) in a swizzled pair tile
  return (uint32_t)((c >> 6) * (BTP80 / 2) + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2);
}
__device__ __forceinline__ void cp_zero16_80(uint32_t dst, const void* src) {      // 16 zero bytes (src-size 0: nothing is read from src)
  const uint32_t zero = 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(zero) : "memory");
}
__device__ __forceinline__ uint4 lds128_80(uint32_t a) {                    // not merged with an earlier read of the same address
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void mma1688_80(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void ldsm_x4_trans80(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ float ldd(const float* p) { return *p; }       // a dbias value / a pair of them from shared memory
__device__ __forceinline__ float ldd(const bf* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float2 ldd2(const float* p) { return *reinterpret_cast<const float2*>(p); }
__device__ __forceinline__ float2 ldd2(const bf* p) { return unpack2(*reinterpret_cast<const uint32_t*>(p)); }

template <int NH, class DBT>
__global__ void __launch_bounds__(256, bwd80_bps<NH>()) pair_bias_bwd80_k(const bf* __restrict__ Z, const DBT* __restrict__ DB, const bf* __restrict__ W,
    bf* __restrict__ DZ, float* __restrict__ AWF, float* __restrict__ AHS, int L, int LP, float eps) {
  constexpr int NT8 = nhp80<NH>() / 8, BW = NT8, BT = bwd80_tile<NH, DBT>(), NSB = bwd80_stages<NH>();   // BW: dx^ B words per tile (one per group)
  constexpr int DCH = 128 * (int)sizeof(DBT) / 16;                          // 16-B granules of one head's 128 dbias values
  constexpr bool YP = NH == 8;                                              // S2 from the projection y (8) or a first dx^ pass (12, 16)
  constexpr int RING = NSB * BT > 8 * (NH * DP + NH) * 4 ? NSB * BT : 8 * (NH * DP + NH) * 4;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~uintptr_t(127));
  uint32_t* bwt = reinterpret_cast<uint32_t*>(smem + RING);               // [u][lane][4 tiles x BW words]: Wf[heads][f(t, g)]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + 127) / 128;
  const long ntiles = (long)L * ntj;
  const uint32_t sbase = su32_80(smem);
  auto issue = [&](long n) {                                                // every thread: its granules of tile n into stage n % NSB (always one group)
    const long t = blockIdx.x + n * (long)gridDim.x;
    if (t < ntiles) {
      const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0);
      const bf* zs = Z + ((long)i * L + j0) * DP;
      const DBT* ds = DB + (long)i * LP + j0;
      const uint32_t st = sbase + (uint32_t)(n % NSB) * BT;
      for (int gr = threadIdx.x; gr < 128 * 16; gr += 256) {                // 128 rows x 16 granules of 8 channels
        const int r = gr >> 4, k = gr & 15;
        const uint32_t dst = st + (uint32_t)(k >> 3) * (BTP80 / 2) + r * 128 + ((((k & 7) ^ (r & 7))) << 4);
        if (r < rows) cp_async16_80(dst, zs + (long)r * DP + k * 8);
        else cp_zero16_80(dst, Z);
      }
      for (int gr = threadIdx.x; gr < NH * DCH; gr += 256) {                // dbias: NH heads x DCH granules
        const int h = gr / DCH, k = gr % DCH;
        cp_async16_80(st + BTP80 + h * (DCH * 16) + k * 16, ds + (long)h * LP * LP + k * (16 / (int)sizeof(DBT)));
      }
    }
    cp_commit80();
  };
#pragma unroll
  for (int s = 0; s < NSB - 1; ++s) issue(s);                               // the first tiles are on their way while Wf's fragments load
  if (warp == 0) {                                                          // dx^ tile t's B: heads 2q, 2q+1 (| 2q+8, 2q+9) at f(t, g)
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int col = 32 * (t >> 2) + 8 * (g >> 1) + 2 * (t & 3) + (g & 1);
#pragma unroll
      for (int k = 0; k < BW; ++k)
        bwt[((t >> 2) * 32 + lane) * 4 * BW + (t & 3) * BW + k] =
            2 * q + 8 * k < NH ? pack2(__bfloat162float(W[(2 * q + 8 * k) * DP + col]), __bfloat162float(W[(2 * q + 1 + 8 * k) * DP + col])) : 0u;
    }
  }
  __syncthreads();
  uint32_t bf_[1][8][2];                                                    // Wf as the forward's B (8 heads only)
  float sw0[NT8], sw1[NT8];
  if constexpr (YP) {
    wf_frags80<1, 8>(WfBf{W}, bf_, sw0, sw1);
  } else {
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {                                      // sum_c Wf of heads 2q, 2q + 1 (+ 8 nt) alone
      float sw = 0.f;
      for (int c = 8 * q; c < DP; c += 32)
#pragma unroll
        for (int e = 0; e < 8; ++e) sw += g + 8 * nt < NH ? __bfloat162float(W[(g + 8 * nt) * DP + c + e]) : 0.f;
      sw = quad_sum(sw);
      sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
    }
  }
  float acc[8][NT8][4] = {};                                                // dWf^T tile mt: cols 16 mt + g (+8), heads 2q, 2q+1 (+8 nt)
  float hsum[NT8] = {}, corr[NT8] = {};                                     // head g + 8 nt: sum dbias, sum a mean
  const uint32_t lrow = (uint32_t)((lane & 7) + ((lane >> 4) << 3)), lcol = (uint32_t)(((lane >> 3) & 1) * 8);
  const int rb = warp * 16;
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * (long)gridDim.x;
    cp_wait80<NSB - 2>();                                                   // tile n has landed
    __syncthreads();                                                        // ... for every thread; the stage (n - 1) % NSB is free
    if (t >= ntiles) break;
    issue(n + NSB - 1);
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0), st = (int)(n % NSB);
    const uint8_t* tile = smem + st * BT;
    const uint32_t tb = su32_80(tile);
    const DBT* dbs = reinterpret_cast<const DBT*>(tile + BTP80);            // [NH][128]
    if (rb < rows) {
      float y[1][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint4 r0 = lds128_80(tb + swzb80(rb + g, 32 * u + 8 * q)), r1 = lds128_80(tb + swzb80(rb + g + 8, 32 * u + 8 * q));
        const uint32_t u0[4] = {r0.x, r0.y, r0.z, r0.w}, u1[4] = {r1.x, r1.y, r1.z, r1.w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (YP) mma16816(y[0], u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], bf_[0][2 * u + h][0], bf_[0][2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float mu0, rs0, mu1, rs1;
      gram_stats(g0, g1, cs, eps, mu0, rs0, mu1, rs1);
      // dbias of rows g, g + 8 at heads 2q + j + 8 nt (A of dx^; the row sums)
      float d0[NT8][2], d1[NT8][2], s1a = 0.f, s1b = 0.f;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const bool live = 2 * q + j + 8 * nt < NH;
          d0[nt][j] = live ? ldd(dbs + (2 * q + j + 8 * nt) * 128 + rb + g) : 0.f;
          d1[nt][j] = live ? ldd(dbs + (2 * q + j + 8 * nt) * 128 + rb + g + 8) : 0.f;
        }
        s1a += d0[nt][0] * sw0[nt] + d0[nt][1] * sw1[nt]; s1b += d1[nt][0] * sw0[nt] + d1[nt][1] * sw1[nt];
      }
      uint32_t ad0[NT8], ad1[NT8];                                          // A of dx^ per 8-head group: rows g, g + 8
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) { ad0[nt] = pack2(d0[nt][0], d0[nt][1]); ad1[nt] = pack2(d1[nt][0], d1[nt][1]); }
      // dx^ of chunk u (tiles 4u .. 4u + 3) into c[k][4]; x words of rows g, g + 8 at the same columns
      auto dxhat = [&](int u, float (&c)[4][4], uint32_t (&u0)[4], uint32_t (&u1)[4]) {
        const uint4 x0v = lds128_80(tb + swzb80(rb + g, 32 * u + 8 * q)), x1v = lds128_80(tb + swzb80(rb + g + 8, 32 * u + 8 * q));
        u0[0] = x0v.x; u0[1] = x0v.y; u0[2] = x0v.z; u0[3] = x0v.w; u1[0] = x1v.x; u1[1] = x1v.y; u1[2] = x1v.z; u1[3] = x1v.w;
        uint32_t bw4[4 * BW];
#pragma unroll
        for (int k = 0; k < BW; ++k) {
          const uint4 bv = lds128_80(su32_80(bwt + (u * 32 + lane) * 4 * BW + 4 * k));
          bw4[4 * k] = bv.x; bw4[4 * k + 1] = bv.y; bw4[4 * k + 2] = bv.z; bw4[4 * k + 3] = bv.w;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {                                       // K = the heads: m16n8k16 per two groups, m16n8k8 for an odd one
          c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0.f;
#pragma unroll
          for (int p = 0; p + 1 < NT8; p += 2)
            mma16816(c[k], ad0[p], ad1[p], ad0[p + 1], ad1[p + 1], bw4[BW * k + p], bw4[BW * k + p + 1]);
          if constexpr (NT8 & 1) mma1688_80(c[k], ad0[NT8 - 1], ad1[NT8 - 1], bw4[BW * k + NT8 - 1]);
        }
      };
      float s2a = 0.f, s2b = 0.f;
      if constexpr (YP) {
        s2a = d0[0][0] * y[0][0] + d0[0][1] * y[0][1]; s2b = d1[0][0] * y[0][2] + d1[0][1] * y[0][3];
      } else {
#pragma unroll
        for (int u = 0; u < 4; ++u) {                                       // first pass: S2 = sum_c dx^ x
          float c[4][4]; uint32_t u0[4], u1[4];
          dxhat(u, c, u0, u1);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 x0 = unpack2(u0[k]), x1 = unpack2(u1[k]);
            s2a = fmaf(c[k][0], x0.x, fmaf(c[k][1], x0.y, s2a)); s2b = fmaf(c[k][2], x1.x, fmaf(c[k][3], x1.y, s2b));
          }
        }
      }
      const float S1a = quad_sum(s1a), S1b = quad_sum(s1b), S2a = quad_sum(s2a), S2b = quad_sum(s2b);
      const float m1a = S1a * (1.f / DP), m1b = S1b * (1.f / DP);
      const float m2a = rs0 * (S2a - mu0 * S1a) * (1.f / DP), m2b = rs1 * (S2b - mu1 * S1b) * (1.f / DP);
      const float ka = -rs0 * rs0 * m2a, kb = -rs1 * rs1 * m2b;              // dpair = rstd dx^ + k x + c0
      const float ca = rs0 * (rs0 * m2a * mu0 - m1a), cb = rs1 * (rs1 * m2b * mu1 - m1b);
      const long R = (long)i * L + j0 + rb;
      const bool va = j0 + rb + g < L, vb = j0 + rb + g + 8 < L;            // the last tile's rows past L are not in dpair
#pragma unroll
      for (int u = 0; u < 4; ++u) {                                         // tiles 4u .. 4u + 3 fill chunk u: one 16-B store
        float c[4][4]; uint32_t u0[4], u1[4];
        dxhat(u, c, u0, u1);
        uint32_t o0[4], o1[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 x0 = unpack2(u0[k]), x1 = unpack2(u1[k]);
          o0[k] = pack2(fmaf(rs0, c[k][0], fmaf(ka, x0.x, ca)), fmaf(rs0, c[k][1], fmaf(ka, x0.y, ca)));
          o1[k] = pack2(fmaf(rs1, c[k][2], fmaf(kb, x1.x, cb)), fmaf(rs1, c[k][3], fmaf(kb, x1.y, cb)));
        }
        if (va) *reinterpret_cast<uint4*>(DZ + (R + g) * DP + 32 * u + 8 * q) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
        if (vb) *reinterpret_cast<uint4*>(DZ + (R + g + 8) * DP + 32 * u + 8 * q) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
      }
      // dWf: B = a = rstd dbias[head g + 8 nt][rows 2q, 2q+1 | 2q+8, 2q+9]; rstd / mean of those rows from their lanes
      const float ra = __shfl_sync(0xffffffffu, rs0, 8 * q), rb_ = __shfl_sync(0xffffffffu, rs0, 8 * q + 4);
      const float rc = __shfl_sync(0xffffffffu, rs1, 8 * q), rd = __shfl_sync(0xffffffffu, rs1, 8 * q + 4);
      const float ma = __shfl_sync(0xffffffffu, mu0, 8 * q), mb = __shfl_sync(0xffffffffu, mu0, 8 * q + 4);
      const float mc = __shfl_sync(0xffffffffu, mu1, 8 * q), md = __shfl_sync(0xffffffffu, mu1, 8 * q + 4);
      uint32_t db0[NT8], db1[NT8];
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const bool live = g + 8 * nt < NH;
        const float2 e01 = live ? ldd2(dbs + (g + 8 * nt) * 128 + rb + 2 * q) : make_float2(0.f, 0.f);
        const float2 e89 = live ? ldd2(dbs + (g + 8 * nt) * 128 + rb + 2 * q + 8) : make_float2(0.f, 0.f);
        const float aa = ra * e01.x, ab = rb_ * e01.y, ac = rc * e89.x, ad = rd * e89.y;
        hsum[nt] += e01.x + e01.y + e89.x + e89.y;
        corr[nt] += aa * ma + ab * mb + ac * mc + ad * md;
        db0[nt] = pack2(aa, ab); db1[nt] = pack2(ac, ad);
      }
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        uint32_t a[4];                                                      // A = x^T: [16 columns][16 rows]
        ldsm_x4_trans80(a, tb + swzb80(rb + (int)lrow, 16 * mt + (int)lcol));
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) mma16816(acc[mt][nt], a[0], a[1], a[2], a[3], db0[nt], db1[nt]);
      }
    }
  }
  // dWf -= the per-head mean term, then block-reduce the warps' tiles and head sums; one atomic add per entry
  constexpr int RS = NH * DP + NH;
  float* red = reinterpret_cast<float*>(smem);                              // the ring is idle now
  float* pw = red + warp * RS;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    const float cq = quad_sum(corr[nt]), hq = quad_sum(hsum[nt]);           // head g + 8 nt
    const float c0 = __shfl_sync(0xffffffffu, cq, 8 * q), c1 = __shfl_sync(0xffffffffu, cq, 8 * q + 4);   // heads 2q, 2q + 1 (+ 8 nt)
    const int h0 = 2 * q + 8 * nt;
    if (h0 < NH) {                                                          // (NH is even: h0 + 1 < NH too)
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        pw[h0 * DP + 16 * mt + g] = acc[mt][nt][0] - c0; pw[(h0 + 1) * DP + 16 * mt + g] = acc[mt][nt][1] - c1;
        pw[h0 * DP + 16 * mt + g + 8] = acc[mt][nt][2] - c0; pw[(h0 + 1) * DP + 16 * mt + g + 8] = acc[mt][nt][3] - c1;
      }
    }
    if (q == 0 && g + 8 * nt < NH) pw[NH * DP + g + 8 * nt] = hq;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < RS; c += 256) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w * RS + c];
    atomicAdd(c < NH * DP ? AWF + c : AHS + (c - NH * DP), v);
  }
}

template <int NH, class DBT>
void pair_bias_bwd80_t(const at::Tensor& z, const at::Tensor& dbias, const at::Tensor& w, at::Tensor& dz, at::Tensor& acc, int64_t L, int64_t LP,
                       double eps, const Geo& G, int64_t grid) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_bwd80_k<NH, DBT>, cudaFuncAttributeMaxDynamicSharedMemorySize, bwd80_smem<NH, DBT>()));
    attr = true;
  }
  const long ntiles = L * ((L + 127) / 128);
  float* awf = P<float>(acc) + 2 * G.d + width(G);
  if (grid <= 0) grid = (long)bwd80_bps<NH>() * nsm_of(z);
  pair_bias_bwd80_k<NH, DBT><<<(unsigned)std::min<long>(grid, ntiles), 256, bwd80_smem<NH, DBT>(), S()>>>(
      CP<bf>(z), CP<DBT>(dbias), CP<bf>(w), P<bf>(dz), awf, awf + NH * DP, (int)L, (int)LP, (float)eps);
}

// z [L L, 128] bf16, dbias [NH, LP, LP] fp32 (the gradient of the NATURAL-unit bias; LP >= L the padded length), w = Wf [NH, 128] bf16 (natural units),
// dz [L L, 128] bf16 out; acc (``acc_n`` floats, zeroed by the caller): dWf [NH, 128] and the per-head dbias sums are added at 2 d + W
// grid: CTAs of the persistent pass (0: the full occupancy, so none is left for a kernel running next to it)
void pair_bias_bwd_grid(at::Tensor z, at::Tensor dbias, at::Tensor w, at::Tensor dz, at::Tensor acc, int64_t L, double eps, int64_t d, int64_t grid) {
  chk(z, at::kBFloat16, "pair"); chk(w, at::kBFloat16, "wf"); chk(dz, at::kBFloat16, "dpair");
  const bool dbf = dbias.scalar_type() == at::kBFloat16;                    // dbias: fp32, or bf16 (read as it is)
  TORCH_CHECK(dbf || dbias.scalar_type() == at::kFloat, "pair_bias_bwd: dbias fp32 or bf16");
  const int64_t nh = w.size(0);
  TORCH_CHECK(nh == 8 || nh == 12 || nh == 16 || nh == 24, "pair_bias_bwd (sm_80): 8, 12, 16 or 24 heads");
  const Geo G = geo(nh, d);
  chk_acc(acc, G);
  const int64_t LP = (int64_t)std::llround(std::sqrt((double)dbias.numel() / (double)nh));
  TORCH_CHECK(L >= 1 && LP >= L && LP % 128 == 0 && LP - L < 128 && dbias.numel() == nh * LP * LP && z.numel() == L * L * DP && dz.numel() == z.numel(),
              "pair_bias_bwd shapes: pair [L L, 128], dbias [H, LP, LP] with LP = L rounded up to 128");
  const at::cuda::CUDAGuard gd(z.device());
#define PB_BWD(NH_) { if (dbf) pair_bias_bwd80_t<NH_, bf>(z, dbias, w, dz, acc, L, LP, eps, G, grid); else pair_bias_bwd80_t<NH_, float>(z, dbias, w, dz, acc, L, LP, eps, G, grid); }
  if (nh == 8) PB_BWD(8)
  else if (nh == 12) PB_BWD(12)
  else if (nh == 16) PB_BWD(16)
  else PB_BWD(24)
#undef PB_BWD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pair_bias_bwd(at::Tensor z, at::Tensor dbias, at::Tensor w, at::Tensor dz, at::Tensor acc, int64_t L, double eps, int64_t d) {
  pair_bias_bwd_grid(z, dbias, w, dz, acc, L, eps, d, 0);
}

// ---------------------------------------------------------------------------------------------------------- single-track rows, bf16 core outputs
// The A100 attention core returns o and the gradients dq / dk / dv in bf16 (the B200 cores: fp32): the same three row passes as gate_rows / gate_bwd /
// qkv_bwd above, over bf16 inputs. A row past M (the padded rows of the core's length) is not touched.

// og = sigmoid(g) o over W = 128 V columns (g: the qkvg row's last quarter)
template <int V>
__global__ void __launch_bounds__(NT) gate_rows_bf_k(const bf* __restrict__ O, const bf* __restrict__ QKVG, bf* __restrict__ OG, int M) {
  constexpr int WD = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  float4 o[V], g[V];
  ldv<V>(O + r * WD, o); ldv<V>(QKVG + r * 4 * WD + 3 * WD, g);
#pragma unroll
  for (int k = 0; k < V; ++k) o[k] = mul4(sig4(g[k]), o[k]);
  stv<V>(OG + r * WD, o);
}

// og = sigmoid(g) o: dO = dog s (bf16, the core's input), dg = dog o s (1 - s) into the qkvg gradient's last quarter
template <int V>
__global__ void __launch_bounds__(NT) gate_bwd_bf_k(const bf* __restrict__ DOG, const bf* __restrict__ O, const bf* __restrict__ QKVG,
    bf* __restrict__ DOB, bf* __restrict__ DQKVG, int M) {
  constexpr int WD = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  float4 dog[V], o[V], s[V], d_o[V], dg[V];
  ldv<V>(DOG + r * WD, dog); ldv<V>(O + r * WD, o); ldv<V>(QKVG + r * 4 * WD + 3 * WD, s);
#pragma unroll
  for (int k = 0; k < V; ++k) {
    s[k] = sig4(s[k]);
    d_o[k] = mul4(dog[k], s[k]);
    const float4 t = mul4(dog[k], o[k]);
    dg[k] = make_float4(t.x * s[k].x * (1.f - s[k].x), t.y * s[k].y * (1.f - s[k].y), t.z * s[k].z * (1.f - s[k].z),
                        t.w * s[k].w * (1.f - s[k].w));
  }
  stv<V>(DOB + r * WD, d_o);
  stv<V>(DQKVG + r * 4 * WD + 3 * WD, dg);
}

// dq | dk | dv bf16 [., W] -> dqkvg[:, 0:3W]; ABQ[c] += sum_r dq
template <int V>
__global__ void __launch_bounds__(NT) qkv_bwd_bf_k(const bf* __restrict__ DQ, const bf* __restrict__ DK, const bf* __restrict__ DV,
    bf* __restrict__ DQKVG, float* __restrict__ ABQ, int M) {
  constexpr int WD = 128 * V;
  __shared__ __align__(16) float red[RW * WD];
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  float4 v[V] = {};
  if (r < M) {
    ldv<V>(DQ + r * WD, v);
    stv<V>(DQKVG + r * 4 * WD, v);
    float4 t[V];
    ldv<V>(DK + r * WD, t); stv<V>(DQKVG + r * 4 * WD + WD, t);
    ldv<V>(DV + r * WD, t); stv<V>(DQKVG + r * 4 * WD + 2 * WD, t);
  }
  colsum_add<V>(v, red, ABQ, 0);  // acc_stride 0: atomic add (the A100 accumulator is one vector)
}

// o [>= M, W] bf16, qkvg [., 4 W] bf16, og [M, W] bf16; W = 384 (8 x 48, 12 x 32) or 512 (16 x 24 padded, 16 x 32)
void gate_rows_bf(at::Tensor o, at::Tensor qkvg, at::Tensor og, int64_t M) {
  const int64_t W = o.size(1);
  chk(o, at::kBFloat16, "o"); chk(qkvg, at::kBFloat16, "qkvg"); chk(og, at::kBFloat16, "og");
  TORCH_CHECK((W == 384 || W == 512) && qkvg.size(1) == 4 * W && og.size(1) == W && o.size(0) >= M && qkvg.size(0) >= M && og.size(0) >= M, "gate_rows_bf shapes");
  const at::cuda::CUDAGuard gd(o.device());
  if (W == 384) gate_rows_bf_k<3><<<blocks(M), NT, 0, S()>>>(CP<bf>(o), CP<bf>(qkvg), P<bf>(og), (int)M);
  else gate_rows_bf_k<4><<<blocks(M), NT, 0, S()>>>(CP<bf>(o), CP<bf>(qkvg), P<bf>(og), (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_bwd_bf(at::Tensor dog, at::Tensor o, at::Tensor qkvg, at::Tensor dob, at::Tensor dqkvg, int64_t M) {
  const int64_t W = o.size(1);
  chk(dog, at::kBFloat16, "dog"); chk(o, at::kBFloat16, "o"); chk(qkvg, at::kBFloat16, "qkvg"); chk(dob, at::kBFloat16, "dob");
  chk(dqkvg, at::kBFloat16, "dqkvg");
  TORCH_CHECK((W == 384 || W == 512) && dog.size(1) == W && dob.size(1) == W && qkvg.size(1) == 4 * W && dqkvg.size(1) == 4 * W && dog.size(0) >= M &&
              o.size(0) >= M && qkvg.size(0) >= M && dob.size(0) >= M && dqkvg.size(0) >= M, "gate_bwd_bf shapes");
  const at::cuda::CUDAGuard gd(o.device());
  if (W == 384) gate_bwd_bf_k<3><<<blocks(M), NT, 0, S()>>>(CP<bf>(dog), CP<bf>(o), CP<bf>(qkvg), P<bf>(dob), P<bf>(dqkvg), (int)M);
  else gate_bwd_bf_k<4><<<blocks(M), NT, 0, S()>>>(CP<bf>(dog), CP<bf>(o), CP<bf>(qkvg), P<bf>(dob), P<bf>(dqkvg), (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// acc: the step's accumulator (d = d_single locates its dbq slice)
void qkv_bwd_bf(at::Tensor dq, at::Tensor dk, at::Tensor dv, at::Tensor dqkvg, at::Tensor acc, int64_t d, int64_t M) {
  const int64_t W = dk.size(1);
  chk(dq, at::kBFloat16, "dq"); chk(dk, at::kBFloat16, "dk"); chk(dv, at::kBFloat16, "dv"); chk(dqkvg, at::kBFloat16, "dqkvg"); chk(acc, at::kFloat, "acc");
  TORCH_CHECK((W == 384 || W == 512) && dq.size(1) == W && dv.size(1) == W && dqkvg.size(1) == 4 * W && dq.size(0) >= M && dk.size(0) >= M && dv.size(0) >= M &&
              dqkvg.size(0) >= M, "qkv_bwd_bf shapes");
  const at::cuda::CUDAGuard gd(dq.device());
  float* abq = P<float>(acc) + 2 * d;
  if (W == 384) qkv_bwd_bf_k<3><<<blocks(M), NT, 0, S()>>>(CP<bf>(dq), CP<bf>(dk), CP<bf>(dv), P<bf>(dqkvg), abq, (int)M);
  else qkv_bwd_bf_k<4><<<blocks(M), NT, 0, S()>>>(CP<bf>(dq), CP<bf>(dk), CP<bf>(dv), P<bf>(dqkvg), abq, (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
