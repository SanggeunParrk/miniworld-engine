// Backbone pair embedding of La-Proteina's autoencoder encoder on sm_90a (H100), forward only; 256 channels.
//
//   pair[b,i,j,:] = M_ij * ( T[sep] + (ca_i ca_j) (T[d_CA-N] + T[d_CA-CA] + T[d_CA-C] + T[d_CA-CB])
//                            + (ca_i ca_j cb_i) (T[theta_12] + T[theta_21] + T[phi_12] + T[phi_21] + T[omega]) + chain_i chain_j T[chain] )
//
// The reference builds 317 binned one-hot channels per pair (sequence separation, four CA-to-backbone distances, five inter-residue
// orientation angles, a chain product), a bias-free Linear over them and a mask.  A one-hot times a matrix is a row gather: this kernel
// bins the pair (distances, dihedrals and bond angles as torch computes them), and sums up to ten rows of the table, which sits in shared
// memory as bf16 (autocast runs the Linear on bf16 weights; the sum is fp32).  Persistent CTAs, one warp per (b, i) row, 32 pairs at a time.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int C = 256;
constexpr int NW = 8;
constexpr int NT = 317;                                  // 127 sep + 4*21 dist + 5*21 angle + 1 chain
constexpr int R_DIST = 127, R_ANG = 127 + 84, R_CHAIN = 316;
constexpr uint32_t MASKED = 1u << 31, F_DIST = 1u << 28, F_ORI = 1u << 29;
constexpr uint32_t LIVE = 1u << 31;          // bit 31 of the stored code word: the pair is live (the forward's word 0 xor MASKED)
constexpr int NT_CA = 157, R_CAD = 127;

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// bf16 store of 8 channels per lane; ln != 0: first the LayerNorm (no affine) over the 256 channels of the pair, on the values as a bf16 Linear output
// would hold them (what torch's layer_norm sees), statistics in fp32
__device__ __forceinline__ void store_pair(__nv_bfloat16* dst, float (&x)[8], bool ln, float eps) {
  if (ln) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) { x[e] = __bfloat162float(__float2bfloat16_rn(x[e])); s += x[e]; }
    float mean = wsum(s) * (1.f / C);
    float q = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) { float d = x[e] - mean; q += d * d; }
    float rstd = rsqrtf(wsum(q) * (1.f / C) + eps);
#pragma unroll
    for (int e = 0; e < 8; e++) x[e] = (x[e] - mean) * rstd;
  }
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; k++) h[k] = __floats2bfloat162_rn(x[2 * k], x[2 * k + 1]);
  *reinterpret_cast<uint4*>(dst) = u;
}

struct V3 { float x, y, z; };
__device__ __forceinline__ V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ __forceinline__ float nrm(V3 a) { return sqrtf(dot(a, a)); }
__device__ __forceinline__ V3 unit(V3 a) { float n = fmaxf(nrm(a), 1e-8f); return {a.x / n, a.y / n, a.z / n}; }
__device__ __forceinline__ float dihedral(V3 a, V3 b, V3 c, V3 d) {
  V3 b0 = sub(b, a), b1 = sub(c, b), b2 = sub(d, c);
  V3 n1 = unit(cross(b0, b1)), n2 = unit(cross(b1, b2));
  float cosv = dot(n1, n2);
  V3 x = cross(n1, n2);
  float s = dot(x, b1), sg = (s > 0.f) - (s < 0.f);
  return atan2f(sg * nrm(x), cosv);
}
__device__ __forceinline__ float bond(V3 a, V3 b, V3 c) {
  V3 b0 = unit(sub(b, a)), b1 = unit(sub(c, a));
  return atan2f(nrm(cross(b0, b1)), dot(b0, b1));
}
__device__ __forceinline__ int bucket(float v, const float* lim, int n) {   // torch.bucketize: limits strictly below v
  int c = 0;
  for (int k = 0; k < n; k++) c += (__ldg(lim + k) < v);
  return c;
}
__device__ __forceinline__ V3 atom(const float* bb, int b, int n, int i, int a) {
  const float* p = bb + (((size_t)b * n + i) * 4 + a) * 3;
  return {p[0], p[1], p[2]};
}

__global__ void __launch_bounds__(NW * 32, 1)
pg_fwd(const __nv_bfloat16* tbl, const float* bb, const uint8_t* ca, const uint8_t* cb, const uint8_t* mask, const int* idx, const int* chains,
       const float* dlim, const float* alim, __nv_bfloat16* out, uint4* codes, int B, int n, int ln, float eps) {
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* T = reinterpret_cast<__nv_bfloat16*>(smem);          // [NT][C]
  for (int k = threadIdx.x; k < NT * C / 8; k += NW * 32) reinterpret_cast<uint4*>(T)[k] = reinterpret_cast<const uint4*>(tbl)[k];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  for (int row = blockIdx.x * NW + warp; row < B * n; row += gridDim.x * NW) {
    int b = row / n, i = row % n;
    bool mi = mask[b * n + i], cai = ca[b * n + i], cbi = cb[b * n + i];
    V3 Ni = atom(bb, b, n, i, 0), CAi = atom(bb, b, n, i, 1), CBi = atom(bb, b, n, i, 3);
    __nv_bfloat16* orow = out + (size_t)row * n * C;
    for (int j0 = 0; j0 < n; j0 += 32) {
      int j = j0 + lane;
      uint32_t w0 = MASKED, w1 = 0;
      float cv = 0.f;
      if (j < n && mi && mask[b * n + j]) {
        int s = min(max(idx[b * n + i] - idx[b * n + j] + 63, 0), 126);
        w0 = (uint32_t)s;
        cv = (float)(chains[b * n + i] * chains[b * n + j]);
        if (cai && ca[b * n + j]) {
          V3 Nj = atom(bb, b, n, j, 0), CAj = atom(bb, b, n, j, 1), Cj = atom(bb, b, n, j, 2), CBj = atom(bb, b, n, j, 3);
          float d0 = nrm(sub(CAi, Nj)), d1 = nrm(sub(CAi, CAj)), d2 = nrm(sub(CAi, Cj)), d3 = cb[b * n + j] ? nrm(sub(CAi, CBj)) : 0.f;
          w0 |= F_DIST | ((uint32_t)bucket(d0, dlim, 20) << 8) | ((uint32_t)bucket(d1, dlim, 20) << 13) | ((uint32_t)bucket(d2, dlim, 20) << 18) |
                ((uint32_t)bucket(d3, dlim, 20) << 23);
          if (cbi) {
            float a0 = dihedral(Ni, CAi, CBi, CBj), a1 = dihedral(Nj, CAj, CBj, CBi), a2 = bond(CAi, CBi, CBj), a3 = bond(CAj, CBj, CBi),
                  a4 = dihedral(CAi, CBi, CBj, CAj);
            w0 |= F_ORI;
            w1 = (uint32_t)bucket(a0, alim, 20) | ((uint32_t)bucket(a1, alim, 20) << 5) | ((uint32_t)bucket(a2, alim, 20) << 10) |
                 ((uint32_t)bucket(a3, alim, 20) << 15) | ((uint32_t)bucket(a4, alim, 20) << 20);
          }
        }
      }
      if (codes != nullptr && j < n) codes[(size_t)row * n + j] = make_uint4(w0 ^ MASKED, w1, __float_as_uint(cv), 0u);   // the backward replays the binning from these (bit 31: live pair)
      int cnt = min(32, n - j0);
      for (int t = 0; t < cnt; t++) {
        uint32_t p0 = __shfl_sync(0xffffffffu, w0, t), p1 = __shfl_sync(0xffffffffu, w1, t);
        float pc = __shfl_sync(0xffffffffu, cv, t);
        float x[8];
#pragma unroll
        for (int e = 0; e < 8; e++) x[e] = 0.f;
        if (!(p0 & MASKED)) {
          auto add = [&](int r, float scale) {
            uint4 u = *reinterpret_cast<const uint4*>(T + (size_t)r * C + c0);
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
            for (int k = 0; k < 4; k++) { float2 f = __bfloat1622float2(h[k]); x[2 * k] += scale * f.x; x[2 * k + 1] += scale * f.y; }
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
        store_pair(orow + (size_t)(j0 + t) * C + c0, x, ln != 0, eps);
      }
    }
  }
}

void forward(torch::Tensor tbl, torch::Tensor bb, torch::Tensor ca, torch::Tensor cb, torch::Tensor mask, torch::Tensor idx, torch::Tensor chains,
             torch::Tensor dlim, torch::Tensor alim, torch::Tensor out, torch::Tensor codes, int64_t ln, double eps) {
  const at::cuda::CUDAGuard guard(tbl.device());
  int B = idx.size(0), n = idx.size(1);
  size_t smem = (size_t)NT * C * sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(pg_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int grid = std::min(sms, (B * n + NW - 1) / NW);
  pg_fwd<<<grid, NW * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>()), bb.data_ptr<float>(), ca.data_ptr<uint8_t>(), cb.data_ptr<uint8_t>(),
      mask.data_ptr<uint8_t>(), idx.data_ptr<int>(), chains.data_ptr<int>(), dlim.data_ptr<float>(), alim.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), codes.defined() && codes.numel() > 0 ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr, B, n, (int)ln, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Decoder pair embedding: pair[b,i,j] = M_ij * (T[sep] + T[127 + bin(|ca_i - ca_j|)]),  table [157][256] (127 sequence separations, 30 CA distances).
// ---------------------------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(NW * 32, 1)
pg_fwd_ca(const __nv_bfloat16* tbl, const float* ca, const uint8_t* mask, const int* idx, const float* dlim, __nv_bfloat16* out, uint4* codes,
          int B, int n, int nd, int ln, float eps) {
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* T = reinterpret_cast<__nv_bfloat16*>(smem);          // [NT_CA][C]
  for (int k = threadIdx.x; k < NT_CA * C / 8; k += NW * 32) reinterpret_cast<uint4*>(T)[k] = reinterpret_cast<const uint4*>(tbl)[k];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  for (int row = blockIdx.x * NW + warp; row < B * n; row += gridDim.x * NW) {
    int b = row / n, i = row % n;
    bool mi = mask[b * n + i];
    const float* pi = ca + (size_t)row * 3;
    float xi = pi[0], yi = pi[1], zi = pi[2];
    __nv_bfloat16* orow = out + (size_t)row * n * C;
    for (int j0 = 0; j0 < n; j0 += 32) {
      int j = j0 + lane;
      uint32_t w0 = MASKED;
      if (j < n && mi && mask[b * n + j]) {
        const float* pj = ca + ((size_t)b * n + j) * 3;
        float dx = xi - pj[0], dy = yi - pj[1], dz = zi - pj[2];
        float d = sqrtf(dx * dx + dy * dy + dz * dz);
        w0 = (uint32_t)min(max(idx[b * n + i] - idx[b * n + j] + 63, 0), 126) | ((uint32_t)bucket(d, dlim, nd) << 8);
      }
      if (codes != nullptr && j < n) codes[(size_t)row * n + j] = make_uint4(w0 ^ MASKED, 0u, 0u, 0u);
      int cnt = min(32, n - j0);
      for (int t = 0; t < cnt; t++) {
        uint32_t p0 = __shfl_sync(0xffffffffu, w0, t);
        float x[8];
#pragma unroll
        for (int e = 0; e < 8; e++) x[e] = 0.f;
        if (!(p0 & MASKED)) {
          auto add = [&](int r) {
            uint4 u = *reinterpret_cast<const uint4*>(T + (size_t)r * C + c0);
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
            for (int k = 0; k < 4; k++) { float2 f = __bfloat1622float2(h[k]); x[2 * k] += f.x; x[2 * k + 1] += f.y; }
          };
          add(p0 & 127);
          add(R_CAD + ((p0 >> 8) & 31));
        }
        store_pair(orow + (size_t)(j0 + t) * C + c0, x, ln != 0, eps);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Backward of both builders w.r.t. the table: dT[r][c] = sum over the pairs that use row r of dpair[pair][c]  (x the chain product for the chain row).
//
// The forward leaves one 16-byte code per pair: word 0 = the packed bins (bit 31: the pair is live, i.e. not padding), word 1 = the five angle bins,
// word 2 = the chain product (float). dT is a product of the one-hot matrix F^T [rows x pairs] and the gradient G [pairs x 256]: the tensor-core
// kernel (impl 1) builds the bf16 one-hot A fragments of each feature group straight from the codes (nothing is materialised), takes the B
// fragments of a 64-pair x 64-channel tile of G with ldmatrix.trans from a cp.async ring, and accumulates in fp32 registers. Warp w owns two
// 16-row m-tiles of one feature group (sequence separation + chain: 4 warps, each of the 4 distances: 1, each of the 5 angles: 1; decoder: 4 + 1),
// CTA (q, t) = channel slice q (64) of pair slice t (a contiguous range, tiles in order); the 33 slice partials are added in slice order by
// pg_reduce_tbl. Bitwise deterministic, no atomics. impl 0 is the earlier shared-memory read-modify-write kernel (kept for A/B).
// ---------------------------------------------------------------------------------------------------------------------------------------
constexpr int NBLK = 33, TU = 4;
__device__ __forceinline__ int code_bin(const uint4& c, int kind, int k) {
  if (!(c.x & LIVE)) return -1;
  switch (kind) {
    case 0: return (int)(c.x & 127);
    case 1: return (c.x & F_DIST) ? (int)((c.x >> (8 + 5 * k)) & 31) : -1;
    case 2: return (c.x & F_ORI) ? (int)((c.y >> (5 * k)) & 31) : -1;
    default: return (int)((c.x >> 8) & 31);          // decoder: CA distance bin
  }
}

template <int MODE>
__global__ void __launch_bounds__(128, 1)
pg_bwd_tbl(const __nv_bfloat16* g, const uint4* codes, long long P, float* tpart) {
  constexpr int NTAB = MODE == 0 ? NT : NT_CA;
  extern __shared__ float acc[];                            // [4 warps][NTAB][32]
  int q = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int k = threadIdx.x; k < 4 * NTAB * 32; k += 128) acc[k] = 0.f;
  __syncthreads();
  float* my = acc + (size_t)w * NTAB * 32 + lane;
  long long per = (P + NBLK - 1) / NBLK, p0 = (long long)t * per, p1 = min(P, p0 + per);
  for (long long p = p0 + w; p < p1; p += 4 * TU) {
    uint4 cd[TU];
    float f[TU];
#pragma unroll
    for (int u = 0; u < TU; u++) {
      long long pp = p + 4 * u;
      cd[u] = pp < p1 ? codes[pp] : make_uint4(0u, 0u, 0u, 0u);
      f[u] = 0.f;
      if (cd[u].x & LIVE) f[u] = __bfloat162float(g[(size_t)pp * C + q * 32 + lane]);
    }
#pragma unroll
    for (int u = 0; u < TU; u++) {
      if (!(cd[u].x & LIVE)) continue;
      uint32_t w0 = cd[u].x ^ MASKED, w1 = cd[u].y;
      float v = f[u];
      if (MODE == 0) {
        int r[11];
        float sc[11];
        int na = 0;
        r[na] = w0 & 127; sc[na++] = 1.f;
        if (w0 & F_DIST) {
          r[na] = R_DIST + ((w0 >> 8) & 31); sc[na++] = 1.f;
          r[na] = R_DIST + 21 + ((w0 >> 13) & 31); sc[na++] = 1.f;
          r[na] = R_DIST + 42 + ((w0 >> 18) & 31); sc[na++] = 1.f;
          r[na] = R_DIST + 63 + ((w0 >> 23) & 31); sc[na++] = 1.f;
        }
        if (w0 & F_ORI) {
#pragma unroll
          for (int k = 0; k < 5; k++) { r[na] = R_ANG + 21 * k + ((w1 >> (5 * k)) & 31); sc[na++] = 1.f; }
        }
        float cv = __uint_as_float(cd[u].z);
        if (cv != 0.f) { r[na] = R_CHAIN; sc[na++] = cv; }
        // the rows of one pair are distinct: read them all, then write them all (no read-after-write chain through shared memory)
        float cur[11];
#pragma unroll
        for (int k = 0; k < 11; k++) if (k < na) cur[k] = my[(size_t)r[k] * 32];
#pragma unroll
        for (int k = 0; k < 11; k++) if (k < na) my[(size_t)r[k] * 32] = cur[k] + sc[k] * v;
      } else {
        int r0 = w0 & 127, r1 = R_CAD + ((w0 >> 8) & 31);
        float c0_ = my[(size_t)r0 * 32], c1_ = my[(size_t)r1 * 32];
        my[(size_t)r0 * 32] = c0_ + v; my[(size_t)r1 * 32] = c1_ + v;
      }
    }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < NTAB * 32; k += 128) {
    float s = acc[k] + acc[NTAB * 32 + k] + acc[2 * NTAB * 32 + k] + acc[3 * NTAB * 32 + k];
    tpart[((size_t)t * NTAB + k / 32) * C + q * 32 + k % 32] = s;
  }
}

// ---- tensor-core kernel -----------------------------------------------------------------------------------------------------------------
constexpr int ST = 4, TP = 64;                              // cp.async stages, pairs per tile

__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  int sz = pred ? 16 : 0;                                   // src-size 0: zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned bf16_bits(float v) { return (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(v)); }

template <int MODE>
__global__ void __launch_bounds__((MODE == 0 ? 13 : 5) * 32, 1)
pg_bwd_mma(const __nv_bfloat16* g, const uint4* codes, long long P, float* tpart) {
  constexpr int NW = MODE == 0 ? 13 : 5;
  constexpr int NTAB = MODE == 0 ? NT : NT_CA;
  __shared__ __align__(128) unsigned char Bs[ST][TP * 128];  // 64 pairs x 64 channels bf16, 16-byte chunks XOR-swizzled by (pair & 7)
  __shared__ __align__(16) uint4 Cs[ST][TP];
  const int q = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const long long per = (P + NBLK - 1) / NBLK, p0 = (long long)t * per, p1 = min(P, p0 + per);
  const int ntiles = p1 > p0 ? (int)((p1 - p0 + TP - 1) / TP) : 0;

  // the feature group of this warp: kind (0 sequence separation [+ chain], 1 distance k, 2 angle k, 3 decoder CA distance), its first row
  int kind, kidx = 0, rb0 = 0, dbase = 0, nrow;
  if (warp < 4) { kind = 0; rb0 = 32 * warp; nrow = 127; }
  else if (MODE == 0 && warp < 8) { kind = 1; kidx = warp - 4; dbase = R_DIST + 21 * kidx; nrow = 21; }
  else if (MODE == 0) { kind = 2; kidx = warp - 8; dbase = R_ANG + 21 * kidx; nrow = 21; }
  else { kind = 3; dbase = R_CAD; nrow = 30; }

  float acc[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

  auto load_tile = [&](int tile, int stage) {
    const long long pb = p0 + (long long)tile * TP;
    for (int c = tid; c < TP * 8; c += NW * 32) {
      int kp = c >> 3, ch8 = c & 7;
      long long pair = pb + kp;
      bool pred = pair < p1;
      const __nv_bfloat16* src = g + (size_t)(pred ? pair : 0) * C + q * 64 + ch8 * 8;
      cp16(&Bs[stage][kp * 128 + ((ch8 ^ (kp & 7)) << 4)], src, pred);
    }
    if (tid < TP) {
      long long pair = pb + tid;
      bool pred = pair < p1;
      cp16(&Cs[stage][tid], codes + (pred ? pair : 0), pred);
    }
  };

#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < ntiles) load_tile(s, s);
    cp_commit();
  }
  const unsigned bs_base = (unsigned)__cvta_generic_to_shared(&Bs[0][0]);
  for (int kt = 0; kt < ntiles; kt++) {
    cp_wait<ST - 2>();
    __syncthreads();
    if (kt + ST - 1 < ntiles) load_tile(kt + ST - 1, (kt + ST - 1) % ST);
    cp_commit();
    const int stage = kt % ST;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int kb = ks * 16 + 2 * tig;
      const uint4 c0 = Cs[stage][kb], c1 = Cs[stage][kb + 1], c2 = Cs[stage][kb + 8], c3 = Cs[stage][kb + 9];
      const int b0 = code_bin(c0, kind, kidx), b1 = code_bin(c1, kind, kidx), b2 = code_bin(c2, kind, kidx), b3 = code_bin(c3, kind, kidx);
      const bool chain = MODE == 0 && kind == 0;
      const unsigned cv0 = chain && (c0.x & LIVE) ? bf16_bits(__uint_as_float(c0.z)) : 0u, cv1 = chain && (c1.x & LIVE) ? bf16_bits(__uint_as_float(c1.z)) : 0u;
      const unsigned cv2 = chain && (c2.x & LIVE) ? bf16_bits(__uint_as_float(c2.z)) : 0u, cv3 = chain && (c3.x & LIVE) ? bf16_bits(__uint_as_float(c3.z)) : 0u;
      unsigned bq[8][2];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned addr = bs_base + stage * (TP * 128) + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * 128 + ((((2 * j + (lane >> 4)) ^ (lane & 7))) << 4);
        ldsm4t(bq[2 * j][0], bq[2 * j][1], bq[2 * j + 1][0], bq[2 * j + 1][1], addr);
      }
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        const int rlo = rb0 + 16 * mt + gid, rhi = rlo + 8;
        auto el = [&](int bin, unsigned cvb, int row) -> unsigned { return bin == row ? 0x3F80u : (row == 127 ? cvb : 0u); };
        unsigned a0 = el(b0, cv0, rlo) | (el(b1, cv1, rlo) << 16);
        unsigned a1 = el(b0, cv0, rhi) | (el(b1, cv1, rhi) << 16);
        unsigned a2 = el(b2, cv2, rlo) | (el(b3, cv3, rlo) << 16);
        unsigned a3 = el(b2, cv2, rhi) | (el(b3, cv3, rhi) << 16);
#pragma unroll
        for (int nt = 0; nt < 8; nt++) mma16816(acc[mt][nt], a0, a1, a2, a3, bq[nt][0], bq[nt][1]);
      }
    }
  }
  cp_wait<0>();
  // partials of this slice: row (within the group) -> table row
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
#pragma unroll
    for (int hi = 0; hi < 2; hi++) {
      const int r = rb0 + 16 * mt + gid + 8 * hi;
      int drow = -1;
      if (kind == 0) drow = r < 127 ? r : (MODE == 0 && r == 127 ? R_CHAIN : -1);
      else if (r < nrow) drow = dbase + r;
      if (drow < 0) continue;
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        float2 v = make_float2(acc[mt][nt][2 * hi], acc[mt][nt][2 * hi + 1]);
        *reinterpret_cast<float2*>(tpart + ((size_t)t * NTAB + drow) * C + q * 64 + nt * 8 + 2 * tig) = v;
      }
    }
  }
}


// ---- LayerNorm backward with the pair recomputed from the codes ---------------------------------------------------------------------------
// hat = LN(bf16(T[rows])): the pair itself is never stored; dx = rstd * (dh - mean(dh) - hat * mean(dh * hat)) per pair, written as bf16 (zeros for padding).
template <int MODE>
__device__ __forceinline__ void gather_rows(const __nv_bfloat16* T, const uint4& cd, int c0, float (&x)[8]) {
#pragma unroll
  for (int e = 0; e < 8; e++) x[e] = 0.f;
  if (!(cd.x & LIVE)) return;
  const uint32_t w0 = cd.x ^ MASKED, w1 = cd.y;
  auto add = [&](int r, float scale) {
    uint4 u = *reinterpret_cast<const uint4*>(T + (size_t)r * C + c0);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int k = 0; k < 4; k++) { float2 f = __bfloat1622float2(h[k]); x[2 * k] += scale * f.x; x[2 * k + 1] += scale * f.y; }
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

template <int MODE>
__global__ void __launch_bounds__(16 * 32, 1)
pg_ln_bwd(const __nv_bfloat16* tbl, const uint4* codes, const __nv_bfloat16* dh, __nv_bfloat16* dx, long long P, float eps) {
  constexpr int NTAB = MODE == 0 ? NT : NT_CA;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* T = reinterpret_cast<__nv_bfloat16*>(smem);          // [NTAB][C]
  for (int k = threadIdx.x; k < NTAB * C / 8; k += 16 * 32) reinterpret_cast<uint4*>(T)[k] = reinterpret_cast<const uint4*>(tbl)[k];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  const long long stride = (long long)gridDim.x * 16;
  long long p = (long long)blockIdx.x * 16 + warp;
  uint4 cd = p < P ? codes[p] : make_uint4(0u, 0u, 0u, 0u);
  uint4 gv = (p < P && (cd.x & LIVE)) ? *reinterpret_cast<const uint4*>(dh + (size_t)p * C + c0) : make_uint4(0u, 0u, 0u, 0u);
  for (; p < P; p += stride) {
    const long long pn = p + stride;                                    // the next pair's loads are in flight while this one is computed
    uint4 cn = pn < P ? codes[pn] : make_uint4(0u, 0u, 0u, 0u);
    uint4 gn = (pn < P && (cn.x & LIVE)) ? *reinterpret_cast<const uint4*>(dh + (size_t)pn * C + c0) : make_uint4(0u, 0u, 0u, 0u);
    uint4 o = make_uint4(0u, 0u, 0u, 0u);
    if (cd.x & LIVE) {
      float x[8], g[8];
      gather_rows<MODE>(T, cd, c0, x);
      const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gv);
#pragma unroll
      for (int k = 0; k < 4; k++) { float2 f = __bfloat1622float2(gh[k]); g[2 * k] = f.x; g[2 * k + 1] = f.y; }
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) { x[e] = __bfloat162float(__float2bfloat16_rn(x[e])); s += x[e]; }
      const float mean = wsum(s) * (1.f / C);
      float q = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) { float d = x[e] - mean; q += d * d; }
      const float rstd = rsqrtf(wsum(q) * (1.f / C) + eps);
      float a = 0.f, bsum = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) { x[e] = (x[e] - mean) * rstd; a += g[e]; bsum += g[e] * x[e]; }
      a = wsum(a) * (1.f / C);
      bsum = wsum(bsum) * (1.f / C);
      __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int k = 0; k < 4; k++)
        oh[k] = __floats2bfloat162_rn(rstd * (g[2 * k] - a - x[2 * k] * bsum), rstd * (g[2 * k + 1] - a - x[2 * k + 1] * bsum));
    }
    *reinterpret_cast<uint4*>(dx + (size_t)p * C + c0) = o;
    cd = cn;
    gv = gn;
  }
}

__global__ void pg_reduce_tbl(const float* tpart, __nv_bfloat16* dtbl, int NTAB) {
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (size_t)NTAB * C) return;
  float s = 0.f;
  for (int r = 0; r < NBLK; r++) s += tpart[(size_t)r * NTAB * C + k];
  dtbl[k] = __float2bfloat16_rn(s);
}

void forward_ca(torch::Tensor tbl, torch::Tensor ca, torch::Tensor mask, torch::Tensor idx, torch::Tensor dlim, torch::Tensor out, torch::Tensor codes, int64_t ln, double eps) {
  const at::cuda::CUDAGuard guard(tbl.device());
  int B = idx.size(0), n = idx.size(1);
  size_t smem = (size_t)NT_CA * C * sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(pg_fwd_ca, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int grid = std::min(sms * 2, (B * n + NW - 1) / NW);
  pg_fwd_ca<<<grid, NW * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>()), ca.data_ptr<float>(), mask.data_ptr<uint8_t>(), idx.data_ptr<int>(),
      dlim.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      codes.defined() && codes.numel() > 0 ? reinterpret_cast<uint4*>(codes.data_ptr<int>()) : nullptr, B, n, (int)dlim.numel(), (int)ln, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// mode 0: geometry table [317][256]; mode 1: decoder table [157][256]. impl 1: tensor cores, impl 0: shared-memory read-modify-write.
// dtbl bf16 [NTAB][256], tpart fp32 scratch [NBLK][NTAB][256].
void backward(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl) {
  const at::cuda::CUDAGuard guard(g.device());
  int B = g.size(0), n = g.size(1);
  long long P = (long long)B * n * n;
  auto st = at::cuda::getCurrentCUDAStream();
  int ntab = mode == 0 ? NT : NT_CA;
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  if (impl == 1) {
    if (mode == 0) pg_bwd_mma<0><<<dim3(4, NBLK), 13 * 32, 0, st>>>(gp, cp, P, tpart.data_ptr<float>());
    else pg_bwd_mma<1><<<dim3(4, NBLK), 5 * 32, 0, st>>>(gp, cp, P, tpart.data_ptr<float>());
  } else {
    size_t smem = (size_t)4 * ntab * 32 * sizeof(float);
    if (mode == 0) {
      cudaFuncSetAttribute(pg_bwd_tbl<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      pg_bwd_tbl<0><<<dim3(8, NBLK), 128, smem, st>>>(gp, cp, P, tpart.data_ptr<float>());
    } else {
      cudaFuncSetAttribute(pg_bwd_tbl<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      pg_bwd_tbl<1><<<dim3(8, NBLK), 128, smem, st>>>(gp, cp, P, tpart.data_ptr<float>());
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pg_reduce_tbl<<<(ntab * C + 255) / 256, 256, 0, st>>>(tpart.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dtbl.data_ptr<at::BFloat16>()), ntab);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_backward(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor dx, int64_t mode, double eps) {
  const at::cuda::CUDAGuard guard(tbl.device());
  long long P = (long long)dh.size(0) * dh.size(1) * dh.size(2);
  int ntab = mode == 0 ? NT : NT_CA;
  size_t smem = (size_t)ntab * C * sizeof(__nv_bfloat16);
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto* tp = reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>());
  auto* hp = reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>());
  auto* xp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  auto st = at::cuda::getCurrentCUDAStream();
  if (mode == 0) {
    cudaFuncSetAttribute(pg_ln_bwd<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    pg_ln_bwd<0><<<sms, 16 * 32, smem, st>>>(tp, cp, hp, xp, P, (float)eps);
  } else {
    cudaFuncSetAttribute(pg_ln_bwd<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    pg_ln_bwd<1><<<sms, 16 * 32, smem, st>>>(tp, cp, hp, xp, P, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("forward_ca", &forward_ca);
  m.def("backward", &backward);
  m.def("ln_backward", &ln_backward);
}
