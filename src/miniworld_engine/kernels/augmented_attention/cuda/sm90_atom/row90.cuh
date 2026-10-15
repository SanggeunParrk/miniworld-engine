// row90.cuh — warp-level building blocks of the H100 atom-block row kernels (mma.sync; bf16 or TF32).
//
// A warp owns 16 rows x 128 columns held as C fragments, f[16][4]: f[nt][0..1] = (row g, cols 8 nt + 2 t, +1), f[nt][2..3] = (row g + 8, same
// columns), g = lane / 4, t = lane % 4. A C fragment is its own next A operand: bf16 m16n8k16 packs (f[2ks], f[2ks+1]) pairs, TF32 m16n8k8
// permutes the k slots (slot t -> column 8 nt + 2 t, slot t + 4 -> 8 nt + 2 t + 1) and the B operand follows the same permutation, so no
// data moves between a GEMM's epilogue and the next GEMM's A operand.
//
// -DTF32=1: activations are fp32 in memory, every MMA operand is rounded to nearest tf32 (cvt.rna) before the tensor core reads it,
// weights are rounded once on the host. Otherwise activations are bf16.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__

#ifndef TF32
#define TF32 0
#endif
// "=f" first products (the accumulators are not live into them): measured correct for the TF32 build; the bf16 build gives wrong
// outputs with them (cause not found), so it keeps the predicated accumulate
#ifndef WG_FIRST
#define WG_FIRST TF32
#endif
#ifndef MINB
#define MINB 1   // minimum CTAs per SM the register allocation is asked for
#endif
#ifndef NWARP
#define NWARP 4   // warps (= 16-row tiles) per CTA: one warpgroup = 64 rows; 8 = two warpgroups, 128 rows
#endif

namespace r90 {
constexpr int NTHR = NWARP * 32, ROWS = NWARP * 16;
#if TF32
typedef float T;
constexpr int ES = 4;     // bytes per activation element
#else
typedef __nv_bfloat16 T;
constexpr int ES = 2;
#endif
constexpr float LOG2E = 1.4426950408889634f;

// programmatic dependent launch: the kernel may start while its predecessor drains; nothing is read or written before the wait
DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEVI uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void cp16(uint32_t d, const void* s) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s) : "memory"); }
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

DEVI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float sigm(float x) { return rcp(1.f + ex2(-LOG2E * x)); }
DEVI float silu(float x) { return x * sigm(x); }
DEVI uint32_t pk(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r; }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI uint32_t tf(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float tff(float x) { return __uint_as_float(tf(x)); }       // x rounded to tf32, as a float
DEVI float2 up(uint32_t v) { return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)); }
DEVI float quad_sum(float v) { v += __shfl_xor_sync(~0u, v, 1); return v + __shfl_xor_sync(~0u, v, 2); }

// ---- mbarrier + TMA (weight ring: one producer warp, the consumers wait on full / empty barriers; no CTA-wide barriers) --------------
DEVI void mb_init(uint32_t bar, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory"); }
DEVI void mb_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mb_expect(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEVI void mb_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
DEVI void mb_wait(uint32_t bar, int parity) {
  asm volatile("{ .reg .pred p; L_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra L_%=; }" ::"r"(bar), "r"(parity) : "memory");
}
DEVI void tma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"(m), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
DEVI void reg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory"); }
DEVI void reg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory"); }
DEVI void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// ---- storage: one element type T, loads / stores of column pairs ---------------------------------------------------------------
DEVI float2 ld2(const T* p) {
#if TF32
  return *reinterpret_cast<const float2*>(p);
#else
  return up(*reinterpret_cast<const uint32_t*>(p));
#endif
}
DEVI void st2(T* p, float x, float y) {
#if TF32
  *reinterpret_cast<float2*>(p) = make_float2(x, y);
#else
  *reinterpret_cast<uint32_t*>(p) = pk(x, y);
#endif
}
// a value that only feeds MMAs: rounded to the operand precision on the way out (tf32: cvt.rna; bf16: the store rounds)
DEVI void st2_op(T* p, float x, float y) {
#if TF32
  *reinterpret_cast<float2*>(p) = make_float2(tff(x), tff(y));
#else
  *reinterpret_cast<uint32_t*>(p) = pk(x, y);
#endif
}
DEVI float round_store(float x) {  // the value a T store followed by a T load would give
#if TF32
  return x;
#else
  return rnb(x);
#endif
}

// ---- 16 x 128 row tile ------------------------------------------------------------------------------------------------------
// Lane layout: a lane's VEC consecutive columns of a group of GS (= 16 fp32 / 32 bf16, i.e. 16 bytes per lane) sit in TPG = GS / 8
// consecutive n-tiles of its fragment: tile nt = TPG gp + s, element e (0, 1) of row g (2, 3: row g + 8) <-> column GS gp + VEC t + 2 s + e.
// Every access of a tile to memory is then one 16-byte vector per row; the weights' K and row axes are permuted the same way when packed
// (``sm90_atom._lperm``), so the MMA sees the logical order 8 nt + 2 t + e.
constexpr int GS = 64 / ES, VEC = GS / 4, TPG = GS / 8;
DEVI int pcol(int nt, int t) { return (nt / TPG) * GS + VEC * t + 2 * (nt % TPG); }  // memory column of tile nt, elements e = 0, 1

struct Tile { float f[16][4]; };

// NT tiles (a multiple of TPG) of rows g, g + 8 starting at column col (a multiple of GS) of p -> v[NT][4]
template <int NT>
DEVI void ld_nt(float (*v)[4], const T* p, size_t ld, int col, int g, int t) {
  const T* r0 = p + (size_t)g * ld + col + VEC * t;
  const T* r1 = r0 + 8 * ld;
#pragma unroll
  for (int gp = 0; gp < NT / TPG; ++gp) {
#if TF32
    const float4 a = *reinterpret_cast<const float4*>(r0 + gp * GS), b = *reinterpret_cast<const float4*>(r1 + gp * GS);
    v[gp * TPG + 0][0] = a.x; v[gp * TPG + 0][1] = a.y; v[gp * TPG + 1][0] = a.z; v[gp * TPG + 1][1] = a.w;
    v[gp * TPG + 0][2] = b.x; v[gp * TPG + 0][3] = b.y; v[gp * TPG + 1][2] = b.z; v[gp * TPG + 1][3] = b.w;
#else
    const uint4 a = *reinterpret_cast<const uint4*>(r0 + gp * GS), b = *reinterpret_cast<const uint4*>(r1 + gp * GS);
    const uint32_t wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const float2 x = up(wa[s]), y = up(wb[s]);
      v[gp * TPG + s][0] = x.x; v[gp * TPG + s][1] = x.y; v[gp * TPG + s][2] = y.x; v[gp * TPG + s][3] = y.y;
    }
#endif
  }
}
// v[NT][4] -> rows g, g + 8 at column col of p; OP: values that only feed MMAs (tf32: rounded to nearest)
template <int NT, bool OP>
DEVI void st_nt(const float (*v)[4], T* p, size_t ld, int col, int g, int t) {
  T* r0 = p + (size_t)g * ld + col + VEC * t;
  T* r1 = r0 + 8 * ld;
#pragma unroll
  for (int gp = 0; gp < NT / TPG; ++gp) {
#if TF32
    float a[4], b[4];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      a[2 * s] = OP ? tff(v[gp * TPG + s][0]) : v[gp * TPG + s][0]; a[2 * s + 1] = OP ? tff(v[gp * TPG + s][1]) : v[gp * TPG + s][1];
      b[2 * s] = OP ? tff(v[gp * TPG + s][2]) : v[gp * TPG + s][2]; b[2 * s + 1] = OP ? tff(v[gp * TPG + s][3]) : v[gp * TPG + s][3];
    }
    *reinterpret_cast<float4*>(r0 + gp * GS) = make_float4(a[0], a[1], a[2], a[3]);
    *reinterpret_cast<float4*>(r1 + gp * GS) = make_float4(b[0], b[1], b[2], b[3]);
#else
    uint32_t wa[4], wb[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) { wa[s] = pk(v[gp * TPG + s][0], v[gp * TPG + s][1]); wb[s] = pk(v[gp * TPG + s][2], v[gp * TPG + s][3]); }
    *reinterpret_cast<uint4*>(r0 + gp * GS) = make_uint4(wa[0], wa[1], wa[2], wa[3]);
    *reinterpret_cast<uint4*>(r1 + gp * GS) = make_uint4(wb[0], wb[1], wb[2], wb[3]);
#endif
  }
}
// the 128 columns of a [rows, ld] array
DEVI void ld_tile(Tile& c, const T* p, size_t ld, int col0, int g, int t) { ld_nt<16>(c.f, p, ld, col0, g, t); }
DEVI void st_tile(const Tile& c, T* p, size_t ld, int col0, int g, int t) { st_nt<16, false>(c.f, p, ld, col0, g, t); }
DEVI void st_tile_op(const Tile& c, T* p, size_t ld, int col0, int g, int t) { st_nt<16, true>(c.f, p, ld, col0, g, t); }
// a float2 of a per-column fp32 vector for tile nt
DEVI float2 colv(const float* v, int nt, int t) { return *reinterpret_cast<const float2*>(v + pcol(nt, t)); }
// ---- per-CTA column sums (bias gradients): a warp's 16 rows -> red[warp][128] (shared), the CTA's 8 warps -> part[set][128] ----------
DEVI void col_partial(const Tile& c, float* red, int warp, int lane, int t) {
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float v = c.f[nt][j] + c.f[nt][2 + j];
      v += __shfl_xor_sync(~0u, v, 4);
      v += __shfl_xor_sync(~0u, v, 8);
      v += __shfl_xor_sync(~0u, v, 16);
      if (lane < 4) red[warp * 128 + pcol(nt, t) + j] += v;
    }
  }
}
// the same for NT tiles nt0 .. nt0 + NT - 1 held in v[NT][4]
template <int NT>
DEVI void col_partial_nt(const float (*v)[4], int nt0, float* red, int warp, int lane, int t) {
#pragma unroll
  for (int i = 0; i < NT; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float x = v[i][j] + v[i][2 + j];
      x += __shfl_xor_sync(~0u, x, 4);
      x += __shfl_xor_sync(~0u, x, 8);
      x += __shfl_xor_sync(~0u, x, 16);
      if (lane < 4) red[warp * 128 + pcol(nt0 + i, t) + j] += x;
    }
  }
}
// NT tiles of a modulation gradient: per-sample rows (T, dT + row_s) or, for a shared conditioning, accumulated over the CTA's samples in fp32
// rows (dF + row_a; the CTA owns them, so no atomics: the first sample stores, the later ones add)
template <int NT>
DEVI void put_nt(const float (*v)[4], T* dT, float* dF, size_t row_s, size_t row_a, int ld, int col, bool shared, bool first, int g, int t) {
  if (!shared) { st_nt<NT, false>(v, dT + row_s * ld, ld, col, g, t); return; }
  float* r0 = dF + (row_a + g) * ld + col;
  float* r1 = r0 + 8 * (size_t)ld;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int pc = pcol(i, t);
    float2 a = make_float2(v[i][0], v[i][1]), b = make_float2(v[i][2], v[i][3]);
    if (!first) {
      const float2 oa = *reinterpret_cast<const float2*>(r0 + pc), ob = *reinterpret_cast<const float2*>(r1 + pc);
      a.x += oa.x; a.y += oa.y; b.x += ob.x; b.y += ob.y;
    }
    *reinterpret_cast<float2*>(r0 + pc) = a;
    *reinterpret_cast<float2*>(r1 + pc) = b;
  }
}

// Grouped sample mode (shared conditioning): warp w of the CTA holds 16 atoms of sample (w & 3) of a group of four samples, atom block
// w >> 2. The four samples' modulation gradients of an atom block are staged through shared memory as 128-column slabs (grp_put, no
// barrier), summed in a fixed order j = 0 .. 3 in fp32 and stored once, as T, into partial row (group, atom) (grp_flush: one barrier per
// slab). bf16 alternates two slab buffers (one barrier per flush); the fp32 build has one buffer and a second barrier.
constexpr int FRS = 128 + (ES == 2 ? 8 : 4);   // slab row stride (elements)
constexpr int GNB = TF32 ? 1 : 2;              // slab buffers
constexpr int FBUF = NWARP * 16 * FRS;         // elements of one buffer
constexpr int GRP_BYTES = GNB * FBUF * ES;
template <int NT>
DEVI void grp_put(const float (*v)[4], T* buf, int c0, int warp, int g, int t) {
  T* mine = buf + (size_t)warp * 16 * FRS + c0;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int pc = pcol(i, t);
    st2(mine + g * FRS + pc, v[i][0], v[i][1]);
    st2(mine + (g + 8) * FRS + pc, v[i][2], v[i][3]);
  }
}
DEVI void grp_flush(T* dT, size_t prow, int ld, int col, const T* buf, int tid) {
  constexpr int VPE = 16 / ES, VPR = 128 / VPE;
  __syncthreads();
  const int sub = tid >> 7;
  for (int it = tid & 127; it < 16 * VPR; it += 128) {
    const int r = it / VPR, vi = it % VPR;
    float sum[VPE];
#pragma unroll
    for (int e = 0; e < VPE; ++e) sum[e] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const T* sp = buf + (size_t)(4 * sub + j) * 16 * FRS + r * FRS + vi * VPE;
#if TF32
      const float4 a = *reinterpret_cast<const float4*>(sp);
      sum[0] += a.x; sum[1] += a.y; sum[2] += a.z; sum[3] += a.w;
#else
      const uint4 a = *reinterpret_cast<const uint4*>(sp);
      const uint32_t w4[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) { const float2 f = up(w4[q]); sum[2 * q] += f.x; sum[2 * q + 1] += f.y; }
#endif
    }
    T* dst = dT + (prow + sub * 16 + r) * ld + col + vi * VPE;
#if TF32
    *reinterpret_cast<float4*>(dst) = make_float4(sum[0], sum[1], sum[2], sum[3]);
#else
    *reinterpret_cast<uint4*>(dst) = make_uint4(pk(sum[0], sum[1]), pk(sum[2], sum[3]), pk(sum[4], sum[5]), pk(sum[6], sum[7]));
#endif
  }
#if TF32
  __syncthreads();   // one buffer: the next slab's writes wait for these reads
#endif
}

// after a CTA barrier: threads 0 .. 127 add the 8 warps' rows in order -> out[tid]
DEVI void col_finish(const float* red, float* out, int tid) {
  if (tid < 128) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) s += red[w * 128 + tid];
    out[tid] = s;
  }
}

// row statistics over the 128 columns of the tile: mean and rstd of rows g (x) and g + 8 (y)
DEVI void ln_stats(const Tile& c, float eps, float& m0, float& r0, float& m1, float& r1) {
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) { s0 += c.f[nt][0] + c.f[nt][1]; s1 += c.f[nt][2] + c.f[nt][3]; }
  m0 = quad_sum(s0) * (1.f / 128.f);
  m1 = quad_sum(s1) * (1.f / 128.f);
  float v0 = 0.f, v1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    float d;
    d = c.f[nt][0] - m0; v0 += d * d; d = c.f[nt][1] - m0; v0 += d * d;
    d = c.f[nt][2] - m1; v1 += d * d; d = c.f[nt][3] - m1; v1 += d * d;
  }
  r0 = rsqrtf(quad_sum(v0) * (1.f / 128.f) + eps);
  r1 = rsqrtf(quad_sum(v1) * (1.f / 128.f) + eps);
}

// ---- tensor-core products: wgmma, A from registers (the C-fragment tile), B from swizzled shared memory --------------------------------
#include "wg90.inc"
#if TF32
constexpr int KSTEPS = 16;   // k8 steps over K = 128
// The A fragments are the tile's own registers: a tile that feeds a product is rounded to tf32 in place first (``rna``), so no converted
// copies stay live while the asynchronous products run (16 k steps x 4 registers was 64 extra registers).
DEVI void afrag(const Tile& c, int kt, uint32_t* a) {
  // slot t -> column 8 kt + 2 t, slot t + 4 -> 8 kt + 2 t + 1: the weights' K columns are permuted the same way (kernels/.../__init__.py pack)
  a[0] = __float_as_uint(c.f[kt][0]); a[1] = __float_as_uint(c.f[kt][2]); a[2] = __float_as_uint(c.f[kt][1]); a[3] = __float_as_uint(c.f[kt][3]);
}
DEVI void rna(Tile& c) {
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) c.f[nt][e] = tff(c.f[nt][e]);
}
#else
constexpr int KSTEPS = 8;    // k16 steps over K = 128
DEVI void rna(Tile&) {}   // bf16: the packed fragments are made per step
DEVI void afrag(const Tile& c, int ks, uint32_t* a) {
  a[0] = pk(c.f[2 * ks][0], c.f[2 * ks][1]); a[1] = pk(c.f[2 * ks][2], c.f[2 * ks][3]);
  a[2] = pk(c.f[2 * ks + 1][0], c.f[2 * ks + 1][1]); a[3] = pk(c.f[2 * ks + 1][2], c.f[2 * ks + 1][3]);
}
#endif
constexpr int HB = 128 / ES;  // columns of one 128-byte swizzle row: 64 bf16 / 32 fp32 (= the SwiGLU hidden chunk)

DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
DEVI void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// K-major B operand, 128-byte swizzle: start address, LBO (ignored) = 1, SBO = 1024 B (8 rows x 128 B), swizzle mode 128 B
DEVI uint64_t wdesc(uint32_t addr) { return uint64_t((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 62); }

template <int NR> DEVI void wg_mma(float* d, const uint32_t* a, uint64_t desc, int acc) {
#if TF32
  if constexpr (NR == 32) wg_rs_tf32_n32(a[0], a[1], a[2], a[3], desc, d, acc);
  else if constexpr (NR == 64) wg_rs_tf32_n64(a[0], a[1], a[2], a[3], desc, d, acc);
  else wg_rs_tf32_n128(a[0], a[1], a[2], a[3], desc, d, acc);
#else
  if constexpr (NR == 32) wg_rs_bf16_n32(a[0], a[1], a[2], a[3], desc, d, acc);
  else if constexpr (NR == 64) wg_rs_bf16_n64(a[0], a[1], a[2], a[3], desc, d, acc);
  else wg_rs_bf16_n128(a[0], a[1], a[2], a[3], desc, d, acc);
#endif
}
template <int NR> DEVI void wg_mma_first(float* d, const uint32_t* a, uint64_t desc) {
#if TF32
  if constexpr (NR == 32) wg_rs_tf32_n32_first(a[0], a[1], a[2], a[3], desc, d);
  else if constexpr (NR == 64) wg_rs_tf32_n64_first(a[0], a[1], a[2], a[3], desc, d);
  else wg_rs_tf32_n128_first(a[0], a[1], a[2], a[3], desc, d);
#else
  if constexpr (NR == 32) wg_rs_bf16_n32_first(a[0], a[1], a[2], a[3], desc, d);
  else if constexpr (NR == 64) wg_rs_bf16_n64_first(a[0], a[1], a[2], a[3], desc, d);
  else wg_rs_bf16_n128_first(a[0], a[1], a[2], a[3], desc, d);
#endif
}
// d[NR / 2] (+)= tile[16 rows of this warp, K = 32 * KB / ES * ... ] * W^T, W a swizzled shared tile of NR output rows;
// KS = number of k steps; the tile's step ks reads W step ks (block ks / 4 of NR x 128 B, 32 B per step inside the block).
template <int NR, int KS>
DEVI void wg_gemm(float* d, const Tile& c, uint32_t w, int accumulate) {
  wg_fence();
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t a[4];
    afrag(c, ks, a);
    const uint64_t desc = wdesc(w + (ks >> 2) * (NR * 128) + (ks & 3) * 32);
#if WG_FIRST
    if (ks > 0) wg_mma<NR>(d, a, desc, 1);
    else if (accumulate) wg_mma<NR>(d, a, desc, 1);
    else wg_mma_first<NR>(d, a, desc);   // "=f": the accumulators are not live into the product
#else
    wg_mma<NR>(d, a, desc, accumulate | (ks > 0));
#endif
  }
  wg_commit();
  wg_wait();
}

// cp.async a [rows, kcols] weight tile (global row stride ld elements) into the swizzled layout [kcols / HB blocks][rows][128 B]
DEVI void load_w(uint32_t dst, const T* src, int ld, int rows, int kcols, int tid, int nthr) {
#ifdef ABL_W
  if (src != nullptr) return;   // ablation: no weight traffic (the shared tiles keep stale contents)
#endif
  const int cpr = kcols * ES / 16;  // 16-byte chunks per row
  for (int i = tid; i < rows * cpr; i += nthr) {
    const int r = i / cpr, j = i % cpr, b = j >> 3, c = j & 7;
    cp16(dst + b * (rows * 128) + r * 128 + ((c ^ (r & 7)) << 4), reinterpret_cast<const char*>(src + (size_t)r * ld) + j * 16);
  }
}
}  // namespace r90
