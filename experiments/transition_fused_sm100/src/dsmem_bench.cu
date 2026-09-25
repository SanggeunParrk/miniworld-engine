// d_xn reduce-scatter cost across a cluster of 8 (one CTA per hidden slice): per "tile" each CTA holds a 128 x 128 fp32 partial,
// one row per thread (128 threads); rows 16p..16p+15 belong to CTA p. Every thread sends its row to the owner with remote
// st.shared::cluster (512 B), then a cluster barrier; the owners sum their 8 contributions. Reports clk per tile.
#include "sm100.cuh"
using namespace s100;
DEVI uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
DEVI void st_cluster_v4(uint32_t a, float4 v) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(128, 1) rs8(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);                    // receive: [2 stages][8 senders][16 rows][128] fp32 = 2 x 64 KB
  const uint32_t me = cluster_rank();
  const int r = threadIdx.x, owner = r >> 4, lrow = r & 15;
  float part[128];
  for (int k = 0; k < 128; ++k) part[k] = 0.001f * (k + r + me);
  cluster_sync();
  const unsigned long long t0 = clock64();
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    const int st = it & 1;
    const uint32_t dst = mapa(su + st * 65536 + (me * 16 + lrow) * 512, owner);
#pragma unroll
    for (int k = 0; k < 32; ++k) st_cluster_v4(dst + k * 16, make_float4(part[4 * k], part[4 * k + 1], part[4 * k + 2], part[4 * k + 3]));
    cluster_sync();                                    // all contributions of this stage have landed
    // owner rows: this CTA's 16 rows x 128 cols; each thread sums one (row, 16-col) strip over the 8 senders
    const int orow = r >> 3, ocol = (r & 7) * 16;
    float s[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) s[k] = 0.f;
#pragma unroll
    for (int snd = 0; snd < 8; ++snd)
#pragma unroll
      for (int k = 0; k < 16; k += 4) {
        const uint4 v = lds128(su + st * 65536 + (snd * 16 + orow) * 512 + (ocol + k) * 4);
        s[k] += __uint_as_float(v.x); s[k + 1] += __uint_as_float(v.y); s[k + 2] += __uint_as_float(v.z); s[k + 3] += __uint_as_float(v.w);
      }
#pragma unroll
    for (int k = 0; k < 16; ++k) acc += s[k];
  }
  cluster_sync();
  if (threadIdx.x == 0) out[blockIdx.x] = clock64() - t0;
  if (acc == 1234.5f) out[0] = 0;
}
