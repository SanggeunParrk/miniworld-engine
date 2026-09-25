// Sustained tcgen05.mma energy per FLOP by operand / accumulator type: every SM issues back-to-back M128 N256 products on resident
// smem operands (one elected thread), 32 bytes of K per instruction (K = 16 for 16-bit, K = 32 for 8-bit operands).
#include "sm100.cuh"
using namespace s100;
// idesc: c_format bit 4 (1 = F32 accumulate, 0 = F16), a/b format bits 7 / 10 (kind::f16: 0 F16, 1 BF16; kind::f8f6f4: 0 E4M3, 1 E5M2)
template <int KIND, uint32_t ID>
__device__ void run(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x < 32) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  if (threadIdx.x == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  // small random-ish values (0x3c/0x38 patterns): finite in every format, no inf / nan accumulation within the run
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = (KIND == 0 ? 0x2c002400u : 0x20281820u) ^ (i * 2654435761u & 0x01010101u);
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (threadIdx.x < 32) {
    unsigned long long t0 = clock64();
    if (elect_one()) {
      for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint64_t off = (uint64_t)(((ks & 3) * 32) >> 4);     // B: 256 rows x 128 B = 32 KB, one K-block reused
          const uint64_t a = desc_k128(su) + off, b = desc_k128(su + 32768) + off;
          const uint32_t d = tm + (it & 1) * 256, acc = ks > 0 ? 1u : 0u;
          if (KIND == 0)
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" :: "r"(d), "l"(a), "l"(b), "r"(ID), "r"(acc) : "memory");
          else
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }" :: "r"(d), "l"(a), "l"(b), "r"(ID), "r"(acc) : "memory");
        }
      }
      tc_commit(&bar);
    }
    __syncwarp();
    mbar_wait(&bar, 0);
    if (threadIdx.x == 0) out[blockIdx.x] = clock64() - t0;
  }
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) { tc_fence_after(); tmem_dealloc(tm, 512); }
}
constexpr uint32_t NM = (256u >> 3) << 17 | (128u >> 4) << 24;
extern "C" __global__ void e_bf16_f32(unsigned long long* o, int n) { run<0, NM | (1u << 4) | (1u << 7) | (1u << 10)>(o, n); }
extern "C" __global__ void e_f16_f32(unsigned long long* o, int n) { run<0, NM | (1u << 4)>(o, n); }
extern "C" __global__ void e_f16_f16(unsigned long long* o, int n) { run<0, NM>(o, n); }
extern "C" __global__ void e_e4m3_f32(unsigned long long* o, int n) { run<1, NM | (1u << 4)>(o, n); }
extern "C" __global__ void e_e4m3_f16(unsigned long long* o, int n) { run<1, NM>(o, n); }
