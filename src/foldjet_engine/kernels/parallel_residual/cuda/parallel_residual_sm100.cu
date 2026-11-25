// parallel_residual_sm100.cu -- the B200 (sm_100a) build of parallel_residual.cu (2026-11-17): the same kernels; a tensor larger than the B200's
// L2 (126 MB; kLdgBytes, per tensor) is read through the default cache path (ld.global.nc, __ldg) instead of the evict-first loads (__ldcs), the
// stores stay streaming (__stcs). Measured on the B200 under the measurer's L2 flush (CUDA graphs, 2026-11-17): the evict-first loads ran 2-4 %
// behind the Triton kernel on the large inference cases (d 256, 2 / 3 branches, L768-2048: 1547 vs 1488 us at L2048 x 3), the cached loads tie
// it (1486 us, 7.2 TB/s) and take 3-7 % off the training step; below the L2 size they were one timer step slower (L256 / L384 x 3: 34.7 / 65.4
// vs 32.6 / 63.4 us), so the evict-first loads stay there. Fewer / more threads per CTA, more vectors in flight or plain stores did not help.
// The shared source (H100 / A100 / RTX 5090 builds) is included unchanged: `n` (elements per tensor) and `T` are in scope at every __ldcs of
// its kernels, and the condition is uniform over a launch.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
constexpr int64_t kLdgBytes = 160ll << 20;
// 3.0.6 routes the loads through PR_LD (a template without n / T): the same condition as a macro at the same call sites
#define PR_LD_CUSTOM
#define PR_LD(p) ((int64_t)(n) * (int64_t)sizeof(T) >= kLdgBytes ? __ldg(p) : __ldcs(p))
#include "parallel_residual.cu"
#undef PR_LD
