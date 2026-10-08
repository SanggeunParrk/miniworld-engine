// bo_pvdpb.cu -- pv dV and the bias gradient of the bias-only DiT's bf16 training backward in ONE launch, sm_100a
// (MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB; the fused backward's B3):
//
//   CTAs [0, NPV)          pv_gate_inf.cu's body with GATE 0: dV = P^T do into dvg's dV half, on the virtual grid (blockIdx.x, NPV)
//   CTAs [NPV, gridDim.x)  dpb_sm100.cu's body, dbias = P o (do v^T - D), on the virtual grid (blockIdx.x - NPV, gridDim.x - NPV)
// Single-CTA bias gradient only: dpbx2_sm100.cu's cta_group::2 MMAs cannot share a function with pv dV's cta_group::1 ones (ptxas,
// bwdf6), so where the host picks the CTA pairs (L768) it launches pv dV and dpbx2 instead.
//
// The two bodies read only the tail's outputs (do, dd) and the forward's P^T / P / v, and write disjoint buffers (dvg's first DA
// columns, dbias): nothing orders one against the other, so the bias gradient's CTAs start on the SMs pv dV leaves idle or frees
// instead of after pv dV's last wave and a launch boundary.
//
// Sync protocol. Each body keeps its own file's protocol unchanged (its mbarriers, its TMEM allocation and dealloc, its exit), all
// inside its own CTA. Programmatic dependent launch: the launch may start before the previous kernel ends (the
// tail's last launch triggers early); every thread runs griddepcontrol.wait before its body (do, dd and the tail's other outputs complete and visible).
// No launch_dependents: the next kernel (the dxa GEMM, cuBLAS) waits for this one to finish. Shared memory: the larger of the two
// bodies' layouts, each laid out from the dynamic base by its own file.
// The bodies' sources are included here: the host passes a hash of them (-DSRC_HASH) so the cubin cache rebuilds when they change.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"

#define BODY_ONLY
#define GATE 0
namespace pvb {
#include "pv_gate_inf.cu"
}
namespace dpbb {
#include "dpb_sm100.cu"
}

extern "C" __global__ void __launch_bounds__(256, 1)
bo_pvdpb_sm100(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mg,
               const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdv,
               const __grid_constant__ CUtensorMap mdp, const __grid_constant__ CUtensorMap mdb, const float* __restrict__ dd, int L,
               int A, int npv) {
  s100::pdl_wait();
  if ((int)blockIdx.x < npv) pvb::pv_gate_body(mp, mv, mg, mo, L, A, (int)blockIdx.x, npv);
  else dpbb::dpb_body(mdo, mdv, mdp, mdb, dd, L, A, (int)blockIdx.x - npv, (int)gridDim.x - npv);
}
