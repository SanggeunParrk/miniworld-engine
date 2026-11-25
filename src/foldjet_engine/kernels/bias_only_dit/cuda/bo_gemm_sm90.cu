// bo_gemm_sm90.cu -- the bias-only token DiT's GEMMs with their row pass in the epilogue, H100 (sm_90a): the fewer-launch alternative
// to the B200 step's fusions (bo_tail_bf16.cu / bo_bwd_tail.cu are tcgen05 kernels; their cluster-split wgmma port, bo_inf3_sm90.cu,
// measured slower). Dispatched: EPI_BIAS (cond_tables, the 12-launch step up to 384 table rows). Measured slower than cuBLAS + the row
// kernel and kept as probes (docs/records/h100_bias_only_dit_inf3_20261019.md): EPI_RESOUT, EPI_GATEBWD, dog_gate_tma -- their
// mainloops run at cuBLAS's rate, the epilogue does not overlap them.
//
//   EPI_BIAS     out = [LN(c) | c] [W1; W2]^T + bias                  both conditioning tables in one launch (inference)
//   EPI_RESOUT   out = x + sigmoid(gate[tok]) (A B^T)                 the squeeze GEMM + the block's last residual (inference)
//   EPI_GATEBWD  dog = A B (B MN-major: Wo [768, DA]), then the gate backward of a = sigmoid(g) o:
//                do = dog sigmoid(g), dg = dog ao (1 - sigmoid(g)), dd[s, h, i] = sum_{head h} dog ao      (training backward)
//
// Persistent CTAs walk 128 x BN output tiles (m fastest; BN 64 .. 256), two consumer warpgroups (rows 0-63 / 64-127: one m64nBNk16 wgmma slab each), a
// producer warp streaming [A 128 x 64 | B BN x 64] k-blocks by TMA through an NST-stage ring. A K-major [M, K] bf16; B K-major [N, K]
// (weights as nn.Linear stores them) or MN-major [K, N] (EPI_GATEBWD: Wo itself, [64 k][64 n] boxes, wgmma transposed B).
// PDL: launch_dependents after setup; the producer waits before its first A load (and, unless BPRE, before the B loads: inference
// weights are packed once and may be prefetched under the previous kernel); the consumers wait before the epilogue's reads/stores.
// Deterministic: no atomics, fixed-order sums.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_ptx.cuh"
using namespace tmn;
#include "wgmma_n.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace bog {
#define DEVI __device__ __forceinline__
using bf = __nv_bfloat16;

// m64nNk16 bf16, K-major A and B (tnspB 0) / transposed B (tnspB 1)
template <int NN, int TB> DEVI void mma(float (&d)[NN / 2], uint64_t a, uint64_t b);
template <> DEVI void mma<64, 0>(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
template <> DEVI void mma<64, 1>(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
template <> DEVI void mma<128, 0>(float (&d)[64], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b));
}
template <> DEVI void mma<128, 1>(float (&d)[64], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b));
}
template <> DEVI void mma<192, 0>(float (&d)[96], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]) : "l"(a), "l"(b));
}
template <> DEVI void mma<192, 1>(float (&d)[96], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, 1, 1, 1, 0, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]) : "l"(a), "l"(b));
}
template <> DEVI void mma<256, 0>(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(a), "l"(b));
}
template <> DEVI void mma<256, 1>(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, 1, 1, 1, 0, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(a), "l"(b));
}

DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }                 // K-major SW128
DEVI uint64_t dmn(uint32_t addr) { return smem_desc(addr, 8192, 1024, 1); }               // MN-major SW128: 64-column groups 8 KB apart
DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEVI void tma_ld(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
DEVI float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
DEVI float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
DEVI uint32_t pk(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint32_t ldg_u32(const void* p) { uint32_t v; asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
DEVI float2 ldg_f2(const void* p) { float2 v; asm volatile("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p)); return v; }

constexpr int QM = 128, KB = 64, SA = QM * 128, SMEM_MAX = 232448;
enum { EPI_RESOUT = 0, EPI_GATEBWD = 1, EPI_NONE = 2, EPI_BIAS = 3 };   // NONE: the mainloop alone (probes); BIAS: out = acc + bias[col] (bf16)

template <int BN> struct Cfg {
  static constexpr int SB = BN * 128, ST = SA + SB;
  static constexpr int NST = (SMEM_MAX - 1024) / ST > 6 ? 6 : (SMEM_MAX - 1024) / ST;
  static constexpr int SMEM = NST * ST + 256;
  static_assert(NST >= 3 && BN % 64 == 0 && BN <= 256, "ring");
};

struct EpiArgs {
  const void* x; int xbf;              // RESOUT: the residual [M, 768] (bf16 or fp32)
  const bf* gate; int gs;              // RESOUT: the gate table rows (tok = row % T), row stride gs; GATEBWD: g [M, DA], stride gs
  void* out; int obf; int os;          // RESOUT: out; GATEBWD: do [M, DA] (stride os)
  const bf* ao; bf* dg; int dgs;       // GATEBWD: the forward's gated output, dg (stride dgs)
  float* dd; int L, nh, dh;            // GATEBWD: dd [S, nh, L] fp32
  const float* bias; int nsplit;       // BIAS: per-column bias (fp32); columns >= nsplit take A from the second map (ma2)
  int T;
};

template <int BN, int EPI, int TB, int DH = 0>
__global__ void __launch_bounds__(384, 1)
gemm_epi(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap ma2, int M, int N,
         int K, int bpre, EpiArgs e) {
  using C = Cfg<BN>;
  constexpr int NST = C::NST, ST = C::ST;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + NST * ST);
  uint64_t* empty = full + NST;
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int MT = M / QM, ntiles = MT * (N / BN), NK = K / KB;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  pdl_launch();
  if (wg == 0) {                                                                 // producer: a persistent walk over the tiles, m fastest
    setmaxnreg_dec<40>();
    if (tid == 0) {
      auto bload = [&](int k, int s, int n0) {
        const uint32_t dst = su + s * ST + SA;
        if constexpr (TB == 0) tma_ld(dst, &mb, full + s, KB * k, n0);
        else
#pragma unroll
          for (int j = 0; j < BN / 64; ++j) tma_ld(dst + j * 8192, &mb, full + s, n0 + 64 * j, KB * k);
      };
      const int t0 = (int)blockIdx.x;
      const int pre = (bpre && t0 < ntiles) ? (NK < NST ? NK : NST) : 0;
      for (int k = 0; k < pre; ++k) { mbar_arrive_expect_tx(full + k, ST); bload(k, k, (t0 / MT) * BN); }   // weights: under the previous kernel
      pdl_wait();
      uint32_t it = 0;
      for (int t = t0; t < ntiles; t += (int)gridDim.x) {
        const int m0 = (t % MT) * QM, n0 = (t / MT) * BN;
        for (int k = 0; k < NK; ++k, ++it) {
          const int s = (int)(it % NST);
          if (it >= (uint32_t)pre) {
            if (it >= NST) mbar_wait(empty + s, ((it / NST) - 1) & 1u);
            mbar_arrive_expect_tx(full + s, ST);
            bload(k, s, n0);
          }
          tma_ld(su + s * ST, (EPI == EPI_BIAS && n0 >= e.nsplit) ? &ma2 : &ma, full + s, KB * k, m0);
        }
      }
    }
    __syncwarp();
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  pdl_wait();
  uint32_t it = 0;
  for (int t = (int)blockIdx.x; t < ntiles; t += (int)gridDim.x) {
  const int m0 = (t % MT) * QM, n0 = (t / MT) * BN;
  float acc[BN / 2];
#pragma unroll
  for (int i = 0; i < BN / 2; ++i) acc[i] = 0.f;
  int prev = -1;
  for (int k = 0; k < NK; ++k, ++it) {
    const int s = (int)(it % NST);
    mbar_wait(full + s, (it / NST) & 1u);
    const uint32_t a0 = su + s * ST + cw * 8192, b0 = su + s * ST + SA;
    fence_regs(acc);
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      if constexpr (TB == 0) mma<BN, 0>(acc, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
      else mma<BN, 1>(acc, dsw(a0 + ks * 32), dmn(b0 + ks * 2048));
    }
    wgmma_commit();
    wgmma_wait<1>();
    fence_regs(acc);
    if (prev >= 0 && wtid == 0) mbar_arrive(empty + prev);
    prev = s;
  }
  wgmma_wait<0>();
  fence_regs(acc);
  if (wtid == 0) mbar_arrive(empty + prev);                                      // the producer refills under this tile's epilogue
  const int r0 = m0 + 64 * cw + 16 * warp + (lane >> 2), cq = 2 * (lane & 3);
  if constexpr (EPI == EPI_BIAS) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int col = n0 + 8 * j + cq;
      const float2 bv = *reinterpret_cast<const float2*>(e.bias + col);
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int ix = 4 * j + 2 * rb;
        *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(e.out) + (size_t)(r0 + 8 * rb) * e.os + col) = pk(acc[ix] + bv.x, acc[ix + 1] + bv.y);
      }
    }
  } else if constexpr (EPI == EPI_NONE) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < BN / 2; ++i) s += acc[i];
    if (s == 12345.f) reinterpret_cast<float*>(e.out)[t] = s;
  } else if constexpr (EPI == EPI_RESOUT) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int row = r0 + 8 * rb, col = n0 + 8 * j + cq, ix = 4 * j + 2 * rb;
        const size_t off = (size_t)row * 768 + col;
        const float2 xv = e.xbf ? bf2(ldg_u32(reinterpret_cast<const bf*>(e.x) + off)) : ldg_f2(reinterpret_cast<const float*>(e.x) + off);
        const float2 g = bf2(ldg_u32(e.gate + (size_t)(row % e.T) * e.gs + col));
        const float o0 = xv.x + sg(g.x) * acc[ix], o1 = xv.y + sg(g.y) * acc[ix + 1];
        const size_t oo = (size_t)row * e.os + col;
        if (e.obf) *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(e.out) + oo) = pk(o0, o1);
        else *reinterpret_cast<float2*>(reinterpret_cast<float*>(e.out) + oo) = make_float2(o0, o1);
      }
  } else {
    // gate backward; dd: per (row, head) the sum over the head's dh columns of dog * ao (BN a multiple of dh: whole heads per tile).
    // A thread holds columns 8 j + cq, + 1 of rows r0, r0 + 8; a head's columns are j in [h dh / 8, (h + 1) dh / 8) over the quad.
    static_assert(EPI != EPI_GATEBWD || (DH % 8 == 0 && BN % DH == 0), "whole heads per tile");
    constexpr int NHB = EPI == EPI_GATEBWD ? BN / DH : 1;
    float hs[2][NHB];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int u = 0; u < NHB; ++u) hs[rb][u] = 0.f;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int row = r0 + 8 * rb, col = n0 + 8 * j + cq, ix = 4 * j + 2 * rb;
        const float2 g = bf2(ldg_u32(e.gate + (size_t)row * e.gs + col)), ao = bf2(ldg_u32(e.ao + (size_t)row * e.os + col));
        const float s0 = sg(g.x), s1 = sg(g.y);
        const float d0 = __bfloat162float(__float2bfloat16_rn(acc[ix])), d1 = __bfloat162float(__float2bfloat16_rn(acc[ix + 1]));
        *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(e.out) + (size_t)row * e.os + col) = pk(d0 * s0, d1 * s1);
        *reinterpret_cast<uint32_t*>(e.dg + (size_t)row * e.dgs + col) = pk(d0 * ao.x * (1.f - s0), d1 * ao.y * (1.f - s1));
        hs[rb][(8 * j) / (DH ? DH : 8)] += d0 * ao.x + d1 * ao.y;                       // (8 j) / dh: 8-column groups never straddle a head
      }
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int u = 0; u < NHB; ++u) {
        float v = hs[rb][u];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        if ((lane & 3) == 0) {
          const int row = r0 + 8 * rb, head = n0 / DH + u;
          e.dd[((size_t)(row / e.L) * e.nh + head) * e.L + row % e.L] = v;
        }
      }
  }
  }  // tiles
}

// ------------------------------------------------------------------------------------------------ dog + gate backward, TMA epilogue
// dog = dy Wo, then the gate backward, with the epilogue's operands and results moved by TMA: the producer loads the tile's g and ao
// ([128][64] SW128 boxes) into an epilogue buffer right after the tile's mainloop k-blocks (they land under its MMAs); the consumers
// read them from shared memory, write do / dg over them in place and one thread stores both tiles by TMA, then frees the buffer
// once the stores have read it (cp.async.bulk.wait_group.read). Persistent over the tiles (m fastest): the next tile's k-blocks
// stream under this tile's epilogue. dd goes out with plain stores (one float per row, head and quad).
DEVI void tma_st(const CUtensorMap* m, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" :: "l"(m), "r"(src), "r"(c0), "r"(c1) : "memory");
}
template <int BN> struct CfgE {
  static constexpr int SB = BN * 128, ST = SA + SB, EB = 2 * QM * BN * 2;          // ring stage; g | ao tiles
  static constexpr int NST = (SMEM_MAX - EB - 1024) / ST > 6 ? 6 : (SMEM_MAX - EB - 1024) / ST;
  static constexpr int O_E = NST * ST, O_BAR = O_E + EB, SMEM = O_BAR + 256;
  static_assert(NST >= 3 && SMEM <= SMEM_MAX && BN % 64 == 0, "ring + epilogue buffer");
};
template <int BN, int DH>
__global__ void __launch_bounds__(384, 1)
dog_gate_tma(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mg,
             const __grid_constant__ CUtensorMap mao, const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdg,
             float* __restrict__ dd, int M, int N, int L, int nh) {
  using C = CfgE<BN>;
  constexpr int NST = C::NST, ST = C::ST, NK = 768 / KB, NB = BN / 64, NHB = BN / DH;
  static_assert(DH % 8 == 0 && BN % DH == 0, "whole heads per tile");
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm), se = su + C::O_E;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::O_BAR);
  uint64_t* empty = full + NST;
  uint64_t* efull = empty + NST;
  uint64_t* efree = efull + 1;
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int MT = M / QM, ntiles = MT * (N / BN);
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    mbar_init(efull, 1);
    mbar_init(efree, 1);
    fence_barrier_init();
  }
  __syncthreads();
  pdl_launch();
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid == 0) {                                                              // the k-block ring
      pdl_wait();
      uint32_t it = 0;
      for (int t = (int)blockIdx.x; t < ntiles; t += (int)gridDim.x) {
        const int m0 = (t % MT) * QM, n0 = (t / MT) * BN;
        for (int k = 0; k < NK; ++k, ++it) {
          const int s = (int)(it % NST);
          if (it >= NST) mbar_wait(empty + s, ((it / NST) - 1) & 1u);
          mbar_arrive_expect_tx(full + s, ST);
          tma_ld(su + s * ST, &ma, full + s, KB * k, m0);
#pragma unroll
          for (int j = 0; j < NB; ++j) tma_ld(su + s * ST + SA + j * 8192, &mb, full + s, n0 + 64 * j, KB * k);
        }
      }
    } else if (tid == 32) {                                                      // the epilogue tiles: issued as soon as the buffer
      pdl_wait();                                                                // is free, so they land under the tile's mainloop
      uint32_t lt = 0;
      for (int t = (int)blockIdx.x; t < ntiles; t += (int)gridDim.x, ++lt) {
        const int m0 = (t % MT) * QM, n0 = (t / MT) * BN;
        if (lt) mbar_wait(efree, (lt - 1) & 1u);                                // the previous tile's stores have read the buffer
        mbar_arrive_expect_tx(efull, C::EB);
#pragma unroll
        for (int j = 0; j < NB; ++j) {
          tma_ld(se + j * 16384, &mg, efull, n0 + 64 * j, m0);
          tma_ld(se + C::EB / 2 + j * 16384, &mao, efull, n0 + 64 * j, m0);
        }
      }
    }
    __syncwarp();
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  uint32_t it = 0, lt = 0;
  for (int t = (int)blockIdx.x; t < ntiles; t += (int)gridDim.x, ++lt) {
    const int m0 = (t % MT) * QM, n0 = (t / MT) * BN;
    float acc[BN / 2];
#pragma unroll
    for (int i = 0; i < BN / 2; ++i) acc[i] = 0.f;
    int prev = -1;
    for (int k = 0; k < NK; ++k, ++it) {
      const int s = (int)(it % NST);
      mbar_wait(full + s, (it / NST) & 1u);
      const uint32_t a0 = su + s * ST + cw * 8192, b0 = su + s * ST + SA;
      fence_regs(acc);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma<BN, 1>(acc, dsw(a0 + ks * 32), dmn(b0 + ks * 2048));
      wgmma_commit();
      wgmma_wait<1>();
      fence_regs(acc);
      if (prev >= 0 && wtid == 0) mbar_arrive(empty + prev);
      prev = s;
    }
    wgmma_wait<0>();
    fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
    mbar_wait(efull, lt & 1u);
    const int rq = 64 * cw + 16 * warp + (lane >> 2), cq = 2 * (lane & 3);      // tile row (and + 8), column pair
    float hs[2][NHB];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int u = 0; u < NHB; ++u) hs[rb][u] = 0.f;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int r = rq + 8 * rb, ix = 4 * j + 2 * rb;
        const uint32_t off = (uint32_t)((j / 8) * 16384 + r * 128 + (((j % 8) ^ (r & 7)) << 4) + cq * 2);
        uint32_t gw, aw;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(gw) : "r"(se + off));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(aw) : "r"(se + C::EB / 2 + off));
        const float2 g = bf2(gw), ao = bf2(aw);
        const float s0 = sg(g.x), s1 = sg(g.y);
        const float d0 = __bfloat162float(__float2bfloat16_rn(acc[ix])), d1 = __bfloat162float(__float2bfloat16_rn(acc[ix + 1]));
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(se + off), "r"(pk(d0 * s0, d1 * s1)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(se + C::EB / 2 + off), "r"(pk(d0 * ao.x * (1.f - s0), d1 * ao.y * (1.f - s1))) : "memory");
        hs[rb][(8 * j) / DH] += d0 * ao.x + d1 * ao.y;
      }
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int u = 0; u < NHB; ++u) {
        float v = hs[rb][u];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        if ((lane & 3) == 0) {
          const int row = m0 + rq + 8 * rb, head = n0 / DH + u;
          dd[((size_t)(row / L) * nh + head) * L + row % L] = v;
        }
      }
    fence_proxy_async();                                                         // do / dg (st.shared) -> the TMA stores
    named_bar_sync(1, 256);
    if (tid == 128) {
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        tma_st(&mdo, se + j * 16384, n0 + 64 * j, m0);
        tma_st(&mdg, se + C::EB / 2 + j * 16384, n0 + 64 * j, m0);
      }
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      mbar_arrive(efree);
    }
  }
  if (tid == 128) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// ------------------------------------------------------------------------------------------------ host
using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static CUtensorMap tmap(const at::Tensor& t, int box_cols, int box_rows) {
  static Encode enc = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "TMA encoder");
    return reinterpret_cast<Encode>(p);
  }();
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) % 8 == 0, "bog: bf16 [rows, cols] operand");
  CUtensorMap m{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)}, str[1] = {(cuuint64_t)t.stride(0) * 2};
  const cuuint32_t box[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows}, el[2] = {1, 1};
  TORCH_CHECK(enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, str, box, el, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
              "bog: TMA encode");
  return m;
}
template <class Kern>
static void launch(Kern kern, int M, int N, int bn, int smem, bool pdl, const CUtensorMap& ma, const CUtensorMap& mb, int K, int bpre, const EpiArgs& e,
                   const CUtensorMap* ma2 = nullptr) {
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount, tiles = (M / QM) * (N / bn);
  const dim3 grid((unsigned)std::min(tiles, nsm));
  TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "bog: shared memory");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kern, ma, mb, ma2 ? *ma2 : ma, M, N, K, bpre, e);
  TORCH_CHECK(err == cudaSuccess, "bog launch: ", cudaGetErrorString(err));
}
// output tiles of 128 x BN: the fewest rounds over the SMs, then the widest tile (per-tile time ~ k-blocks x (16 KB + BN 128 B))
static int pick_bn(int64_t M, int64_t N, int nsm, int dh) {
  int best = 0;
  double bc = 1e30;
  for (int bn : {256, 192, 128, 64}) {
    if (N % bn || (dh && bn % dh)) continue;
    const int64_t tiles = (M / QM) * (N / bn), rounds = (tiles + nsm - 1) / nsm;
    const double c = (double)rounds * (16384.0 + bn * 128.0) / ((double)bn);
    const double cost = c * bn;                                                  // per-round time ~ bytes per k-block
    if (cost < bc - 1e-9) { bc = cost; best = bn; }
  }
  TORCH_CHECK(best, "bog: no tile width for N = ", N);
  return best;
}

// out [M, 768] = x + sigmoid(gate[row % T]) (h Wsq^T); h [M, K] bf16, wsq [768, K] bf16, x [M, 768] bf16 / fp32, gate [T, >= 768]
// bf16 view, out bf16 / fp32 [M, 768] contiguous.
void sq_resout(at::Tensor h, at::Tensor wsq, at::Tensor x, at::Tensor gate, at::Tensor out, bool pdl, bool bpre, int64_t bn_force) {
  const at::cuda::CUDAGuard guard(h.device());
  const int64_t M = h.size(0), K = h.size(1), N = wsq.size(0), T = gate.size(0);
  TORCH_CHECK(N == 768 && wsq.size(1) == K && K % KB == 0 && M % QM == 0 && x.is_contiguous() && x.size(0) == M && x.size(1) == 768 &&
              out.is_contiguous() && out.size(0) == M && out.size(1) == 768 && gate.stride(1) == 1 && gate.scalar_type() == at::kBFloat16 &&
              T % QM == 0 && M % T == 0, "sq_resout: shapes");
  EpiArgs e{};
  e.x = x.data_ptr(); e.xbf = x.scalar_type() == at::kBFloat16;
  e.gate = (const bf*)gate.data_ptr(); e.gs = (int)gate.stride(0);
  e.out = out.data_ptr(); e.obf = out.scalar_type() == at::kBFloat16; e.os = 768; e.T = (int)T;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int bn = bn_force ? (int)bn_force : pick_bn(M, N, nsm, 0);
  const auto ma = tmap(h, 64, QM);
  switch (bn) {
    case 256: launch(gemm_epi<256, EPI_RESOUT, 0>, (int)M, 768, 256, Cfg<256>::SMEM, pdl, ma, tmap(wsq, 64, 256), (int)K, bpre, e); break;
    case 192: launch(gemm_epi<192, EPI_RESOUT, 0>, (int)M, 768, 192, Cfg<192>::SMEM, pdl, ma, tmap(wsq, 64, 192), (int)K, bpre, e); break;
    case 128: launch(gemm_epi<128, EPI_RESOUT, 0>, (int)M, 768, 128, Cfg<128>::SMEM, pdl, ma, tmap(wsq, 64, 128), (int)K, bpre, e); break;
    default: launch(gemm_epi<64, EPI_RESOUT, 0>, (int)M, 768, 64, Cfg<64>::SMEM, pdl, ma, tmap(wsq, 64, 64), (int)K, bpre, e); break;
  }
}

// dog = dy Wo (Wo [768, DA] as stored: MN-major B), then do = dog sigmoid(g), dg = dog ao (1 - sigmoid(g)), dd [S, nh, L] =
// per-head sums of dog ao. dy [M, 768] bf16; g [M, DA] bf16 view; ao [M, DA]; do / dg [M, DA] bf16 views.
void dog_gatebwd(at::Tensor dy, at::Tensor wo, at::Tensor g, at::Tensor ao, at::Tensor dout, at::Tensor dg, at::Tensor dd, int64_t L, int64_t nh,
                 bool pdl, int64_t bn_force) {
  const at::cuda::CUDAGuard guard(dy.device());
  const int64_t M = dy.size(0), K = dy.size(1), DA = wo.size(1), dh = DA / nh;
  TORCH_CHECK(K == 768 && wo.size(0) == 768 && wo.is_contiguous() && M % QM == 0 && M % L == 0 && DA % nh == 0 && dh % 8 == 0 &&
              g.stride(1) == 1 && ao.is_contiguous() && ao.size(1) == DA && dout.stride(1) == 1 && dout.stride(0) == DA && dg.stride(1) == 1 &&
              dd.scalar_type() == at::kFloat && dd.is_contiguous() && dd.numel() == M * nh, "dog_gatebwd: shapes");
  EpiArgs e{};
  e.gate = (const bf*)g.data_ptr(); e.gs = (int)g.stride(0);
  e.out = dout.data_ptr(); e.os = (int)DA; e.ao = (const bf*)ao.data_ptr(); e.dg = (bf*)dg.data_ptr(); e.dgs = (int)dg.stride(0);
  e.dd = dd.data_ptr<float>(); e.L = (int)L; e.nh = (int)nh; e.dh = (int)dh; e.T = (int)M;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int bn = bn_force ? (int)bn_force : pick_bn(M, DA, nsm, (int)dh);
  TORCH_CHECK(bn % dh == 0, "dog_gatebwd: whole heads per tile");
  const auto ma = tmap(dy, 64, QM), mb = tmap(wo, 64, 64);
#define GB(BN_, DH_) \
  if (bn == BN_ && dh == DH_) { launch(gemm_epi<BN_, EPI_GATEBWD, 1, DH_>, (int)M, (int)DA, BN_, Cfg<BN_>::SMEM, pdl, ma, mb, 768, 0, e); return; }
  GB(192, 48) GB(256, 32) GB(192, 32) GB(128, 32) GB(64, 32) GB(256, 64) GB(192, 64) GB(128, 64) GB(64, 64)
#undef GB
  TORCH_CHECK(false, "dog_gatebwd: no kernel for tile ", bn, " head width ", dh);
}
// The conditioning tables in one launch: out [T, N] bf16 = [cn | c] W^T + bias, cn = LN(c) for the columns below nsplit, the raw
// conditioning c above (W = [W1; W2] [N, 384] bf16, bias fp32 [N]); the weights are prefetched before the PDL wait.
void cond_tables(at::Tensor cn, at::Tensor c, at::Tensor w, at::Tensor bias, at::Tensor out, int64_t nsplit, bool pdl, int64_t bn_force) {
  const at::cuda::CUDAGuard guard(c.device());
  const int64_t T = c.size(0), K = c.size(1), N = w.size(0);
  TORCH_CHECK(T % QM == 0 && K % KB == 0 && w.size(1) == K && cn.sizes() == c.sizes() && bias.scalar_type() == at::kFloat && bias.numel() == N &&
              bias.is_contiguous() && out.is_contiguous() && out.size(0) == T && out.size(1) == N && out.scalar_type() == at::kBFloat16,
              "cond_tables: shapes");
  EpiArgs e{};
  e.out = out.data_ptr(); e.os = (int)N; e.bias = bias.data_ptr<float>(); e.nsplit = (int)nsplit;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int bn = bn_force ? (int)bn_force : pick_bn(T, N, nsm, 0);
  while (nsplit % bn) bn /= 2;
  const auto ma = tmap(cn, 64, QM), ma2 = tmap(c, 64, QM);
  switch (bn) {
    case 256: launch(gemm_epi<256, EPI_BIAS, 0>, (int)T, (int)N, 256, Cfg<256>::SMEM, pdl, ma, tmap(w, 64, 256), (int)K, 1, e, &ma2); break;
    case 192: launch(gemm_epi<192, EPI_BIAS, 0>, (int)T, (int)N, 192, Cfg<192>::SMEM, pdl, ma, tmap(w, 64, 192), (int)K, 1, e, &ma2); break;
    case 128: launch(gemm_epi<128, EPI_BIAS, 0>, (int)T, (int)N, 128, Cfg<128>::SMEM, pdl, ma, tmap(w, 64, 128), (int)K, 1, e, &ma2); break;
    default: launch(gemm_epi<64, EPI_BIAS, 0>, (int)T, (int)N, 64, Cfg<64>::SMEM, pdl, ma, tmap(w, 64, 64), (int)K, 1, e, &ma2); break;
  }
}

// dog_gatebwd with the TMA epilogue (dog_gate_tma): the same contract; g / dg may be column views of v | g / d(v | g)
void dog_gatebwd_tma(at::Tensor dy, at::Tensor wo, at::Tensor g, at::Tensor ao, at::Tensor dout, at::Tensor dg, at::Tensor dd, int64_t L,
                     int64_t nh, bool pdl) {
  const at::cuda::CUDAGuard guard(dy.device());
  const int64_t M = dy.size(0), DA = wo.size(1), dh = DA / nh;
  TORCH_CHECK(dy.size(1) == 768 && wo.size(0) == 768 && wo.is_contiguous() && M % QM == 0 && M % L == 0 && DA % nh == 0 &&
              ao.size(1) == DA && dout.size(1) == DA && g.size(1) == DA && dg.size(1) == DA && dd.scalar_type() == at::kFloat &&
              dd.is_contiguous() && dd.numel() == M * nh, "dog_gatebwd_tma: shapes");
  const auto ma = tmap(dy, 64, QM), mb = tmap(wo, 64, 64), mg = tmap(g, 64, QM), mao = tmap(ao, 64, QM), mdo = tmap(dout, 64, QM),
             mdg = tmap(dg, 64, QM);
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto go = [&](auto kern, int bn, int smem) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "bog: shared memory");
    const int tiles = (int)(M / QM) * (int)(DA / bn);
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)std::min(tiles, nsm));
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, kern, ma, mb, mg, mao, mdo, mdg, dd.data_ptr<float>(), (int)M, (int)DA, (int)L, (int)nh);
    TORCH_CHECK(err == cudaSuccess, "dog_gate_tma launch: ", cudaGetErrorString(err));
  };
  if (dh == 48) go(dog_gate_tma<192, 48>, 192, CfgE<192>::SMEM);
  else if (dh == 32) go(dog_gate_tma<128, 32>, 128, CfgE<128>::SMEM);
  else { TORCH_CHECK(dh == 64, "dog_gatebwd_tma: head width 32 / 48 / 64"); go(dog_gate_tma<128, 64>, 128, CfgE<128>::SMEM); }
}

// probes: the mainloop alone, C [M, N] = A B^T (tb 0, B [N, K]) or A B (tb 1, B [K, N]) with no epilogue
void mainloop(at::Tensor a, at::Tensor b, int64_t N, int64_t tb, int64_t bn, at::Tensor sink) {
  const at::cuda::CUDAGuard guard(a.device());
  const int64_t M = a.size(0), K = a.size(1);
  EpiArgs e{};
  e.out = sink.data_ptr();
  const auto ma = tmap(a, 64, QM);
#define ML(BN_)                                                                                                       \
  if (bn == BN_) {                                                                                                    \
    if (tb) launch(gemm_epi<BN_, EPI_NONE, 1>, (int)M, (int)N, BN_, Cfg<BN_>::SMEM, false, ma, tmap(b, 64, 64), (int)K, 0, e);   \
    else launch(gemm_epi<BN_, EPI_NONE, 0>, (int)M, (int)N, BN_, Cfg<BN_>::SMEM, false, ma, tmap(b, 64, BN_), (int)K, 0, e); \
    return;                                                                                                           \
  }
  ML(256) ML(192) ML(128) ML(64)
#undef ML
}
}  // namespace bog

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sq_resout", &bog::sq_resout);
  m.def("dog_gatebwd", &bog::dog_gatebwd);
  m.def("mainloop", &bog::mainloop);
  m.def("dog_gatebwd_tma", &bog::dog_gatebwd_tma);
  m.def("cond_tables", &bog::cond_tables);
}
