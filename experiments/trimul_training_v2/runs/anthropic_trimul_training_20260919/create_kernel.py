from pathlib import Path
import hashlib
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
u=E/'third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc/tmn_kernels.cuh'
s=u.read_text()
b=s[s.index('template <class G, int LNM, bool UPD = false>'):s.index('\n}  // namespace sm90',s.index('template <class G, int LNM, bool UPD = false>'))]
b=b.replace('template <class G, int LNM, bool UPD = false>\nTMN_DEVI void k3_body(const K3Params& p) {','template <class G>\nTMN_DEVI void k3_training_body(const TrainParams& tp) {\n  const auto& p = tp.base;\n  constexpr int LNM = 1; constexpr bool UPD = false;')
b=b.replace('constexpr bool R240 = (TMN_K3_REGS_24_240 != 0) && !G::ZF32 && !G::OUT32 && !G::RESG && !G::PRENORM;', 'constexpr bool R240 = MWK3_REGS == 240;')
b=b.replace('if ((TMN_K3_BULK_STORE != 0) && !G::ZF32 && !G::OUT32 && !G::RESG && !G::PRENORM) tma_prefetch_desc(&p.tm_out);','tma_prefetch_desc(&p.tm_out);')
b=b.replace('constexpr bool K3ST = (TMN_K3_BULK_STORE != 0) && !ZF32 && !OUT32 && !RESG && !PRENORM;','constexpr bool K3ST = true;')
a=b.index('#ifndef TMN_DEV_NOLN');e=b.index('#endif',a)+len('#endif')
b=b[:a]+'''    const LnStats stats = ln_fragment<KSP, MWK3_LNSERIAL>(fx, sGout, sBout, lane, p.eps);
    // Save exactly the BF16 operand consumed by WGMMA, plus FP32 LN stats.
    // A split-N tile has duplicate operands: only consumer 0 owns these stores.
    if (!SPLITN || cw == 0) {
      const int ja = jw + 16 * wiw + gq, jb = ja + 8;
      if (iw < p.N) {
        if ((lane & 3) == 0) {
          if (ja < p.N) { tp.mean[iw * p.N + ja] = stats.mA; tp.rstd[iw * p.N + ja] = stats.rA; }
          if (jb < p.N) { tp.mean[iw * p.N + jb] = stats.mB; tp.rstd[iw * p.N + jb] = stats.rB; }
        }
#pragma unroll
        for (int ks = 0; ks < KSP; ++ks) {
          const int c = 16 * ks + q2;
          if (ja < p.N) {
            stg32(tp.norm + ((size_t)iw * p.N + ja) * CH + c, fx[ks][0]);
            stg32(tp.norm + ((size_t)iw * p.N + ja) * CH + c + 8, fx[ks][2]);
          }
          if (jb < p.N) {
            stg32(tp.norm + ((size_t)iw * p.N + jb) * CH + c, fx[ks][1]);
            stg32(tp.norm + ((size_t)iw * p.N + jb) * CH + c + 8, fx[ks][3]);
          }
        }
      }
    }
''' +b[e:]
a=b.index('    // o = bf16(sigmoid(g) * p)');e=b.index('    // vector pass over',a)
b=b[:a]+'''    // Training rounding contract matches engine F567: BF16 projection/logit,
    // FP32 sigmoid/product/drop-scale/residual, then one BF16 output rounding.
    // Saves use BF16 (the established engine backward contract).
    auto stage = [&](float (&accP)[16], float (&accG)[16], int h, int b0) {
      fence_regs(accP); fence_regs(accG);
      uint32_t fr[2][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int jr = jw + 16 * wiw + gq + 8 * r;
          const int c = BN * (b0 + h) + 8 * j + q2;
          const float p0 = math::round_bf16(accP[4*j+2*r]);
          const float p1 = math::round_bf16(accP[4*j+2*r+1]);
          const float g0 = math::sigmoid(math::round_bf16(accG[4*j+2*r]));
          const float g1 = math::sigmoid(math::round_bf16(accG[4*j+2*r+1]));
          if (iw < p.N && jr < p.N) {
            const size_t off = ((size_t)iw * p.N + jr) * CZ + c;
            stg32(tp.proj + off, pack_bf16(p0, p1));
            stg32(tp.gate + off, pack_bf16(g0, g1));
            const uint32_t ds = ldg32(tp.dropscale + (size_t)jr * CZ + c);
            const uint32_t res = ldg32(reinterpret_cast<const __nv_bfloat16*>(p.zres) + off);
            v[2*r] = fmaf(p0 * g0, bf16lo(ds), bf16lo(res));
            v[2*r+1] = fmaf(p1 * g1, bf16hi(ds), bf16hi(res));
          }
        }
        fr[j >> 1][2*(j&1)] = pack_bf16(v[0], v[1]);
        fr[j >> 1][2*(j&1)+1] = pack_bf16(v[2], v[3]);
      }
      stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4*h+(mat>>1))*16)), fr[0][0],fr[0][1],fr[0][2],fr[0][3]);
      stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4*h+2+(mat>>1))*16)), fr[1][0],fr[1][1],fr[1][2],fr[1][3]);
    };
''' +b[e:]
p=E/'src/miniworld_engine/kernels/trimul_inproj/cuda'
p.mkdir(exist_ok=True)
(p/'__init__.py').write_text('"""Experimental CUDA TriMul derivatives of Anthropic inference kernels."""\n')
header='''// SPDX-License-Identifier: Apache-2.0
// Derived from Anthropic uplifting-biomolecular-modeling, revision
// f4f62fa6592ae4938d49b1757bea0cfeff9f468e, trimul/native/pkg/v5/csrc/tmn_kernels.cuh.
// Upstream SHA256: '''+hashlib.sha256(u.read_bytes()).hexdigest()+'''
// MiniWorld changes: prenormalized input, output LN saves, projection/gate saves,
// shared-row dropout and separate residual; preserves upstream TMA/WGMMA pipeline.
// Original vendored source is unchanged. See docs/anthropic-trimul-training.md.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
struct TrainParams {
  K3Params base;
  __nv_bfloat16 *norm, *proj, *gate;
  float *mean, *rstd;
  const __nv_bfloat16 *dropscale;
};
static_assert(sizeof(K3Params) == 768, "upstream ABI");
static_assert(sizeof(TrainParams) == 832, "training ABI");
struct TrainCfg : K3Cfg<MWK3_CZ, MWK3_CH, 0, MWK3_BI, MWK3_BJ, MWK3_NSLOT, MWK3_NACC> {
  static constexpr bool PRENORM = true;
};
'''
(p/'anthropic_k3_training.cu').write_text(header+b+'''
}} // tmn::sm90
extern "C" __global__ __launch_bounds__(384, 1)
void mw_k3_train(__grid_constant__ const tmn::sm90::TrainParams p) {
  tmn::sm90::k3_training_body<tmn::sm90::TrainCfg>(p);
}
''')
print(p/'anthropic_k3_training.cu')
