"""Derive a no-save K3 from Anthropic's TMA/WGMMA inference pipeline.

Only the epilogue changes: preserve Miniworld training BF16 rounding, row
dropout and fused residual. Input and output LN remain fused, with no saves.
The independent K1/K3 development sources are never edited.
"""
from pathlib import Path
import hashlib
import json

R = Path(__file__).resolve().parent
source = R.parent / 'anthropic_ln_inference_20260919/k3.cu'
s = source.read_text()
s = s.replace('// Anthropic f4f62fa K3. Always fused output LN; optional shared x_n input.\n'
              '// Split-input path reads original residual from global, no training saves.',
              '// Anthropic f4f62fa K3; Miniworld training epilogue, no activation saves.\n'
              '// Derived by derive.py. Apache-2.0 upstream attribution retained.')
s = s.replace('using Cfg=K3Cfg',
              'struct RecomputeParams { K3Params base; const __nv_bfloat16* dropscale; };\nusing Cfg=K3Cfg')
s = s.replace('infer_k3_body(const K3Params& p) {',
              'infer_k3_body(const RecomputeParams& tp) {\n  const K3Params& p=tp.base;')
a = s.index('    auto stage = ')
b = s.index('    // vector pass over', a)
s = s[:a] + r'''    // Same BF16 projection/logit and FP32 epilogue as saved training K3.
    // Original input z stays in shared memory until the residual is consumed.
    auto stage = [&](float (&accP)[16], float (&accG)[16], int h, int b0) {
      static_assert(K3ST && MW_FUSED && !UPD, "BF16 fused training epilogue only");
      fence_regs(accP); fence_regs(accG);
      uint32_t fr[2][4], rz[2][4];
#pragma unroll
      for (int kb=0;kb<2;++kb) {
        ldsm_x4(rz[kb],sZ_u+(uint32_t)(b0>>1)*CHB+
          swz128((uint32_t)(rho0+lrow),(uint32_t)(((2*h+kb)*16+((mat&2)?8:0))*2)));
        zdep ^= rz[kb][0] ^ rz[kb][3];
      }
#pragma unroll
      for (int j=0;j<4;++j) {
        float v[4]={0.f,0.f,0.f,0.f};
#pragma unroll
        for (int r=0;r<2;++r) {
          const int jr=jw+16*wiw+gq+8*r, c=BN*(b0+h)+8*j+q2;
          const float p0=math::round_bf16(accP[4*j+2*r]);
          const float p1=math::round_bf16(accP[4*j+2*r+1]);
          const float g0=math::sigmoid(math::round_bf16(accG[4*j+2*r]));
          const float g1=math::sigmoid(math::round_bf16(accG[4*j+2*r+1]));
          if (iw<p.N && jr<p.N) {
            const uint32_t ds=ldg32(tp.dropscale+(size_t)jr*CZ+c);
            const uint32_t res=rz[j>>1][2*(j&1)+r];
            v[2*r]=fmaf(p0*g0,bf16lo(ds),bf16lo(res));
            v[2*r+1]=fmaf(p1*g1,bf16hi(ds),bf16hi(res));
          }
        }
        fr[j>>1][2*(j&1)]=pack_bf16(v[0],v[1]);
        fr[j>>1][2*(j&1)+1]=pack_bf16(v[2],v[3]);
      }
      stsm_x4(stg_u+swz128((uint32_t)lrow,(uint32_t)((4*h+(mat>>1))*16)),fr[0][0],fr[0][1],fr[0][2],fr[0][3]);
      stsm_x4(stg_u+swz128((uint32_t)lrow,(uint32_t)((4*h+2+(mat>>1))*16)),fr[1][0],fr[1][1],fr[1][2],fr[1][3]);
    };
''' + s[b:]
assert 'const tmn::K3Params p' in s
s = s.replace('const tmn::K3Params p', 'const tmn::sm90::RecomputeParams p')
(R/'no_save_k3.cu').write_text(s)
(R/'derivation.json').write_text(json.dumps(dict(
    source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    derived_sha256=hashlib.sha256(s.encode()).hexdigest(),
    changes=['no activation saves', 'shared-row dropout', 'training rounding epilogue'],
    upstream_revision='f4f62fa6592ae4938d49b1757bea0cfeff9f468e'), indent=2))
