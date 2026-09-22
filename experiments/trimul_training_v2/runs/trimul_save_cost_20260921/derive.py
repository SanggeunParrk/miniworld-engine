from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
U=R.parent/'trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc/tmn_kernels.cuh'
s=U.read_text();a=s.index('template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false>');b=s.index('// ============================================================================================================ K3',a)
k=s[a:b];k=k.replace('void k1_body(const K1Params& p)', 'void save_front_body(const SaveFrontParams& tp)')
k=k.replace('  constexpr int CZ =', '  const K1Params& p=tp.base;\n  constexpr int CZ =',1)
k=k.replace('float* sGamma = reinterpret_cast<float*>(sStage + G::SMEM_STAGE);', 'uint8_t* sGate=sStage+G::SMEM_STAGE; uint8_t* sProj=sGate+G::SMEM_STAGE;\n  float* sGamma = reinterpret_cast<float*>(sStage + G::SMEM_STAGE*(1+2*MW_SAVE_PG*(MW_PG_METHOD==0)));')
mark='      uint32_t pk[4][2];'
assert k.count(mark)==1
pg='''
#if MW_SAVE_PG
      uint32_t pg[4][2],pp[4][2];
#pragma unroll
      for(int q=0;q<4;++q){pg[q][0]=pack_bf16(ac[4*q],ac[4*q+1]);pg[q][1]=pack_bf16(ac[4*q+2],ac[4*q+3]);pp[q][0]=pack_bf16(ac[4*(q+4)],ac[4*(q+4)+1]);pp[q][1]=pack_bf16(ac[4*(q+4)+2],ac[4*(q+4)+3]);}
#if MW_PG_METHOD == 0
      const uint32_t sg=smem_u32(sGate)+cw*8192+(b&1)*4096,sp=smem_u32(sProj)+cw*8192+(b&1)*4096;
      stsm_x4_t(sg+sts_off0,pg[0][0],pg[0][1],pg[1][0],pg[1][1]);stsm_x4_t(sg+sts_off1,pg[2][0],pg[2][1],pg[3][0],pg[3][1]);
      stsm_x4_t(sp+sts_off0,pp[0][0],pp[0][1],pp[1][0],pp[1][1]);stsm_x4_t(sp+sts_off1,pp[2][0],pp[2][1],pp[3][0],pp[3][1]);
#else
      // Reuse the original a/b stage, including read-completion before overwrite.
      const uint32_t sg=stage_u+(b&1)*4096;
      if(st_elect)tma_store_wait_read<0>();named_bar_sync(bar_id,128);
      stsm_x4_t(sg+sts_off0,pg[0][0],pg[0][1],pg[1][0],pg[1][1]);stsm_x4_t(sg+sts_off1,pg[2][0],pg[2][1],pg[3][0],pg[3][1]);
      fence_proxy_async();named_bar_sync(bar_id,128);
      if(st_elect){tma_store_3d(&tp.tm_gate,sStage+cw*8192+(b&1)*4096,jw,iw,32*b);tma_store_commit();tma_store_wait_read<0>();}named_bar_sync(bar_id,128);
      stsm_x4_t(sg+sts_off0,pp[0][0],pp[0][1],pp[1][0],pp[1][1]);stsm_x4_t(sg+sts_off1,pp[2][0],pp[2][1],pp[3][0],pp[3][1]);
      fence_proxy_async();named_bar_sync(bar_id,128);
      if(st_elect){tma_store_3d(&tp.tm_proj,sStage+cw*8192+(b&1)*4096,jw,iw,32*b);tma_store_commit();tma_store_wait_read<0>();}named_bar_sync(bar_id,128);
#endif
#endif
'''
k=k.replace(mark,pg+mark)
line='if (st_elect) { tma_store_3d(&p.tm_ab, sStage + cw * 8192 + (b & 1) * 4096, jw, iw, 32 * b); tma_store_commit(); }'
assert line in k
k=k.replace(line,'''if(st_elect){
#if MW_SAVE_PG && MW_PG_METHOD == 0
          tma_store_3d(&tp.tm_gate,sGate+cw*8192+(b&1)*4096,jw,iw,32*b);
          tma_store_3d(&tp.tm_proj,sProj+cw*8192+(b&1)*4096,jw,iw,32*b);
#endif
          tma_store_3d(&p.tm_ab,sStage+cw*8192+(b&1)*4096,jw,iw,32*b);tma_store_commit();}''')
head='''// SPDX-License-Identifier: Apache-2.0
// Anthropic native v5 K1 with optional BF16 projection/gate-logit stores.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
struct SaveFrontParams { K1Params base; CUtensorMap tm_gate,tm_proj; };
using Base=K1Cfg<128,256,false,2,64,8,2>;
struct SaveFrontCfg:Base {static constexpr int SMEM=Base::SMEM+2*Base::SMEM_STAGE*MW_SAVE_PG*(MW_PG_METHOD==0);};
'''
# The extracted section includes closing comments but remains within sm90 namespace.
foot='''\n}}\nextern "C" __global__ __launch_bounds__(tmn::sm90::SaveFrontCfg::NTHR,tmn::sm90::SaveFrontCfg::MINB)
void save_k1(__grid_constant__ const tmn::sm90::SaveFrontParams p){tmn::sm90::save_front_body<tmn::sm90::SaveFrontCfg,true,1,false,false>(p);}\n'''
(R/'save_k1.cu').write_text(head+k+foot)
P=R.parent/'trimul_ln_only_save_20260921/ln_only_k3.cu';s=P.read_text();s=s.replace('no activation saves.','optional LN statistics and output projection/gate saves.')
s=s.replace('__nv_bfloat16 *lnin,*lnout;','__nv_bfloat16 *lnin,*lnout; float *mean_in,*rs_in,*mean_out,*rs_out; CUtensorMap tm_proj,tm_gate; __nv_bfloat16 *proj,*gate;')
s=s.replace('float* sGin = reinterpret_cast<float*>(sOut + G::SMEM_OUT);','uint8_t* sProj=sOut+G::SMEM_OUT; uint8_t* sGate=sProj+G::SMEM_OUT;\n  float* sGin = reinterpret_cast<float*>(sOut + G::SMEM_OUT*(1+2*MW_SAVE_PG*(MW_PG_METHOD==0)));')
start=s.index('#ifndef TMN_DEV_NOLN',s.index('uint32_t fx[KSP][4];'));end=s.index('#endif',start)+len('#endif')
s=s[:start]+'''    LnStats stats_out=ln_fragment<KSP,MW_SERIAL>(fx,sGout,sBout,lane,p.eps);
#if MW_SAVE_STATS_OUT
    if((!SPLITN||cw==0)&&(lane&3)==0 && iw<p.N){int r=jw+16*wiw+(lane>>2);if(r<p.N){tp.mean_out[(size_t)iw*p.N+r]=stats_out.mA;tp.rs_out[(size_t)iw*p.N+r]=stats_out.rA;}if(r+8<p.N){tp.mean_out[(size_t)iw*p.N+r+8]=stats_out.mB;tp.rs_out[(size_t)iw*p.N+r+8]=stats_out.rB;}}
#endif'''+s[end:]
start=s.index('#ifndef TMN_DEV_NOLN',s.index('uint32_t fz[KSG][4];'));end=s.index('#endif',start)+len('#endif')
s=s[:start]+'''    LnStats stats_in=ln_fragment<KSG,MW_SERIAL>(fz,sGin,sBin,lane,p.eps);
#if MW_SAVE_STATS_IN
    if((!SPLITN||cw==0)&&(lane&3)==0 && iw<p.N){int r=jw+16*wiw+(lane>>2);if(r<p.N){tp.mean_in[(size_t)iw*p.N+r]=stats_in.mA;tp.rs_in[(size_t)iw*p.N+r]=stats_in.rA;}if(r+8<p.N){tp.mean_in[(size_t)iw*p.N+r+8]=stats_in.mB;tp.rs_in[(size_t)iw*p.N+r+8]=stats_in.rB;}}
#endif'''+s[end:]
s=s.replace('uint32_t fr[2][4], rz[2][4];','uint32_t fr[2][4], rz[2][4];\n#if MW_SAVE_PG && MW_PG_METHOD==0\n      uint32_t pp[2][4],gg[2][4];\n#endif')
needle='''          if (iw<p.N && jr<p.N) {
            const uint32_t ds='''
assert needle in s
s=s.replace(needle,'''#if MW_SAVE_PG
#if MW_PG_METHOD==0
          pp[j>>1][2*(j&1)+r]=pack_bf16(p0,p1);gg[j>>1][2*(j&1)+r]=pack_bf16(g0,g1);
#else
          if(iw<p.N && jr<p.N){stg32(tp.proj+((size_t)iw*p.N+jr)*CZ+c,pack_bf16(p0,p1));stg32(tp.gate+((size_t)iw*p.N+jr)*CZ+c,pack_bf16(g0,g1));}
#endif
#endif
          if (iw<p.N && jr<p.N) {
            const uint32_t ds=''')
needle='''      stsm_x4(stg_u+swz128((uint32_t)lrow,(uint32_t)((4*h+(mat>>1))*16)),fr[0][0],fr[0][1],fr[0][2],fr[0][3]);'''
assert needle in s
s=s.replace(needle,'''#if MW_SAVE_PG && MW_PG_METHOD==0
      const uint32_t ps=smem_u32(sProj)+(4*cw+wiw)*OB,gs=smem_u32(sGate)+(4*cw+wiw)*OB;
      stsm_x4(ps+swz128(lrow,(4*h+(mat>>1))*16),pp[0][0],pp[0][1],pp[0][2],pp[0][3]);stsm_x4(ps+swz128(lrow,(4*h+2+(mat>>1))*16),pp[1][0],pp[1][1],pp[1][2],pp[1][3]);
      stsm_x4(gs+swz128(lrow,(4*h+(mat>>1))*16),gg[0][0],gg[0][1],gg[0][2],gg[0][3]);stsm_x4(gs+swz128(lrow,(4*h+2+(mat>>1))*16),gg[1][0],gg[1][1],gg[1][2],gg[1][3]);
#endif
'''+needle)
needle='''        tma_store_3d(&p.tm_out, sOut + (4 * cw + wiw) * OB, BN * b0, jw + 16 * wiw, iw);'''
assert needle in s
s=s.replace(needle,'''#if MW_SAVE_PG && MW_PG_METHOD==0
        tma_store_3d(&tp.tm_proj,sProj+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
        tma_store_3d(&tp.tm_gate,sGate+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
#endif
'''+needle)
s=s.replace('void infer_k3(__grid_constant__','void save_k3(__grid_constant__')
(R/'save_k3.cu').write_text(s)
(R/'derivation.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (U,P)},indent=2))
print('Generated K1/K3 save experiments')
