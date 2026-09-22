from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'trimul_b1_shared_20260921';S=R.parent/'trimul_save_cost_20260921'
for p in [*OLD.glob('*.cuh'),*OLD.glob('*.inc')]: (R/p.name).write_bytes(p.read_bytes())
up=R.parent/'trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc'
t=(up/'tmn_kernels.cuh').read_text();a=t.index('template <int KS, bool SERIAL = false, int CLS = math::REF>\nTMN_DEVI LnStats ln_fragment');b=t.index('\n}',a)+2;ln=t[a:b].replace('ln_fragment(', 'ln_save_xhat(').replace('int lane, float eps)', 'int lane, float eps, void* output, int row, int M)')
needle='    const uint32_t k0 ='
insert='''    // Save pre-affine normalization without changing the forward operand.
    #pragma unroll
    for(int j=0;j<4;++j){int rr=row+8*(j&1),cc=ks*16+2*(lane&3)+8*(j>>1);float mu=(j&1)?meanB:meanA,rs=(j&1)?rB:rA;
     float a=__fmul_rn(__fsub_rn(bf16lo(fa[ks][j]),mu),rs),b=__fmul_rn(__fsub_rn(bf16hi(fa[ks][j]),mu),rs);
#if XHAT_FP32
     reinterpret_cast<float*>(output)[(size_t)cc*M+rr]=a;reinterpret_cast<float*>(output)[(size_t)(cc+1)*M+rr]=b;
#else
     reinterpret_cast<__nv_bfloat16*>(output)[(size_t)cc*M+rr]=__float2bfloat16_rn(a);reinterpret_cast<__nv_bfloat16*>(output)[(size_t)(cc+1)*M+rr]=__float2bfloat16_rn(b);
#endif
    }
'''
ln=ln.replace(needle,insert+needle)
s=(S/'save_k3.cu').read_text().replace('using Cfg=',ln+'\nusing Cfg=')
s=s.replace('LnStats stats_out=ln_fragment<KSP,MW_SERIAL>(fx,sGout,sBout,lane,p.eps);','LnStats stats_out=ln_save_xhat<KSP,MW_SERIAL>(fx,sGout,sBout,lane,p.eps,tp.lnout,iw*p.N+jw+16*wiw+(lane>>2),p.N*p.N);')
# mean is unnecessary; keep ABI field but do not store it.
s=s.replace('tp.mean_out[(size_t)iw*p.N+r]=stats_out.mA;','').replace('tp.mean_out[(size_t)iw*p.N+r+8]=stats_out.mB;','')
(R/'save_k3.cu').write_text(s)
# Clone forward adapter: no lnout TMA stores, custom scalar save above.
s=(S/'save_cost_core.py').read_text().replace('defs=dict(MW_SAVE_PG=int(pg),MW_PG_METHOD=method)','defs=dict(MW_SAVE_PG=int(pg),MW_PG_METHOD=method,XHAT_FP32=method)')
s=s.replace('MW_SAVE_OUT=(ln>>1)&1','MW_SAVE_OUT=0')
s=s.replace('xnout=e(256) if ln&2 else None','xnout=torch.empty((256,m),device=x.device,dtype=torch.float32 if method else x.dtype) if ln&2 else None')
s=s.replace('mo=f() if stats&2 else None','mo=None')
s=s.replace("om('xnout',256)","maps[-1]")
(R/'replace_core.py').write_text(s)
# B1: replace tri descriptor by xhat; no tri descriptor or pointer remains.
s=(OLD/'b1_fused.cu').read_text().replace('dy,x,tri,wp','dy,x,xhat,wp').replace('*gamma,*bo;','*gamma,*bo,*saved_rs;').replace('&p.tri','&p.xhat')
s=s.replace('#include "b1_lowreg.inc"','#include "xhat_read.inc"\n#include "b1_lowreg.inc"')
s=s.replace('tma_load_2d(dst+16384,&p.xhat,bar+slot,row,0);','''
#if XHAT_FP32
 for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+114688+k*16384+c*8192,&p.wg,bar+slot,c*64,k*64);
#else
 tma_load_2d(dst+16384,&p.xhat,bar+slot,row,0);
#endif
''')
a=s.index(' if(wi==1){',s.index('TMN_DEVI void prepare_shared'));b=s.index(' // Input x_n',a);s=s[:a]+s[b:]
needle=' fence_proxy_async();allsync();\n recompute_gemm<256,16384>'
insert=''' fence_proxy_async();allsync();
#if XHAT_FP32
 // Wg has finished: reuse its 32 KiB tile for the second half of xhat.
 if(threadIdx.x==0){mbar_arrive_expect_tx(bars+4,65536);tma_load_2d(x+16384,&p.xhat,bars+4,row,0);tma_load_2d(sm+114688,&p.xhat,bars+4,row+32,0);}
 mbar_wait(bars+4,phase);
#endif
 if(wi==1){
  int ra=w*16+lane/4,rb=ra+8;
  if(lane%4==0){float* rs=reinterpret_cast<float*>(sm+225280)+64;rs[ra]=p.saved_rs[row+ra];rs[rb]=p.saved_rs[row+rb];}
  #pragma unroll 4
  for(int k=0;k<16;++k){uint32_t f[4];int c=16*k+2*(lane&3);
   #pragma unroll
   for(int j=0;j<4;++j){int r=(j&1)?rb:ra,cc=c+8*(j>>1);float a=xhat_at(sm,slot,r,cc),b=xhat_at(sm,slot,r,cc+1);f[j]=pack_bf16(__fmaf_rn(a,ps[256+cc],ps[512+cc]),__fmaf_rn(b,ps[257+cc],ps[513+cc]));}
   stsm_x4(smem_u32(norm+(k/4)*8192)+swz128(w*16+lane%8+8*(mat&1),(2*(k%4)+(mat>>1))*16),f[0],f[1],f[2],f[3]);
  }
 }
 fence_proxy_async();allsync();
 recompute_gemm<256,16384>'''
assert needle in s;s=s.replace(needle,insert)
(R/'b1_fused.cu').write_text(s)
(R/'xhat_read.inc').write_text('''// Pre-affine xhat: one global load per element; no raw tri input.
static_assert(!PRODUCER && LOWREG && GATE_PHASE, "selected shared schedule only");
TMN_DEVI float xhat_at(uint8_t* sm,int slot,int row,int c){
#if XHAT_FP32
 uint8_t* p=row<32?sm+32768+slot*49152:sm+114688;
 return *reinterpret_cast<float*>(p+swz128(c,(row%32)*4));
#else
 return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(sm+32768+slot*49152+swz128(c,row*2)));
#endif
}
''')
s=(OLD/'b1_lowreg.inc').read_text()
s=s.replace('float mu[2]={mus[ra],mus[rb]},rs[2]', 'float rs[2]')
s=s.replace('uint32_t fx[4],dn[4];','uint32_t dn[4];').replace('uint32_t fx[4],dn[4],out[4];','uint32_t dn[4],out[4];')
import re
s=re.sub(r'   ldsm_x4_t\(fx,[^\n]*\n','',s)
s=s.replace('float xa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[rr]),rs[rr]);','float xa=xhat_at(sm,slot,rr?rb:ra,c),xb=xhat_at(sm,slot,rr?rb:ra,c+1);')
s=s.replace('float xaa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[0]),rs[0]);','float xaa=xhat_at(sm,slot,ra,c),xab=xhat_at(sm,slot,ra,c+1);')
s=s.replace('float xba=__fmul_rn(__fsub_rn(bf16lo(fx[j+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(fx[j+1]),mu[1]),rs[1]);','float xba=xhat_at(sm,slot,rb,c),xbb=xhat_at(sm,slot,rb,c+1);')
s=s.replace('   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);','''#if XHAT_FP32
   uint8_t* dst=wi==0?sm+16384+slot*49152:sm;
   stsm_x4_t(smem_u32(dst)+swz128(nl*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
#else
   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
#endif''')
s=s.replace('tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);','''tma_store_3d(&p.dtri,
#if XHAT_FP32
 (wi==0?sm+16384+slot*49152:sm)+(ch-wi*128)*128,
#else
 sx+ch*128,
#endif
 m0,ch,0);''')
# BF16 too: avoid overwriting xhat while other thread reads scalar element from another lane.
# Mapping reads each lane's own elements before corresponding stores as original.
(R/'b1_lowreg.inc').write_text(s)
s=(OLD/'plan.py').read_text().replace('def __init__(self,d,dy,tri,count=132,part=2,defines=None):','def __init__(self,d,dy,xhat,rstd,count=132,part=2,defines=None):\n  self.xhat,self.rstd=xhat,rstd').replace('self.bind(dy,tri)','self.bind(dy,xhat,rstd)')
s+='''
 def bind(self,dy,xhat,rstd):
  self.xhat,self.rstd=xhat,rstd;d=self.d;n=d['n'];m=n*n;L=T._launch_module();self.wpt=d['wp'].t().contiguous();self.wgt=d['wt'][4];es=xhat.element_size()
  tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  row=lambda t,c:tm(t,[64,64],[c,m],[c*2])
  dg,dwg,dt,dgo,dbo,dwp=self.outputs
  maps=[row(dy,128),row(d['x'],128),tm(xhat,[128//es,256],[m,256],[m*es]),tm(self.wpt,[64,64],[128,256],[256]),tm(self.wgt,[64,64],[128,128],[256]),tm(dt,[64,16,1],[m,256,1],[m*2,m*512]),row(dg,128)]
  self.p=L.Struct([*maps,d['ds'],d['gi'],d['bi'],d['go'],d['bo'],rstd,dg,dwg,dwp,dgo,dbo,self.partw,self.partln,self.counts,m,n,m//64])
  self.inputs=(dy,self.wpt,self.wgt,xhat,rstd)
'''
(R/'replace_plan.py').write_text(s)
print('generated')
