from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_onewg.cu').read_text();ops=','.join('%%%d'%i for i in range(64));outs=','.join('"+f"(d[%d])'%i for i in range(64))
helper='''template<int AO,int BO> TMN_DEVI void mma_weight_off(float (&d)[64],uint32_t al,uint32_t ah,uint32_t bl,uint32_t bh,int scale){
 asm volatile("{.reg .pred p;.reg .b32 la,lb;.reg .b64 ad,bd;setp.ne.b32 p,%%68,0;add.u32 la,%%64,%%69;add.u32 lb,%%66,%%70;mov.b64 ad,{la,%%65};mov.b64 bd,{lb,%%67};wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%s},ad,bd,p,1,1,0,1;}" : %s : "r"(al),"r"(ah),"r"(bl),"r"(bh),"r"(scale),"n"(AO>>4),"n"(BO>>4));
}
template<int OFF> TMN_DEVI void store_off(float* base,float a,float b){asm volatile("{.reg .b64 p;add.u64 p,%%0,%%3;st.global.v2.f32 [p],{%%1,%%2};}"::"l"(base),"f"(a),"f"(b),"n"(OFF):"memory");}
'''%(ops,outs)
a=s.index('struct Params');s=s[:a]+helper+s[a:];a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void input_role',a);f=s[a:b]
f=f.replace('constexpr int kind=decltype(qi)::value;fence_regs(acc[kind]);','constexpr int kind=decltype(qi)::value;uint64_t ad=smem_desc(smem_u32(sm+40960+kind*8192),16,1024,1),bd=smem_desc(smem_u32(sm+24576),8192,1024,1);fence_regs(acc[kind]);')
f=f.replace('mma_weight128(acc[kind],smem_desc(smem_u32(sm+40960+kind*8192+k*32),16,1024,1),smem_desc(smem_u32(sm+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);','mma_weight_off<k*32,k*2048>(acc[kind],uint32_t(ad),uint32_t(ad>>32),uint32_t(bd),uint32_t(bd>>32),r%segmentRounds>0||k>0);')
f=f.replace('kind*8192;static_for<16>','kind*8192;float* dst=out+(w*16+lane/4)*128+2*(lane%4);static_for<16>')
f=f.replace('int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[kind][q*4],acc[kind][q*4+1]);stg64f(out+(rr+8)*128+c,acc[kind][q*4+2],acc[kind][q*4+3]);','store_off<q*8*4>(dst,acc[kind][q*4],acc[kind][q*4+1]);store_off<(8*128+q*8)*4>(dst,acc[kind][q*4+2],acc[kind][q*4+3]);')
s=s[:a]+f+s[b:];s=s.replace('#pragma unroll 2','#pragma unroll 1');(p/'front_onewg_lowreg.cu').write_text(s);(p/'front_onewg_lowreg.launch.json').write_text((p/'front_onewg.launch.json').read_text())
