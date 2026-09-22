from pathlib import Path
r=Path(__file__).resolve().parent;s=(r/'dual_ratio12.cu').read_text()
a=s.index('struct MaskCycle');b=s.index('template<bool DW> TMN_DEVI void cluster_b1',a)
s=s[:a]+r'''
// Lossless shared mask cache supports up to12 phases without register growth.
// DW slots occupy0..196608; DX slots/Wp leave73728..98304 free. Max12KiB.
struct MaskCycle{uint32_t scale,cache;int period;bool cached;};
template<int STRIDE,bool DW> TMN_DEVI MaskCycle mask_cycle(const Params& p,int first,uint8_t* sm){
 MaskCycle v={0,smem_u32(sm+(DW?196608:73728)),0,false};int a=64*STRIDE,b=p.L;while(b){int r=a%b;a=b;b=r;}
 v.period=p.L/a;v.cached=v.period<=12;
 if(v.cached&&first<p.tiles)for(int z=0;z<v.period;++z){uint32_t bits=0;int j0=((first+z*STRIDE)*64)%p.L;
  for(int i=threadIdx.x;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){uint32_t lo=dd[q]&65535u,hi=dd[q]>>16;int bit=8*(i/256)+2*q;bits|=(uint32_t(lo!=0)<<bit)|(uint32_t(hi!=0)<<(bit+1));v.scale|=lo|hi;}
  }
  // Each thread owns and later reads only its own mask word; no peer sync.
  sts32(v.cache+z*1024+threadIdx.x*4,bits);
 }
 return v;
}
'''+s[b:]
s=s.replace('uint32_t bits=mi==0?mask.b0:mi==1?mask.b1:mask.b2;', 'uint32_t bits=0;if(mask.cached)asm volatile("ld.shared.b32 %0,[%1];":"=r"(bits):"r"(mask.cache+mi*1024+threadIdx.x*4):"memory");')
s=s.replace('mask_cycle<DWCOUNT>(p,split)','mask_cycle<DWCOUNT,true>(p,split,sm)').replace('mask_cycle<DXCOUNT>(p,split)','mask_cycle<DXCOUNT,false>(p,split,sm)')
for dw,dx in [(10,23),(13,31),(41,91),(19,47),(4,7)]:
 name=f'dual_smaskratio_{dw}_{dx}';src=s.replace('#define DW_RATIO 1',f'#define DW_RATIO {dw}').replace('#define DX_RATIO 2',f'#define DX_RATIO {dx}')
 (r/(name+'.cu')).write_text(src)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,*args,**kwargs):\n        super().__init__(*args,source="'+name+'",**kwargs)\n')
