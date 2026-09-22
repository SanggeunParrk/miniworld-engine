from pathlib import Path
import json
p=Path(__file__).resolve().parent
for both in [False,True]:
 s=(p/('front_prefetch_dxqueue.cu' if both else 'front_prefetch_lnpair.cu')).read_text()
 if both:s=s.replace('queue_next=DXCOUNT+atomicAdd(p.counts+2,1u)','queue_next=DXCOUNT+atomicAdd(p.counts+10,1u)')
 a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a)
 body=r'''TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 __shared__ int jobs[2];int group=blockIdx.x%8,split=blockIdx.x/8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int mid=((p.tiles+DW_SPLITS-1)/DW_SPLITS+1)/2,r=0;float acc[64]={};
 if(threadIdx.x==0){jobs[0]=split;jobs[1]=split+DW_SPLITS;}allsync();
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(;;++r){int slot=r&1,tile=jobs[slot];if(tile>=p.tiles)break;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc,smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),(r!=0&&r!=mid)||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(threadIdx.x==0){int next=2*DW_SPLITS+atomicAdd(p.counts+2+group,1u);jobs[slot]=next;if(next<p.tiles)load_dw(p,sm,b,slot,next*64,group);}
  if(r+1==mid){float* out=p.partw+((group*DW_SPLITS+split)*2)*16384+wi*8192;static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}allsync();
 }
 int segment=r>mid?1:0;float* out=p.partw+((group*DW_SPLITS+split)*2+segment)*16384+wi*8192;
 static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});
 if(segment==0){float* zero=p.partw+((group*DW_SPLITS+split)*2+1)*16384;for(int i=threadIdx.x;i<16384;i+=256)zero[i]=0.f;}
}
'''
 s=s[:a]+body+s[b:]
 s=s.replace('atomicExch(p.counts+2,0u);','')
 s=s.replace('atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);','for(int i=0;i<EXTRA_COUNTS;++i)atomicExch(p.counts+2+i,0u);atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);')
 extra=9 if both else 8;s='#define EXTRA_COUNTS %d\n'%extra+s;name='front_prefetch_'+('dualqueue' if both else 'dwqueue');(p/(name+'.cu')).write_text(s);c=json.loads((p/'front_kindprefetch.launch.json').read_text());c['extra_counts']=extra;(p/(name+'.launch.json')).write_text(json.dumps(c))
