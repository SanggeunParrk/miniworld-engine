from pathlib import Path
r=Path(__file__).resolve().parent
outs=','.join(f'%{i}' for i in range(64));args=','.join(f'"+f"(d[{i}])' for i in range(64))
head=f'''// NVIDIA WGMMA SS N128 operand contract; raw TMA buffers stay MN-major.
TMN_DEVI void mma_ss128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){{
 asm volatile("{{ .reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {{{outs}}}, %64, %65, p, 1, 1, 1, 1; }}"
 : {args} : "l"(a),"l"(b),"r"(accumulate));
}}
'''
s=(r/'wgrad_ws.cuh').read_text().replace('void wgrad_ws(','void wgrad_ws128(')
s=s.replace('bool wg=tile<4;int t=wg?tile:tile-4,ncols=wg?2:4,mch=(t/ncols)*64,nch=(t%ncols)*64;','bool wg=tile<2;int t=wg?tile:tile-2,mch=wg?t*64:(t/2)*64,nch=wg?0:(t%2)*128;')
s=s.replace('stage*32768','stage*57344')
a=s.index('   if(tid==0){mbar_arrive_expect_tx');b=s.index('   fence_proxy_async();',a)
s=s[:a]+'''   if(tid==0){mbar_arrive_expect_tx(raw+stage,wg?57344:32768);
    if(wg){for(int n=0;n<2;++n){tma_load_2d(buf+n*8192,&p.dy,raw+stage,n*64,row);tma_load_2d(buf+16384+n*8192,&p.gate,raw+stage,n*64,row);tma_load_2d(buf+32768+n*8192,&p.proj,raw+stage,n*64,row);}tma_load_2d(buf+49152,&p.xn,raw+stage,mch,row);}
    else{tma_load_2d(buf,&p.dy,raw+stage,mch,row);tma_load_2d(buf+16384,&p.gate,raw+stage,mch,row);for(int n=0;n<2;++n)tma_load_2d(buf+32768+n*8192,&p.norm,raw+stage,nch+n*64,row);}
   }
   sync_group();mbar_wait(raw+stage,phase);
   int j0=row%p.L;
   for(int i=tid;i<(wg?4096:2048);i+=128){int n=i/2048,r=(i%2048)/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
    uint32_t y=pair_get(buf+n*8192,r,c),g=pair_get(buf+16384+n*8192,r,c),v=wg?pair_get(buf+32768+n*8192,r,c):0,ds=ldg32(p.ds+jr*128+(wg?n*64:mch)+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
    pair_put(buf+n*8192,r,c,pack_bf16(wg?((ya*bf16lo(v))*ga)*(1.f-ga):ya*ga,wg?((yb*bf16hi(v))*gb)*(1.f-gb):yb*gb));
   }
'''+s[b:]
s=s.replace('float acc[32]={}','float acc[64]={}').replace('uint8_t* sa=wg?buf+24576:buf;uint8_t* sb=wg?buf:buf+24576;','uint8_t* sa=wg?buf+49152:buf;uint8_t* sb=wg?buf:buf+32768;').replace('mma_ss(acc,','mma_ss128(acc,').replace('smem_desc(smem_u32(sb+k*2048),16,1024,1)','smem_desc(smem_u32(sb+k*2048),8192,1024,1)')
a=s.index(' float* part=');s=s[:a]+''' float* part=p.partw+(tile*SPLITS+split)*8192;
#pragma unroll
 for(int q=0;q<16;++q){int r=w*16+lane/4,c=q*8+2*(lane%4);part[r*128+c]=acc[4*q];part[r*128+c+1]=acc[4*q+1];part[(r+8)*128+c]=acc[4*q+2];part[(r+8)*128+c+1]=acc[4*q+3];}
 if(ticket(p.counts+tile,SPLITS,last)){
  for(int i=tid;i<8192;i+=128){float v=0;for(int s=0;s<SPLITS;++s)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*SPLITS+s)*8192+i];int r=mch+i/128,c=nch+i%128;(wg?p.dwg:p.dwp)[r*(wg?128:256)+c]=__float2bfloat16_rn(v);}
  sync_group();if(tid==0)atomicExch(p.counts+tile,0u);
 }
 }
}
'''
(r/'wgrad_ws128.cuh').write_text(head+s)
p=r/'fused.cu';s=p.read_text().replace('#include "wgrad_ws.cuh"','#include "wgrad_ws.cuh"\n#include "wgrad_ws128.cuh"').replace('if(WSS==3)wgrad_ws','if(WSS==4)wgrad_ws128(p,sm,bar,&last[1]);else if(WSS==3)wgrad_ws');p.write_text(s)
p=r/'core.py';s=p.read_text().replace("+(R/'wgrad_ws.cuh').read_bytes()","+(R/'wgrad_ws.cuh').read_bytes()+(R/'wgrad_ws128.cuh').read_bytes()").replace('(65536 if wss==3 else','(114688 if wss==4 else 65536 if wss==3 else').replace('(4 if fullw else 12)','(6 if wss==4 else 4 if fullw else 12)').replace('1 if wss==3 else wgroups','1 if wss>=3 else wgroups');p.write_text(s)
