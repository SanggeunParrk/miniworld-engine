"""DX: two independent warpgroups, each owns a complete 64x256 LN tile.

DW unchanged. DX sacrifices intra-WG double buffering for independent WG
progress and no cross-WG row-statistic barrier. Two 64KiB slots remain, Wp
shared/resident; 16KiB warp partials and 4KiB running sums in the old hole.
"""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'dual_balanced.cu').read_text()

# Add a distinct 128-thread mask/gate implementation; DW keeps its existing
# 32-bit mask cache and register lifetime, so it cannot spill due to this.
start=s.index('struct MaskCycle')
end=s.index('// Anthropic WGMMA/ldmatrix',start)
extra=s[start:end]
extra=extra.replace('MaskCycle','MaskCycleDX').replace('mask_cycle','mask_cycle_dx').replace('gate_backward','gate_backward_dx')
extra=extra.replace('uint32_t b0,b1,b2,scale;', 'uint64_t b0,b1,b2;uint32_t scale;')
extra=extra.replace('uint32_t bits=0', 'uint64_t bits=0').replace('uint32_t bits=mi', 'uint64_t bits=mi')
extra=extra.replace('int i=threadIdx.x;i<1024;i+=256', 'int i=threadIdx.x%128;i<1024;i+=128')
extra=extra.replace('8*(i/256)', '8*(i/128)')
extra=extra.replace('uint32_t(lo!=0)', 'uint64_t(lo!=0)').replace('uint32_t(hi!=0)', 'uint64_t(hi!=0)')
extra=extra.replace('fence_proxy_async();allsync();', 'fence_proxy_async();sync_group();')
s=s[:end]+extra+s[end:]

start=s.index('TMN_DEVI void dual_dgrad')
end=s.index('// DW: two 96 KiB',start)
d=s[start:end]
d=d.replace('gam[threadIdx.x]=p.gamma[threadIdx.x];',
            'for(int c=tid;c<256;c+=128)gam[c]=p.gamma[c];')
d=d.replace('if(threadIdx.x<64)', 'if(tid<64)')
d=d.replace('allsync(); // Publish saved mean/rstd/gamma before either WG reads them.',
            'sync_group(); // Saved stats are private to this WG and its tile.')
d=d.replace('s1[2]={},s2[2]={};', 's1[2]={},s2[2]={},v1[2][2]={},v2[2][2]={};')
d=d.replace('fx[2][4][4],dn[2][4][4]', 'fx[4][4][4],dn[4][4][4]')
d=d.replace('static_for<2>([&](auto ni)', 'static_for<4>([&](auto ni)')
d=d.replace('int n=wi*2+nlocal;', 'int n=nlocal;').replace('int n=wi*2+nl;', 'int n=nl;')
d=d.replace('s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;',
            'v1[nlocal/2][rr]+=ha*xa+hb*xb;v2[nlocal/2][rr]+=ha+hb;')
a=d.index(' s1[0]=quad_sum')
b=d.index(' // 65536..73728',a)
d=d[:a]+''' // Preserve the two channel-half sum trees and BF16 dnorm rounding.
 float c1a=quad_sum(v1[0][0])/256.f+quad_sum(v1[1][0])/256.f;
 float c1b=quad_sum(v1[0][1])/256.f+quad_sum(v1[1][1])/256.f;
 float c2a=quad_sum(v2[0][0])/256.f+quad_sum(v2[1][0])/256.f;
 float c2b=quad_sum(v2[0][1])/256.f+quad_sum(v2[1][1])/256.f;
'''+d[b:]
d=d.replace('sm+65536)', 'sm+65536+wi*8192)')
d=d.replace('allsync(); // Publish all four warps\' parameter partials (and all stmatrix writes).',
            'sync_group(); // Publish this WG\'s four warps and stmatrix writes.')
d=d.replace('int c=threadIdx.x;float* red=reinterpret_cast<float*>(sm+229376);',
            'float* red=reinterpret_cast<float*>(sm+81920+wi*2048);\n for(int c=tid;c<256;c+=128){')
d=d.replace(' fence_proxy_async();sync_group();', ' }\n fence_proxy_async();sync_group();')
d=d.replace('int ch=wi*128;ch<(wi+1)*128;', 'int ch=0;ch<256;')
s=s[:start]+d+s[end:]

start=s.index('// DX: two64KiB')
end=s.index('extern "C" __global__ __launch_bounds__',start)
s=s[:start]+'''// Independent WG slots. Each slot is reused only after its own TMA store
// completes. WG1 also waits for the initial shared Wp load on full[0].
TMN_DEVI void issue_dx(const Params& p,uint8_t* sm,uint64_t* full,int slot,int tile,bool initial){
 if(threadIdx.x%128!=0)return;
 uint8_t* s=sm+slot*98304;int row=tile*64;
#pragma unroll
 for(int c=0;c<2;++c){tma_load_2d(s+c*8192,&p.dy,full+slot,c*64,row);tma_load_2d(s+16384+c*8192,&p.gate,full+slot,c*64,row);}
 tma_load_2d(s+32768,&p.tri,full+slot,row,0);
 if(initial)for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sm+163840+n*16384+k*8192,&p.wp,full+slot,k*64,n*64);
}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* full){
 int split=blockIdx.x-DWCOUNT,wi=threadIdx.x/128,tid=threadIdx.x%128;
 int first=split*2+wi,mi=0,round=0;
 MaskCycleDX mask=mask_cycle_dx<DXCOUNT*2>(p,first);
 if(split*2<p.tiles)mbar_wait(full,0); // Both WGs acquire resident Wp once.
 allsync(); // WG1 acquires Wp before WG0 may reuse full[0].
 for(int it=first;it<p.tiles;it+=DXCOUNT*2,++round){
  mbar_wait(full+wi,round&1);
  gate_backward_dx<false>(p,sm+wi*98304,it*64,mask,mi);
  dual_dgrad(p,sm,it*64,wi);
  bool next=it+DXCOUNT*2<p.tiles;
  if(next&&tid==0)mbar_arrive_expect_tx(full+wi,65536);
  sync_group(); // All readers/store issuers finished before local slot refill.
  if(next)issue_dx(p,sm,full,wi,it+DXCOUNT*2,false);
  if(++mi==mask.period)mi=0;
 }
 allsync(); // Each WG has completed its independent, possibly ragged loop.
 for(int j=threadIdx.x;j<512;j+=256){
  float* red=reinterpret_cast<float*>(sm+81920);
  p.partln[split*512+j]=red[j]+red[512+j];
 }
}
'''+s[end:]
s=s.replace('if(split<p.tiles)mbar_arrive_expect_tx(full,dw?98304:131072);',
            'if((dw?split:split*2)<p.tiles)mbar_arrive_expect_tx(full,dw?98304:131072);')
s=s.replace('if(split+stride<p.tiles)mbar_arrive_expect_tx(full+1,dw?98304:65536);',
            'if((dw?split+stride:split*2+1)<p.tiles)mbar_arrive_expect_tx(full+1,dw?98304:65536);')
s=s.replace(' // Publish initialized slot barriers',
            ' if(!dw)for(int i=threadIdx.x;i<1024;i+=256)reinterpret_cast<float*>(sm+81920)[i]=0;\n // Publish initialized slot barriers')
s=s.replace('if(split<p.tiles)issue_slot<false>(p,sm,full,0,split,true);',
            'if(threadIdx.x==0&&split*2<p.tiles)issue_dx(p,sm,full,0,split*2,true);')
s=s.replace('if(split+stride<p.tiles)issue_slot<false>(p,sm,full,1,split+stride,false);',
            'if(threadIdx.x==128&&split*2+1<p.tiles)issue_dx(p,sm,full,1,split*2+1,false);')
(r/'dual_wg_tiles.cu').write_text('// Experiment: independent full-channel DX warpgroups.\n'+s)
(r/'dual_wg_tiles.py').write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,"dual_wg_tiles")\n')
