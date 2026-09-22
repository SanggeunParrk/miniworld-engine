from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'front_prefetch_lnpair.cu').read_text()
s=s.replace('#include "warp_primitives.cuh"','#define allsync allsync_256\n#include "warp_primitives.cuh"\n#undef allsync\nTMN_DEVI void allsync(){named_bar_sync(0,512);}')
ops=','.join('%%%d'%i for i in range(16));outs=','.join('"+f"(d[%d])'%i for i in range(16))
helper=''
for name,ta in [('mma_gate32',0),('mma_input32',1)]:
 helper+='TMN_DEVI void %s(float (&d)[16],uint64_t a,uint64_t b,int scale){asm volatile("{.reg .pred p;setp.ne.b32 p,%%18,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%s},%%16,%%17,p,1,1,%d,0;}" : %s : "l"(a),"l"(b),"r"(scale));}\n'%(name,ops,ta,outs)
a=s.index('TMN_DEVI void glu_small');s=s[:a]+helper+s[a:]
a=s.index('TMN_DEVI void glu_small');b=s.index('TMN_DEVI void load_dw',a);v=s[a:b].replace('#pragma unroll 8','#pragma unroll 4').replace('q<8','q<4').replace('q*256','q*512');s=s[:a]+v+s[b:]
a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b]
v=v.replace('float acc[64]','float acc[32]').replace('mma_weight128(acc','mma_weight64(acc').replace('s+40960+wi*8192','s+40960+(wi/2)*8192').replace('s+24576+k*2048','s+24576+(wi%2)*8192+k*2048').replace('),8192,1024,1)','),16,1024,1)')
v=v.replace('+wi*8192;', '+(wi/2)*8192+(wi%2)*64;').replace('static_for<16>','static_for<8>');s=s[:a]+v+s[b:]
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
v=v.replace('uint32_t gate_packed[16];float acc[32]','uint32_t gate_packed[8];float acc[16]').replace('float gate[32]','float gate[16]').replace('mma_dgrad(gate','mma_gate32(gate').replace('sm+16384+wi*16384+(k/4)*8192','sm+16384+(wi/2)*16384+(wi%2)*4096+(k/4)*8192')
v=v.replace('static_for<16>','static_for<8>').replace('mma_input64(acc','mma_input32(acc').replace('s+24576+wi*8192','s+24576+wi*4096').replace('s+wi*8192+k*32','s+wi*4096+k*32')
# Only the LN loops have qi as their compile-time outer index.
v=v.replace('static_for<4>([&](auto qi)','static_for<2>([&](auto qi)')
v=v.replace('gamma[wi*64+c]','gamma[wi*32+c]').replace('gamma[wi*64+c+1]','gamma[wi*32+c+1]').replace('gc=wi*64+c','gc=wi*32+c')
v=v.replace('get(lnsm+wi*8192,r,c)','get(lnsm+(wi/2)*8192,r,(wi%2)*32+c)').replace('get(lnsm+wi*8192,r,c+1)','get(lnsm+(wi/2)*8192,r,(wi%2)*32+c+1)')
for r in ['ra','rb']:
 for c in ['c','c+1']:
  v=v.replace('get(lnsm+wi*8192,%s,%s)'%(r,c),'get(lnsm+(wi/2)*8192,%s,(wi%%2)*32+%s)'%(r,c))
 v=v.replace('pair_get(lnsm+16384+wi*8192,%s,c)'%r,'pair_get(lnsm+16384+(wi/2)*8192,%s,(wi%%2)*32+c)'%r)
 v=v.replace('lnsm+wi*8192+swz128(%s,c*2)'%r,'lnsm+(wi/2)*8192+swz128(%s,((wi%%2)*32+c)*2)'%r)
v=v.replace('stats[ra*2]+stats[128+ra*2]','(stats[ra*2]+stats[128+ra*2])+(stats[256+ra*2]+stats[384+ra*2])').replace('stats[rb*2]+stats[128+rb*2]','(stats[rb*2]+stats[128+rb*2])+(stats[256+rb*2]+stats[384+rb*2])')
v=v.replace('stats[ra*2+1]+stats[128+ra*2+1]','(stats[ra*2+1]+stats[128+ra*2+1])+(stats[256+ra*2+1]+stats[384+ra*2+1])').replace('stats[rb*2+1]+stats[128+rb*2+1]','(stats[rb*2+1]+stats[128+rb*2+1])+(stats[256+rb*2+1]+stats[384+rb*2+1])')
v=v.replace('  running+=','  if(threadIdx.x<256)running+=').replace(' p.partln[split*256+threadIdx.x]=running;',' if(threadIdx.x<256)p.partln[split*256+threadIdx.x]=running;')
s=s[:a]+v+s[b:];s=s.replace('__launch_bounds__(256,2)','__launch_bounds__(512,2)').replace('for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)','for(int i=blockIdx.x*512+threadIdx.x;i<131328;i+=UCOUNT*512)')
name='front_prefetch_fourwg';(p/(name+'.cu')).write_text(s);c=json.loads((p/'front_kindprefetch.launch.json').read_text());c['threads']=512;(p/(name+'.launch.json')).write_text(json.dumps(c))
